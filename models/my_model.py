import torch
from models.Momu import MoMu
from models.fp_emb import FPMLP
from models.smiles_emb import SmilesModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer

import rdkit.Chem as Chem
import numpy as np
import torch.optim as optim
from models.graphormer import GraphormerModel, GraphormerEncoder
import argparse


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class GraphEncoder(nn.Module):

    def __init__(self, config=None,graph_model="bi_graphormer", bra_size=7):
        super().__init__()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_model=graph_model
        self.bra_size=bra_size
        self.graph2d_encoder = MoMu(config["graph"]).graph_encoder
        args = argparse.Namespace()
        args.dropout = 0.2
        args.attention_dropout = 0.15
        args.act_dropout = 0.05
        args.encoder_ffn_embed_dim = 2048
        args.encoder_layers = 12
        args.encoder_attention_heads = 16
        args.encoder_embed_dim = 512
        args.share_encoder_input_output_embed = False
        args.no_token_positional_embeddings = True
        args.apply_graphormer_init = True
        args.activation_fn = "relu"
        args.encoder_normalize_before = False
        args.tokens_per_sample = 35
        args.num_atoms = 512 * 9
        args.num_in_degree = 512
        args.num_out_degree = 512
        args.num_edges = 512 * 3
        args.num_spatial = 512
        args.num_edge_dis = 128
        args.edge_type = "multi_hop"
        args.multi_hop_max_dist = 0
        args.pre_layernorm = False
        args.num_classes = 1
        args.max_nodes = 768
        args.graph_model = self.graph_model
        args.bra_size=self.bra_size
        encoder = GraphormerEncoder(args)
        self.graphormer = GraphormerModel(args=args, encoder=encoder, graph_model=self.graph_model)

        for param in self.graph2d_encoder.parameters():
            param.requires_grad = False

        self.num_features = 400
        self.hidden_size = 768
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)

    def forward(self, mol, graphormer_datas=None):
        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model == 'no_graphormer_mpnn' or self.graph_model=='original_graphormer_mpnn':
            graph_feats, node_feats, node_feats_mask = self.graph2d_encoder(mol)
        # concatenated = graph_feats
        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model=='original_graphormer_mpnn' or self.graph_model=='bi_graphormer_no_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            graphormer_datas["x"].to("cuda:0")
            graphormer_datas["in_degree"].to("cuda:0")
            graphormer_datas["out_degree"].to("cuda:0")
            graphormer_datas["attn_bias"].to("cuda:0")
            graphormer_datas["spatial_pos"].to("cuda:0")
            graphormer_datas["edge_input"].to("cuda:0")
            graphormer_datas["attn_edge_type"].to("cuda:0")
            graphormer_output = self.graphormer(graphormer_datas)
            # print(f"After graphormer, output size: {graphormer_output.shape}")
            graphormer_output_flat = graphormer_output.squeeze(-1)
            # print(f"After graphormer, post process 1: {graphormer_output_flat.shape}")
            if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
                graphormer_output_flat = graphormer_output_flat.squeeze(1)
                graphormer_output_flat = graphormer_output_flat.transpose(0,1)
            # print(f"After graphormer, post process 2: {graphormer_output_flat.shape}")
            linear_layer = nn.Linear(graphormer_output_flat.size(1), 768).to("cuda:0")
            tensor2_expanded = linear_layer(graphormer_output_flat)
            # print(f"After graphormer, post process 3 & final output: {tensor2_expanded.shape}")
            # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
            # Concatenate along the feature dimension (dimension 1)
        
        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model=='original_graphormer_mpnn':
            # print(graph_feats.shape)
            # print(tensor2_expanded.shape)
            return torch.cat((graph_feats, tensor2_expanded), dim=1)
        elif self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            # print(len(graphormer_datas["x"]))
            return tensor2_expanded
        elif self.graph_model == 'no_graphormer_mpnn':
            return graph_feats


class FpEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        input_size = 2048  # 假设我们的指纹长度为 2048
        hidden_size = 768  # 隐藏层大小
        output_size = 768  # 输出大小，根据你的任务来定
        self.fp_encoder = FPMLP(input_size, hidden_size, output_size)
        self.num_features = 768
        self.hidden_size = 768
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)

    def get_fp_from_mol(self, mol):
        fingerprints = torch.randn(1, 2048)
        return fingerprints

    def forward(self, mol):
        # fp = self.get_fp_from_mol(self, mol)
        fp_embeddings = self.fp_encoder(mol)
        # fp_embeddings = self.fc_hidden(fp_embeddings)
        return fp_embeddings


class SmilesEncoder(nn.Module):
    def __init__(self, config=None, vocab=None):
        super().__init__()
        self.config = config
        self.num_features = 2362
        self.hidden_size = 768
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)
        self.smiles_encoder = SmilesModule(config=self.config, vocab=vocab)

    def forward(self, mol):
        smiles_feats = self.smiles_encoder(mol)
        smiles_feats = self.fc_hidden(smiles_feats)
        return smiles_feats


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


# Combined Model
class CombinedModel(nn.Module):
    def __init__(self, gcn, bert, fingerprint_nn):
        super(CombinedModel, self).__init__()
        self.gcn = gcn
        self.bert = bert
        self.fingerprint_nn = fingerprint_nn
        self.fc = nn.Linear(256 + 768 + 128, 128)  # Adjust based on output sizes

    def forward(self, graph_data, smiles, fingerprints):
        gcn_out = self.gcn(graph_data.x, graph_data.edge_index)
        smiles_out = self.bert(smiles)
        fingerprint_out = self.fingerprint_nn(fingerprints)
        combined = torch.cat([gcn_out, smiles_out, fingerprint_out], dim=1)
        return self.fc(combined)


# # Training loop
# criterion = ContrastiveLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# labels = torch.tensor([0, 1])  # 0 for negative pair, 1 for positive pair

# for epoch in range(100):
#     optimizer.zero_grad()
#     output1 = model(graph_data, smiles, fingerprints)
#     output2 = model(graph_data, smiles, fingerprints)
#     loss = criterion(output1, output2, labels)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch}, Loss: {loss.item()}")
