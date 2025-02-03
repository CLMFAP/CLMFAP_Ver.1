import os

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from models.my_model import *
import json
import pandas as pd
from ogb.utils.mol import smiles2graph
from typing import Callable
from torch_geometric.data import InMemoryDataset, Data
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from args import *
from utils.utils import *
from utils.data_utils import *
from graphormer_data.wrapper import preprocess_item
from graphormer_data.collator import collator
import regex as re
# Define the Model
# from args import args
import argparse
from models.pubchem_encoder import Encoder
import math
from nt_xent import NTXentLoss
from nt_xent_3 import NTXentLoss3
from nt_logistic import NTLogisticLoss
from margin_triplet_loss import MarginTriplet
from BCEWithLogits_Loss import BCEWithLogitsLoss
from functools import lru_cache


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Home device: {}".format(device))
torch.multiprocessing.set_sharing_strategy('file_system')


# Get args:
main_args = parse_args()
config_path = "configs/config.json"
config = json.load(open(config_path))
if main_args.batch_size:
    config["pretrain"]["batch_size"] = main_args.batch_size

print(config["pretrain"]["batch_size"])
result_path = f'{config["result"]["pretrain_path"]}_{main_args.graph_model}_{main_args.save_model_folder}'

if main_args.data_path:
    data_path = main_args.data_path
else:
    data_path = config['data']['pretrain_path']

print(f"Data path: {data_path}")
ensure_folder_exists(result_path)

nt_xent_criterion = NTXentLoss(device, config["pretrain"]["batch_size"], 0.1, True)
nt_logistic_criterion = NTLogisticLoss(device, config["pretrain"]["batch_size"], 0.1, True)
margin_triplet_criterion = MarginTriplet()
nt_xent_3_mean_criterion = NTXentLoss3(device, config["pretrain"]["batch_size"], 0.1, True,"mean")
nt_xent_3_random_criterion = NTXentLoss3(device, config["pretrain"]["batch_size"], 0.1, True,"random")
nt_xent_3_sum_criterion = NTXentLoss3(device, config["pretrain"]["batch_size"], 0.1, True,"sum")
bce_with_logits_criterion = BCEWithLogitsLoss(device, config["pretrain"]["batch_size"], True)

# Combined dataset class for SMILES, fingerprints, and graphs
class CombinedMoleculeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data["canonical_smiles"] = data["SMILES"].apply(
            lambda smi: normalize_smiles(smi, canonical=True, isomeric=False)
        )

        self.smiles_list_good = self.data.dropna()
        self.smiles_list_good.reset_index(drop=True, inplace=True)

        assert (
            self.smiles_list_good["canonical_smiles"].isna().sum() == 0
        ), "There are still NaN values in 'canonical_smiles'"
        print(
            "Dropped {} invalid smiles".format(
                len(self.data) - len(self.smiles_list_good)
            )
        )
        print("Valid smiles {}".format(len(self.smiles_list_good)))

        self.smiles_list = self.smiles_list_good["SMILES"]
        self.text_encoder = Encoder(1000)

        print("Loading data...")
        print(f"Fingerprint using radius {main_args.fp_radius}")
        self.encodings = self.text_encoder.process(self.smiles_list)
        self.fingerprints = generate_fingerprints(self.smiles_list, main_args.fp_radius)
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='original_graphormer_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn' or main_args.graph_model=='only_original_graphormer':
            self.graphs, self.graphormer_datas, self.graphs_remove_subgraph = generate_3d_graphs_pretrain(self.smiles_list)
        elif main_args.graph_model == 'no_graphno_graphormer_mpnnormer':
            self.graphs = generate_3d_graphs_no_graphormer(self.smiles_list)

    def __len__(self):
        return len(self.encodings[0])

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[0][idx]

        item["fingerprints"] = self.fingerprints[idx]
        item["graphs"] = self.graphs[idx]
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='original_graphormer_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn' or main_args.graph_model=='only_original_graphormer':
            self.graphormer_datas[idx].idx = idx
            item["graphormer_datas"] = self.graphormer_datas[idx]
            if main_args.save_model_folder and ( "remove_sub" in main_args.save_model_folder):
                self.graphs_remove_subgraph[idx].idx = idx
                item["graphormer_datas_remove_subgraph"] = self.graphs_remove_subgraph[idx]
        return item

class MolecularEmbeddingModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50):
        super(MolecularEmbeddingModel, self).__init__()
        cur_config = main_args
        text_encoder = Encoder(cur_config.max_len)
        vocab = text_encoder.char2id
        self.bert = SmilesEncoder(cur_config, vocab)
        self.fp_embedding = FpEncoder()
        self.graph_embedding = GraphEncoder(config=config["mpnn"], graph_model=main_args.graph_model, bra_size=main_args.bra_size, graph_n_head=main_args.graph_n_head, graph_n_layer=main_args.graph_n_layer)
        # self.cls_embedding = nn.Linear(
        #     self.bert.hidden_size, 128
        # )  # Added to match combined embedding size
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model=='original_graphormer_mpnn':
            self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        elif main_args.graph_model == 'no_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
            self.mid_liner_graph = nn.Linear(1 * 768, 1 * 768)


    def forward(self, input_ids, fingerprints, graphs, graphormer_datas=None, graphormer_datas_remove_subgraph=None):

        # Smiles embedding
        logits = self.bert(input_ids)
        cls_output = logits[:, 0, :]

        # Fingerprint embedding
        fp_embedding = self.fp_embedding(fingerprints)

        # 3D Graph embedding
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='original_graphormer_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
            graph_embedding = self.graph_embedding(graphs, graphormer_datas)

            if graphormer_datas_remove_subgraph:
                graph_embedding_1 = self.graph_embedding(graphs, graphormer_datas_remove_subgraph)

                graph_embedding = self.mid_liner_graph(graph_embedding)
                graphormer_datas_remove_subgraph = self.mid_liner_graph(graph_embedding_1)
                return cls_output, fp_embedding, graph_embedding, graphormer_datas_remove_subgraph
        elif main_args.graph_model == 'no_graphormer_mpnn':
            graph_embedding = self.graph_embedding(graphs)
        graph_embedding = self.mid_liner_graph(graph_embedding)

        # print(cls_output.shape)
        # print(fp_embedding.shape)
        # print(graph_embedding.shape)

        return cls_output, fp_embedding, graph_embedding

def contrastive_loss(smiles_emb, fingerprint_emb, graph_emb, graph_embedding_remove_subgraph=None,save_model_folder=None, temperature=0.1):
    
    if main_args.loss_function and main_args.loss_function == "nt_xent":
        loss_function = nt_xent_criterion
    elif main_args.loss_function and main_args.loss_function == "loss_nt_logistic":
        loss_function = nt_logistic_criterion
    elif main_args.loss_function and main_args.loss_function == "loss_margin_triplet":
        loss_function = margin_triplet_criterion
    else:
        loss_function = nt_xent_criterion

    smiles_emb = F.normalize(smiles_emb, dim=1)
    fingerprint_emb = F.normalize(fingerprint_emb, dim=1)
    graph_emb = F.normalize(graph_emb, dim=1)


    if not save_model_folder or save_model_folder == "original":
        smiles_graph_loss = loss_function(smiles_emb, graph_emb)
        smiles_fingerprint_loss = loss_function(smiles_emb, fingerprint_emb)
        fingerprint_graph_loss = loss_function(fingerprint_emb, graph_emb)

        if main_args.skip_modal == 'smiles':
            return fingerprint_graph_loss
        elif main_args.skip_modal == 'fp':
            return smiles_graph_loss
        elif main_args.skip_modal == 'graph':
            return smiles_fingerprint_loss

        return (
            main_args.smiles_graph_weight * smiles_graph_loss
            + main_args.smiles_fp_weight * smiles_fingerprint_loss
            + main_args.graph_fp_weight * fingerprint_graph_loss
        )
    elif graph_embedding_remove_subgraph is not None and save_model_folder == "remove_sub_sf_gg":
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)
        sf_loss = loss_function(smiles_emb, fingerprint_emb)
        gg_loss = loss_function(graph_emb, graph_emb_remove_subgraph)
        return (
            0.5 * sf_loss
            + 0.5 * gg_loss
        )
    elif graph_embedding_remove_subgraph is not None and save_model_folder == "remove_sub_sfgg":
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

        smiles_graph_loss = loss_function(smiles_emb, graph_emb)
        smiles_fingerprint_loss = loss_function(smiles_emb, fingerprint_emb)
        fingerprint_graph_loss = loss_function(fingerprint_emb, graph_emb)
        gg_loss = loss_function(graph_emb, graph_emb_remove_subgraph)
        return (
            0.25 * smiles_graph_loss
            + 0.25 * smiles_fingerprint_loss
            + 0.25 * fingerprint_graph_loss
            + 0.25 * gg_loss
        )
    elif graph_embedding_remove_subgraph is not None and save_model_folder == "nt_xent_3_mean_remove_sub":
        # print("$$$$$$$$$$$  nt_xent_3_mean_remove_sub ")
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

        smiles_GRA_loss = nt_xent_3_mean_criterion(smiles_emb, graph_emb, graph_emb_remove_subgraph)
        fingerprint_GRA_loss = nt_xent_3_mean_criterion(fingerprint_emb, graph_emb, graph_emb_remove_subgraph)

        smiles_fingerprint_loss = nt_xent_criterion(smiles_emb, fingerprint_emb)
        gg_loss = nt_xent_criterion(graph_emb, graph_emb_remove_subgraph)
        return (
            0.25 * smiles_GRA_loss
            + 0.25 * smiles_fingerprint_loss
            + 0.25 * fingerprint_GRA_loss
            + 0.25 * gg_loss
        )

    elif graph_embedding_remove_subgraph is not None and save_model_folder == "nt_xent_3_random_remove_sub":
        # print("$$$$$$$$$$$  nt_xent_3_random_remove_sub ")
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

        smiles_GRA_loss = nt_xent_3_random_criterion(smiles_emb, graph_emb, graph_emb_remove_subgraph)
        fingerprint_GRA_loss = nt_xent_3_random_criterion(fingerprint_emb, graph_emb, graph_emb_remove_subgraph)

        smiles_fingerprint_loss = nt_xent_criterion(smiles_emb, fingerprint_emb)
        gg_loss = nt_xent_criterion(graph_emb, graph_emb_remove_subgraph)
        return (
            0.25 * smiles_GRA_loss
            + 0.25 * smiles_fingerprint_loss
            + 0.25 * fingerprint_GRA_loss
            + 0.25 * gg_loss
        )

    elif graph_embedding_remove_subgraph is not None and save_model_folder == "nt_xent_3_sum_remove_sub":
        # print("$$$$$$$$$$$  nt_xent_3_sum_remove_sub ")
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

        smiles_GRA_loss = nt_xent_3_sum_criterion(smiles_emb, graph_emb, graph_emb_remove_subgraph)
        fingerprint_GRA_loss = nt_xent_3_sum_criterion(fingerprint_emb, graph_emb, graph_emb_remove_subgraph)

        smiles_fingerprint_loss = nt_xent_criterion(smiles_emb, fingerprint_emb)
        gg_loss = nt_xent_criterion(graph_emb, graph_emb_remove_subgraph)
        return (
            0.25 * smiles_GRA_loss
            + 0.25 * smiles_fingerprint_loss
            + 0.25 * fingerprint_GRA_loss
            + 0.25 * gg_loss
        )

    elif graph_embedding_remove_subgraph is not None and "bce_with_logits_remove_sub" in save_model_folder:
        # print("$$$$$$$$$$$  bce_with_logits_remove_sub ")
        graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

        smiles_GRA_loss = bce_with_logits_criterion(smiles_emb, graph_emb, graph_emb_remove_subgraph)
        fingerprint_GRA_loss = bce_with_logits_criterion(fingerprint_emb, graph_emb, graph_emb_remove_subgraph)

        smiles_fingerprint_loss = nt_xent_criterion(smiles_emb, fingerprint_emb)
        gg_loss = nt_xent_criterion(graph_emb, graph_emb_remove_subgraph)
        return (
            0.25 * smiles_GRA_loss
            + 0.25 * smiles_fingerprint_loss
            + 0.25 * fingerprint_GRA_loss
            + 0.25 * gg_loss
        )

    # elif graph_emb_remove_subgraph and save_model_folder == "loss_nt_logistic":
    #     graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

    # elif graph_emb_remove_subgraph and save_model_folder == "loss_margin_triplet":
    #     graph_emb_remove_subgraph = F.normalize(graph_embedding_remove_subgraph, dim=1)

    

    smiles_graph_loss = loss_function(smiles_emb, graph_emb)
    smiles_fingerprint_loss = loss_function(smiles_emb, fingerprint_emb)
    fingerprint_graph_loss = loss_function(fingerprint_emb, graph_emb)

    if main_args.skip_modal == 'smiles':
        return fingerprint_graph_loss
    elif main_args.skip_modal == 'fp':
        return smiles_graph_loss
    elif main_args.skip_modal == 'graph':
        return smiles_fingerprint_loss

    return (
        main_args.smiles_graph_weight * smiles_graph_loss
        + main_args.smiles_fp_weight * smiles_fingerprint_loss
        + main_args.graph_fp_weight * fingerprint_graph_loss
    )

def load_cluster_data(cluster_data):
    train_val_data, test_data = train_test_split(
        cluster_data, test_size=0.1, random_state=42
    )

    # Split train+validation into train and validation data
    train_data, valid_data = train_test_split(
        train_val_data, test_size=0.1111, random_state=42
    )

    train_dataset = CombinedMoleculeDataset(data=train_data)
    valid_dataset = CombinedMoleculeDataset(data=valid_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["pretrain"]["num_workers"],
        drop_last=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["pretrain"]["num_workers"],
        drop_last=True,
        persistent_workers=True
    )

    return train_dataloader, valid_dataloader

# @lru_cache(maxsize=None)
def load_cluster_to_cache(start_idx, end_idx):
    cluster_data = df[start_idx:end_idx]
    train_dataloader, valid_dataloader = load_cluster_data(cluster_data)
    return train_dataloader, valid_dataloader


def train_bi_graphormer(model, df, optimizer, epochs=10):
    model.train()
    # df = []
    # loss_file = open(f"{result_path}/pretrain_loss.txt", "a")

    with open(f"{result_path}/pretrain_loss.txt", "a") as loss_file:
    # data = f.read()
        for epoch in range(epochs):
            torch.cuda.empty_cache()

            total_data_size = len(df)
            cluster_size = 1000
            total_clusters = math.ceil(total_data_size / cluster_size)
            for i in range(0,total_data_size, cluster_size):
                torch.cuda.empty_cache()
                current_cluster = i // cluster_size
                start_idx = i
                end_idx = min(i+cluster_size, total_data_size)
                # cluster_data = df[start_idx:end_idx]
                # train_dataloader, valid_dataloader = load_cluster_data(cluster_data)
                train_dataloader, valid_dataloader = load_cluster_to_cache(start_idx, end_idx)
                total_loss = 0.0
                for batch in tqdm(train_dataloader):
                    input_ids = batch["input_ids"]
                    # attention_mask = batch["attention_mask"]
                    fingerprints = batch["fingerprints"]
                    graphs = batch["graphs"]
                    
                    graphormer_datas = batch["graphormer_datas"]
                    
                    # if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    fingerprints = fingerprints.to(device)
                    # attention_mask = attention_mask.to("cuda")
                    graphs = graphs.to(device)

                    optimizer.zero_grad()

                    if not main_args.save_model_folder or main_args.save_model_folder == "original":
                        cls_output, fp_embedding, graph_embedding = model(
                            input_ids, fingerprints, graphs, graphormer_datas
                        )
                        loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, save_model_folder=main_args.save_model_folder)
                    elif main_args.save_model_folder == "remove_sub_sf_gg":
                        graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                        cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                            input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                        )
                        loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph, save_model_folder=main_args.save_model_folder)
                    elif main_args.save_model_folder == "remove_sub_sfgg":
                        graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                        cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                            input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                        )
                        loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph,save_model_folder=main_args.save_model_folder)
                    elif "remove_sub" in main_args.save_model_folder:
                        graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                        cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                            input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                        )
                        loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph,save_model_folder=main_args.save_model_folder)

                    else:
                        cls_output, fp_embedding, graph_embedding = model(
                            input_ids, fingerprints, graphs, graphormer_datas
                        )
                        loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, save_model_folder=main_args.save_model_folder)
                    

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_dataloader)
                print(f"Epoch {epoch+1}/{epochs}; Cluster {current_cluster}/{total_clusters}: Loss: {avg_loss}")
                

                torch.save(
                    model.state_dict(),
                    f"{result_path}/checkpoint_{(epoch +1 )}.ckpt",
                )
                if epoch % 1 == 0:
                    total_loss = 0.0
                    for batch in tqdm(valid_dataloader):
                        input_ids = batch["input_ids"]
                        fingerprints = batch["fingerprints"]
                        graphs = batch["graphs"]
                        
                        graphormer_datas = batch["graphormer_datas"]
                        # if torch.cuda.is_available():
                        input_ids = input_ids.to(device)
                        fingerprints = fingerprints.to(device)
                        graphs = graphs.to(device)

                        optimizer.zero_grad()

                    
                        # cls_output, fp_embedding, graph_embedding = model(
                        #     input_ids, fingerprints, graphs, graphormer_datas
                        # )


                        # loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)

                        if not main_args.save_model_folder or main_args.save_model_folder == "original":
                            cls_output, fp_embedding, graph_embedding = model(
                                input_ids, fingerprints, graphs, graphormer_datas
                            )
                            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, save_model_folder=main_args.save_model_folder)
                        elif main_args.save_model_folder == "remove_sub_sf_gg":
                            graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                            cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                                input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                            )
                            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph, save_model_folder=main_args.save_model_folder)
                        elif main_args.save_model_folder == "remove_sub_sfgg":
                            graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                            cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                                input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                            )
                            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph,save_model_folder=main_args.save_model_folder)
                        elif "remove_sub" in main_args.save_model_folder:
                            graphormer_datas_remove_subgraph = batch["graphormer_datas_remove_subgraph"]
                            cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph = model(
                                input_ids, fingerprints, graphs, graphormer_datas, graphormer_datas_remove_subgraph
                            )
                            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, graph_embedding_remove_subgraph,save_model_folder=main_args.save_model_folder)

                        else:
                            cls_output, fp_embedding, graph_embedding = model(
                                input_ids, fingerprints, graphs, graphormer_datas
                            )
                            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding, save_model_folder=main_args.save_model_folder)
                        

                        total_loss += loss.item()
                    avg_loss = total_loss / len(valid_dataloader)
                    print(f"Validation Epoch {epoch+1}/{epochs}; Cluster {current_cluster}/{total_clusters}: Loss: {avg_loss}")
                    loss_file.write(str(avg_loss) + "\n")

def train_no_graphormer(model, train_dataloader, valid_dataloader, optimizer, epochs=10):
    model.train()
    # df = []
    loss_file = open(f"{result_path}/pretrain_loss.txt", "a")
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        total_loss = 0.0

        for batch in tqdm(train_dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]
            # if main_args.graph_model == 'bi_graphormer_mpnn':
            #     graphormer_datas = batch["graphormer_datas"]
            # if torch.cuda.is_available():
            input_ids = input_ids.to(device)
            fingerprints = fingerprints.to(device)
            # attention_mask = attention_mask.to("cuda")
            graphs = graphs.to(device)

            optimizer.zero_grad()
            # if main_args.graph_model == 'bi_graphormer_mpnn':
            #     cls_output, fp_embedding, graph_embedding = model(
            #         input_ids, fingerprints, graphs, graphormer_datas
            #     )
            # elif main_args.graph_model == 'no_graphormer_mpnn':
            cls_output, fp_embedding, graph_embedding = model(
                input_ids, fingerprints, graphs
            )

            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss}")
        loss_file.write(str(avg_loss) + "\n")
        if epoch % 4 == 0:
            total_loss = 0.0
            for batch in tqdm(valid_dataloader):
                input_ids = batch["input_ids"]
                fingerprints = batch["fingerprints"]
                graphs = batch["graphs"]
                # if main_args.graph_model == 'bi_graphormer_mpnn':
                    # graphormer_datas = batch["graphormer_datas"]
                # if torch.cuda.is_available():
                input_ids = input_ids.to(device)
                fingerprints = fingerprints.to(device)
                graphs = graphs.to(device)

                optimizer.zero_grad()

                # if main_args.graph_model == 'bi_graphormer_mpnn':
                #     cls_output, fp_embedding, graph_embedding = model(
                #         input_ids, fingerprints, graphs, graphormer_datas
                #     )
                # elif main_args.graph_model == 'no_graphormer_mpnn':
                cls_output, fp_embedding, graph_embedding = model(
                    input_ids, fingerprints, graphs
                )

                loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)

                total_loss += loss.item()
            avg_loss = total_loss / len(valid_dataloader)
            print(f"Validation Epoch {epoch+1}/{epochs}; Loss: {avg_loss}")

        if epoch % 4 == 0:
            torch.save(
                model.state_dict(),
                f"{result_path}/checkpoint_{epoch}.ckpt",
            )


def train_only_graphormer(model, df, optimizer, epochs=10):
    model.train()
    # df = []
    loss_file = open(f"{result_path}/pretrain_loss.txt", "a")
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        total_loss = 0.0

        total_data_size = len(df)
        cluster_size = 10000
        total_clusters = math.ceil(total_data_size / cluster_size)
        for i in range(0,total_data_size, cluster_size):
            current_cluster = i // cluster_size
            start_idx = i
            end_idx = min(i+cluster_size, total_data_size)
            cluster_data = df[start_idx:end_idx]
            train_dataloader, valid_dataloader = load_cluster_data(cluster_data)

            for batch in tqdm(train_dataloader):
                input_ids = batch["input_ids"]
                # attention_mask = batch["attention_mask"]
                fingerprints = batch["fingerprints"]
                graphs = batch["graphs"]
                
                graphormer_datas = batch["graphormer_datas"]
                # if torch.cuda.is_available():
                input_ids = input_ids.to(device)
                fingerprints = fingerprints.to(device)
                # attention_mask = attention_mask.to("cuda")
                graphs = graphs.to(device)

                optimizer.zero_grad()

                print(graphormer_datas)

                cls_output, fp_embedding, graph_embedding = model(
                    input_ids, fingerprints, graphs, graphormer_datas
                )


                # loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimizer.step()

                # total_loss += loss.item()

            # avg_loss = total_loss / len(train_dataloader)
            # print(f"Epoch {epoch+1}/{epochs}; Cluster {current_cluster}/{total_clusters}: Loss: {avg_loss}")
            # loss_file.write(str(avg_loss) + "\n")
            # if epoch % 4 == 0:
            #     total_loss = 0.0
            #     for batch in tqdm(valid_dataloader):
            #         input_ids = batch["input_ids"]
            #         fingerprints = batch["fingerprints"]
            #         graphs = batch["graphs"]
                    
            #         graphormer_datas = batch["graphormer_datas"]
            #         # if torch.cuda.is_available():
            #         input_ids = input_ids.to(device)
            #         fingerprints = fingerprints.to(device)
            #         graphs = graphs.to(device)

            #         optimizer.zero_grad()

                   
            #         cls_output, fp_embedding, graph_embedding = model(
            #             input_ids, fingerprints, graphs, graphormer_datas
            #         )


            #         loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)

            #         total_loss += loss.item()
            #     avg_loss = total_loss / len(valid_dataloader)
            #     print(f"Validation Epoch {epoch+1}/{epochs}; Cluster {current_cluster}/{total_clusters}: Loss: {avg_loss}")

            # if epoch % 4 == 0:
            #     torch.save(
            #         model.state_dict(),
            #         f"{result_path}/checkpoint_{epoch}.ckpt",
            #     )

def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        # print(key)
        collated_batch[key] = [item[key] for item in batch]
        if key in ["graphs"]:
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ["graphormer_datas", "graphormer_datas_remove_subgraph"]:
            graphormer_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = graphormer_datas
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


df = pd.read_csv(data_path, sep=" ", header=None, names=["Index", "SMILES"])
# Extract SMILES to a list
def round_down_to_nearest_10K(n):
    return n//10000*10000

keep_lines = round_down_to_nearest_10K(len(df))
smiles_list = df[:keep_lines]


# Instantiate and train the model
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model = MolecularEmbeddingModel()
# model.apply(weights_init)
if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model=='original_graphormer_mpnn':
    # model.load_state_dict(
    #     torch.load(f"result/pretrain_result_bi_graphormer/checkpoint_16.ckpt"), strict=False
    # )

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr_start"])

    train_bi_graphormer(model, smiles_list, optimizer, epochs=config["pretrain"]["epochs"])
elif main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
    # model.load_state_dict(
    #     torch.load(f"{result_path}/checkpoint_3.ckpt"), strict=False
    # )

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr_start"])

    train_bi_graphormer(model, smiles_list, optimizer, epochs=config["pretrain"]["epochs"])

elif main_args.graph_model == 'no_graphormer_mpnn':
    train_val_data, test_data = train_test_split(
        smiles_list, test_size=0.2, random_state=42
    )

    # Split train+validation into train and validation data
    train_data, valid_data = train_test_split(
        train_val_data, test_size=0.25, random_state=42
    )

    # Create dataset and dataloader
    train_dataset = CombinedMoleculeDataset(data=train_data)
    test_dataset = CombinedMoleculeDataset(data=test_data)
    valid_dataset = CombinedMoleculeDataset(data=valid_data)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["pretrain"]["num_workers"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["pretrain"]["num_workers"],
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config["pretrain"]["num_workers"],
    )

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr_start"])

    train_no_graphormer(model, train_dataloader, valid_dataloader, optimizer, epochs=config["pretrain"]["epochs"])

elif main_args.graph_model=='only_original_graphormer':

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr_start"])

    train_only_graphormer(model, smiles_list, optimizer, epochs=config["pretrain"]["epochs"])
