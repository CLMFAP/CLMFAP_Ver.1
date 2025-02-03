from models.retrosys_model import RetroSysModel
from argparse import ArgumentParser, ArgumentTypeError, ArgumentError, Namespace
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
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

main_args = parse_args()
config_path = "configs/config.json"
config = json.load(open(config_path))
data_path = "/home/user/workspace/MY_MODEL/data/chemformer_pred.csv"
graph_model = "bi_graphormer_mpnn"
model_path = config['pretrained_model'][graph_model]['path']


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

        self.smiles_list_good["reactants"] = self.smiles_list_good["reactants"].apply(lambda x: x.split('.')[0])

        self.smiles_list_good["canonical_smiles"] = self.smiles_list_good["reactants"].apply(
            lambda smi: normalize_smiles(smi, canonical=True, isomeric=False)
        )

        self.smiles_list_good_final = self.smiles_list_good.dropna()
        self.smiles_list_good_final.reset_index(drop=True, inplace=True)

        assert (
            self.smiles_list_good_final["canonical_smiles"].isna().sum() == 0
        ), "There are still NaN values in 'canonical_smiles'"
        print(
            "Dropped {} invalid smiles".format(
                len(self.smiles_list_good) - len(self.smiles_list_good_final)
            )
        )
        print("Valid smiles {}".format(len(self.smiles_list_good_final)))


        # For smiles
        self.smiles_list = self.smiles_list_good_final["SMILES"]
        self.text_encoder = Encoder(1000, mlm_probability = 0.0)

        print("Loading data...")
        self.encodings = self.text_encoder.process(self.smiles_list)
        self.fingerprints = generate_fingerprints(self.smiles_list)
        self.graphs, self.graphormer_datas = generate_3d_graphs(self.smiles_list)

        # For reactions
        self.target_smiles_list = self.smiles_list_good_final["reactants"]

        print("Loading data...")
        self.target_encodings = self.text_encoder.process(self.target_smiles_list)
        self.target_fingerprints = generate_fingerprints(self.target_smiles_list)
        self.target_graphs, self.target_graphormer_datas = generate_3d_graphs(self.target_smiles_list)



    def __len__(self):
        return len(self.encodings[0])

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[0][idx]
        item["fingerprints"] = self.fingerprints[idx]
        item["graphs"] = self.graphs[idx]
        self.graphormer_datas[idx].idx = idx
        item["graphormer_datas"] = self.graphormer_datas[idx]

        item["target_input_ids"] = self.target_encodings[0][idx]
        item["target_fingerprints"] = self.target_fingerprints[idx]
        item["target_graphs"] = self.target_graphs[idx]
        self.target_graphormer_datas[idx].idx = idx
        item["target_graphormer_datas"] = self.target_graphormer_datas[idx]

        item["target_smiles"] = self.target_smiles_list[idx]
        return item


class MolecularEmbeddingModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50):
        super(MolecularEmbeddingModel, self).__init__()
        cur_config = main_args
        text_encoder = Encoder(cur_config.max_len)
        vocab = text_encoder.char2id
        self.bert = SmilesEncoder(cur_config, vocab)
        self.fp_embedding = FpEncoder()
        self.graph_embedding = GraphEncoder(config=config["mpnn"], graph_model=main_args.graph_model)
        # self.cls_embedding = nn.Linear(
        #     self.bert.hidden_size, 128
        # )  # Added to match combined embedding size
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model=='original_graphormer_mpnn':
            self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        elif main_args.graph_model == 'no_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
            self.mid_liner_graph = nn.Linear(1 * 768, 1 * 768)

        self.mid_liner = nn.Linear(3 * 768, 1 * 768)


    def forward(self, input_ids, fingerprints, graphs, graphormer_datas=None):

        # Smiles embedding
        logits = self.bert(input_ids)
        cls_output = logits[:, 0, :]

        # Fingerprint embedding
        fp_embedding = self.fp_embedding(fingerprints)

        # 3D Graph embedding
        if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model=='original_graphormer_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
            graph_embedding = self.graph_embedding(graphs, graphormer_datas)
        elif main_args.graph_model == 'no_graphormer_mpnn':
            graph_embedding = self.graph_embedding(graphs)
        graph_embedding = self.mid_liner_graph(graph_embedding)

        # print(cls_output.shape)
        # print(fp_embedding.shape)
        # print(graph_embedding.shape)
        molecule_emb = torch.cat((cls_output, fp_embedding, graph_embedding), dim=1)
        molecule_emb = self.mid_liner(molecule_emb)

        return molecule_emb

def train(model, cl_model, data_loader, num_epochs, device, lr=1e-4):
    """
    Trains the RetroSysModel.

    Args:
        model (RetroSysModel): The model to train.
        data_loader (DataLoader): DataLoader that yields training batches.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to train on ('cpu' or 'cuda').
        lr (float): Learning rate for the optimizer.
    """
    # Move model to the device
    model.to(device)
    

    # model.val()  # Set model to training mode
    
    for batch in tqdm(data_loader):
        # Unpack batch
        # src_tokens = batch['src_tokens'].to(device)
        # src_lengths = batch['src_lengths'].to(device)
        # prev_output_tokens = batch['prev_output_tokens'].to(device)

        input_ids = batch["input_ids"]
        fingerprints = batch["fingerprints"]
        graphs = batch["graphs"]
        graphormer_datas = batch["graphormer_datas"]
        input_ids = input_ids.to(device)
        fingerprints = fingerprints.to(device)
        graphs = graphs.to(device)
        target_tokens = batch["target_smiles"]
        target_input_ids = batch["target_input_ids"]

        cls_output = cl_model(
            input_ids, fingerprints, graphs, graphormer_datas
        )

        # plm_input = {"net_input0": {"src_tokens": src_tokens}}  # Assuming this structure for PLM input
        plm_input = cls_output
        # Forward pass
        output = model(
            src_tokens=cls_output,
            src_lengths=768,
            prev_output_tokens=cls_output,
            plm_input=plm_input,
            return_all_hiddens=False
        )

        logits, _ = output

        encoded_sequences = logits.tolist()
        

        decoded_sequences = []
        text_encoder = Encoder(1000)

        for encoded_seq in encoded_sequences:
            generated_smiles = text_encoder.decode(encoded_seq)
            decoded_sequences.append(generated_smiles)

        print(target_tokens)
        # print(target_input_ids[0])
        # print(encoded_sequences[0])
        print(decoded_sequences)     


# Example usage:
# Assuming a custom dataset that provides batches of data with 'src_tokens', 'src_lengths', 'prev_output_tokens', and 'target_tokens'.

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example data batch
data = [
    {
        "src_tokens": torch.randint(0, 100, (768,)),  # Example tokenized sentence
        "src_lengths": torch.tensor(10),  # Length of the source sequence
        "prev_output_tokens": torch.randint(0, 100, (768,)),  # Tokenized target with start-of-sequence tokens
        "target_tokens": torch.randint(0, 100, (768,))  # Target sequence tokens (ground truth)
    }
    for _ in range(100)  # Example dataset of 100 sentences
]

for item in data:
    item["src_tokens"] = item["src_tokens"].float()
    item["src_lengths"] = item["src_lengths"].float()
    item["prev_output_tokens"] = item["prev_output_tokens"].float()
    item["target_tokens"] = item["target_tokens"].float()

# DataLoader to handle batching
data_loader = DataLoader(CustomDataset(data), batch_size=8, shuffle=True)

# Initialize the model and training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
lr = 1e-3


# Initialize the model
src_dict = {"padding_idx": 0}
tgt_dict = {"padding_idx": 0}
args = Namespace(
    encoder_embed_dim=8,
    decoder_embed_dim=8,
    encoder_embed_path=None,
    decoder_embed_path=None,
    no_cross_attention=False,
    padding_idx=0,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads = 8,
    decoder_attention_heads = 8,
    attention_dropout = 0.0,
    plm_encoder_embed_dim = 768,
    dropout = 0.1,
    encoder_ffn_embed_dim = 768,
    decoder_ffn_embed_dim = 768
)

model = RetroSysModel(args)

if torch.cuda.is_available():
    model = model.cuda()

model.load_state_dict(
    torch.load(f"result_retrosys/{graph_model}/finetune/checkpoint_39.ckpt"), strict=False
)

cl_model = MolecularEmbeddingModel()


if torch.cuda.is_available():
    cl_model = cl_model.cuda()

# print(model.state_dict().keys())
# print(torch.load(f"checkpoint_12.ckpt").keys)
cl_model.load_state_dict(
    torch.load(model_path), strict=False
)

def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        # print(key)
        collated_batch[key] = [item[key] for item in batch]
        if key in ["graphs", "target_graphs"]:
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ["graphormer_datas", "target_graphormer_datas"]:
            graphormer_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = graphormer_datas
        elif key in ["target_smiles"]:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch



df = pd.read_csv(data_path, sep=",", header=None, names=["idx","target","reactants","likelihood"])
# Extract SMILES to a list
df = df.rename(columns={"target": "SMILES"})
smiles_list = df[10001:10010]

train_data = smiles_list
train_dataset = CombinedMoleculeDataset(data=train_data)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["pretrain"]["batch_size"],
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=config["pretrain"]["num_workers"],
    drop_last=True
)

# Train the model
train(model,cl_model, train_dataloader, num_epochs, device, lr)
