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
from bi_graph_data.wrapper import preprocess_item
from bi_graph_data.collator import collator
import regex as re
# Define the Model
# from args import args
import argparse
from models.pubchem_encoder import Encoder


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Home device: {}".format(device))


# Get args:
main_args = parse_args()
config_path = "configs/config.json"
config = json.load(open(config_path))
result_path = config["result"]["pretrain_path"]
data_path = config['data']['pretrain_path']
ensure_folder_exists(result_path)


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
        self.encodings = self.text_encoder.process(self.smiles_list)
        self.fingerprints = generate_fingerprints(self.smiles_list)
        self.graphs, self.bi_graph_datas = generate_3d_graphs(self.smiles_list)

    def __len__(self):
        return len(self.encodings[0])

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[0][idx]

        item["fingerprints"] = self.fingerprints[idx]
        item["graphs"] = self.graphs[idx]
        self.bi_graph_datas[idx].idx = idx
        item["bi_graph_datas"] = self.bi_graph_datas[idx]
        return item

class MolecularEmbeddingModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50):
        super(MolecularEmbeddingModel, self).__init__()
        cur_config = main_args
        text_encoder = Encoder(cur_config.max_len)
        vocab = text_encoder.char2id
        self.bert = SmilesEncoder(cur_config, vocab)
        self.fp_embedding = FpEncoder()
        self.graph_embedding = GraphEncoder(config=config["mpnn"])
        # self.cls_embedding = nn.Linear(
        #     self.bert.hidden_size, 128
        # )  # Added to match combined embedding size
        self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)


    def forward(self, input_ids, fingerprints, graphs, bi_graph_datas):

        # Smiles embedding
        logits = self.bert(input_ids)
        cls_output = logits[:, 0, :]

        # Fingerprint embedding
        fp_embedding = self.fp_embedding(fingerprints)

        # 3D Graph embedding
        graph_embedding = self.graph_embedding(graphs, bi_graph_datas)
        graph_embedding = self.mid_liner_graph(graph_embedding)

        return cls_output, fp_embedding, graph_embedding


#     return loss
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def criterion(out_1, out_2, batch_size, temperature=0.5):
    # device = torch.device("cuda:0" if out_1.is_cuda else "cpu")
    # neg score
    out = torch.cat([out_1, out_2], dim=0).to(device)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0).to(device)

    # negative samples similarity scoring
    Ng = neg.sum(dim=-1)
    loss = (-torch.log(pos / (pos + Ng))).mean()

    return loss


def contrastive_loss(smiles_emb, fingerprint_emb, graph_emb, temperature=0.1):
    # device = torch.device("cuda")
    smiles_graph_loss = torch.nn.functional.cosine_embedding_loss(
        smiles_emb,
        graph_emb,
        torch.ones(smiles_emb.size(0)).to(device),
        margin=temperature,
    )
    smiles_fingerprint_loss = torch.nn.functional.cosine_embedding_loss(
        smiles_emb,
        fingerprint_emb,
        torch.ones(smiles_emb.size(0)).to(device),
        margin=temperature,
    )
    fingerprint_graph_loss = torch.nn.functional.cosine_embedding_loss(
        fingerprint_emb,
        graph_emb,
        torch.ones(fingerprint_emb.size(0)).to(device),
        margin=temperature,
    )
    return (
        main_args.smiles_graph_weight * smiles_graph_loss
        + main_args.smiles_fp_weight * smiles_fingerprint_loss
        + main_args.graph_fp_weight * fingerprint_graph_loss
    )


def train(model, dataloader, valid_dataloader, optimizer, epochs=10):
    model.train()
    loss_file = open(f"{result_path}/pretrain_loss.txt", "a")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]
            bi_graph_datas = batch["bi_graph_datas"]
            # if torch.cuda.is_available():
            input_ids = input_ids.to(device)
            fingerprints = fingerprints.to(device)
            # attention_mask = attention_mask.to("cuda")
            graphs = graphs.to(device)

            optimizer.zero_grad()

            cls_output, fp_embedding, graph_embedding = model(
                input_ids, fingerprints, graphs, bi_graph_datas
            )

            loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        loss_file.write(str(avg_loss) + "\n")
        if epoch % 4 == 0:
            total_loss = 0.0
            for batch in tqdm(valid_dataloader):
                input_ids = batch["input_ids"]
                fingerprints = batch["fingerprints"]
                graphs = batch["graphs"]
                bi_graph_datas = batch["bi_graph_datas"]
                # if torch.cuda.is_available():
                input_ids = input_ids.to(device)
                fingerprints = fingerprints.to(device)
                graphs = graphs.to(device)

                optimizer.zero_grad()

                cls_output, fp_embedding, graph_embedding = model(
                    input_ids, fingerprints, graphs, bi_graph_datas
                )

                loss = contrastive_loss(cls_output, fp_embedding, graph_embedding)

                total_loss += loss.item()
            avg_loss = total_loss / len(valid_dataloader)
            print(f"Validation Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        if epoch % 4 == 0:
            torch.save(
                model.state_dict(),
                f"{result_path}/checkpoint_{epoch}.ckpt",
            )


def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        # print(key)
        collated_batch[key] = [item[key] for item in batch]
        if key in ["graphs"]:
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ["bi_graph_datas"]:
            bi_graph_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = bi_graph_datas
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


df = pd.read_csv(data_path, sep=" ", header=None, names=["Index", "SMILES"])
# Extract SMILES to a list
smiles_list = df[:500000]

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


# Instantiate and train the model
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


model = MolecularEmbeddingModel()
# model.apply(weights_init)

if torch.cuda.is_available():
    model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config["pretrain"]["lr_start"])

train(model, train_dataloader, valid_dataloader, optimizer, epochs=config["pretrain"]["epochs"])
