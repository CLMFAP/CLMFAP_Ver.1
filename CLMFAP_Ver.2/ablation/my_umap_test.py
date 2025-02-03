#Import numpy
import numpy as np

#Import scikitlearn for machine learning functionalities
import sklearn
from sklearn.manifold import TSNE 
from sklearn.datasets import load_digits # For the UCI ML handwritten digits dataset

# Import matplotlib for plotting graphs ans seaborn for attractive graphics.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import seaborn as sb

import os

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score
from utils.utils import *
from graphormer_data.wrapper import preprocess_item
from graphormer_data.collator import collator
from utils.data_utils import *
import regex as re
# Define the Model
from args import *
import argparse
from models.pubchem_encoder import Encoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score

config_path = "configs/config.json"
config = json.load(open(config_path))
measure_name = "scaffold_number"
measure_name = "BINARY_ACTIVITY"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# "bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"
graph_model = "bi_graphormer_no_mpnn"
X_list = []
Y_list = []
digits = []

total = 5000
# n_components = 64
# perplexity = 10
# n_iter = 1000


# bbbp
df = pd.read_csv("/home/user/workspace/MY_MODEL/data/MIC/valid.csv")

# scaffolds
# df = pd.read_csv("/home/user/workspace/MY_MODEL/data/top_10_scaffolds_chembl.csv")

class CombinedMoleculeDataset(Dataset):
    def __init__(self, data, graph_model):
        self.data = data
        self.data["canonical_smiles"] = data["smiles"].apply(
            lambda smi: normalize_smiles(smi, canonical=True, isomeric=False)
        )
        self.smiles_list_good = self.data.dropna(subset=['canonical_smiles'])
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

        self.smiles_list = self.smiles_list_good["smiles"]
        self.targets = self.smiles_list_good[measure_name]
        self.text_encoder = Encoder(100000)
        print("Loading data...")
        self.encodings = self.text_encoder.process(self.smiles_list)
        self.fingerprints = generate_fingerprints(self.smiles_list,2)
        # self.graphs, self.graphormer_datas = generate_3d_graphs(
        #     self.smiles_list, smiles2graph=smiles2graph
        # )

        if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
            self.graphs, self.graphormer_datas = generate_3d_graphs(self.smiles_list)
        elif graph_model == 'no_graphormer_mpnn':
            self.graphs = generate_3d_graphs_no_graphormer(self.smiles_list)

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        # item = {key: val[idx] for key, val in self.encodings.items()}
        item = {}

        item["input_ids"] = self.encodings[0][idx]
        item["fingerprints"] = self.fingerprints[idx]
        item["graphs"] = self.graphs[idx]
        item[measure_name] = self.targets[idx]
        # self.graphormer_datas[idx].idx = idx
        # item["graphormer_datas"] = self.graphormer_datas[idx]

        if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
            self.graphormer_datas[idx].idx = idx
            item["graphormer_datas"] = self.graphormer_datas[idx]
        return item


class MolecularEmbeddingModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50, graph_model="bi_graphormer_mpnn"):
        super(MolecularEmbeddingModel, self).__init__()
        # self.bert = BertModel.from_pretrained(bert_model_name)
        cur_config = parse_args()
        text_encoder = Encoder(cur_config.max_len)
        vocab = text_encoder.char2id
        self.bert = SmilesEncoder(cur_config, vocab)
        self.fp_embedding = FpEncoder()
        self.graph_embedding = GraphEncoder(config=config["mpnn"], graph_model=graph_model)
        self.cls_embedding = nn.Linear(
            self.bert.hidden_size, 128
        )  # Added to match combined embedding size

        if graph_model == 'bi_graphormer_mpnn' or graph_model== 'original_graphormer_mpnn':
            self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        elif graph_model == 'no_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model=='bi_graphormer_no_mpnn':
            self.mid_liner_graph = nn.Linear(1 * 768, 1 * 768)
        # self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        self.mid_liner = nn.Linear(3 * 768, 1 * 768)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, fingerprints, graphs, graphormer_datas=None):

        logits = self.bert(input_ids)
        cls_output = logits[:, 0, :]

        # Fingerprint embedding
        fp_embedding = self.fp_embedding(fingerprints)

        # Flatten and embed 3D Graph
        if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
            graph_embedding = self.graph_embedding(graphs, graphormer_datas)
        elif graph_model == 'no_graphormer_mpnn':
            graph_embedding = self.graph_embedding(graphs)
        # graph_embedding = self.graph_embedding(graphs, graphormer_datas)
        graph_embedding = self.mid_liner_graph(graph_embedding)

        molecule_emb = torch.cat((cls_output, fp_embedding, graph_embedding), dim=1)
        # molecule_emb = self.mid_liner(molecule_emb)
        
        return molecule_emb
        return cls_output, fp_embedding, graph_embedding
    
def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    for key in elem_keys:
        # print(key)
        collated_batch[key] = [item[key] for item in batch]
        if key in ["graphs"]:
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in [measure_name]:
            collated_batch[key] = [int(item[key]) for item in batch]
        elif key in ["graphormer_datas"]:
            graphormer_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = graphormer_datas
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


### User My Model
test_dataset = CombinedMoleculeDataset(data=df, graph_model=graph_model)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=custom_collate_fn,
    num_workers=config["pretrain"]["num_workers"],
)


model = MolecularEmbeddingModel(graph_model=graph_model)

model_path = config['pretrained_model'][graph_model]['path']
model_path = "/home/user/workspace/MY_MODEL/pre_trained_models/pretrain_result_bi_graphormer_no_mpnn_original/checkpoint_24.ckpt"
model_path = "/home/user/workspace/MY_MODEL/pre_trained_models/pretrain_result_bi_graphormer_no_mpnn_remove_sub_sfgg/checkpoint_24.ckpt"
model.load_state_dict(
    torch.load(model_path), strict=False
)

if torch.cuda.is_available():
    model = model.cuda()

for batch in tqdm(test_dataloader):
    input_ids = batch["input_ids"]
    # attention_mask = batch["attention_mask"]
    fingerprints = batch["fingerprints"]
    graphs = batch["graphs"]
    
    if graph_model != 'no_graphormer_mpnn':
        graphormer_datas = batch["graphormer_datas"]

    label = torch.tensor(batch[measure_name]).to("cuda")
    # if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")
    fingerprints = fingerprints.to("cuda")
    # attention_mask = attention_mask.to("cuda")
    graphs = graphs.to("cuda")

    if graph_model == 'no_graphormer_mpnn':
        molecule_emb = model(input_ids, fingerprints, graphs)
    else:
        # embedding = model(
        #     input_ids, fingerprints, graphs, graphormer_datas
        # )

        molecule_emb = model(
            input_ids, fingerprints, graphs, graphormer_datas
        )

    X_list.append(molecule_emb.cpu().detach().numpy())
    Y_list.append(label.cpu().detach().numpy())

X = np.vstack(X_list)
Y = np.hstack(Y_list)

sample_silhouette_values = silhouette_samples(X, Y)

# print('Silhouette scores for each sample:', sample_silhouette_values)

average_silhouette_score = np.mean(sample_silhouette_values)

print("Average Silhouette Score:", average_silhouette_score)

calinski_harabasz_score = metrics.calinski_harabasz_score(X, Y)

print("Calinski-Harabasz Index:", calinski_harabasz_score)

db_index = davies_bouldin_score(X, Y)

print("Davies-Bouldin Index:", db_index)

import umap

# Define the UMAP-based dimensionality reduction function
def apply_umap(n_neighbors, min_dist, n_components, X):
    # Initialize UMAP with the given parameters
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    # Apply UMAP's fit_transform to reduce dimensions
    embedding = reducer.fit_transform(X)
    return embedding

# Function for plotting the embeddings remains the same
def plot(x, colors):
    # Choose color palette
    palette = np.array(sb.color_palette("hls", 2))  
    # print(palette)
    colors = np.array(colors)
    print(colors)
    print(colors.astype(int))

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    
    # Update: Replace np.int with int to avoid deprecation warning
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=15, c=palette[colors.astype(int)])

# Execution loop for UMAP
for n_components in [2]:  # UMAP typically used for 2D or 3D visualization
    for n_neighbors in [15]:  # Common UMAP parameter # 15, 
        for min_dist in [0.1]:  # Common UMAP parameter # 0.1
            # Apply UMAP to reduce dimensions
            embedding = apply_umap(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, X=X)
            # Plot the UMAP result
            plot(embedding, Y)

            # Save the plot with UMAP parameters
            # plt.savefig(f"umap/{graph_model}_scaffolds.png", format="png", dpi=100) 
            plt.show()
# 0
# Average Silhouette Score: 0.05227825
# Calinski-Harabasz Index: 40.50420062836853
# Davies-Bouldin Index: 5.068289901951287

# 4
# Average Silhouette Score: 0.05121869
# Calinski-Harabasz Index: 31.535532037564717
# Davies-Bouldin Index: 5.767997788339832

# 8
# Average Silhouette Score: 0.04248556
# Calinski-Harabasz Index: 26.227057414898187
# Davies-Bouldin Index: 6.330935531780278

# 12
# Average Silhouette Score: 0.044869594
# Calinski-Harabasz Index: 26.445892301536098
# Davies-Bouldin Index: 6.283341986071239

# 16
# Average Silhouette Score: 0.037749626
# Calinski-Harabasz Index: 21.367790228912554
# Davies-Bouldin Index: 6.9400874279809965

# 24
# Average Silhouette Score: 0.038654
# Calinski-Harabasz Index: 17.621564444374606
# Davies-Bouldin Index: 7.49524647310827

# 0
# Average Silhouette Score: -0.018156538
# Calinski-Harabasz Index: 69.0979684323854
# Davies-Bouldin Index: 4.706026741327359

# 4
# Average Silhouette Score: -0.013801487
# Calinski-Harabasz Index: 294.91058915368865
# Davies-Bouldin Index: 4.494946140119676
