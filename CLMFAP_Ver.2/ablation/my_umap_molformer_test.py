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
measure_name = "p_np"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# "bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"
graph_model = "original_graphormer_no_mpnn"
X_list = []
Y_list = []
digits = []

total = 5000
# n_components = 64
# perplexity = 10
# n_iter = 1000



df = pd.read_csv("/home/user/workspace/MY_MODEL/data/bbbp/train.csv")
# df = pd.read_csv("/home/user/workspace/MY_MODEL/data/top_10_scaffolds_chembl.csv")
data_feature = 'bbbp'




### Use smiles_transformer
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("ibm/smiles_transformer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/smiles_transformer-XL-both-10pct", trust_remote_code=True)

smiles = df['smiles'].tolist()
labels = df[measure_name].tolist()
batch_size = 1000
batches = [smiles[i:i + batch_size] for i in range(0, len(smiles), batch_size)]
embedding_list = []
with torch.no_grad():
    for batch in batches:
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        outputs = model(**inputs)
        embedding_list.extend(outputs.pooler_output.detach().cpu().numpy())

X = embedding_list
Y = labels

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
    print(colors.astype(int)-1)

    # Create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    
    # Update: Replace np.int with int to avoid deprecation warning
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=15, c=palette[colors.astype(int)])
    
# Execution loop for UMAP
for n_components in [2]:  # UMAP typically used for 2D or 3D visualization
    for n_neighbors in [15]:  # Common UMAP parameter
        for min_dist in [0.1]:  # Common UMAP parameter
            # Apply UMAP to reduce dimensions
            embedding = apply_umap(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, X=embedding_list)
            # Plot the UMAP result
            plot(embedding, labels)

            # Save the plot with UMAP parameters
            plt.savefig(f"umap/smiles_transformer_graph_scaffolds.png", format="png", dpi=100) 
            plt.show()
