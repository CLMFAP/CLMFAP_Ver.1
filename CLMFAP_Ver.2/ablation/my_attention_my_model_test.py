from visualizer import get_local
get_local.activate() # 激活装饰器

import torch
import matplotlib.pyplot as plt
import numpy as np


#Import numpy
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.my_model import *
import json
import pandas as pd
from torch_geometric.data import Batch
from tqdm import tqdm
from utils.utils import *
from graphormer_data.collator import collator
from utils.data_utils import *
import regex as re
# Define the Model
from args import *
from models.pubchem_encoder import Encoder
from sklearn.decomposition import PCA
import regex as re

config_path = "configs/config.json"
config = json.load(open(config_path))
measure_name = "scaffold_number"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# "bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"
graph_model = "bi_graphormer_no_mpnn"
X_list = []
Y_list = []
digits = []

total = 1
# n_components = 64
# perplexity = 10
# n_iter = 1000



df = pd.read_csv("/home/user/workspace/MY_MODEL/data/smiles_attention_weight.csv")[0:1]
smiles = df.iloc[0]['smiles']
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
        self.fingerprints = generate_fingerprints(self.smiles_list)
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


# class MolecularEmbeddingModel(nn.Module):
#     def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50, graph_model="bi_graphormer_mpnn"):
#         super(MolecularEmbeddingModel, self).__init__()
#         # self.bert = BertModel.from_pretrained(bert_model_name)
#         cur_config = parse_args()
#         text_encoder = Encoder(cur_config.max_len)
#         vocab = text_encoder.char2id
#         self.bert = SmilesEncoder(cur_config, vocab)
#         self.fp_embedding = FpEncoder()
#         self.graph_embedding = GraphEncoder(config=config["mpnn"], graph_model=graph_model)
#         self.cls_embedding = nn.Linear(
#             self.bert.hidden_size, 128
#         )  # Added to match combined embedding size

#         if graph_model == 'bi_graphormer_mpnn' or graph_model== 'original_graphormer_mpnn':
#             self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
#         elif graph_model == 'no_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model=='bi_graphormer_no_mpnn':
#             self.mid_liner_graph = nn.Linear(1 * 768, 1 * 768)
#         # self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
#         self.mid_liner = nn.Linear(3 * 768, 1 * 768)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, input_ids, fingerprints, graphs, graphormer_datas=None):

#         logits = self.bert(input_ids)
#         cls_output = logits[:, 0, :]

#         # Fingerprint embedding
#         fp_embedding = self.fp_embedding(fingerprints)

#         # Flatten and embed 3D Graph
#         if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
#             graph_embedding = self.graph_embedding(graphs, graphormer_datas)
#         elif graph_model == 'no_graphormer_mpnn':
#             graph_embedding = self.graph_embedding(graphs)
#         # graph_embedding = self.graph_embedding(graphs, graphormer_datas)
#         graph_embedding = self.mid_liner_graph(graph_embedding)

#         molecule_emb = torch.cat((cls_output, fp_embedding, graph_embedding), dim=1)
#         molecule_emb = self.mid_liner(molecule_emb)
        
#         # return graph_embedding
#         return cls_output, fp_embedding, graph_embedding


class MolecularEmbeddingModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", max_atoms=50, graph_model="bi_graphormer_mpnn"):
        super(MolecularEmbeddingModel, self).__init__()
        # self.bert = BertModel.from_pretrained(bert_model_name)
        self.graph_model = graph_model
        cur_config = parse_args()
        text_encoder = Encoder(cur_config.max_len)
        vocab = text_encoder.char2id
        self.bert = SmilesEncoder(cur_config, vocab)
        self.fp_embedding = FpEncoder()
        self.graph_embedding = GraphEncoder(config=config["mpnn"], graph_model=self.graph_model)
        self.cls_embedding = nn.Linear(
            self.bert.hidden_size, 128
        )  # Added to match combined embedding size

        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model== 'original_graphormer_mpnn':
            self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        elif self.graph_model == 'no_graphormer_mpnn' or self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            self.mid_liner_graph = nn.Linear(1 * 768, 1 * 768)
        # self.mid_liner_graph = nn.Linear(2 * 768, 1 * 768)
        self.mid_liner = nn.Linear(3 * 768, 1 * 768)

        self.net = self.Net(768, 2)

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
        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model== 'original_graphormer_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            graph_embedding = self.graph_embedding(graphs, graphormer_datas)
        elif self.graph_model == 'no_graphormer_mpnn':
            graph_embedding = self.graph_embedding(graphs)
        # graph_embedding = self.graph_embedding(graphs, graphormer_datas)
        graph_embedding = self.mid_liner_graph(graph_embedding)

        molecule_emb = torch.cat((cls_output, fp_embedding, graph_embedding), dim=1)
        molecule_emb = self.mid_liner(molecule_emb)
        pred = self.net(molecule_emb)
        return pred

    class Net(nn.Module):
        dims = [150, 50, 50, 2]

        def __init__(self, smiles_embed_dim, num_classes, dims=dims, dropout=0.5):
            super().__init__()
            self.desc_skip_connection = True
            self.fcs = []  # nn.ModuleList()
            print("dropout is {}".format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, num_classes)  # classif

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)

            # z = self.layers(smiles_emb)
            return z


    
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
model_path = "/home/user/workspace/MY_MODEL/result/pretrain_result_bi_graphormer_no_mpnn_wholechembl_442/checkpoint_3.ckpt"
model_path = "/home/user/workspace/MY_MODEL/result/finetune_result/data/MIC/bi_graphormer_no_mpnn/checkpoint_0.ckpt"
model.load_state_dict(
    torch.load(model_path), strict=False
)

if torch.cuda.is_available():
    model = model.cuda()

def regex_smiles(smiles):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    raw_regex = regex.findall(smiles.strip("\n"))
    return raw_regex


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
        embedding = model(input_ids, fingerprints, graphs)
    else:
        embedding = model(
            input_ids, fingerprints, graphs, graphormer_datas
        )

        # cls_output, fp_embedding, graph_embedding = model(
        #     input_ids, fingerprints, graphs, graphormer_datas
        # )
    # print("Embedding: ")
    # print(embedding.cpu().detach().numpy())
    X_list.append(embedding.cpu().detach().numpy())
    Y_list.append(label.cpu().detach().numpy())

X = np.vstack(X_list)
Y = np.hstack(Y_list)

attentions = model.state_dict()
cache = get_local.cache
attention_maps = cache['RotateAttentionLayer.forward']
# print(len(attention_maps))
# print(attention_maps[-1].shape)
# attention_maps = [torch.from_numpy(arr) for arr in attention_maps]
# attention_maps = torch.stack(attention_maps, dim=0)
# print(type(attention_maps))
# print(attention_maps[-1].shape)
# attentions = attention_maps[-1]

def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image, cmap='gray')
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

# print(len(attention_maps))
last_layer_attentions = attention_maps[-1]
np.save(f"last_layer_weight/{smiles}_lastlayer_weight.npy", last_layer_attentions)
# print(last_layer_attentions.shape)
non_letters_with_indices = [(i, char) for i, char in enumerate(regex_smiles(smiles)) if not char.isalpha()]

# Split the indices and characters into separate lists for convenience
non_letter_indices = [i for i, char in non_letters_with_indices]
non_letter_chars = [char for i, char in non_letters_with_indices]
# print(non_letter_chars)
# print(last_layer_attentions.shape)
# rows_to_remove = [2,4,6,8,9,11]
last_layer_attentions = np.delete(last_layer_attentions, non_letter_indices, axis=2)
# print(last_layer_attentions.shape)
last_layer_attentions = np.delete(last_layer_attentions, non_letter_indices, axis=3)
# print(last_layer_attentions.shape)
# visualize_heads(last_layer_attentions, cols=4)
np.save(f"last_layer_weight/{smiles}_lastlayer_weight_only element.npy", last_layer_attentions)

# To only show the average head attention:
# The different is that above visualize use numpy array, and below 
# average_attention = torch.mean(attentions, dim=1)  # Average over heads and select the first item in batch

average_attention = np.mean(last_layer_attentions, axis=1)
average_attention = np.squeeze(average_attention)
# print(type(average_attention))
# print(average_attention.shape)
# np.save('Cc1c(-c2ccc(Cl)cc2)c(=O)oc2cc(OCCN3CCOCC3)ccc12.npy', average_attention)
# labels = ["C", "C", "C", "C", "C", "C", "O"]
# Visualize the average attention matrix
plt.figure(figsize=(10, 8))
# plt.imshow(average_attention, cmap='viridis')
plt.imshow(average_attention, cmap='gray')
plt.title(f"CL Average Attention Map for SMILES: {smiles}")
plt.xlabel("Token Positions")
plt.ylabel("Token Positions")

# plt.xticks(ticks=np.arange(len(labels)), labels=labels)
# plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.colorbar()
# plt.show()
smiles_without_non_letters = [char for char in regex_smiles(smiles) if char.isalpha()]
array = np.load("/home/user/workspace/MY_MODEL/last_layer_weight/COc1ccccc1N1CCN(c2ncc3c(n2)C[C@@H](c2ccccc2Cl)CC3=O)CC1_lastlayer_weight_only element.npy")
print(array.shape)
print(non_letter_indices)
print(non_letter_chars)
print(smiles_without_non_letters)
print(len(smiles_without_non_letters))
print(len(smiles))
print(len(non_letter_chars))