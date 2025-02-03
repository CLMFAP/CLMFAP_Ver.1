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

print("Hwllo world")

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Home device: {}".format(device))

main_args = parse_args()
config_path = "configs/config.json"
config = json.load(open(config_path))
dataset_name = main_args.dataset_name

# Combined dataset class for SMILES, fingerprints, and graphs
class CombinedMoleculeDataset(Dataset):
    def __init__(self, data, graph_model):
        self.data = data
        self.graph_model = graph_model
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
        # self.targets = self.smiles_list_good[measure_name]
        self.text_encoder = Encoder(100000)
        print("Loading data...")
        self.encodings = self.text_encoder.process(self.smiles_list)
        self.fingerprints = generate_fingerprints(self.smiles_list)
        # self.graphs, self.graphormer_datas = generate_3d_graphs(
        #     self.smiles_list, smiles2graph=smiles2graph
        # )

        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model== 'original_graphormer_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            self.graphs, self.graphormer_datas = generate_3d_graphs(self.smiles_list)
        elif self.graph_model == 'no_graphormer_mpnn':
            self.graphs = generate_3d_graphs_no_graphormer(self.smiles_list)

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        # item = {key: val[idx] for key, val in self.encodings.items()}
        item = {}
        item["smiles"] = self.smiles_list[idx]
        item["input_ids"] = self.encodings[0][idx]
        item["fingerprints"] = self.fingerprints[idx]
        item["graphs"] = self.graphs[idx]
        # item[measure_name] = self.targets[idx]
        # self.graphormer_datas[idx].idx = idx
        # item["graphormer_datas"] = self.graphormer_datas[idx]

        if self.graph_model == 'bi_graphormer_mpnn' or self.graph_model == 'original_graphormer_no_mpnn' or self.graph_model== 'original_graphormer_mpnn' or self.graph_model=='bi_graphormer_no_mpnn':
            self.graphormer_datas[idx].idx = idx
            item["graphormer_datas"] = self.graphormer_datas[idx]
        return item


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

        self.net = self.Net(768, num_classes)

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


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")

def prediction(
    model,
    valid_dataloader
):

    with open(result_folder + "results" + "_test.csv", 'a') as file:
        headers = "smiles,y_pred"
        file.write(headers + "\n")

    model.eval()
    batch_id = 0
    for batch in tqdm(valid_dataloader):
        batch_id += 1
        smiles = batch["smiles"]
        input_ids = batch["input_ids"]
        fingerprints = batch["fingerprints"]
        graphs = batch["graphs"]
        graphormer_datas = batch["graphormer_datas"]

        # label = torch.tensor(batch[measure_name]).to("cuda")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            fingerprints = fingerprints.to("cuda")
            # attention_mask = attention_mask.to("cuda")
            graphs = graphs.to("cuda")

        measure = model(input_ids, fingerprints, graphs, graphormer_datas)

        measure = F.softmax(measure, dim=1)

        # actuals_cpu = label.detach().cpu().numpy()
        preds_cpu = measure.detach().cpu().numpy()

        # classif
        preds_0_cpu = preds_cpu[:, 0]
        preds_cpu = preds_cpu[:, 1]
        y_pred = np.where(preds_cpu >= 0.5, 1, 0)

        for i in range(len(y_pred)):
            append_to_file(
                os.path.join(result_folder, "results" + "_test.csv"),
                f"{smiles[i]},"
                # + f"{preds_cpu[i]},"
                # + f"{label[i]},"
                + f"{y_pred[i]}"
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
        elif key in [measure_name]:
            collated_batch[key] = [int(item[key]) for item in batch]
        elif key in ["graphormer_datas"]:
            graphormer_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = graphormer_datas
        elif key in ["smiles"]:
            collated_batch[key] = [elem[key] for elem in batch]
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


print(main_args.dataset_name)

if main_args.prediction_test:

    dataset_name = main_args.dataset_name
    graph_model = main_args.graph_model
    print(f"Dataset name: {dataset_name}")

    data_path = config['data'][dataset_name]['data_root']
    measure_name = config['data'][dataset_name]['measure_name']
    num_classes = config['data'][dataset_name]['num_classes']
    result_folder = f"{config['result']['test_path']}/"

    ensure_folder_exists(result_folder)

    test_data = pd.read_csv(
        f"{data_path}/test.csv",
        sep=",",
        header=0
    )
    test_dataset = CombinedMoleculeDataset(data=test_data, graph_model=graph_model)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=False
    )

        # Instantiate and train the model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    model = MolecularEmbeddingModel(graph_model=graph_model)
    
    model_path = main_args.model_path
    print(f"Model path: {model_path}")
    model.load_state_dict(
        torch.load(model_path), strict=False
    )

    if torch.cuda.is_available():
        model = model.cuda()
            
    prediction(model=model, valid_dataloader=test_dataloader)
