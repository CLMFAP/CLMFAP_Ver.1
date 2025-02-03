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
from scipy import stats
from utils.utils import *
from graphormer_data.wrapper import preprocess_item
from graphormer_data.collator import collator
from utils.data_utils import *
import regex as re
# Define the Model
from args import *
import argparse
from models.pubchem_encoder import Encoder
from repra.repra import Repra
from ptflops import get_model_complexity_info

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

data = {
    'Model': ['CustomModel'],
    'Param(M)': [0.04],  # Example value
    'Flops(G)': [0.01]   # Example value
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存或追加数据到 CSV 文件
csv_file = 'model_metrics_param_flops.csv'

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
        # print(self.targets)

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
        self.mid_liner = nn.Linear(3 * 768, 1 * 512)

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

def train(
    model,
    dataloader,
    measure_name="hiv",
    model_name = "model"
):
        model.eval()

        # 1. 计算参数数量
        def get_model_param_count(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params / 1e6  # 以百万为单位

        # 2. 计算 FLOPs
        def get_model_flops(model, input_ids, fingerprints, graphs, graphormer_datas):

            flops, _ = get_model_complexity_info(model, (input_ids, fingerprints, graphs, graphormer_datas), as_strings=False, print_per_layer_stat=False, input_constructor=lambda _: { 'input_ids': input_ids,'fingerprints': fingerprints, 'graphs': graphs, 'graphormer_datas': graphormer_datas})
            print("flops:")
            print(flops)
            return flops / 1e9  # 以十亿为单位

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]
            graphormer_datas = batch["graphormer_datas"]

            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                fingerprints = fingerprints.to("cuda")
                # attention_mask = attention_mask.to("cuda")
                graphs = graphs.to("cuda")

            # measure = model(input_ids, fingerprints, graphs, graphormer_datas)

                        # 获取模型的参数数量和 FLOPs
            param_count = get_model_param_count(model)
            flops = get_model_flops(model, input_ids, fingerprints, graphs, graphormer_datas)

            # 生成表格
            data = {
                'Model': [model_name],
                'Param(M)': [param_count],
                'Flops(G)': [flops]
            }
            df = pd.DataFrame(data)

            # 保存 TO CSV
            df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)

            print("Results saved to model_metrics_param_flops.csv")
            print(df)
            break


      
def train_no_graphormer(
    model,
    dataloader,
    measure_name="hiv",
    model_name = "model"
):
        model.eval()

        # 1. 计算参数数量
        def get_model_param_count(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params / 1e6  # 以百万为单位

        # 2. 计算 FLOPs
        def get_model_flops(model, input_ids, fingerprints, graphs):

            flops, _ = get_model_complexity_info(model, (input_ids, fingerprints, graphs), as_strings=False, print_per_layer_stat=False, input_constructor=lambda _: { 'input_ids': input_ids,'fingerprints': fingerprints, 'graphs': graphs})
            print("flops:")
            print(flops)
            return flops / 1e9  # 以十亿为单位

        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]

            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                fingerprints = fingerprints.to("cuda")
                # attention_mask = attention_mask.to("cuda")
                graphs = graphs.to("cuda")

            # measure = model(input_ids, fingerprints, graphs, graphormer_datas)

                        # 获取模型的参数数量和 FLOPs
            param_count = get_model_param_count(model)
            flops = get_model_flops(model, input_ids, fingerprints, graphs)

            # 生成表格
            data = {
                'Model': [model_name],
                'Param(M)': [param_count],
                'Flops(G)': [flops]
            }

            # 创建 DataFrame
            df = pd.DataFrame(data)

            # 保存为 CSV
            df.to_csv(csv_file, mode='a', header=not pd.io.common.file_exists(csv_file), index=False)

            print("Results saved to model_metrics_param_flops.csv")
            print(df)
            break

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
            collated_batch[key] = [item[key] for item in batch]
        elif key in ["graphormer_datas"]:
            graphormer_datas = collator([elem[key] for elem in batch])
            collated_batch[key] = graphormer_datas
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


main_args.dataset_name = 'ALL'
print(main_args.dataset_name)
if main_args.dataset_name == 'ALL':
    print("Finetune all dataset: Bioavailability,HIA,PAMPA,clintox")
    for dataset_name in ["esol"]:
        print(f"Dataset name: {dataset_name}")
        for graph_model in ["bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"]:
        # for graph_model in ["no_graphormer_mpnn"]:
            print(f"Model name: {graph_model}")

            data_path = config['data'][dataset_name]['data_root']
            measure_name = config['data'][dataset_name]['measure_name']
            num_classes = config['data'][dataset_name]['num_classes']
            result_folder = f"{config['result']['finetune_path']}/{data_path}/while_model/{graph_model}/"
            model_path = config['pretrained_model'][graph_model]['path']
            ensure_folder_exists(result_folder)

            valid_data = pd.read_csv(
                f"{data_path}/valid.csv",
                sep=",",
                header=0
            )

            valid_dataset = CombinedMoleculeDataset(data=valid_data, graph_model=graph_model)

            valid_dataloader = DataLoader(
                valid_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
            )

            model = MolecularEmbeddingModel(graph_model=graph_model)
            model.load_state_dict(
                torch.load(model_path), strict=False
            )

            if torch.cuda.is_available():
                model = model.cuda()

            if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
                train(model, valid_dataloader, measure_name=measure_name, model_name=graph_model)
            elif graph_model == 'no_graphormer_mpnn':
                train_no_graphormer(model, valid_dataloader,  measure_name=measure_name, model_name=graph_model)
