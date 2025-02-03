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
# data_path = config['data'][dataset_name]['data_root']
# measure_name = config['data'][dataset_name]['measure_name']
# num_classes = config['data'][dataset_name]['num_classes']
# result_folder = f"{config['result']['finetune_path']}/{data_path}/{main_args.graph_model}/"
# ensure_folder_exists(result_folder)

# Combined dataset class for SMILES, fingerprints, and graphs
# class CombinedMoleculeDataset(Dataset):
#     def __init__(self, data, graph_model):
#         self.data = data
#         self.data["canonical_smiles"] = data["smiles"].apply(
#             lambda smi: normalize_smiles(smi, canonical=True, isomeric=False)
#         )
#         self.smiles_list_good = self.data.dropna(subset=['canonical_smiles'])
#         self.smiles_list_good.reset_index(drop=True, inplace=True)

#         assert (
#             self.smiles_list_good["canonical_smiles"].isna().sum() == 0
#         ), "There are still NaN values in 'canonical_smiles'"
#         print(
#             "Dropped {} invalid smiles".format(
#                 len(self.data) - len(self.smiles_list_good)
#             )
#         )
#         print("Valid smiles {}".format(len(self.smiles_list_good)))

#         self.smiles_list = self.smiles_list_good["smiles"]
#         self.targets = self.smiles_list_good[measure_name]
#         self.text_encoder = Encoder(100000)
#         print("Loading data...")
#         self.encodings = self.text_encoder.process(self.smiles_list)
#         self.fingerprints = generate_fingerprints(self.smiles_list)
#         # self.graphs, self.graphormer_datas = generate_3d_graphs(
#         #     self.smiles_list, smiles2graph=smiles2graph
#         # )

#         if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
#             self.graphs, self.graphormer_datas = generate_3d_graphs(self.smiles_list)
#         elif graph_model == 'no_graphormer_mpnn':
#             self.graphs = generate_3d_graphs_no_graphormer(self.smiles_list)

#     def __len__(self):
#         return len(self.fingerprints)

#     def __getitem__(self, idx):
#         # item = {key: val[idx] for key, val in self.encodings.items()}
#         item = {}

#         item["input_ids"] = self.encodings[0][idx]
#         item["fingerprints"] = self.fingerprints[idx]
#         item["graphs"] = self.graphs[idx]
#         item[measure_name] = self.targets[idx]
#         # self.graphormer_datas[idx].idx = idx
#         # item["graphormer_datas"] = self.graphormer_datas[idx]

#         if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
#             self.graphormer_datas[idx].idx = idx
#             item["graphormer_datas"] = self.graphormer_datas[idx]
#         return item

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
        self.fingerprints = generate_fingerprints(self.smiles_list,main_args.fp_radius)
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

        self.net = self.Net(768, 1)

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


#     return loss
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def criterion(out_1, out_2, batch_size, temperature=0.5):
    device = torch.device("cuda:0" if out_1.is_cuda else "cpu")
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
    # contrastive loss
    loss = (-torch.log(pos / (pos + Ng))).mean()

    return loss


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def train(
    model,
    dataloader,
    valid_dataloader,
    optimizer,
    epochs=22,
    measure_name="hiv",
):
    min_loss = {
        measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
        measure_name + "min_epoch": 0,
    }
    dataset = "valid"
    # results_dir = f"finetune_{dataset}_{measure_name}/weight442"
    fc = nn.Linear(128, 2).to("cuda")
    loss_fun = nn.L1Loss()
    # loss_fun = nn.BCELoss()

    loss_file = open(f"{result_folder}/loss.txt", "a")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]
            graphormer_datas = batch["graphormer_datas"]

            label = torch.tensor(batch[measure_name]).to("cuda")
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                fingerprints = fingerprints.to("cuda")
                # attention_mask = attention_mask.to("cuda")
                graphs = graphs.to("cuda")

            optimizer.zero_grad()
            measure = model(input_ids, fingerprints, graphs, graphormer_datas)
            measure = measure.squeeze()
            label = label.float()
            loss = loss_fun(measure, label)
            # print("$$$$ Log output")
            # print(label)
            # print(measure)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        loss_file.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}" + "\n")
        tensorboard_logs = {}
        if epoch % 1 == 0:
            total_loss = 0.0
            total_r2 = 0.0
            total_pearsonr = 0.0
            model.eval()
            batch_id = 0
            for batch in tqdm(valid_dataloader):
                batch_id += 1
                input_ids = batch["input_ids"]
                # attention_mask = batch["attention_mask"]
                fingerprints = batch["fingerprints"]
                graphs = batch["graphs"]
                graphormer_datas = batch["graphormer_datas"]
                # if len(set(batch[measure_name])) != 1:
                #     print("All labels are not the same: ", batch[measure_name])
                #     print(len(valid_dataloader))
                #     print(len(batch))

                label = torch.tensor(batch[measure_name]).to("cuda")
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                    fingerprints = fingerprints.to("cuda")
                    # attention_mask = attention_mask.to("cuda")
                    graphs = graphs.to("cuda")

                measure = model(input_ids, fingerprints, graphs, graphormer_datas)
                measure = measure.squeeze()
                label = label.float()

                loss = loss_fun(measure, label)

                actuals_cpu = label.detach().cpu().numpy()
                preds_cpu = measure.detach().cpu().numpy()

                # classif
                # preds_0_cpu = preds_cpu[:, 0]
                # preds_cpu = preds_cpu[:, 1]

                # fpr, tpr, threshold = roc_curve(actuals_cpu, preds_cpu)
                # roc_auc = auc(fpr, tpr)
                # prec = average_precision_score(actuals_cpu, preds_cpu)

                # y_pred = np.where(preds_cpu >= 0.5, 1, 0)
                # accuracy = accuracy_score(actuals_cpu, y_pred)
                # print(actuals_cpu)
                # print(preds_cpu)
                pearson_r = stats.pearsonr(actuals_cpu, preds_cpu)
                r2 = r2_score(actuals_cpu, preds_cpu)

                # print(pearson_r)


                tensorboard_logs.update(
                    {
                        # dataset + "_avg_val_loss": avg_loss,
                        measure_name + "_" + dataset + "_loss": loss.item(),
                        measure_name + "_" + dataset + "_r2": r2,
                        measure_name + "_" + dataset + "_pearsonr": pearson_r
                    }
                )


                append_to_file(
                    os.path.join(result_folder, "results" + "_batch.csv"),
                    f"{measure_name}, epoch {epoch+1}, batch {batch_id} "
                    + f"{loss},"
                    # + f"{tensorboard_logs[measure_name + '_test_loss']},"
                    + f"{r2},"
                    + f"{pearson_r[0]},"
                )

                total_loss += loss.item()
                total_r2 += r2
                total_pearsonr += pearson_r[0]
                # print(f"Sub Validation Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            avg_loss = total_loss / len(valid_dataloader)
            avg_r2 = total_r2 / len(valid_dataloader)
            avg_pearsonr = total_pearsonr / len(valid_dataloader)

            print(f"Validation Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
            print(f"Validation Epoch {epoch+1}/{epochs}, avg_r2: {avg_r2}")
            print(f"Validation Epoch {epoch+1}/{epochs}, avg_pearsonr: {avg_pearsonr}")

            if (
                tensorboard_logs[measure_name + "_valid_loss"]
                < min_loss[measure_name + "min_valid_loss"]
            ):
                min_loss[measure_name + "min_valid_loss"] = tensorboard_logs[
                    measure_name + "_valid_loss"
                ]
                # min_loss[measure_name + "min_test_loss"] = tensorboard_logs[
                #     measure_name + "_test_loss"
                # ]
                min_loss[measure_name + "min_epoch"] = epoch + 1
                min_loss[measure_name + "max_valid_r2"] = tensorboard_logs[
                    measure_name + "_valid_r2"
                ]
                # min_loss[measure_name + "max_test_rocauc"] = tensorboard_logs[
                #     measure_name + "_test_rocauc"
                # ]
                min_loss[measure_name + "max_valid_pearsonr"] = tensorboard_logs[
                    measure_name + "_valid_pearsonr"
                ]
                # min_loss[measure_name + "max_test_prec"] = tensorboard_logs[
                #     measure_name + "_test_prec"
                # ]

            tensorboard_logs[measure_name + "_min_valid_loss"] = min_loss[
                measure_name + "min_valid_loss"
            ]
            # tensorboard_logs[measure_name + "_min_test_loss"] = min_loss[
            #     measure_name + "min_test_loss"
            # ]
            # tensorboard_logs[measure_name + "_max_test_rocauc"] = min_loss[
            #     measure_name + "max_test_rocauc"
            # ]
            tensorboard_logs[measure_name + "_max_valid_r2"] = min_loss[
                measure_name + "max_valid_r2"
            ]
            tensorboard_logs[measure_name + "_max_valid_pearsonr"] = min_loss[
                measure_name + "max_valid_pearsonr"
            ]
            # tensorboard_logs[measure_name + "_max_test_prec"] = min_loss[
            #     measure_name + "max_test_prec"
            # ]

            append_to_file(
                os.path.join(result_folder, "results" + ".csv"),
                f"{measure_name}, {epoch+1},"
                + f"{tensorboard_logs[measure_name + '_valid_loss']},"
                # + f"{tensorboard_logs[measure_name + '_test_loss']},"
                + f"{min_loss[measure_name + 'min_epoch']},"
                + f"{min_loss[measure_name + 'min_valid_loss']},"
                + f"{min_loss[measure_name + 'max_valid_r2']},"
                # + f"{min_loss[measure_name + 'max_test_rocauc']},"
                + f"{min_loss[measure_name + 'max_valid_pearsonr']},"
                # + f"{min_loss[measure_name + 'max_test_prec']},",
                + f"######## "
                + f"{avg_loss}, "
                + f"{avg_r2}, "
                + f"{avg_pearsonr}, "
            )
        torch.save(
            model.state_dict(),
            f"{result_folder}/checkpoint.ckpt",
        )
        
def train_no_graphormer(
    model,
    dataloader,
    valid_dataloader,
    optimizer,
    epochs=10,
    measure_name="hiv",
):
    min_loss = {
        measure_name + "min_valid_loss": torch.finfo(torch.float32).max,
        measure_name + "min_epoch": 0,
    }
    dataset = "valid"
    # results_dir = f"finetune_{dataset}_{measure_name}/weight442"
    fc = nn.Linear(128, 2).to("cuda")
    loss_fun = nn.L1Loss()
    # loss_fun = nn.BCELoss()

    loss_file = open(f"{result_folder}/loss.txt", "a")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"]
            # attention_mask = batch["attention_mask"]
            fingerprints = batch["fingerprints"]
            graphs = batch["graphs"]
            # graphormer_datas = batch["graphormer_datas"]

            label = torch.tensor(batch[measure_name]).to("cuda")
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
                fingerprints = fingerprints.to("cuda")
                # attention_mask = attention_mask.to("cuda")
                graphs = graphs.to("cuda")

            optimizer.zero_grad()
            measure = model(input_ids, fingerprints, graphs)
            measure = measure.squeeze()
            label = label.float()

            loss = loss_fun(measure, label)
            # print("$$$$ Log output")
            # print(label)
            # print(measure)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
        loss_file.write(str(avg_loss) + "\n")
        tensorboard_logs = {}
        if epoch % 1 == 0:
            total_loss = 0.0
            total_r2 = 0.0
            total_pearsonr = 0.0
            model.eval()
            batch_id = 0
            for batch in tqdm(valid_dataloader):
                batch_id += 1
                input_ids = batch["input_ids"]
                # attention_mask = batch["attention_mask"]
                fingerprints = batch["fingerprints"]
                graphs = batch["graphs"]
                # graphormer_datas = batch["graphormer_datas"]
                label = torch.tensor(batch[measure_name]).to("cuda")
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                    fingerprints = fingerprints.to("cuda")
                    # attention_mask = attention_mask.to("cuda")
                    graphs = graphs.to("cuda")

                measure = model(input_ids, fingerprints, graphs)
                measure = measure.squeeze()
                label = label.float()

                loss = loss_fun(measure, label)
                # print("$$$$ Log output")
                # print(label)
                # print(measure)
                # measure = F.softmax(measure, dim=1)

                actuals_cpu = label.detach().cpu().numpy()
                preds_cpu = measure.detach().cpu().numpy()

                # classif
                # preds_0_cpu = preds_cpu[:, 0]
                # preds_cpu = preds_cpu[:, 1]

                pearson_r = stats.pearsonr(actuals_cpu, preds_cpu)
                r2 = r2_score(actuals_cpu, preds_cpu)

                # print(pearson_r)

                tensorboard_logs.update(
                    {
                        # dataset + "_avg_val_loss": avg_loss,
                        measure_name + "_" + dataset + "_loss": loss.item(),
                        measure_name + "_" + dataset + "_r2": r2,
                        measure_name + "_" + dataset + "_pearsonr": pearson_r
                    }
                )

                append_to_file(
                    os.path.join(result_folder, "results" + "_batch.csv"),
                    f"{measure_name}, epoch {epoch+1}, batch {batch_id} "
                    + f"{loss},"
                    # + f"{tensorboard_logs[measure_name + '_test_loss']},"
                    + f"{r2},"
                    + f"{pearson_r[0]},"
                )

                total_loss += loss.item()
                total_r2 += r2
                total_pearsonr += pearson_r[0]
                # print(f"Sub Validation Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            avg_loss = total_loss / len(valid_dataloader)
            avg_r2 = total_r2 / len(valid_dataloader)
            avg_pearsonr = total_pearsonr / len(valid_dataloader)

            print(f"Validation Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")
            print(f"Validation Epoch {epoch+1}/{epochs}, avg_r2: {avg_r2}")
            print(f"Validation Epoch {epoch+1}/{epochs}, avg_pearsonr: {avg_pearsonr}")

            if (
                tensorboard_logs[measure_name + "_valid_loss"]
                < min_loss[measure_name + "min_valid_loss"]
            ):
                min_loss[measure_name + "min_valid_loss"] = tensorboard_logs[
                    measure_name + "_valid_loss"
                ]
                # min_loss[measure_name + "min_test_loss"] = tensorboard_logs[
                #     measure_name + "_test_loss"
                # ]
                min_loss[measure_name + "min_epoch"] = epoch + 1
                min_loss[measure_name + "max_valid_r2"] = tensorboard_logs[
                    measure_name + "_valid_r2"
                ]
                # min_loss[measure_name + "max_test_rocauc"] = tensorboard_logs[
                #     measure_name + "_test_rocauc"
                # ]
                min_loss[measure_name + "max_valid_pearsonr"] = tensorboard_logs[
                    measure_name + "_valid_pearsonr"
                ]
                # min_loss[measure_name + "max_test_prec"] = tensorboard_logs[
                #     measure_name + "_test_prec"
                # ]

            tensorboard_logs[measure_name + "_min_valid_loss"] = min_loss[
                measure_name + "min_valid_loss"
            ]
            # tensorboard_logs[measure_name + "_min_test_loss"] = min_loss[
            #     measure_name + "min_test_loss"
            # ]
            # tensorboard_logs[measure_name + "_max_test_rocauc"] = min_loss[
            #     measure_name + "max_test_rocauc"
            # ]
            tensorboard_logs[measure_name + "_max_valid_r2"] = min_loss[
                measure_name + "max_valid_r2"
            ]
            tensorboard_logs[measure_name + "_max_valid_pearsonr"] = min_loss[
                measure_name + "max_valid_pearsonr"
            ]
            # tensorboard_logs[measure_name + "_max_test_prec"] = min_loss[
            #     measure_name + "max_test_prec"
            # ]

            append_to_file(
                os.path.join(result_folder, "results" + ".csv"),
                f"{measure_name}, {epoch+1},"
                + f"{tensorboard_logs[measure_name + '_valid_loss']},"
                # + f"{tensorboard_logs[measure_name + '_test_loss']},"
                + f"{min_loss[measure_name + 'min_epoch']},"
                + f"{min_loss[measure_name + 'min_valid_loss']},"
                + f"{min_loss[measure_name + 'max_valid_r2']},"
                # + f"{min_loss[measure_name + 'max_test_rocauc']},"
                + f"{min_loss[measure_name + 'max_valid_pearsonr']},"
                # + f"{min_loss[measure_name + 'max_test_prec']},",
                + f"######## "
                + f"{avg_loss}, "
                + f"{avg_r2}, "
                + f"{avg_pearsonr}, "
            )

        torch.save(
            model.state_dict(),
            f"{result_folder}/checkpoint.ckpt",
        )

def test_with_graphormer(
    model,
    valid_dataloader,
    measure_name="hiv",
):

    with open(result_folder + "results" + "_test.csv", 'a') as file:
        headers = "label,y_pred"
        file.write(headers + "\n")

    model.eval()
    for batch in tqdm(valid_dataloader):
        input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        fingerprints = batch["fingerprints"]
        graphs = batch["graphs"]
        graphormer_datas = batch["graphormer_datas"]

        label = torch.tensor(batch[measure_name]).to("cuda")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            fingerprints = fingerprints.to("cuda")
            # attention_mask = attention_mask.to("cuda")
            graphs = graphs.to("cuda")

        # print(input_ids.device.type)
        # print(fingerprints.device.type)
        # print(graphs.device.type)
        # print(graphormer_datas.device.type)

        measure = model(input_ids, fingerprints, graphs, graphormer_datas)

        actuals_cpu = label.detach().cpu().numpy()
        preds_cpu = measure.detach().cpu().numpy()

        for i in range(len(preds_cpu)):
            append_to_file(
                os.path.join(result_folder, "results" + "_test.csv"),
                f"{actuals_cpu[i]},"
                + f"{preds_cpu[i][0]}"
            )


def test_no_graphormer(
    model,
    valid_dataloader,
    measure_name="hiv",
):

    with open(result_folder + "results" + "_test.csv", 'a') as file:
        headers = "label,y_pred"
        file.write(headers + "\n")

    model.eval()
    batch_id = 0
    for batch in tqdm(valid_dataloader):
        batch_id += 1
        input_ids = batch["input_ids"]
        fingerprints = batch["fingerprints"]
        graphs = batch["graphs"]

        label = torch.tensor(batch[measure_name]).to("cuda")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            fingerprints = fingerprints.to("cuda")
            graphs = graphs.to("cuda")

        measure = model(input_ids, fingerprints, graphs)

        actuals_cpu = label.detach().cpu().numpy()
        preds_cpu = measure.detach().cpu().numpy()

        for i in range(len(preds_cpu)):
            append_to_file(
                os.path.join(result_folder, "results" + "_test.csv"),
                f"{actuals_cpu[i]},"
                + f"{preds_cpu[i][0]}"
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
        else:
            padded_data = torch.stack([elem[key] for elem in batch])
            padded_data = padded_data.squeeze(1)
            collated_batch[key] = padded_data

    return collated_batch


print(main_args.dataset_name)
print(main_args.dataset_name == 'ALL')
print(main_args.test)

if main_args.test:
    # graph_model='bi_graphormer_mpnn'
    # data_path = config['data'][dataset_name]['data_root']
    # measure_name = config['data'][dataset_name]['measure_name']
    # num_classes = config['data'][dataset_name]['num_classes']
    # result_folder = f"{config['result']['test_path']}/{data_path}/{graph_model}/"
    # model_path = "workspace/MY_MODEL/result/finetune_result/data/clintox/bi_graphormer_mpnn/checkpoint.ckpt"
    # ensure_folder_exists(result_folder)

    # test_data = pd.read_csv(
    #     f"{data_path}/test.csv",
    #     sep=",",
    #     header=0
    # )
    # test_dataset = CombinedMoleculeDataset(data=test_data, graph_model=graph_model)
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=False
    # )

    #     # Instantiate and train the model
    # def weights_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)


    # model = MolecularEmbeddingModel(graph_model=graph_model)
    # model.load_state_dict(
    #     torch.load(model_path), strict=False
    # )

    # if torch.cuda.is_available():
    #     model = model.cuda()

    for dataset_name in ["PPBR", "Caco2"]:
        print(f"Dataset name: {dataset_name}")
        for graph_model in ["bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"]:
        # for graph_model in ["no_graphormer_mpnn"]:
            data_path = config['data'][dataset_name]['data_root']
            measure_name = config['data'][dataset_name]['measure_name']
            num_classes = config['data'][dataset_name]['num_classes']
            result_folder = f"{config['result']['test_path']}/{data_path}/{graph_model}/"
    
            ensure_folder_exists(result_folder)

            test_data = pd.read_csv(
                f"{data_path}/test.csv",
                sep=",",
                header=0
            )
            test_dataset = CombinedMoleculeDataset(data=test_data, graph_model=graph_model)
            test_dataloader = DataLoader(
                test_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn
            )

                # Instantiate and train the model
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


            model = MolecularEmbeddingModel(graph_model=graph_model)
            
            model_path = f"/home/user/workspace/MY_MODEL/result/finetune_result/data/{dataset_name}/{graph_model}/checkpoint.ckpt"
            print(f"Model path: {model_path}")

            model.load_state_dict(
                torch.load(model_path), strict=False
            )

            if torch.cuda.is_available():
                model = model.cuda()
            if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
                test_with_graphormer(model=model, valid_dataloader=test_dataloader,measure_name=measure_name)
            elif graph_model == 'no_graphormer_mpnn':
                test_no_graphormer(model=model, valid_dataloader=test_dataloader,measure_name=measure_name)
    

elif main_args.dataset_name == 'ALL':
    print("Finetune all dataset: Bioavailability,HIA,PAMPA,clintox")
    # for dataset_name in ["Bioavailability","HIA","PAMPA","clintox"]:
    # for dataset_name in ["PPBR","Caco2"]:
    for dataset_name in ["esol"]:
        print(f"Dataset name: {dataset_name}")
        # for graph_model in ["bi_graphormer_mpnn","bi_graphormer_no_mpnn","no_graphormer_mpnn","original_graphormer_mpnn","original_graphormer_no_mpnn"]:
        for graph_model in ["bi_graphormer_no_mpnn"]:
            print(f"Model name: {graph_model}")

            data_path = config['data'][dataset_name]['data_root']
            measure_name = config['data'][dataset_name]['measure_name']
            num_classes = config['data'][dataset_name]['num_classes']
            result_folder = f"{config['result']['finetune_path']}/{data_path}/{graph_model}/"
            model_path = config['pretrained_model'][graph_model]['path']
            model_path = f"/home/user/workspace/MY_MODEL/result/pretrain_result_bi_graphormer_no_mpnn_bce_with_logits_remove_sub_2/checkpoint_4.ckpt"
            ensure_folder_exists(result_folder)

            train_data = pd.read_csv(
                f"{data_path}/train.csv",
                sep=",",
                header=0
            )
            test_data = pd.read_csv(
                f"{data_path}/test.csv",
                sep=",",
                header=0
            )
            valid_data = pd.read_csv(
                f"{data_path}/valid.csv",
                sep=",",
                header=0
            )

            # Create dataset and dataloader
            train_dataset = CombinedMoleculeDataset(data=train_data, graph_model=graph_model)
            test_dataset = CombinedMoleculeDataset(data=test_data, graph_model=graph_model)
            valid_dataset = CombinedMoleculeDataset(data=valid_data, graph_model=graph_model)

            # train_labels=train_dataset.targets
            # train_class_sample_count=np.bincount(train_labels)
            # train_weights=1.0/train_class_sample_count
            # train_sample_weights=train_weights[train_labels]
            # train_sampler=WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

            # valid_labels=valid_dataset.targets
            # valid_class_sample_count=np.bincount(valid_labels)
            # valid_weights=1.0/valid_class_sample_count
            # valid_sample_weights=valid_weights[valid_labels]
            # valid_sampler=WeightedRandomSampler(weights=valid_sample_weights, num_samples=len(valid_sample_weights), replacement=True)

            # dataset = CombinedMoleculeDataset()
            train_dataloader = DataLoader(
                train_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
            )
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
            )

            # Instantiate and train the model
            def weights_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


            model = MolecularEmbeddingModel(graph_model=graph_model)

            # print(model.state_dict().keys())
            print(f"Load Model from: {model_path}")
            model.load_state_dict(
                torch.load(model_path), strict=False
            )
            # model.apply(weights_init)

            if torch.cuda.is_available():
                model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

            if graph_model == 'bi_graphormer_mpnn' or graph_model == 'original_graphormer_no_mpnn' or graph_model== 'original_graphormer_mpnn' or graph_model=='bi_graphormer_no_mpnn':
                train(model, train_dataloader, valid_dataloader, optimizer, epochs=15, measure_name=measure_name)
            elif graph_model == 'no_graphormer_mpnn':
                train_no_graphormer(model, train_dataloader, valid_dataloader, optimizer, epochs=5, measure_name=measure_name)

else:
    data_path = config['data'][dataset_name]['data_root']
    measure_name = config['data'][dataset_name]['measure_name']
    num_classes = config['data'][dataset_name]['num_classes']
    result_folder = f"{config['result']['finetune_path']}/{data_path}/{main_args.graph_model}/"
    ensure_folder_exists(result_folder)

    train_data = pd.read_csv(
        f"{data_path}/train.csv",
        sep=",",
        header=0
    )
    test_data = pd.read_csv(
        f"{data_path}/test.csv",
        sep=",",
        header=0
    )
    valid_data = pd.read_csv(
        f"{data_path}/valid.csv",
        sep=",",
        header=0
    )

    # Create dataset and dataloader
    train_dataset = CombinedMoleculeDataset(data=train_data,graph_model=main_args.graph_model)
    test_dataset = CombinedMoleculeDataset(data=test_data,graph_model=main_args.graph_model)
    valid_dataset = CombinedMoleculeDataset(data=valid_data,graph_model=main_args.graph_model)

    # train_labels=train_dataset.targets
    # train_class_sample_count=np.bincount(train_labels)
    # train_weights=1.0/train_class_sample_count
    # train_sample_weights=train_weights[train_labels]
    # train_sampler=WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    # valid_labels=valid_dataset.targets
    # valid_class_sample_count=np.bincount(valid_labels)
    # valid_weights=1.0/valid_class_sample_count
    # valid_sample_weights=valid_weights[valid_labels]
    # valid_sampler=WeightedRandomSampler(weights=valid_sample_weights, num_samples=len(valid_sample_weights), replacement=True)

    # dataset = CombinedMoleculeDataset()
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config["finetune"]["batch_size"], shuffle=False, collate_fn=custom_collate_fn,drop_last=True
    )

    # Instantiate and train the model
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    model = MolecularEmbeddingModel(graph_model=main_args.graph_model)

    # print(model.state_dict().keys())
    # print(torch.load(f"checkpoint_12.ckpt").keys)
    model.load_state_dict(
        torch.load(main_args.model_path), strict=False
    )
    # model.apply(weights_init)

    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    if main_args.graph_model == 'bi_graphormer_mpnn' or main_args.graph_model == 'original_graphormer_no_mpnn' or main_args.graph_model== 'original_graphormer_mpnn' or main_args.graph_model=='bi_graphormer_no_mpnn':
        train(model, train_dataloader, valid_dataloader, optimizer, epochs=20)
    elif main_args.graph_model == 'no_graphormer_mpnn':
        train_no_graphormer(model, train_dataloader, valid_dataloader, optimizer, epochs=20)
