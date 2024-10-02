
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from models.my_model import *
from ogb.utils.mol import smiles2graph
from typing import Callable
from torch_geometric.data import  Data

from tqdm import tqdm
from utils.utils import *
from utils.data_utils import *
from bi_graph_data.wrapper import preprocess_item
import regex as re
from models.pubchem_encoder import Encoder


def normalize_smiles(smi, canonical, isomeric):

    text_encoder = Encoder(1000)
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
        
        tokens = text_encoder.process_text([smi])
        tokens = regex.findall(smi.strip("\n"))
        
        if len(tokens)<=0:
            print("find one: ", smi)
        
        text_encoder.encode(tokens)
        
    except Exception as e:
        print(f"An error occurred while normalizing '{smi}': {e}")
        normalized = None
    
    return normalized
    
# Generate Fingerprints
def generate_fingerprints(smiles_list):
    fingerprints = []
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprints.append(fp)
    return torch.tensor(fingerprints, dtype=torch.float32)


# Generate 3D Graphs with Padding
def generate_3d_graphs(
    smiles_list, max_atoms=50, smiles2graph: Callable = smiles2graph
):
    smiles2graph = smiles2graph
    graphs = []
    bi_graph_datas = []
    for smi in tqdm(smiles_list):
        graph = smiles2graph(smi)
        # print("$$$$$$$$$$$$$$$")
        # print(smi)
        # print(f"Original smi size: {len(smi)}")
        data = Data()
        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        # print(data.x.shape)
        graphs.append(data)
        bi_graph_data = preprocess_item(data)
        bi_graph_datas.append(bi_graph_data)
        # print(data.x.shape)
        # print("$$$$$$$$$$$$$$$$")
    return graphs, bi_graph_datas

# Generate 3D Graphs with Padding
def generate_3d_graphs_no_bi_graph(
    smiles_list, max_atoms=50, smiles2graph: Callable = smiles2graph
):
    smiles2graph = smiles2graph
    graphs = []
    for smi in tqdm(smiles_list):
        graph = smiles2graph(smi)
        data = Data()
        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        graphs.append(data)
    return graphs