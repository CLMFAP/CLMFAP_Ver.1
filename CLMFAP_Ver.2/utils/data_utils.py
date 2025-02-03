
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from models.my_model import *
from ogb.utils.mol import smiles2graph
import ogb
from typing import Callable
from torch_geometric.data import  Data

from tqdm import tqdm
from utils.utils import *
from utils.data_utils import *
from graphormer_data.wrapper import preprocess_item
import regex as re
from models.pubchem_encoder import Encoder
import networkx as nx
from torch_geometric.utils import subgraph


def normalize_smiles(smi, canonical, isomeric):

    text_encoder = Encoder(max_length=100)
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
        if len(tokens)>100:
            # print("find long smi: ", smi)
            normalized = None

        percent = 0.25
        graph = smiles2graph(smi)
        data = Data()
        data.__num_nodes__ = int(graph["num_nodes"])
        data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
        data_remove_graph = removeSubgraph(graph=graph, percent=percent)
        data.data_remove_graph = data_remove_graph
        preprocess_item(data)
        preprocess_item(data_remove_graph)
        text_encoder.encode(tokens)
        
    except Exception as e:
        print(f"An error occurred while normalizing '{smi}': {e}")
        normalized = None
    
    return normalized
    
# Generate Fingerprints
def generate_fingerprints(smiles_list, radius):
    fingerprints = []
    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)
        fingerprints.append(fp)
    return torch.tensor(fingerprints, dtype=torch.float32)


def removeSubgraph(graph, percent=0.25):
    assert percent <= 1

    # return G, removed
   # Extract graph components
    edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
    edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.float)
    x = torch.tensor(graph['node_feat'], dtype=torch.float)
    num_nodes = graph['num_nodes']
    
    # Determine the maximum number of nodes to remove
    max_remove_nodes = max(1, int(percent * num_nodes))  # Ensure at least one node can be removed

    # Randomly select a node (atom) to remove
    # atom_to_remove = random.randint(0, num_nodes - 1)
    
    # # Find the neighbors of the selected atom
    # neighbors = edge_index[1][edge_index[0] == atom_to_remove].tolist() + \
    #             edge_index[0][edge_index[1] == atom_to_remove].tolist()
    
    # # Combine the atom and its neighbors into the list of nodes to remove
    # nodes_to_remove = set(neighbors + [atom_to_remove])

    # # Limit the nodes to remove if it exceeds the max_remove_nodes
    # if len(nodes_to_remove) > max_remove_nodes:
    #     nodes_to_remove = set(random.sample(nodes_to_remove, max_remove_nodes))

    nodes_to_remove = set()

    # While we have not reached the limit of nodes to remove
    while len(nodes_to_remove) < max_remove_nodes:
        # Randomly select a new atom (node) to remove
        atom_to_remove = random.randint(0, num_nodes - 1)

        # Add the atom and its neighbors to the removal set
        neighbors = edge_index[1][edge_index[0] == atom_to_remove].tolist() + \
                    edge_index[0][edge_index[1] == atom_to_remove].tolist()
        nodes_to_remove.update(neighbors + [atom_to_remove])

        # If we exceed the max_remove_nodes, randomly sample to fit the limit
        if len(nodes_to_remove) > max_remove_nodes:
            nodes_to_remove = set(random.sample(nodes_to_remove, max_remove_nodes))

    # Create a mask for nodes to keep
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[list(nodes_to_remove)] = False  # Mark nodes to remove as False
    
    # Use PyTorch Geometric's subgraph utility to filter nodes and edges
    new_edge_index, new_edge_attr = subgraph(mask, edge_index, edge_attr=edge_attr, relabel_nodes=True)
    new_x = x[mask]
    
    # Create a new Data object
    data = Data(x=new_x.to(torch.int64), edge_index=new_edge_index.to(torch.int64), edge_attr=new_edge_attr.to(torch.int64))
    
    return data

# Generate 3D Graphs with Padding
def generate_3d_graphs(
    smiles_list, max_atoms=50, smiles2graph: Callable = smiles2graph
):
    percent = 0.25
    smiles2graph = smiles2graph
    graphs = []
    # graphs_remove_subgraph = []
    graphormer_datas = []
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
        # data_remove_graph = removeSubgraph(graph=graph, percent=percent)
        # data.data_remove_graph = data_remove_graph

        graphs.append(data)
        graphormer_data = preprocess_item(data)
        # graphormer_data_remove_subgraph = preprocess_item(data_remove_graph)
        # graphormer_data.graphormer_remove_subgraph = graphormer_data_remove_subgraph
        graphormer_datas.append(graphormer_data)

        
        # graphs_remove_subgraph.append(graphormer_data_remove_subgraph)
        
        # print(data.x.shape)
        # print("$$$$$$$$$$$$$$$$")
    return graphs, graphormer_datas
    # return graphs, graphormer_datas, graphs_remove_subgraph

    # Generate 3D Graphs with Padding
def generate_3d_graphs_pretrain(
    smiles_list, max_atoms=50, smiles2graph: Callable = smiles2graph
):
    percent = 0.25
    smiles2graph = smiles2graph
    graphs = []
    graphs_remove_subgraph = []
    graphormer_datas = []
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
        data_remove_graph = removeSubgraph(graph=graph, percent=percent)
        data.data_remove_graph = data_remove_graph

        graphs.append(data)
        graphormer_data = preprocess_item(data)
        graphormer_data_remove_subgraph = preprocess_item(data_remove_graph)
        graphormer_data.graphormer_remove_subgraph = graphormer_data_remove_subgraph
        graphormer_datas.append(graphormer_data)

        
        graphs_remove_subgraph.append(graphormer_data_remove_subgraph)
        
        # print(data.x.shape)
        # print("$$$$$$$$$$$$$$$$")
    # return graphs, graphormer_datas
    return graphs, graphormer_datas, graphs_remove_subgraph

# Generate 3D Graphs with Padding
def generate_3d_graphs_no_graphormer(
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