from nt_xent import NTXentLoss
from nt_xent_3 import NTXentLoss3
from nt_xent_4 import NTXentLoss4
from margin_triplet_loss import MarginTriplet
from BCEWithLogits_Loss import BCEWithLogitsLoss
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 2
feature_dim = 3

nt_xent_criterion = NTXentLoss(device, batch_size, 0.1, True)
nt_xent_criterion_3 = NTXentLoss3(device, batch_size, 0.1, True, multi_label_fun="mean")
nt_xent_criterion_4 = NTXentLoss4(device, batch_size, 0.1, True)
bceWithLogitsLoss = BCEWithLogitsLoss(device, batch_size, True)

zis = torch.randn(batch_size, feature_dim).to(device)
zjs = torch.randn(batch_size, feature_dim).to(device)
zks = torch.randn(batch_size, feature_dim).to(device)
zls = torch.randn(batch_size, feature_dim).to(device)


# for 2
# print(zis)
# print(zjs)
# print("$$$$$$$$$$$$$")
# loss = nt_xent_criterion(zis, zjs)
# print(loss)
# # similarity_matrix
# tensor([[ 1.0000, -0.7512,  0.8121, -0.9685],
#         [-0.7512,  1.0000, -0.2462,  0.6085],
#         [ 0.8121, -0.2462,  1.0000, -0.9188],
#         [-0.9685,  0.6085, -0.9188,  1.0000]], device='cuda:0')

# # positives
# tensor([[0.8121],
#         [0.6085],
#         [0.8121],
#         [0.6085]], device='cuda:0')

# # negatives
# tensor([[-0.7512, -0.9685],
#         [-0.7512, -0.2462],
#         [-0.2462, -0.9188],
#         [-0.9685, -0.9188]], device='cuda:0')

# logits
# tensor([[ 8.1213, -7.5122, -9.6852],
#         [ 6.0852, -7.5122, -2.4619],
#         [ 8.1213, -2.4619, -9.1878],
#         [ 6.0852, -9.6852, -9.1878]], device='cuda:0')

# labels
# tensor([0, 0, 0, 0], device='cuda:0')


# for 3
# print(zis)
# print(zjs)
# print(zks)
# print("$$$$$$$$$$$$$")
# loss = nt_xent_criterion_3(zis, zjs, zks, "sum")

# similarity_matrix
# tensor([[ 1.0000, -0.7512,  0.8121, -0.9685,  0.1961, -0.0434],
#         [-0.7512,  1.0000, -0.2462,  0.6085, -0.7316,  0.6520],
#         [ 0.8121, -0.2462,  1.0000, -0.9188, -0.4093,  0.4166],
#         [-0.9685,  0.6085, -0.9188,  1.0000,  0.0421, -0.0684],
#         [ 0.1961, -0.7316, -0.4093,  0.0421,  1.0000, -0.6945],
#         [-0.0434,  0.6520,  0.4166, -0.0684, -0.6945,  1.0000]],
#        device='cuda:0')


# positives
# tensor([[ 0.8121,  0.6085],
#         [-0.4093, -0.0684],
#         [ 0.8121,  0.6085],
#         [-0.4093, -0.0684],
#         [ 0.1961,  0.6520],
#         [ 0.1961,  0.6520]], device='cuda:0')


# negatives
# tensor([[-0.7512, -0.9685, -0.0434],
#         [-0.7512, -0.2462, -0.7316],
#         [-0.2462, -0.9188,  0.4166],
#         [-0.9685, -0.9188,  0.0421],
#         [-0.7316,  0.0421, -0.6945],
#         [-0.0434,  0.4166, -0.6945]], device='cuda:0')

# labels
# tensor([0, 0, 0, 0, 0, 0], device='cuda:0')

# for 4
# print(zis)
# print(zjs)
# print(zks)
# print("$$$$$$$$$$$$$")
# loss = nt_xent_criterion_4(zis, zjs, zks, zls)

# similarity_matrix


# positives


# negatives


def nt_xent_loss(zi, zj):
        """
        NT-Xent Loss implementation.
        
        Args:
            zi (torch.Tensor): Embedding batch for the first view, shape (batch_size, embed_dim).
            zj (torch.Tensor): Embedding batch for the second view, shape (batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Scalar NT-Xent loss.
        """
        temperature = 0.1
        batch_size = zi.size(0)
        
        # Normalize embeddings
        zi = F.normalize(zi, p=2, dim=1)
        zj = F.normalize(zj, p=2, dim=1)
        
        # Concatenate embeddings for similarity computation
        representations = torch.cat([zi, zj], dim=0)  # Shape: (2 * batch_size, embed_dim)
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        
        # Compute pairwise similarity
        # similarity_matrix = torch.mm(representations, representations.T)  # Shape: (2 * batch_size, 2 * batch_size)
        similarity_matrix = cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0))
        
        # Scale similarities with temperature
        logits = similarity_matrix / temperature
        
        # Create labels for positives
        labels = torch.arange(batch_size).repeat(2).to(zi.device)  # Shape: (2 * batch_size,)
        
        # Exclude self-similarities (mask diagonal)
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(zi.device)
        logits = logits[mask].view(2 * batch_size, -1)  # Exclude diagonal
        print(similarity_matrix)
        print(logits)
        print(labels)
        print(mask)
        
        # Compute NT-Xent loss
        loss = F.cross_entropy(logits, labels)
        return loss

# print(nt_xent_loss(zi=zis, zj=zjs))

def nt_logistic_loss(zi, zj):
        """
        NT-Logistic Loss implementation.
        
        Args:
            zi (torch.Tensor): Embedding batch for the first view, shape (batch_size, embed_dim).
            zj (torch.Tensor): Embedding batch for the second view, shape (batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Scalar NT-Logistic loss.
        """
        temperature = 0.1
        batch_size = zi.size(0)
        
        # Normalize embeddings
        zi = F.normalize(zi, p=2, dim=1)
        zj = F.normalize(zj, p=2, dim=1)
        
        # Compute positive similarities
        positive_similarities = torch.sum(zi * zj, dim=1) / temperature  # Shape: (batch_size,)
        
        # Compute negative similarities
        negatives_zi = torch.mm(zi, zj.T) / temperature  # Shape: (batch_size, batch_size)
        negatives_zj = torch.mm(zj, zi.T) / temperature  # Shape: (batch_size, batch_size)
        
        # Mask to exclude self-pairs
        mask = ~torch.eye(batch_size, dtype=torch.bool).to(zi.device)
        negatives_zi = negatives_zi[mask].view(batch_size, -1)  # Exclude self-pairs
        negatives_zj = negatives_zj[mask].view(batch_size, -1)  # Exclude self-pairs
        
        # Compute NT-Logistic loss
        positive_loss = -torch.log(torch.sigmoid(positive_similarities)).mean()
        negative_loss_zi = -torch.log(1 - torch.sigmoid(negatives_zi)).mean()
        negative_loss_zj = -torch.log(1 - torch.sigmoid(negatives_zj)).mean()
        
        total_loss = positive_loss + (negative_loss_zi + negative_loss_zj) / 2
        return total_loss

# print(nt_logistic_loss(zi=zis, zj=zjs))



# loss = MarginTriplet()
# print(loss(zis, zjs))

# def find_longest_line(file_path):
#     longest_line = ""
#     longest_length = 0
    
#     with open(file_path, 'r') as file:
#         for line in file:
#             stripped_line = line.strip()
#             if len(stripped_line) > longest_length:
#                 longest_line = stripped_line
#                 longest_length = len(stripped_line)
    
#     return longest_line, longest_length

# longest_line, longest_length = find_longest_line("/home/user/workspace/MY_MODEL/data/CID-SMILES-CANONICAL.smi")
# print(longest_line)
# print(longest_length)

# loss = bceWithLogitsLoss(zis, zjs,zks)
# import json
# config_path = "configs/config.json"
# config = json.load(open(config_path))
# print(config["pretrain"]["batch_size"])
# config["pretrain"]["batch_size"] = 50
# print(config["pretrain"]["batch_size"])


from ogb.utils.mol import smiles2graph
smi="NNC(=O)c1ccncc1"
graph = smiles2graph(smi)
print(graph)