import torch
import numpy as np
import torch.nn.functional as F


class NTLogisticLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTLogisticLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, zi, zj):
        
        """
        NT-Logistic Loss implementation.
        
        Args:
            zi (torch.Tensor): Embedding batch for the first view, shape (batch_size, embed_dim).
            zj (torch.Tensor): Embedding batch for the second view, shape (batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Scalar NT-Logistic loss.
        """
        
        # Normalize embeddings
        zi = F.normalize(zi, p=2, dim=1)
        zj = F.normalize(zj, p=2, dim=1)
        
        # Compute positive similarities
        positive_similarities = torch.sum(zi * zj, dim=1) / self.temperature  # Shape: (batch_size,)
        
        # Compute negative similarities
        negatives_zi = torch.mm(zi, zj.T) / self.temperature  # Shape: (batch_size, batch_size)
        negatives_zj = torch.mm(zj, zi.T) / self.temperature  # Shape: (batch_size, batch_size)
        
        # Mask to exclude self-pairs
        mask = ~torch.eye(self.batch_size, dtype=torch.bool).to(zi.device)
        negatives_zi = negatives_zi[mask].view(self.batch_size, -1)  # Exclude self-pairs
        negatives_zj = negatives_zj[mask].view(self.batch_size, -1)  # Exclude self-pairs
        
        # Compute NT-Logistic loss
        positive_loss = -torch.log(torch.sigmoid(positive_similarities)).mean()
        negative_loss_zi = -torch.log(1 - torch.sigmoid(negatives_zi)).mean()
        negative_loss_zj = -torch.log(1 - torch.sigmoid(negatives_zj)).mean()
        
        total_loss = positive_loss + (negative_loss_zi + negative_loss_zj) / 2
        return total_loss
