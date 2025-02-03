import torch
import numpy as np
import torch.nn as nn


class BCEWithLogitsLoss(torch.nn.Module):

    def __init__(self, device, batch_size, use_cosine_similarity):
        super(BCEWithLogitsLoss, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = nn.BCEWithLogitsLoss()

    
    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(3 * self.batch_size)
        l1 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=self.batch_size)
        k1 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=2 * self.batch_size)
        k2 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=-2 * self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2 + k1 + k2))
        # print(diag)
        # print(l1)
        # print(l2)
        # print(k1)
        # print(k2)
        # print(mask)
        mask = (1 - mask).type(torch.bool)
        # print(mask)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v
    

    def forward(self, zis, zjs, zks):
        """
        Forward function to calculate the contrastive loss for three inputs zis, zjs, and zks.
        """
        # Concatenate all representations
        representations = torch.cat([zjs, zis, zks], dim=0)

        # Compute similarity matrix for all representations
        similarity_matrix = self.similarity_function(representations, representations)

        # Calculate positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        kl_pos = torch.diag(similarity_matrix, 2 * self.batch_size)
        kr_pos = torch.diag(similarity_matrix, -2 * self.batch_size)

        # Combine positives from all three directions
        # positives = torch.cat([l_pos, r_pos, kl_pos, kr_pos]).view(3 * self.batch_size, -1)
        positives_1 = torch.cat([l_pos, kl_pos]).view(3 * self.batch_size, 1)
        positives_2 = torch.cat([kr_pos, r_pos]).view(3 * self.batch_size, 1)
        positives = torch.cat([positives_1, positives_2], dim=1)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(3 * self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)

        # 创建目标标签：正样本为 1，负样本为 0
        targets = torch.cat((
            torch.ones_like(positives),  # 正样本标签（1）
            torch.zeros_like(negatives)  # 负样本标签（0）
        ), dim=1)

        # 计算损失
        loss = self.criterion(logits, targets)

        # print(similarity_matrix)
        # print(positives)
        # print(negatives)
        # print(logits)
        # print(targets)
        # print(loss)

        return loss
