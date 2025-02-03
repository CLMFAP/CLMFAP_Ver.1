import torch
import numpy as np
import torch.nn.functional as F


class MarginTriplet(torch.nn.Module):

    def __init__(self):
        super(MarginTriplet, self).__init__()
        self.margin = 1.0
        self.loss = torch.nn.TripletMarginLoss(margin=self.margin,p=2)
        

    def forward(self, zi, zj):
        
        anchors = zi

        # 正例为同一个batch的fingerprints_embeddings
        positives = zj

        # 负例为随机选择或顺序错位的fingerprints_embeddings，这里我们假设错位选择
        # 即每个元素选择下一个，最后一个选择第一个
        negatives = torch.cat((zj[1:], zj[0].unsqueeze(0)), dim=0)

        # 定义损失函数
        margin = 1.0
        loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

        # 计算损失
        loss = loss_fn(anchors, positives, negatives)
        return loss
