import torch
import torch.nn as nn
import torch.optim as optim


# 定义 MLP 模型
class FPMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FPMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    # def __init__(self, input_size, hidden_sizes, output_size):
    #     super(FPMLP, self).__init__()
    #     layers = []
    #     for i in range(5):
    #         if i == 0:
    #             layers.append(nn.Linear(input_size, hidden_sizes))
    #         else:
    #             layers.append(nn.Linear(hidden_sizes, hidden_sizes))
    #         layers.append(nn.BatchNorm1d(hidden_sizes))
    #         layers.append(nn.ReLU())
    #         layers.append(nn.Dropout(0.5))
    #     layers.append(nn.Linear(hidden_sizes, output_size))
    #     self.network = nn.Sequential(*layers)

    # def forward(self, x):
    #     return self.network(x)
