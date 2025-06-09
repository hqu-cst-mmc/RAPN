import torch.nn as nn
import torch
# modify due to 228 pytorch version.
# module 'torch' has no attribute 'log_softmax'
# use F.log_softmax instead
import torch.nn.functional as F


class MLP_tf(nn.Module):
    def __init__(self, in_channel):
        super(MLP_tf, self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(in_channel, 512,bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 1, bias=True),
        )

    def forward(self, input_feature):
        predict = self.mlp(input_feature)
        return predict
