import torch.nn as nn
import torch
# modify due to 228 pytorch version.
# module 'torch' has no attribute 'log_softmax'
# use F.log_softmax instead
import torch.nn.functional as F


class EVnet(nn.Module):
    def __init__(self, in_channel):
        super(EVnet, self).__init__()
        self.alphaNet = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_channel, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 1),
        )
        self.betaNet = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_channel, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 1),
        )


    def forward(self, x):
        alpha = self.alphaNet(x)
        beta = self.betaNet(x)

        return alpha, beta
