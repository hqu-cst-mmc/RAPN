import torch.nn as nn
import torch
# modify due to 228 pytorch version.
# module 'torch' has no attribute 'log_softmax'
# use F.log_softmax instead
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_channel):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            # in_channel = 1024 ,out_channel = 512
            # dim=512
            # nn.Linear(in_channel,256),
            # nn.ReLU(inplace=True),
            # nn.Linear(256,128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128,64),
            # nn.ReLU(inplace=True),
            # nn.Linear(64,1)
            # dim=1024
            nn.Linear(in_channel, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, input_feature):
        predict = self.mlp(input_feature)
        return predict
