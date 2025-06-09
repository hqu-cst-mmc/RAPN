import torch.nn as nn
import torch
# modify due to 228 pytorch version.
# module 'torch' has no attribute 'log_softmax'
# use F.log_softmax instead
import torch.nn.functional as F


class Degree(nn.Module):
    def __init__(self, in_channel):
        super(Degree, self).__init__()
        self.rater = nn.Sequential(
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(in_channel, 512,bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 6, bias=True),
        )
        self.scorer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_channel, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 6, bias=True),
            nn.Sigmoid()
        )

    def forward(self, decoder_res):
        rate = self.rater(decoder_res)
        rate_prob =F.log_softmax(rate)
        score = self.scorer(decoder_res)

        return rate_prob,score
