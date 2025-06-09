import torch.nn as nn
import torch
# modify due to 228 pytorch version.
# module 'torch' has no attribute 'log_softmax'
# use F.log_softmax instead
import torch.nn.functional as F


class Judge(nn.Module):
    def __init__(self, in_channel, out_channel=11):
        super(Judge, self).__init__()
        # self.cls = nn.Sequential(
        #     nn.Dropout(p=0.2,inplace=False),
        #     nn.Linear(in_channel, 512,bias=True),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(p=0.2, inplace=False),
        #     nn.Linear(512, 512, bias=True),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Dropout(p=0.2, inplace=False),
        #     nn.Linear(512, 21, bias=True),
        # )
        self.reg = nn.Sequential(
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(in_channel, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(in_channel, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(128, out_channel),
            nn.Sigmoid()
        )

    def forward(self, decoder_res):
        reg = self.reg(decoder_res)
        cls_prob = self.cls(decoder_res)
        cls_prob_lf = F.log_softmax(cls_prob, dim=-1)
        # cls_prob = F.softmax(cls_prob, dim=-1)
        # prob = F.softmax(cls,dim=-1)
        # log_prob = F.log_softmax(cls,dim=-1)

        return reg, cls_prob, cls_prob_lf
