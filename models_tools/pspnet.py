import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models.resnet import model_urls

import models_tools.extractors as extractors
# import extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',
                 pretrained=True):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained)
        state_dict = load_state_dict_from_url(model_urls[backend], progress=True)
        self.feats.load_state_dict(state_dict, strict=False)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(2048,1024)

    def forward(self, x):
        # x.size=[bs,3,3,16,3,224,224]
        # bs,3,3,16,3,224,224
        B,V,S,T,C,H,W = x.size()

        x = x.view(-1,*x.shape[4:])
        # x.size=[bs*3*3*16,3,224,224]
        f, class_f = self.feats(x)
        # x.size=[bs*3*3*16,2048,28,28]
        f = self.pool(f)
        # x.size=[bs*3*3*16,2048,1,1]
        f = f.squeeze()
        # f.size=[bs*3*3*16,2048]
        f = self.linear(f)
        # f.size=[bs*3*3*16,1024]
        # f.size=[B,V,S,T,1024]
        return f.reshape(B,V,S,T,f.shape[-1])



if __name__ == '__main__':
    a = torch.rand(16, 3, 112, 112)
    model = PSPNet(psp_size=512)
    print(model(a).shape)
