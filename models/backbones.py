'''
File containing backbone models 
'''
import torch
import torch.nn as nn
import torchvision

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)



class EfficientNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights = EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:1]))

    def forward(self,x):

        return  self.backbone(x)

class BackboneModel_FPN(nn.Module):
    '''
    This model stacks feautures from different level of extractor
    '''
    def __init__(self):
        super().__init__()
        
        self.model = EfficientNetV2()

        self.seq0 = self.model.backbone[0][0]
        self.seq1 = self.model.backbone[0][1]
        self.seq2 = self.model.backbone[0][2]
        self.seq3 = self.model.backbone[0][3]
        self.seq4 = self.model.backbone[0][4]
        self.seq5 = self.model.backbone[0][5]
        self.seq6 = self.model.backbone[0][6]
        self.seq7 = self.model.backbone[0][7]


        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
    def forward(self,x):

        out0 = self.seq0(x)
        out1 = self.seq1(out0)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        out4 = self.seq4(out3)
        out5 = self.seq5(out4)
        out6 = self.seq6(out5)
        out7 = self.seq7(out6)

        merge67= torch.cat([out6,out7], dim=1)
        merge67 = self.upsamle(merge67)
        merge45= torch.cat([out4,out5], dim=1)
        merge4567= torch.cat([merge45,merge67], dim=1)
        merge4567 = self.upsamle(merge4567)
        merge34567 = torch.cat([out3,merge4567], dim=1)
        merge34567 = self.upsamle(merge34567)
        merge234567 = torch.cat([out2,merge34567], dim=1)
        merge234567 = self.upsamle(merge234567)

        final = merge01234567 = torch.cat([out0,out1,merge234567], dim=1)

        return  final

class BackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s(weights = EfficientNet_V2_S_Weights)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))

    def forward(self,x):
        
        return self.backbone(x)
        
class HighLowFeauturesBck(nn.Module):
    def __collect(self,m, i, o):
  
        self.act["exp_after_linear"] = o.detach()
        return self.act["exp_after_linear"]

    def __init__(self) -> None:
        super().__init__()

        self.model = EfficientNetV2()
        self.model.backbone[0][2].register_forward_hook(self.__collect)
        self.act ={}

    def forward(self,x):
        
        out_high = self.model(x)
        out_low = self.act["exp_after_linear"]
        
        return {
            'low_feautures': out_low,
            'high_feautures': out_high
        }