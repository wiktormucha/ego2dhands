'''
File to store models' heads
'''

import torch
import torch.nn as nn
import torchvision

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import numpy as np


class DeconvolutionLayer(nn.Module):
    '''
    Class for deconvolutional layer 
    '''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2) -> None:
        super().__init__()

        self.deconv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ConvTranspose2d(
                                        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                                    )

    def forward(self, x):
        out = self.deconv(x)
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(in_channels=int(
            in_ch), out_channels=int(in_ch/4), kernel_size=4, stride=4)
        self.deconv2 = DeconvolutionLayer(in_channels=int(
            in_ch/4), out_channels=int(in_ch/16), kernel_size=4, stride=4)

    def forward(self, x):

        out = self.deconv1(x)
        out = self.deconv2(out)
        return out


class Waterfall(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.l1 = nn.Conv2d(
            in_channels=1280, out_channels=1280, kernel_size=1, dilation=1)
        self.l2 = nn.Conv2d(
            in_channels=1280, out_channels=1280, kernel_size=2, dilation=2)
        self.l3 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        # self.norm = nn.BatchNorm2d(3840)
        # self.in_norm = nn.BatchNorm2d(1280)

    def forward(self, x):

        # x = self.in_norm(x)
        out1 = self.pool2(self.l1(x))
        out2 = self.l2(x)
        out3 = self.l3(x)

        final_out = torch.cat([out1, out2, out3], dim=1)
        # final_out = self.norm(final_out)

        return final_out


class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3,
                      padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SimpleHead_FPN(nn.Module):
    '''
    This head network works for FPN model 
    '''

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(
            in_channels=in_channels, out_channels=256)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class SimpleHead5(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(
            in_channels=in_channels, out_channels=256)
        self.deconv2 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv3 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv4 = DeconvolutionLayer(in_channels=256, out_channels=256)
        self.deconv5 = DeconvolutionLayer(in_channels=256, out_channels=256)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class SimpleHead5_small(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.deconv1 = DeconvolutionLayer(in_channels=int(
            in_channels), out_channels=int(in_channels/2), kernel_size=4, stride=4)
        self.deconv2 = DeconvolutionLayer(in_channels=int(
            in_channels/2), out_channels=int(in_channels/4), kernel_size=4, stride=4)
        self.deconv3 = DeconvolutionLayer(in_channels=int(
            in_channels/4), out_channels=int(in_channels/8), kernel_size=4, stride=4)

        self.final = torch.nn.Conv2d(in_channels=int(
            in_channels/8), out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


class SimpleHead_trans(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=256, kernel_size=2, stride=2,
                                                padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.deconv5 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0,
                                                output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

        self.final = torch.nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.deconv1(x)

        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = self.final(x)

        x = self.sigmoid(x)

        return x


class PredictionHead(nn.Module):
    '''
    Baseline head model
    '''

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.convup1 = ConvBlock(1280, output_dim)

        self.prob_out = nn.Sigmoid()

        self.upsample128 = nn.Upsample(
            scale_factor=128, mode="bilinear", align_corners=False)

    def forward(self, x):

        x = self.convup1(self.upsample128(x))

        out = self.prob_out(x)
        return out


class PredictionHeadPoints(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(input_dim)
        self.regression = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        self.norm(x)
        x = torch.squeeze(x)

        out = self.regression(x)
        return out
