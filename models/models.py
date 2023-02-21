'''
This file contains network models that are ready to be inported to other pytohn files 
'''
import torch
import torch.nn as nn
from config import MODEL_NEURONS
from models.heads import SimpleHead5_small, SimpleHead_FPN, SimpleHead5, PredictionHead, SimpleHead_trans, Waterfall, DeconvolutionLayer, Encoder
from models.backbones import BackboneModel_FPN, BackboneModel, ConvBlock, HighLowFeauturesBck


class EfficientWaterfall(nn.Module):
    def __init__(self, out_channels=21) -> None:
        super().__init__()

        self.backbone = HighLowFeauturesBck()
        self.waterfall = Waterfall()
        self.encoder = Encoder(in_ch=3840)
        self.deconv = DeconvolutionLayer(
            in_channels=288, out_channels=288, kernel_size=4, stride=4)
        self.point_conv = nn.Conv2d(
            in_channels=288, out_channels=out_channels, kernel_size=1, stride=1)
        self.prob = nn.Sigmoid()
        # self.norm1 = nn.BatchNorm2d(288)
        # self.norm2 = nn.BatchNorm2d(21)

    def forward(self, x):

        feautures = self.backbone(x)
        out_high = self.waterfall(feautures['high_feautures'])
        out_high = self.encoder(out_high)
        all_feautures = torch.cat(
            [out_high, feautures['low_feautures']], dim=1)
        # all_feautures = self.norm1(all_feautures)
        all_feautures = self.deconv(all_feautures)
        out = self.point_conv(all_feautures)
        # out = self.norm2(out)
        out = self.prob(out)

        return out


class CustomHeatmapsModel_FPN(nn.Module):
    '''
    Model with feature stacking form different level during extracting
    '''

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel_FPN()
        self.head = SimpleHead_FPN(1984, 21)

    def forward(self, x):

        bck = self.backbone(x)
        out = self.head(bck)

        return out


class CustomHeatmapsModel(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel()
        self.head = SimpleHead5(1280, 21)
        # self.head = SimpleHead5_small(1280, 21)

    def forward(self, x):

        bck = self.backbone(x)
        out = self.head(bck)

        return out


class CustomHeatmapsModel_trans(nn.Module):

    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        self.backbone = BackboneModel()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=1280, num_heads=2, batch_first=True)
        self.head = SimpleHead_trans(1280, 21)

    def forward(self, x):

        bck = self.backbone(x)
        re = bck.reshape(x.shape[0], 1280)
        att, _ = self.att(key=re, value=re, query=re)
        att = att.reshape(x.shape[0], 1280, 1, 1)
        out = self.head(bck + att)

        return out


class BaselineFutureStaking(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_down4 = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)
        self.conv_down5 = ConvBlock(MODEL_NEURONS * 8, MODEL_NEURONS * 16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.upsamle4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False)
        self.upsamle8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=False)
        self.upsamle16 = nn.Upsample(
            scale_factor=16, mode="bilinear", align_corners=False)

        self.head = PredictionHead(input_dim=496, output_dim=out_channel)

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_d4 = self.conv_down4(self.maxpool(conv_d3))
        conv_d5 = self.conv_down5(self.maxpool(conv_d4))

        conv_d2 = self.upsamle(conv_d2)
        conv_d3 = self.upsamle4(conv_d3)
        conv_d4 = self.upsamle8(conv_d4)
        conv_d5 = self.upsamle16(conv_d5)

        concat = torch.cat(
            [conv_d1, conv_d2, conv_d3, conv_d4, conv_d5], dim=1)

        out = self.head(concat)

        return out
