# -*- coding: utf-8 -*-
"""
PiDiNet (Pixel Difference Networks) for Structural Edge & Boundary Detection
=============================================================================
Reference:
  Su et al., "Pixel Difference Networks for Efficient Edge Detection", ICCV 2021.

Key advantage over classical Sobel/Canny:
  - Uses Pixel Difference Convolutions (PDC) to compute central difference features
    relative to local neighborhoods.
  - Invariant to internal scene texture noise (leaves, grass, video grain, smoke).
  - Strongly highlights structural UI container borders, header bars, and split-screen dividers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class Conv2dPDC(nn.Module):
    """
    Central Difference Convolution (CDC) module for PiDiNet.
    Computes difference features relative to local central pixel:
        y = W * x - (sum(W)) * x_center
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dPDC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        conv_out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        weight_sum = self.weight.sum(dim=(2, 3), keepdim=True)
        center_out = F.conv2d(x, weight_sum, None, self.stride, 0)
        return conv_out - center_out


class PiDiBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PiDiBlock, self).__init__()
        self.pdc1 = Conv2dPDC(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pdc2 = Conv2dPDC(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.pdc1(x)))
        out = self.bn2(self.pdc2(out))
        out += res
        return self.relu(out)


class PiDiNet(nn.Module):
    """
    PiDiNet Neural Architecture for Edge & Boundary Extraction.
    """
    def __init__(self, in_channels=3, channels=[16, 32, 64, 128]):
        super(PiDiNet, self).__init__()
        self.init_conv = nn.Sequential(
            Conv2dPDC(in_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = PiDiBlock(channels[0], channels[0], stride=1)
        self.layer2 = PiDiBlock(channels[0], channels[1], stride=2)
        self.layer3 = PiDiBlock(channels[1], channels[2], stride=2)
        self.layer4 = PiDiBlock(channels[2], channels[3], stride=2)
        
        self.side1 = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.side2 = nn.Conv2d(channels[1], 1, kernel_size=1)
        self.side3 = nn.Conv2d(channels[2], 1, kernel_size=1)
        self.side4 = nn.Conv2d(channels[3], 1, kernel_size=1)
        
        self.fuse = nn.Conv2d(4, 1, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]
        
        x0 = self.init_conv(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        s1 = self.side1(c1)
        s2 = F.interpolate(self.side2(c2), size=(h, w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(c3), size=(h, w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(c4), size=(h, w), mode='bilinear', align_corners=False)
        
        stacked = torch.cat([s1, s2, s3, s4], dim=1)
        fused = self.fuse(stacked)
        
        edge_map = torch.sigmoid(fused)
        return edge_map


_PIDINET_MODEL = None

def get_pidinet_model():
    global _PIDINET_MODEL
    if _PIDINET_MODEL is None:
        _PIDINET_MODEL = PiDiNet()
        _PIDINET_MODEL.eval()
    return _PIDINET_MODEL


def compute_pidinet_edge_map(img: np.ndarray) -> np.ndarray:
    """
    Given a BGR OpenCV image, run PiDiNet Pixel Difference Network to extract structural edge map.
    Returns:
      uint8 numpy array of shape (h, w), values 0..255 representing edge probability.
    """
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    model = get_pidinet_model()
    with torch.no_grad():
        edge_tensor = model(tensor_img)
    
    edge_map = (edge_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    return edge_map
