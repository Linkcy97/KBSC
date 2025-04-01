# -*- coding: utf-8 -*-   
# Author       : Chongyang Li
# Email        : lichongyang2016@163.com
# Date         : 2025-03-18 16:05:58
# LastEditors  : Chongyang Li
# LastEditTime : 2025-03-28 21:17:21
# FilePath     : /KBSC/model/net.py

import torch
import torch.nn as nn
from utils.channel import Channel


class FeatureTranser(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.original_feature
        out_dim = config.compressed_feature
        self.en = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.ReLU(),
            nn.Linear(384, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.de = nn.Sequential(
            nn.Linear(out_dim, 384),
            nn.ReLU(),
            nn.Linear(384, in_dim),
            nn.BatchNorm1d(in_dim),
        )
        self.channel = Channel(config)
    
    def forward(self, x, snr):
        x = self.en(x)
        if snr != 'no':
            x = self.channel(x, snr)
        x = self.de(x)
        return x