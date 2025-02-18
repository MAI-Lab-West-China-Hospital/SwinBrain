#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:05:03 2024

@author: gn3
"""

import torch
import torch.nn as nn
from monai.networks.nets import ViTAutoEnc, SwinUNETR

class ViTClassifier(nn.Module):   
    def __init__(self):
        super().__init__()
        self.model = ViTAutoEnc(
                    in_channels=3,
                    out_channels=3,
                    img_size=(128, 128, 64),
                    patch_size=(16, 16, 16),
                    proj_type='conv',
                    num_heads=32,
                    num_layers=16,
                    hidden_size=2048,
                    mlp_dim=3072,
        )
        self.pool = nn.AvgPool2d(3,stride=3)
        self.FC = nn.Sequential(
            nn.Linear(85*682, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            )
    def forward(self, x):
        x = self.model(x)[1][15]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.FC(x)
        return x

def get_features_hook(module, input, output):
    global features
    features = output
    
class SwinClassifier(nn.Module):    
    def __init__(self):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=3,
            out_channels=3,
            spatial_dims=3,
            feature_size=24,    
            drop_rate=0.0,
            attn_drop_rate=0.0
            )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.FC = nn.Sequential(
            nn.Linear(24, 12),
            nn.ReLU(),        
            nn.BatchNorm1d(12),
            nn.Linear(12, 2)
            )
    def forward(self, x):
        hook = self.model.decoder1.register_forward_hook(get_features_hook)            
        out = self.model(x)            
        hook.remove()                       
        x = self.pool(features)
        x = torch.flatten(x, 1)
        x = self.FC(x)        
        return x
