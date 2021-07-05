# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:26:14 2021

@author: Admin
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp       = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        model = self.resnet_dict[model_name]
        return model

    def forward(self, x):
        features = self.backbone(x)
        features = F.normalize(features, dim=1)
        return features