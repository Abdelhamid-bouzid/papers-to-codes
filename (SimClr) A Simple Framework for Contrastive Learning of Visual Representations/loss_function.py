# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:41:38 2021

@author: Admin
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function,Variable
from config import config

class loss_function(Function):
    def __init__(self):
        self.loss   = torch.nn.CrossEntropyLoss()
        self.SMOOTH = 1e-6
    
    def forward(self, features):
        logits, labels = self.info_nce_loss(features)
        loss = self.loss(logits, labels)
        
        return loss
    
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(config["batch_size"]) for i in range(config["n_views"])], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(config["device"])

        #features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(config["device"])
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives 
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(config["device"])

        logits = logits / config["temperature"]
        return logits, labels