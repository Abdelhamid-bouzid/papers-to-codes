# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:45:25 2021

@author: Admin
"""
import torch
config = {
    
    "iteration"      : 100000,      # iterataions
    "learning_rate"  : 10**-3,     # learning rate
    "decay_lr"       : 0.5,        # deacy learing rate factor
    "decay_lr_iter"  : 20000,        # deacy learning rate iterataion
    "min_lr"         : 10**-5,     # min learing
    "batch_size"     : 32,         # batch size
    "optimizer_flag" : 'Adam',     # Optimizer
    
    
    "threshold loss" : 10,         # threshold loss
    
    
    "transform"      : [False, False, True], # flip, rnd crop, gaussian noise
    
    "test_model_cycel" :250,
    
    "device"          : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "out_dim"         : 128,
    "arch"            : "resnet18",
    "n_views"         : 2,
    "temperature"     : 0.07,
}