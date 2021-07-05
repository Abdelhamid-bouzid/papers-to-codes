
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:45:05 2021

@author: Admin
"""
import torch
from load_data import load_data
from learning_function import learning_function
from torchsummary import summary
from ResNetSimCLR import ResNetSimCLR
from config import config
from plot import plot


#####################################################################################################
######################################## load data ##################################################
#####################################################################################################
l_train = load_data(r"D:/MHS data segmentation labeling/data/patches_train.npy")
test    = load_data(r"D:/MHS data segmentation labeling/data/patches_test.npy")

#####################################################################################################
#################################### student model ##################################################
#####################################################################################################
model = ResNetSimCLR(base_model=config["arch"], out_dim=config["out_dim"])
#summary(model, (3, 240 ,320))


Loss_train,Loss_test = learning_function(model,l_train,test)

plot(Loss_train,Loss_test)
