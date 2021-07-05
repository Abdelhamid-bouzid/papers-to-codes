# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:23:32 2021

@author: Admin
"""

import transform
from loss_function import loss_function
import torch
import torch.nn.functional as F
from config import config
import numpy as np
from torch.utils.data import DataLoader
from RandomSampler import RandomSampler
import math
import os

def learning_function(model,l_train,test):
    
    transform_fn = transform.transform(*config["transform"])
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    ''' #################################################  set up optim  ################################################### '''
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device      = torch.device("cpu")
    model       = model.to(device)
    model.train()
    
    optimizer     = torch.optim.Adam(model.parameters(),lr = config['learning_rate'])
    
    ''' #################################################  Dtat loaders  ################################################### '''
    train_sampler=RandomSampler(len(l_train), config["iteration"] * config["batch_size"])
    l_loader = DataLoader(l_train, config["batch_size"],drop_last=True,sampler=train_sampler)
    
    test_loader = DataLoader(test, config["batch_size"],drop_last=True)
    
    ''' #################################################  initialization  ################################################### '''
        
    Loss_train,Loss_test = [],[]
    best_loss  = 10**6
    iteration  = 0
    train_loss = []
    for l_input, l_target in l_loader:
        
        iteration += 1
        
        l_input, l_target = l_input.to(device).float()/255, torch.squeeze(l_target).to(device).long()
        aug_input = transform_fn(l_input).to(device)
        
        l_input = torch.cat((l_input,aug_input))
        
        outputs = model(l_input)
        
        loss = loss_function().forward(outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        
        if iteration%config["decay_lr_iter"]==0:
            if optimizer.param_groups[0]["lr"]>config["min_lr"]:
                optimizer.param_groups[0]["lr"] *= config["decay_lr"]
        
        print('##########################################################################################################')
        print("   #####  Train iteration: {} train_loss: {:0.4f}".format(iteration,loss.item()))
        print('##########################################################################################################')
        
        if iteration%config["test_model_cycel"]==0:
            model.eval()
            test_loss = 0
            with torch.no_grad(): 
                for l_input, l_target in test_loader:
                    
                    l_input, l_target = l_input.to(device).float()/255, torch.squeeze(l_target).to(device).long()
                    aug_input = transform_fn(l_input).to(device)
                    
                    l_input = torch.cat((l_input,aug_input))
                    
                    outputs = model(l_input)
                    loss    = loss_function().forward(outputs)
                    
                    test_loss += loss.item()
                    
                    
            test_loss = test_loss/len(test_loader)
            Loss_test.append(test_loss)
            Loss_train.append(sum(train_loss)/len(train_loss))
            train_loss = []
            print('**********************************************************************************************************')
            print("   #####  Train iteration: {} test_iou: {:0.4f}  best_iou: {:0.4f}".format(iteration,test_loss,best_loss))
            print('**********************************************************************************************************')
        
            if test_loss<best_loss:
                best_loss = test_loss
                torch.save(model,'models/model.pth')
            
        
        model.train()
    return Loss_train,Loss_test