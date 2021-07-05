# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:26:58 2021

@author: Admin
"""
import matplotlib.pyplot as plt
import numpy as np
#from config import config

def plot(Loss_train,Loss_test):
    x1 = np.arange(len(Loss_train))
    x2 = np.arange(len(Loss_train))
    plt.plot(x1,Loss_train,label='Train loss', c='r')
    plt.plot(x2,Loss_test,label='Test loss', c='b')
    #plt.axhline(config["threshold loss"],0,len(Loss_train),label='loss threshold',c='k')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()