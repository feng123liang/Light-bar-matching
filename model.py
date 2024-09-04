import math
import warnings
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
import random as r
import copy

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3), # [16, 398]
            nn.BatchNorm1d(16,1e-6),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 16, 5]) [16,199]
            nn.Conv1d(16, 32, 3),# [32, 196] 
            nn.BatchNorm1d(32,1e-6),
            nn.ReLU(),
            nn.MaxPool1d(4),  # torch.Size([32, 99]) 
            nn.Flatten(),  # torch.Size([128, 32])    (假如上一步的结果为[128, 32, 2]， 那么铺平之后就是[128, 64])
        )
        self.model1_2 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3), # [16, 398]
            nn.BatchNorm1d(16,1e-6),
            nn.ReLU(),
            nn.MaxPool1d(2),  # torch.Size([128, 16, 5]) [16,199]
            nn.Conv1d(16, 32, 3),# [32, 196] 
            nn.BatchNorm1d(32,1e-6),
            nn.ReLU(),
            nn.MaxPool1d(4),  # torch.Size([32, 99]) 
            nn.Flatten(),  # torch.Size([128, 32])    (假如上一步的结果为[128, 32, 2]， 那么铺平之后就是[128, 64])
        )
        self.model2 = nn.Sequential(
            nn.Linear(in_features=1568+1568, out_features=80, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=10, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # print(input,input.shape)
        input = input.reshape(-1,1,400)   #结果为[128,1,21]  目的是把二维变为三维数据
        x = self.model1(input)
        # print(x.shape)
        x = self.model2(x)
        return x
    
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ft = nn.Flatten()
        self.fc1 = nn.Linear(16 * 17 * 17, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = x.reshape(-1,1,80,80)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)
        x = self.ft(x)
        # x = x.view(-1, 16 * 17 * 17)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x