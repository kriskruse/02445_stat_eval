# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:35:45 2022

@author: Christoffer
"""

def get_class_distribution(obj):
    count_dict = {
        "experiment_1": 0,
        "experiment_2": 0,
        "experiment_3": 0,
        "experiment_4": 0,
        "experiment_5": 0,
        "experiment_6": 0,
        "experiment_7": 0,
        "experiment_8": 0,
        "experiment_9": 0,
        "experiment_10": 0,
        "experiment_11": 0,
        "experiment_12": 0,
        "experiment_13": 0,
        "experiment_14": 0,
        "experiment_15": 0,
        "experiment_16": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['experiment_1'] += 1
        elif i == 1:
            count_dict['experiment_2'] += 1
        elif i == 2:
            count_dict['experiment_3'] += 1
        elif i == 3:
            count_dict['experiment_4'] += 1
        elif i == 4:
            count_dict['experiment_5'] += 1
        elif i == 5:
            count_dict['experiment_6'] += 1
        elif i == 6:
            count_dict['experiment_7'] += 1
        elif i == 7:
            count_dict['experiment_8'] += 1
        elif i == 8:
            count_dict['experiment_9'] += 1
        elif i == 9:
            count_dict['experiment_10'] += 1
        elif i == 10:
            count_dict['experiment_11'] += 1
        elif i == 11:
            count_dict['experiment_12'] += 1
        elif i == 12:
            count_dict['experiment_13'] += 1
        elif i == 13:
            count_dict['experiment_14'] += 1
        elif i == 14:
            count_dict['experiment_15'] += 1
        elif i == 15:
            count_dict['experiment_16'] += 1
        else:
            print("Check classes.")

    return count_dict

#%%

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


#%%

import torch
import torch.nn as nn

class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

#%%

import numpy as np
import itertools as it
from sklearn.preprocessing import LabelEncoder

def pickCoordinates(df,listOfCoordinates):
    le = LabelEncoder()
    Y = le.fit_transform(df["experiment"])
    Y = np.sort(np.array(Y))
    
    X = np.array([df[name] for name in listOfCoordinates])
    X = np.transpose(X)
    val = np.array([])
    print(np.shape(X)[1])
    
    
    
    for lst in X:  
        
        toms = list(it.chain.from_iterable([lst[i] for i in range(np.shape(X)[1])]))
        
        val = np.append(val, toms)
    val.shape = (1600, 100*np.shape(X)[1])
    
    X = val
    
    return X,Y
