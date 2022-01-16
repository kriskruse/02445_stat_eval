# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:55:12 2022

@author: Christoffer
"""
import numpy as np
import pandas as pd
import itertools as it
import random
import os
from keras.utils import np_utils

import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from funcsAndClasses import get_class_distribution,ClassifierDataset,MulticlassClassification,multi_acc,pickCoordinates
    
def KfoldNN(X,Y,numFolds,EPOCHS,random_state,device):
    
    kf = StratifiedKFold(numFolds, shuffle=True, random_state=random_state)
    
    predictions=[]
    trueVals=[]
    trainIndex=[]
    testIndex=[]
    
    fold=0
    for train, test in kf.split(X,Y):  
        fold+=1
        
        
        
        trainIndex.extend(train)
        testIndex.extend(test)
        
        #noticed that its not shuffled in the batches, so hereby shuffling the train indices
        
        np.random.shuffle(train)
        print(f"Fold #{fold}")       
        X_train = X[train]
        Y_train = Y[train]
        X_test = X[test]
        Y_test = Y[test]
        
        #print(Y_train)
        samples, features = X_train.shape
        classes = np.unique(Y_train).size
        
        #print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        #print(samples, features, classes)
        
        yhot = np_utils.to_categorical(Y)
        yhot_train = np_utils.to_categorical(Y_train)
        yhot_test = np_utils.to_categorical(Y_test)
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
        
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
        val_dataset = ClassifierDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())
        test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long())
    
        class2idx = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            10: 9,
            11: 10,
            12: 11,
            13: 12,
            14: 13,
            15: 14,
            16: 15
        }
    
        idx2class = {v: k for k, v in class2idx.items()}
        
        target_list = []
        for _, t in train_dataset:
            target_list.append(t)
    
        target_list = torch.tensor(target_list)
    
        class_count = [i for i in get_class_distribution(Y_train).values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        #print(class_weights)
        class_weights_all = class_weights[target_list]
    
        weighted_sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )
        
        #EPOCHS = 10
        BATCH_SIZE = 128
        LEARNING_RATE = 0.1
        NUM_SAMPLES, NUM_FEATURES = X_train.shape
        NUM_CLASSES = np.unique(Y).size
    
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  sampler=weighted_sampler)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    
        model = MulticlassClassification(num_feature=NUM_FEATURES, num_class=NUM_CLASSES)
        model.to(device)
    
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print(model)
        
        # For visualization
        accuracy_stats = {
            'train': [],
            "val": []
        }
        loss_stats = {
            'train': [],
            "val": []
        }
    
        print(f"Beginning training in fold {fold}.")
        
        for e in (range(1, EPOCHS + 1)):
    
            # TRAINING
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
    
                y_train_pred = model(X_train_batch)
    
                train_loss = criterion(y_train_pred, y_train_batch)
                train_acc = multi_acc(y_train_pred, y_train_batch)
    
                train_loss.backward()
                optimizer.step()
    
                train_epoch_loss += train_loss.item()
                train_epoch_acc += train_acc.item()
    
            # VALIDATION
            with torch.no_grad():
    
                val_epoch_loss = 0
                val_epoch_acc = 0
    
                model.eval()
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
    
                    y_val_pred = model(X_val_batch)
    
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)
    
                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))
    
            print(f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f}\
             | Val Loss: {val_epoch_loss / len(val_loader):.5f}\
             | Train Acc: {train_epoch_acc / len(train_loader):.3f}\
             | Val Acc: {val_epoch_acc / len(val_loader):.3f}')
        
        y_pred_list = []
        with torch.no_grad():
            model.eval()
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                y_test_pred = model(X_batch)
                _, y_pred_tags = torch.max(y_test_pred, dim=1)
                y_pred_list.append(y_pred_tags.cpu().numpy())
                # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
                
# =============================================================================
#         confusion_matrix_df = pd.DataFrame(confusion_matrix(Y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
#         sns.heatmap(confusion_matrix_df, annot=True)
#         #print(classification_report(Y_test, y_pred_list))
# =============================================================================
        
        #print(y_pred_list)
        #print(Y_test)
        predictions.append([arr[0] for arr in y_pred_list])
        trueVals.append(Y_test)
        
    return predictions,trueVals,trainIndex,testIndex
        
        