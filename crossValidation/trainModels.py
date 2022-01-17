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
from KfoldNN import KfoldNN
from KfoldMultiNomial import KfoldMultiNomial


# Script starts here
seed = 42069
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(seed)

device = "cpu"
print("Device : {}".format(device))


#load the data 
df = pd.read_pickle("DataFrame.pkl")

listOfCoordCombinations=[["x","y","z"],
                         ["x","y"],
                         ["x","z"],
                         ["y","z"]]

#the trainIndices is just at check to see if each time a model is run, it is
#trained on the same indices. The same with test indices
#
trainIndices=[]
testIndices=[]
predictionClassifiers=[]
modelType=[]
trueValsList=[]


#%% Train the neural networks. 

for coordCombination in listOfCoordCombinations:                                  


    
    
    X,Y=pickCoordinates(df, coordCombination)
    
    coords="".join(coordCombination)
    modelType.append("NN"+coords)
    print(f"Training on combination of coordinates {coords}")
    print(f"Shape of X: {np.shape(X)}")
    print(f"Datatype X: {X.dtype}")
    print(f"Shape of Y: {np.shape(Y)}")
    print(f"Datatype Y: {Y.dtype}")
    
    
    #choose number of folds and epochs and randomstate for the folds
    numFolds=5
    EPOCHS=100
    
    #keep this the same so we can compare with different combinations of coordinates. 
    random_state=42
    
    predictions,trueVals,trainIDX,testIDX=KfoldNN(X,Y,numFolds,EPOCHS,random_state,device)
    
    #change the predictions and trueVals to normal lists
    predictions = list(it.chain.from_iterable([fold for fold in predictions]))
    trueVals = list(it.chain.from_iterable([fold for fold in trueVals]))
    
    print(classification_report(predictions, trueVals))
        
    predictionClassifiers.append(predictions)
    trueValsList.append(trueVals)
    trainIndices.append(trainIDX)
    testIndices.append(testIDX)
    
    conMatrix=confusion_matrix(trueVals, predictions)
    
    df_cm = pd.DataFrame(conMatrix, index = [i for i in range(1,17)],
                      columns = [i for i in range(1,17)])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.show()

#create af dataframe where each column contains the classifications for different models on same folds
NNpredictions=np.array(predictionClassifiers).T
    
classificationsUnordered= pd.DataFrame(NNpredictions, columns=modelType)

#%%Now train the multinomial classifier, but only for three dimensions xyz. 

#choose lambda value
lamda=1000


multiNomialPredictions,trueVals,trainIndex,testIndex=KfoldMultiNomial(df,numFolds,random_state,lamda)

predictionClassifiers.append(multiNomialPredictions)
trueValsList.append(trueVals)
trainIndices.append(trainIDX)
testIndices.append(testIDX)

#add the classifications of the multinomial to the dataframe.
classificationsUnordered["MultiNomial"]=multiNomialPredictions


#%%
classificationsUnordered["TrueVals"]=trueValsList[0]   
classificationsUnordered["testIndices"]=testIndices[0]

#save the df unordered, only uncomment if you do a proper run with many epochs
#classificationsUnordered.to_csv('Classifications_Unordered.csv', index=False)

#sort whole datafram by testindices, such that we have the orginal order back
classificationsOrdered=classificationsUnordered.sort_values('testIndices')

#save the ordered DF only uncomment if you do a proper run with many epochs
#classificationsOrdered.to_csv('Classifications_Ordered.csv', index=False)

print(classification_report(classificationsOrdered["NNxyz"], classificationsOrdered["TrueVals"]))

#%% for good measure this part confirms whether every model was trained on 
#identical train and test splits. This is to make sure we can do mcNemar test

for i in range(len(trainIndices)):
    print(trainIndices[0]==trainIndices[i])

for i in range(len(testIndices)):
    print(testIndices[0]==testIndices[i])
    
for i in range(len(trueValsList)):
     print(trueValsList[0]==trueValsList[i])
           



