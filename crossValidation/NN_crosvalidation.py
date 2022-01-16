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


# Script starts here
seed = 42069
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : {}".format(device))


#load the data 
df = pd.read_pickle("DataFrame.pkl")

listOfCoordCombinations=[["x","y","z"],
                         ["x","y"],
                         ["x","z"],
                         ["y","z"]]

#the trainIndices is just at check to see if each time a model is run, it is
#trained on the same indices. The same test indices
#
trainIndices=[]
testIndices=[]
predictionClassifiers=[]
modelType=[]
trueValsList=[]
for coordCombination in listOfCoordCombinations:                                  


    #%%
    
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
    EPOCHS=300
    
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
predictionClassifiers = np.array(predictionClassifiers).T
    
classificationsUnordered= pd.DataFrame(predictionClassifiers, columns=modelType)
classificationsUnordered["TrueVals"]=trueValsList[0]   
classificationsUnordered["testIndices"]=testIndices[0]

#save the df unorded 
classificationsUnordered.to_csv('Classifications_Unordered.csv', index=False)

#sort whole datafram by testindices, such that we have the orginal order back
classificationsOrdered=classificationsUnordered.sort_values('testIndices')

#save the ordered DF
classificationsOrdered.to_csv('Classifications_Ordered.csv', index=False)

