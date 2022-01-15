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

coordCombination=["x","y","z"]


#try other combinations if you please
#coordCombination=["y","z"]


X,Y=pickCoordinates(df, coordCombination)

print(f"Shape of X: {np.shape(X)}")
print(f"Datatype X: {X.dtype}")
print(f"Shape of Y: {np.shape(Y)}")
print(f"Datatype Y: {Y.dtype}")


#choose number of folds and epochs and randomstate for the folds
numFolds=5
EPOCHS=5

#keep this the same so we can compare with different combinations of coordinates. 
random_state=42

predictions,trueVals=KfoldNN(X,Y,EPOCHS,random_state,device)

#change the predictions and trueVals to normal lists
predictions = list(it.chain.from_iterable([fold for fold in predictions]))
trueVals = list(it.chain.from_iterable([fold for fold in trueVals]))

print(classification_report(predictions, trueVals))
    

conMatrix=confusion_matrix(trueVals, predictions)

df_cm = pd.DataFrame(conMatrix, index = [i for i in range(1,17)],
                  columns = [i for i in range(1,17)])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
