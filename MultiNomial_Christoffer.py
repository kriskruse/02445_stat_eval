# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:34:08 2022

@author: Christoffer
"""

import numpy as np
import pyreadr
import pandas as pd
import statistics
from statistics import mode

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures


#load the data from pickle file
df = pd.read_pickle("DataFrame.pkl")

#%%

#Create a new Dataframe which holds the area 
#under the curves as well as zMax, ZMaxIdx and zMaxXValue
#zMaxXValue is the x-value where the z-coordiante reaches its max

dfArea=pd.DataFrame()

dfArea["xAUC"]=[np.trapz(rep) for rep in df["x"]]
dfArea["yAUC"]=[np.trapz(rep) for rep in df["y"]]
dfArea["zAUC"]=[np.trapz(rep) for rep in df["z"]]
dfArea["zMax"]=[max(rep) for rep in df["z"]]
dfArea["zMaxIdx"]=[list(rep).index(max(rep))for rep in df["z"]]

dfArea["zMaxXValue"]=[rep[(dfArea["zMaxIdx"][i])] for i,rep in enumerate(df["x"])]

dfArea["experiment"] = [experiment for experiment in range(16) for rep in range(100)]

Size=[]
for experiment in dfArea["experiment"]:
    if experiment in [0,3,6,9,12]:
        Size.append(1)
        
    elif experiment in [1,4,7,10,13]:
        Size.append(2)
        
    elif experiment in [2,5,8,11,14]:
        Size.append(3) 
        
    elif experiment == 15:
        Size.append(0)   

dfArea["size"]=Size

#%%
#reduce the dimension by computing each
#persons mean AUC. mean zMAx and the mode of ZMaxId for each experiment.

dfMean = pd.DataFrame()
dfMean["xAUC"] = [np.mean(dfArea["xAUC"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["yAUC"] = [np.mean(dfArea["yAUC"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["zAUC"] = [np.mean(dfArea["zAUC"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["zMax"] = [np.mean(dfArea["zMax"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["zMaxIdx"] = [mode(dfArea["zMaxIdx"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["zMaxXValue"]=[np.mean(dfArea["zMaxXValue"][i:i+10]) for i in range(0,len(dfArea),10)]
dfMean["experiment"] = [person for person in range(16) for experiment in range(10)]

Size=[]
for experiment in dfMean["experiment"]:
    if experiment in [0,3,6,9,12]:
        Size.append(1)
        
    elif experiment in [1,4,7,10,13]:
        Size.append(2)
        
    elif experiment in [2,5,8,11,14]:
        Size.append(3) 
        
    elif experiment == 15:
        Size.append(0)   

dfMean["size"]=Size


#Think we should use some feature selection algorithm here. Would be nice
#Choose attributes,
#Following three attributes seems to peform the best for overall classification
#attributes=["yAUC","zMax","zMaxXValue"]

#Following three attributes seems to peform the best for object height
attributes=["xAUC","yAUC","zAUC","zMax","zMaxXValue"]


#different types of attributes

#attributes=["yAUC","zMax","zMaxXValue"]
#attributes=["xAUC","yAUC","zAUC","zMax","zMaxIdx","zMaxXValue"]
#attributes=["xAUC","yAUC","zAUC","zMax","zMaxXValue"]
#attributes=["yAUC","zAUC","zMax","zMaxXValue"]


#%%

#change between mean values or all values

#mean values
X=np.array(dfMean[attributes])

#standardize
X = stats.zscore(X)

#change if wheter we want classifcation on exerpiment or size classes
#Y=np.array(dfMean["experiment"])

Y=np.array(dfMean["size"])


#all values

# X =np.array(dfArea[attributes])
# X = stats.zscore(X)

# #change if wheter we want classifcation on exerpiment or size classes
# Y=np.array(dfArea["experiment"])
# #Y=np.array(dfArea["size"])




#%% Model fitting and prediction
seed = 42069

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=seed)

# =============================================================================
# logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=seed)
# logreg.fit(X_train,Y_train)
# 
# print('Number of miss-classifications for Multinormal regression:\n\t {0} out of {1}'.format(np.sum(logreg.predict(X_test)!=Y_test),len(Y_test)))
# 
# =============================================================================
#%%
# note: 0.01 seems to be the best C value, of the tested
test_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,100000]
#lam = 0.01
for lam in test_lst:  

    lr2 = LogisticRegression(multi_class="multinomial", C=lam, max_iter=100000)
    model2 = lr2.fit(X_train, Y_train)
    y_pred2 = model2.predict(X_test)

    print("")
    print(f"lambda value: {lam}")
    print(f"train score {model2.score(X_train, Y_train)}")
    print(f"test score {model2.score(X_test, Y_test)}")

