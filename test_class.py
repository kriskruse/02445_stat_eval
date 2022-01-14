#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import itertools as it
from torch.utils.data import Dataset, DataLoader
from cross_validator import CrossValidator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn

from NNvalid import NeuralNetworkClass
#from Torch_NN_file_xy import ClassifierDataset


#%%


#    
#e = 0
#p = 0
#r = 0
#k = 0
#
#
#for (e, p, r, k) in range(16):
#    before_data[e, p, r, k]
#    
#
#exp = []
#
#for e in range(16):
#    per = []
#    exp.append(per)
#    for p in range(10):
#        rep = []
#        per.append(rep)
#        for r in range(1,11):
#            k = 100*r
#            if k > 100:
#                x = my_data[k - 100:k,]
#            else:
#                x = my_data[0:k,]
#            y = my_data[k:k+100 ,]
#            z = my_data[k + 100: k + 200 ,]
#            rep.append(np.hstack((x,y,z)))
#    

#%%

armydata = np.array(exp)

#for r in range(1,160000, 3):
#    k = 100*r
#    if k > 100:
#        x = my_data[k - 100:k,]
#    else:
#        x = my_data[0:k,]
#    y = my_data[k:k+100 ,]
#    z = my_data[k + 100: k + 100 + 100 ,]
#    array_me_this = np.column_stack((x,y,z))
#
##f = np.reshape(my_data, ())









#%%








































#%%















































































df = pd.read_pickle("DataFrame.pkl")

# for col in df.columns:
# print(col)

le = LabelEncoder()
Y = le.fit_transform(df["experiment"])
y = np.sort(np.array(Y))
X = np.array([df["x"], df["y"]])
X = np.transpose(X)

val = np.array([])
for lst in X:
    toms = list(it.chain.from_iterable([lst[0], lst[1]]))
    val = np.append(val, toms)
val.shape = (1600, 200)
X = val

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=9999)




#Define loss function
def loss_fn(pred, label):
    #Subtract predections from labels.
    # Incorrect will result in row summing to 2
    # Correct will result in row being zero
    # Sum all rows (equivalent to summing everything), 
    # then divide by two to get amount of mispredictions.
    # Divide by total to get error rate.
    return (np.abs(pred - label).sum() / 2) / len(pred)


cv = CrossValidator(n_outer=0, n_inner=10, n_workers=1, stratified= True,
                    verbose = True, randomize_seed = 9999)

#Define tester
def test(models):
    #Cross validate
    result = cv.cross_validate(X, y, models, loss_fn)
    
    #Return
    return result


#%%

h = NeuralNetworkClass(3).train_predict(X_train, Y_train, X_test)

plt.hist(h)


#%%
nn_params = [i for i in range(2, 4)]
nn_models = [NeuralNetworkClass(p) for p in nn_params]



results = test(nn_models)
# %%






#%%


idx_label = np.argsort(y)
labels_predict = NeuralNetworkClass(3).train_predict(X_train, Y_train, X_test)

plt.plot(y[idx_label], labels_predict[idx_label],"-")
plt.plot(y[idx_label], y[idx_label])
plt.show()
# %%
