#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import itertools as it

from cross_validator import CrossValidator
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from NNvalid import NeuralNetworkClass
#from Torch_NN_file_xy import ClassifierDataset

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

NeuralNetworkClass(2).train_predict(X_train, Y_train, X_test)




#%%
nn_params = [i for i in range(2, 16)]
nn_models = [NeuralNetworkClass(p) for p in nn_params]



results = test(nn_models)
# %%
