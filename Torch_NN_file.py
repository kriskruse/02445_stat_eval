import itertools as it
import numpy as np
import os
import torch
import pandas as pd
import random
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

seed = 42069
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device : {}".format(device))

df = pd.read_pickle("DataFrame.pkl")

# for col in df.columns:
# print(col)

le = LabelEncoder()
Y = le.fit_transform(df["experiment"])
Y = np.sort(np.array(Y))
X = np.array([df["x"], df["y"], df["z"]])
X = np.transpose(X)

val = np.array([])
for lst in X:
    toms = list(it.chain.from_iterable([lst[0], lst[1], lst[2]]))
    # print(np.shape(toms))

    val = np.append(val, toms)
val.shape = (1600, 300)
X = val

print(f"Shape of X: {np.shape(X)}")
print(f"Datatype X: {X.dtype}")
print(f"Shape of Y: {np.shape(Y)}")
print(f"Datatype Y: {Y.dtype}")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=seed)

samples, features = X_train.shape
classes = np.unique(Y_train).size

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
print(samples, features, classes)

yhot = np_utils.to_categorical(Y)
yhot_train = np_utils.to_categorical(Y_train)
yhot_test = np_utils.to_categorical(Y_test)




