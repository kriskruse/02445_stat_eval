import itertools as it
import numpy as np
import os
import torch as T
import pandas as pd
import random
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # important that the input size matches the feature size
    # and that the output is the size of the amount of classes
    self.hid1 = T.nn.Linear(300, 600)  # 300-(600-600)-16
    self.hid2 = T.nn.Linear(600, 600)
    self.oupt = T.nn.Linear(600, 16)

    T.nn.init.xavier_uniform_(self.hid1.weight)
    T.nn.init.zeros_(self.hid1.bias)
    T.nn.init.xavier_uniform_(self.hid2.weight)
    T.nn.init.zeros_(self.hid2.bias)
    T.nn.init.xavier_uniform_(self.oupt.weight)
    T.nn.init.zeros_(self.oupt.bias)

  def forward(self, x):
    z = T.tanh(self.hid1(x))
    z = T.tanh(self.hid2(z))
    z = self.oupt(z)  # no softmax: CrossEntropyLoss()
    return z








seed = 42069
np.random.seed(seed)
T.manual_seed(seed)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(seed)


device = "cuda" if T.cuda.is_available() else "cpu"
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

X_train, X_test, yhot_train, yhot_test = T.tensor(X_train, dtype=T.float32).to(device), \
                                   T.tensor(X_test, dtype=T.float32).to(device), \
                                   T.tensor(yhot_train, dtype=T.float32).to(device), \
                                   T.tensor(yhot_test, dtype=T.float32).to(device)

print("\nBegin test ")
net = Net().to(device)
y = net(X_train)

print("\ninput = ")
print(X_train)
print("output = ")
print(y)

print("\nEnd test ")
