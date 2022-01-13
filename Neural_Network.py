import os
import itertools as it
import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def InitializeWeights(layer_sizes, features):
    weights = []
    for i, units in enumerate(layer_sizes):
        if i == 0:
            w = torch.rand(units, features, dtype=torch.float32, requires_grad=True)  ## First Layer
        else:
            w = torch.rand(units, layer_sizes[i - 1], dtype=torch.float32, requires_grad=True)  ## All other layers
        b = torch.rand(units, dtype=torch.float32, requires_grad=True)  ## Bias
        weights.append([w, b])

    return weights


def main():
    seed = 42069

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device : {}".format(device))

    df = pd.read_pickle("DataFrame.pkl")

    #for col in df.columns:
        #print(col)

    le = LabelEncoder()
    Y = le.fit_transform(df["experiment"])
    Y = np.sort(np.array(Y))
    X = np.array([df["x"], df["y"], df["z"]])
    X = np.transpose(X)

    val = np.array([])
    for lst in X:
        toms = list(it.chain.from_iterable([lst[0], lst[1], lst[2]]))
        #print(np.shape(toms))

        val = np.append(val, toms)
    val.shape = (1600,300)
    X = val

    print(f"Shape of X: {np.shape(X)}")
    print(f"Datatype X: {X.dtype}")
    print(f"Shape of Y: {np.shape(Y)}")
    print(f"Datatype Y: {Y.dtype}")


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=seed)
    X_train, X_test, Y_train, Y_test = torch.tensor(X_train, dtype=torch.float32), \
                                       torch.tensor(X_test, dtype=torch.float32), \
                                       torch.tensor(Y_train, dtype=torch.float32), \
                                       torch.tensor(Y_test, dtype=torch.float32)

    samples, features = X_train.shape
    classes = Y_train.unique()

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    print(samples, features, classes)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std


if __name__ == "__main__":
    main()
