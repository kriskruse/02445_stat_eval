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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def get_class_distribution(obj):
    count_dict = {
        "experiment_1": 0,
        "experiment_2": 0,
        "experiment_3": 0,
        "experiment_4": 0,
        "experiment_5": 0,
        "experiment_6": 0,
        "experiment_7": 0,
        "experiment_8": 0,
        "experiment_9": 0,
        "experiment_10": 0,
        "experiment_11": 0,
        "experiment_12": 0,
        "experiment_13": 0,
        "experiment_14": 0,
        "experiment_15": 0,
        "experiment_16": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['experiment_1'] += 1
        elif i == 1:
            count_dict['experiment_2'] += 1
        elif i == 2:
            count_dict['experiment_3'] += 1
        elif i == 3:
            count_dict['experiment_4'] += 1
        elif i == 4:
            count_dict['experiment_5'] += 1
        elif i == 5:
            count_dict['experiment_6'] += 1
        elif i == 6:
            count_dict['experiment_7'] += 1
        elif i == 7:
            count_dict['experiment_8'] += 1
        elif i == 8:
            count_dict['experiment_9'] += 1
        elif i == 9:
            count_dict['experiment_10'] += 1
        elif i == 10:
            count_dict['experiment_11'] += 1
        elif i == 11:
            count_dict['experiment_12'] += 1
        elif i == 12:
            count_dict['experiment_13'] += 1
        elif i == 13:
            count_dict['experiment_14'] += 1
        elif i == 14:
            count_dict['experiment_15'] += 1
        elif i == 15:
            count_dict['experiment_16'] += 1
        else:
            print("Check classes.")

    return count_dict


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


# Script starts here
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

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).long())

class2idx = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:4,
    6:5,
    7:6,
    8:7,
    9:8,
    10:9,
    11:10,
    12:11,
    13:12,
    14:13,
    15:14,
    16:15
}

idx2class = {v: k for k, v in class2idx.items()}
#df['quality'].replace(class2idx, inplace=True)


target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)

class_count = [i for i in get_class_distribution(Y_train).values()]
class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
print(class_weights)
class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

# This is where it starts
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
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

print("Begin training.")
for e in tqdm(range(1, EPOCHS + 1)):

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

# plot the stuff
# Create dataframes
# train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(
#     columns={"index": "epochs"})
# train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(
#     columns={"index": "epochs"})  # Plot the dataframes
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
# sns.lineplot(data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]).set_title(
#     'Train-Val Accuracy/Epoch')
# sns.lineplot(data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]).set_title(
#     'Train-Val Loss/Epoch')

y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix_df = pd.DataFrame(confusion_matrix(Y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
sns.heatmap(confusion_matrix_df, annot=True)
