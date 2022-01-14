import itertools as it
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

seed = 42069


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
# Dosnt seem to be able to run onehot encoded stuff with the LR models

# yhot_train = np_utils.to_categorical(Y_train)
# yhot_test = np_utils.to_categorical(Y_test)

# note: 0.01 seems to be the best C value, of the tested
test_lst = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
lam = 0.01
# for lam in test_lst:

lr1 = LogisticRegression(multi_class="ovr", C=lam, max_iter=100000)
model = lr1.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print("")
print(f"lambda value: {lam}")
print(f"train score {model.score(X_train, Y_train)}")
print(f"test score {model.score(X_test, Y_test)}")

lr2 = LogisticRegression(multi_class="multinomial", C=lam, max_iter=100000)
model2 = lr2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

print("")
print(f"lambda value: {lam}")
print(f"train score {model2.score(X_train, Y_train)}")
print(f"test score {model2.score(X_test, Y_test)}")

# We now test with less labels
# moving the classes values from experiment number, to distance classes
# so we now have a class for each of the obstacles 6 distances
# 0 no obstacle, 1 = 15cm, 2 = 22.5cm, 3 = 30cm, 4 = 37.5cm, 5 = 45cm
# this is a bit hacky but we know the data so it should be ok
clas_lst = [1, 2, 3, 4, 5, 0]
Y = []
for item in clas_lst:
    if item == 0:
        for i in range(100):
            Y.append(item)
    else:
        for i in range(300):
            Y.append(item)
Y = np.array(Y)
# print(np.size(Y))
# print(Y)

# we do the split again to get the right sizes for the new Y_train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

print("")
print("new sizes for simpler classification task")
print(np.size(Y_train))
print(np.size(Y_test))

for lam in test_lst:
    lr3 = LogisticRegression(multi_class="multinomial", C=lam, max_iter=1000)
    model3 = lr3.fit(X_train, Y_train)
    y_pred3 = model2.predict(X_test)

    print("")
    print(f"lambda value: {lam}")
    print(f"train score {model2.score(X_train, Y_train)}")
    print(f"test score {model2.score(X_test, Y_test)}")


# print(classification_report(Y_test, y_pred))
# print(classification_report(Y_test, y_pred2))
