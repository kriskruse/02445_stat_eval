import numpy as np
# import pyreadr
import pandas as pd
# import statistics
# from statistics import mode
import itertools as it

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
# import sklearn.linear_model as lm
from scipy import stats

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import confusion_matrix, classification_report

# THIS is a script to test attributes that is used for multinomial classifications
# When you have found the optimal attributes, please change it in the file called
# KfoldMultiNomial

df = pd.read_pickle("DataFrame.pkl")
numFolds = 5
random_state = 42
lamda = 10000


def combinations_att(att):
    attributes_comblist = list()
    for n in range(1, len(att)):
        attributes_comblist += list(it.combinations(att, n + 1))
    return attributes_comblist


# start of function
def KfoldMultiNomial(df, numFolds, random_state, lamda):
    # Create a new Dataframe which holds the area
    # under the curves as well as zMax, ZMaxIdx and zMaxXValue
    # zMaxXValue is the x-value where the z-coordiante reaches its max

    dfArea = pd.DataFrame()

    dfArea["xAUC"] = [np.trapz(rep) for rep in df["x"]]
    dfArea["yAUC"] = [np.trapz(rep) for rep in df["y"]]
    dfArea["zAUC"] = [np.trapz(rep) for rep in df["z"]]
    dfArea["zMax"] = [max(rep) for rep in df["z"]]
    dfArea["zMaxIdx"] = [list(rep).index(max(rep)) for rep in df["z"]]

    dfArea["zMaxXValue"] = [rep[(dfArea["zMaxIdx"][i])] for i, rep in enumerate(df["x"])]

    dfArea["experiment"] = [experiment for experiment in range(16) for rep in range(100)]

    # Think we should use some feature selection algorithm here. Would be nice
    # Choose attributes,
    # Following three attributes seems to peform the best for overall classification
    # attributes = ["yAUC", "zMax", "zMaxXValue"]
    print("Training")

    attributes = ["xAUC", "yAUC", "zAUC", "zMax", "zMaxIdx", "zMaxXValue"]
    attributes_comblist = combinations_att(attributes)
    print(f"We going over all of these,"
          f"{attributes_comblist}")

    df_results = pd.DataFrame()

    for attributes in attributes_comblist:
        attributes = list(attributes)
        X = np.array(dfArea[attributes])
        X = stats.zscore(X)

        Y = np.array(dfArea["experiment"])

        kf = StratifiedKFold(numFolds, shuffle=True, random_state=random_state)

        predictions = []
        trueVals = []
        trainIndex = []
        testIndex = []

        fold = 0
        for train, test in kf.split(X, Y):
            fold += 1

            # print(len(train))
            trainIndex.extend(train)
            testIndex.extend(test)

            # noticed that its not shuffled in the batches, so hereby shuffling the train indices

            np.random.shuffle(train)

            print(f"Training multinomial classifier with attributes {attributes}")
            print(f"Fold #{fold}")
            X_train = X[train]
            Y_train = Y[train]
            X_test = X[test]
            Y_test = Y[test]

            lr = LogisticRegression(multi_class="multinomial", C=lamda, max_iter=100000)
            model = lr.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            score_train = model.score(X_train, Y_train)
            score_test = model.score(X_test, Y_test)

            print("")
            print(f"lambda value: {lamda}")
            print(f"train score {score_train}")
            print(f"test score {score_test}")

            predictions.extend(y_pred)
            trueVals.extend(Y_test)
            df_results[str(attributes)] = [lamda, score_train, score_test]

    return df_results


df_result = KfoldMultiNomial(df, numFolds, random_state, lamda)
df_result.to_csv("multinomial_att_test.csv")
