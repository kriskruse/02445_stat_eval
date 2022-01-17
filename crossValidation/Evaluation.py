import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Classifications_Ordered.csv")

trueVals = df["TrueVals"]
preds = [df["NNxyz"], df["NNxy"], df["NNxz"], df["NNyz"], df["MultiNomial"]]
names = ["NNxyz", "NNxy", "NNxz", "NNyz", "MultiNomial"]

for i in range(len(preds)) :
    conMatrix = confusion_matrix(trueVals, preds[i])

    df_cm = pd.DataFrame(conMatrix, index=[i for i in range(1, 17)],
                         columns=[i for i in range(1, 17)])
    plt.figure(figsize=(10, 7))
    heat = sns.heatmap(df_cm, annot=True, xticklabels=True, yticklabels=True)
    heat.set(xlabel='True value', ylabel='Predicted value', title=names[i])
    plt.savefig(f"Confusionmatrix_{names[i]}", dpi=450)
