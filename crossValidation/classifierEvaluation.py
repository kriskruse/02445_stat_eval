import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Classifications_Ordered.csv')

#confusion matrix for each of the classifiers

for classifier in df.iloc[: , :5]:
    
    conMatrix=confusion_matrix(df["TrueVals"], df[classifier])
    #print(conMatrix)
    df_cm = pd.DataFrame(conMatrix, index = [i for i in range(1,17)],
                          columns = [i for i in range(1,17)])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.title(f"Confusion matrix {classifier} classifier")
    plt.show()

#%%
# classification rapports for each of the classifiers
for classifier in df.iloc[: , :5]:
    print((f"classification_report {classifier}"))
    print(classification_report(df[classifier],df["TrueVals"]))
          
          
#%% here we could calculate some confidence intervals for the different models


#%%
#pairwise mcnemar test and Cochran's Q Test
import numpy as np
from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

#add columns to the data frame with boolean values corresponding to correct or
#wrong classification
for classifier in df.iloc[: , :5]:
    df[f"{classifier}boolean"]=df[classifier]==df["TrueVals"]

#%% 

q, p_value = cochrans_q(np.asarray(df["TrueVals"]), 
                        np.asarray(df["NNxyz"]), 
                        np.asarray(df["MultiNomial"]))


#%%
print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)