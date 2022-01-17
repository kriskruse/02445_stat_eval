import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Classifications_Ordered.csv')

#confusion matrix for each of the classifiers

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


# trueVals = df["TrueVals"]
# for classifier in df.iloc[: , :5]:
    
#     conMatrix=confusion_matrix(df["TrueVals"], df[classifier])
#     #print(conMatrix)
#     df_cm = pd.DataFrame(conMatrix, index = [i for i in range(1,17)],
#                           columns = [i for i in range(1,17)])
#     plt.figure(figsize = (10,7))
#     sns.heatmap(df_cm, annot=True)
#     plt.title(f"Confusion matrix {classifier} classifier")
#     plt.show()
    
    
    

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
#from mlxtend.evaluate import mcnemar
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
import itertools as it


#%% 
classifiersPredictions =[np.asarray(df[classifier]) for classifier in df.iloc[: , :5]]


#cochrans_q test
q, p_value = cochrans_q(np.asarray(df["TrueVals"]),
                        classifiersPredictions[0],
                        classifiersPredictions[1],
                        classifiersPredictions[2],
                        classifiersPredictions[3],
                        classifiersPredictions[4])
print("cochrans_q test")
print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)

#%%
#do all the pairwise McNemar test and store p-values

combinations=list(it.combinations(df.iloc[: , :5], 2))

pValues = []
for classifierCombination in combinations:  
    print(classifierCombination)
    table=mcnemar_table(np.asarray(df["TrueVals"]), 
                        np.asarray(df[classifierCombination[0]]),
                        np.asarray(df[classifierCombination[1]]),
                        )
    # calculate mcnemar test
    result = mcnemar(table, exact=True)
    # summarize the finding
    pValues.append(result.pvalue)
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
    	print('Same proportions of errors (fail to reject H0)')
    else:
    	print('Different proportions of errors (reject H0)')
        
    # chi2, p_value = mcnemar(table, corrected=True)
    # # summarize the finding
    # print('p-value=%.3f' % (p_value))
    # print('q-value=%.3f' % (p_value))
    # pValues.append(p_value)
    # # interpret the p-value
    # alpha = 0.05
    # if p_value > alpha:
    # 	print('Same proportions of errors (fail to reject H0)')
    # else:
    # 	print('Different proportions of errors (reject H0)')
        
#make af df over the pValues
names=[f"{classifierName[0]} vs {classifierName[1]}" for classifierName in combinations ]
McNemarTests = pd.DataFrame([pValues], columns=names)

print(McNemarTests)
#%%
boolearnvals,correctedpVals=fdrcorrection(McNemarTests.iloc[0])
McNemarTests.loc[1]=correctedpVals
McNemarTests.loc[2]=[bool(val) for val in boolearnvals]
McNemarTests.loc[2]=[bool(val) for val in boolearnvals]

McNemarTestsTransposed = McNemarTests.T
McNemarTestsTransposed.columns = ["p-value", "adjusted P-value", "Reject H0"]
McNemarTestsSorted=McNemarTestsTransposed.sort_values(by = "adjusted P-value", axis=0)

McNemarTestsSorted.to_csv('McNemarTable.csv', index=True)
