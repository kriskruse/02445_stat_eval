import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Classifications_Ordered.csv')

#confusion matrix for each of the classifiers

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
#from mlxtend.evaluate import mcnemar as mlxmcnemar
#from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import fdrcorrection
from mcnemar import mcnemar as mcnemarTest
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
exactDifferences=[]
CIs=[]
for classifierCombination in combinations:  
    print(classifierCombination)
    table=mcnemar_table(np.asarray(df["TrueVals"]), 
                        np.asarray(df[classifierCombination[0]]),
                        np.asarray(df[classifierCombination[1]]),
                        )
    # calculate mcnemar test
    # result = mcnemar(table, exact=False)
    # # summarize the finding
    # pValues.append(result.pvalue)
    # print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # # interpret the p-value
    # alpha = 0.05
    # if result.pvalue > alpha:
    # 	print('Same proportions of errors (fail to reject H0)')
    # else:
    # 	print('Different proportions of errors (reject H0)')
    thetahat, CI, p = mcnemarTest(df["TrueVals"],df[classifierCombination[0]],df[classifierCombination[1]])
    exactDifferences.append(thetahat)
    CIs.append(CI)
    pValues.append(p)
    
        
   
#%%        
#make af df over the pValues
names=[f"{classifierName[0]} vs {classifierName[1]}" for classifierName in combinations ]
McNemarTests = pd.DataFrame([pValues,exactDifferences,CIs], columns=names)

print(McNemarTests)


#%%
boolearnvals,correctedpVals=fdrcorrection(McNemarTests.iloc[0])

McNemarTests.loc[3]=correctedpVals
McNemarTests.loc[4]=[bool(val) for val in boolearnvals]

#%%

McNemarTestsTransposed = McNemarTests.T
McNemarTestsTransposed.columns = ["p-value","thetaHat","CI","adjusted P-value", "Reject H0"]
McNemarTestsSorted=McNemarTestsTransposed.sort_values(by = "adjusted P-value", axis=0)

betterModel=[]
for num, CI in enumerate(McNemarTestsSorted["CI"]):
    if CI[0] < 0 and CI[1] < 0:
        modelsNames=McNemarTestsSorted.index[num]       
        betterModel.append(modelsNames.split(" ")[-1])
        
    elif CI[0] > 0 and CI[1] > 0:
        modelsNames=McNemarTestsSorted.index[num]  
        betterModel.append(modelsNames.split(" ")[0])
    elif CI[0] < 0 and CI[1] > 0:
        betterModel.append("None")
        
        
McNemarTestsSorted["Better model"]=betterModel
#finally choose appropiate order
columns_titles = ["p-value","adjusted P-value","Reject H0","thetaHat","CI","Better model"]
McNemarTestsSorted=McNemarTestsSorted.reindex(columns=columns_titles)
McNemarTestsSorted.to_csv('McNemarTable.csv', index=True)

#%%
#calculate accuracies

for model in df.iloc[: , :5]  :
    acc=sum(df[model]==df["TrueVals"])/len(df[model])
    print(f"The accuracy of model {model} is {acc}")


