data=read.csv("Classifications_Ordered.csv")


#install.packages("nonpar")
library(nonpar)

boolMatrix=cbind(data[1:5])

for (colname in colnames(data[1:5])){
  boolMatrix[colname] = data[colname]==data$TrueVals
}

#do the cochrans.q test. 
cochrans.q(boolMatrix)

#p-value 0, understandable when we look at the diffence in
# sums 

for (classifer in boolMatrix){
  print(sum(classifer))
}

#now try every possible combination of classifier
#and do mcnemar (this is the same as cochrans.q on just two columns )
pvals=c()
options(scipen = 999)
combinations=combn(colnames(data[1:5]), 2)
for (comb in 1:ncol(combinations)){
  
  pair=combinations[,comb]
  print(pair)
  
  testResult=cochrans.q(cbind(boolMatrix[pair[1]],boolMatrix[pair[2]]))
  print(testResult)
  p=testResult@PVal
  #p=as.list(strsplit(p, '\\s+')[[1]])[p[length(p)] ]  
  print(p)
  pvals=c(pvals,p)
} 
options(scipen = 0)
test=cochrans.q(cbind(boolMatrix[pair[1]],boolMatrix[pair[2]]))

test@PVal