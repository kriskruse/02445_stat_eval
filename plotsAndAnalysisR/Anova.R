###### Packages, if not downloaded run install.packages()
# if needed
library(abind)
library(dplyr)
library(rgl)
library(caret)
require(pracma)

############### data load
load(file = "fixedarmdata.RData")
df <- read.csv("Dataframefile")
df = subset(df, select = -c(X) )
## access with armdata


### Data processing for ANOVA
# pre defines
xauc <- c()
yauc <- c()
zauc <- c()
expid = c()
persid = c()
repeid = c()
mean_xauc <- c()
mean_yauc <- c()
mean_zauc <- c()
compexpid <- c()
compers <- c()

# data manipulation, Goal here is to get a mean of AUC for each person in each experiment
# and add it to a dataframe
for (expe in 1:16) {
  for (pers in 1:10) {
    xval =c()
    yval =c()
    zval =c()
    compexpid = c(compexpid, expe)
    compers = c(compers, pers)
    for (repe in 1:10) {
      xval <- c(xval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,1]))
      yval <- c(yval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,2]))
      zval <- c(zval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,3]))
      xauc <- c(xauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,1]))
      yauc <- c(yauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,2]))
      zauc <- c(zauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,3]))
      expid = c(expid, expe)
      persid = c(persid, pers)
      repeid = c(repeid, repe)
    }
    mean_xauc <- c(mean_xauc,mean(xval))
    mean_yauc <- c(mean_yauc,mean(yval))
    mean_zauc <- c(mean_zauc,mean(zval))
  }
}


# first dataframe contains the raw values of AUC, second contains the means
df_auc <- data.frame(expid, persid, repeid, xauc, yauc, zauc) # 1600 obs
df_mean_auc <- data.frame(experiment = as.factor(compexpid),person = as.factor(compers), mean_xauc, mean_yauc, mean_zauc)
# 160 obs

# making three linear models with the dependent variables being xauc,yauc and zauc
Lx <- lm(mean_xauc ~ experiment + person, data = df_mean_auc)
Ly <- lm(mean_yauc ~ experiment + person, data = df_mean_auc)
Lz <- lm(mean_zauc ~ experiment + person, data = df_mean_auc)

#anova
#Lx
summary(Lx)
anova(Lx)
drop1(Lx, test = "F")

#hist of residuals
hist(resid(Lx))
#qqplot of residuals
plot(Lx,2)

#chec whether the residuals are normally distributed.
#null hyp is that they are normally distributed, therefore in this case a high
#p-value is good. 

# Extract the residuals
Lx_residuals <- residuals(object = Lx )
# Run Shapiro-Wilk test
shapiro.test(x = Lx_residuals )

#plot that shows that the variance of the residuals is the same for each factor (experiment)
plot(Lx,5)


#Ly
summary(Ly)
anova(Ly)
drop1(Ly, test = "F")

#hist of residuals
hist(resid(Ly))
#qqplot of residuals
plot(Ly,2)
# Run Shapiro-Wilk test
Ly_residuals <- residuals(object = Ly )
shapiro.test(x = Ly_residuals )

#plot that shows that the variance of the residuals is the same for each factor (experiment)
plot(Ly,5)


#Lz
summary(Lz)
anova(Lz)
drop1(Lz, test = "F")

#hist of residuals
hist(resid(Lz))
#qqplot of residuals
plot(Lz,2)
# Run Shapiro-Wilk test
Lz_residuals <- residuals(object = Lz )
shapiro.test(x = Lz_residuals )

#plot that shows that the variance of the residuals is the same for each factor (experiment)
plot(Lz,5)




#alternative test. 
#kruskal.test(mean_xauc ~ person, data = df_mean_auc)

#testing the effect of person in one experiment with all samples included
exp_1<-df_auc[1:100,]
exp_1$persid  <- as.factor(exp_1$persid)
exp_1$repeid   <- as.factor(exp_1$persid)

one.way <- aov(xauc ~ persid , data = exp_1)
anova(one.way)
