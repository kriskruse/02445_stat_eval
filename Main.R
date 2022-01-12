###### Packages, if not downloaded run install.packages()
# if needed
library(abind)
library(dplyr)
library(rgl)
require(pracma)


############### data load
load(file = "fixedarmdata.RData")
df <- read.csv("Dataframefile")
df = subset(df, select = -c(X) )
## access with armdata

# Unpack the Experiments into their own lists
# and add the person numbers to the data
i = 1
pnum = c(paste0("per",1:10))
for (item in armdata){
  names(item) = pnum
  assign(paste0("eks", i), item)
  i = i + 1
}
# we now have the data as experiment individual lists, with "eksx" being the specific experimenter
# to access a specific person from a experiment, use example "eks1$per1" for 
# experiment 1 person 1, for each person there is 10 repetitions, unlabeled
# for these access them with "eks1$per1[1]" for rep 1


#### Fancy shit under here
model <- aov( )
anova(model)
summary(model)

aov(afvigelse_gen ~ person)





# Find mean line for experiment

xlist = c()
ylist = c()
zlist = c()
eks = 1

for (persnr in 1:10){
  for (gen in 1:10){
  xlist = abind(xlist, armdata[[eks]][[persnr]][[gen]][,1], along = 2)
  ylist = abind(ylist, armdata[[eks]][[persnr]][[gen]][,2], along = 2)
  zlist = abind(zlist, armdata[[eks]][[persnr]][[gen]][,3], along = 2)
  }
}
x_mean = rowMeans(array(xlist, dim = c(100,100)))
y_mean = rowMeans(array(ylist, dim = c(100,100)))
z_mean = rowMeans(array(zlist, dim = c(100,100)))

xyz_mean = array(c(x_mean,y_mean,z_mean), dim = c(100,3))







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

df_auc <- data.frame(expid, persid, repeid, xauc, yauc, zauc)

#df_mean_auc <- data.frame(experiment = compexpid,person = compers, mean_xauc, mean_yauc, mean_zauc)

df_mean_auc <- data.frame(experiment = as.factor(compexpid),person = as.factor(compers), mean_xauc, mean_yauc, mean_zauc)


L <- lm(mean_xauc ~ experiment + person, data = df_mean_auc)
L_aov <- aov(mean_xauc ~ experiment + person, data = df_mean_auc)

anova(L)
anova(L_aov)
drop1(L, test = "F")
drop1(L_aov, test = "F")


Lx <- lm(mean_xauc ~ person + experiment, data = df_mean_auc)
Ly <- lm(mean_yauc ~ person + experiment, data = df_mean_auc)
Lz <- lm(mean_zauc ~ person + experiment, data = df_mean_auc)

anova(Lx)
drop1(Lx, test = "F")

anova(Ly)
drop1(Ly, test = "F")

anova(Lz)
drop1(Lz, test = "F")

Lx <- lm(mean_xauc ~ person, data = df_mean_auc)
anova(Lx)
drop1(Lx, test = "F")

#testing the effect of person in one experiment with all samples included
exp_1<-df_auc[1:100,]
exp_1$persid  <- as.factor(exp_1$persid)
exp_1$repeid   <- as.factor(exp_1$persid)

one.way <- aov(xauc ~ persid , data = exp_1)

anova(one.way)

###
#trying to see if the area under the curve varies between the experiments. 
mean_pr_experimentx<-c()
mean_pr_experimenty<-c()
mean_pr_experimentz<-c()
for (experimentNum in 1:16){
    temp_df<-df_mean_auc[df_mean_auc$experiment == experimentNum, ]
    mean_pr_experimentx<-c(mean_pr_experimentx,mean(temp_df$mean_xauc))
    mean_pr_experimenty<-c(mean_pr_experimenty,mean(temp_df$mean_yauc))
    mean_pr_experimentz<-c(mean_pr_experimentz,mean(temp_df$mean_zauc))
}

hist(mean_pr_experimentx)
hist(mean_pr_experimenty)
hist(mean_pr_experimentz)

#idea behind the following is to see if the different experiments are distinguishable in 3-d space. 

### Plotting the average of x,y,z area for each experiment. 
rgl.open() # Open a new RGL device
rgl.points(mean_pr_experimentx, mean_pr_experimenty, mean_pr_experimentz, color ="lightgray") # Scatter plot
rgl.bbox(color = "#333377") # Add bounding box decoration

### Plotting all the 160 observations in 3d by xarea,yarea and zarea. 
rgl.open() # Open a new RGL device
rgl.points(df_mean_auc$mean_xauc, df_mean_auc$mean_yauc, df_mean_auc$mean_zauc, color ="lightgray") # Scatter plot
rgl.bbox(color = "#333377") # Add bounding box decoration
