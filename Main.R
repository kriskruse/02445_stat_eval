###### Packages, if not downloaded run install.packages()
# if needed
library(abind)
library(dplyr)


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




df[1,1]



?read.csv()

