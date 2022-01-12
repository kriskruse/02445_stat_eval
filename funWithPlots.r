###### Packages, if not downloaded run install.packages()
# if needed
library("ggplot2")
library(grid)
library(rgl)
library(robustsHD)

############### data load
armdata <- get(load(file = "fixedarmdata.RData"))
## access with armdata


#change wheter you want x,y,z values
#x=1, y=2, z=3

coordinate=1

for (expNum in 1:16) {
  df <-as.data.frame(matrix(nrow=10,ncol=100))
  for (obs in 1:100){
    for (pers in 1:10) {
      vals <- numeric(10)
      for (repnr in 1:10){
        vals[repnr]=armdata[[expNum]][[pers]][[repnr]][obs,coordinate]
      }
      #xvals <- scale(xvals)
      df[pers,obs]<-(mean(vals))
    }
  }
  boxplot(df, main= paste0("Experiment",expNum," ", "coordinate", coordinate )) 
}


xauc <- c()
yauc <- c()
zauc <- c()
expid = c()
persid = c()
repeid = c()
mean_xauc <- c()
mean_yauc <- c()
mean_zauc <- c()
for (expe in 1:16) {
  for (pers in 1:10) {
    for (repe in 1:10) {
      xauc <- c(xauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,1]))
      yauc <- c(yauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,2]))
      zauc <- c(zauc,trapz(1:100,armdata[[expe]][[pers]][[repe]][,3]))
      expid = c(expid, expe)
      persid = c(persid, pers)
      repeid = c(repeid, repe)
    }
    mean_xauc <- c(mean_xauc,mean(xauc[-10]))
    mean_yauc <- c(mean_yauc,mean(yauc))
    mean_zauc <- c(mean_zauc,mean(zauc))
  }
}

xauc <- c()
yauc <- c()
zauc <- c()
expid = c()
persid = c()
repeid = c()
mean_xauc <- c()
mean_yauc <- c()
mean_zauc <- c()
for (expe in 1:16) {
  for (pers in 1:10) {
    xval =c()
    yval =c()
    zval =c()
    for (repe in 1:10) {
      xval <- c(xval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,1]))
      yval <- c(yval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,2]))
      zval <- c(zval,trapz(1:100,armdata[[expe]][[pers]][[repe]][,3]))
      expid = c(expid, expe)
      persid = c(persid, pers)
      repeid = c(repeid, repe)
    }
    mean_xauc <- c(mean_xauc,mean(xval))
    mean_yauc <- c(mean_yauc,mean(yval))
    mean_zauc <- c(mean_zauc,mean(zval))
  }
}



