###### Packages, if not downloaded run install.packages()
# if needed
library("ggplot2")
library(grid)
library(rgl)
library(robustsHD)

is.na(armdata)

############### data load
armdata <- get(load(file = "armdata.RData"))
## access with armdata


#change wheter you want x,y,z values
#x=1, y=2, z=3

coordinate=3

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



