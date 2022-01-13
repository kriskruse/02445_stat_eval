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


#helper function, should be moved to different script
#' Get colors for the different levels of 
#' a factor variable
#' 
#' @param groups a factor variable containing the groups
#'  of observations
#' @param colors a vector containing the names of 
#   the default colors to be used
get_colors <- function(groups, group.col = palette()){
  groups <- as.factor(groups)
  ngrps <- length(levels(groups))
  if(ngrps > length(group.col)) 
    group.col <- rep(group.col, ngrps)
  color <- group.col[as.numeric(groups)]
  names(color) <- as.vector(groups)
  return(color)
}


### Plotting all the 160 observations in 3d by xarea,yarea and zarea. 
rgl.open() # Open a new RGL device
#rgl.points(df_mean_auc$mean_xauc, df_mean_auc$mean_yauc, df_mean_auc$mean_zauc, color ="lightgray") # Scatter plot

palette1 <- rainbow(16)
palette1  

cols <- get_colors(df_mean_auc$experiment, palette1)
rgl.points(df_mean_auc$mean_xauc, df_mean_auc$mean_yauc, df_mean_auc$mean_zauc, color = cols) # Scatter plot
rgl.bbox(color = "#333377") # Add bounding box decoration

scatter3d(x = mean_pr_experimentx, y = mean_pr_experimenty, z = mean_pr_experimentz, groups = df_mean_auc$experiment,
          grid = FALSE, surface = FALSE)

palette1 <- rainbow(16)
palette1  



# density plots for each attribute by class value
xvariables<-df_mean_auc[,3:5]
yvariables<-df_mean_auc$experiment
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=xvariables, y=yvariables, plot="density", scales=scales)
