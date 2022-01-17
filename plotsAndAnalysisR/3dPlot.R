###### Packages, if not downloaded run install.packages()
# if needed
library("ggplot2")
library(grid)
library(rgl)

############### data load
armdata <- get(load(file = "fixedarmdata.RData"))
# <- read.csv("Dataframefile")
#df = subset(df, select = -c(X) )


#####  plot 3d images of lines 
# create the cylinders
rgl.open()
start_cyl <- cylinder3d(cbind(0, 0, seq(0, 10, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
target_cyl <- cylinder3d(cbind(60, 0, seq(0, 10, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
cyl1 <- cylinder3d(cbind(0, 0, 10 + seq(0, 12.5, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
cyl2 <- cylinder3d(cbind(60, 0, 10 + seq(0, 12.5, length = 10)), radius = c(3,3,3), sides = 20, closed = -2)
shade3d(addNormals(subdivision3d(start_cyl)), col = 'darkgreen')
shade3d(addNormals(subdivision3d(target_cyl)), col = 'darkgreen', alpha = 0.5) # Slut
shade3d(addNormals(subdivision3d(cyl1)), col = 'pink')
shade3d(addNormals(subdivision3d(cyl2)), col = 'pink', alpha = 0.5) # slut
surface3d(c(-7, 67), c(-20, 20), matrix(0, 2, 2), col = "brown", alpha = 0.9, specular = "black")

s = 20
m = 27.5
t = 30

d1=15
d2=22.5
d3=30
d4=37.5
d5=45

obstacleInfo=list(c(s,d1),c(m,d1),c(t,d1)
               ,c(s,d2),c(m,d2),c(t,d2)
               ,c(s,d3),c(m,d3),c(t,d3)
               ,c(s,d4),c(m,d4),c(t,d4)
               ,c(s,d5),c(m,d5),c(t,d5))

#choose experiment number betweenn 1 and 16

experiment=15

if (experiment == 16) {
  # go through and draw all the lines 
  fav <- c("red", "green", "blue", "magenta", "yellow", "black", "gray", "orange", "pink", "darkolivegreen4")
  for (x in experiment){
    for (y in 1:10){
      for (z in 1:10) {
        lines3d(armdata[[x]][[y]][[z]], col = fav[y])
      }
    }
  }
  
} else {
  # Obstacle
  cyl3 <- cylinder3d(cbind(obstacleInfo[[experiment]][[2]], 0, seq(0,obstacleInfo[[experiment]][[1]] , length = 10)), radius = c(3,3,3), sides = 10, closed = -2)
  shade3d(addNormals(subdivision3d(cyl3)), col = 'lightblue')
  
  # go through and draw all the lines 
  fav <- c("red", "green", "blue", "magenta", "yellow", "black", "gray", "orange", "pink", "darkolivegreen4")
  for (x in experiment){
    for (y in 1:10){
      for (z in 1:10) {
        lines3d(armdata[[x]][[y]][[z]], col = fav[y])
      }
    }
  }
}
