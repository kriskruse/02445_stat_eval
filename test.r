###### Packages, if not downloaded run install.packages()
# if needed
library("ggplot2")
library(grid)
library(rgl)



############### data load
armdata <- get(load(file = "armdata.RData"))
## access with armdata



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
size = t
# Obstacle
cyl3 <- cylinder3d(cbind(15, 0, seq(0, size, length = 10)), radius = c(3,3,3), sides = 10, closed = -2)
shade3d(addNormals(subdivision3d(cyl3)), col = 'lightblue')


fav <- c("red", "green", "blue", "magenta", "yellow", "black", "gray", "orange", "pink", "darkolivegreen4")
for (x in 13){
  for (y in 1:10){
    for (z in 1:10) {
      lines3d(armdata[[x]][[y]][[z]], col = fav[y])
    }
  }
}








