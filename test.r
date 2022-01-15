###### Packages, if not downloaded run install.packages()
# if needed
library("ggplot2")
library(grid)
library(rgl)



############### data load
armdata <- get(load(file = "fixedarmdata.RData"))
df <- read.csv("Dataframefile")
df = subset(df, select = -c(X) )


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

# assign the size of obstacle
s = 20
m = 27.5
t = 30
size = t
# Obstacle
cyl3 <- cylinder3d(cbind(45, 0, seq(0, size, length = 10)), radius = c(3,3,3), sides = 10, closed = -2)
shade3d(addNormals(subdivision3d(cyl3)), col = 'lightblue')

# go through and draw all the lines 
fav <- c("papayawhip", "green", "blue", "magenta", "yellow", "black", "gray", "orange", "pink", "darkolivegreen4")
for (x in 1){
  for (y in 1:10){
    for (z in 5) {
      lines3d(armdata[[x]][[y]][[z]], col = fav[y])
    }
  }
}

# This is the mean of lines in experiment 1
# the variable is gotten from the main.R file
# lines3d(xyz_mean, col = "red", lwd = 12000, alpha = 0.5)



#### QQ-plots of the data
# takes the same x,y,z val for all the repetitions and plot them together
xval <- c()
yval <- c()
zval <- c()
for (x in 1:16) {
  for (y in 1:10) {
    for (z in 1:10) {
      xval <- c(xval, armdata[[x]][[y]][[z]][69,1])
      yval <- c(yval, armdata[[x]][[y]][[z]][69,2])
      zval <- c(zval, armdata[[x]][[y]][[z]][69,3])
    }
  }
  print(length(xval))
  qqnorm(xval)
  qqline(xval, col = "steelblue", lwd = 2)
  qqnorm(yval)
  qqline(yval, col = "red", lwd = 2)
  qqnorm(zval)
  qqline(zval, col = "magenta", lwd = 2)
  xval <- c()
  yval <- c()
  zval <- c()
}




resave <- function(file){
  e <- new.env(parent = emptyenv())
  load(file, envir = e)
  objs <- ls(envir = e, all.names = TRUE)
  for(obj in objs) {
    .x <- get(obj, envir =e)
    message(sprintf('Saving %s as %s.csv', obj,obj) )
    write.csv(.x, file = paste0(obj, '.csv'))
  }
}

  resave('fixedarmdata.RData')



tib <- tibble(armdata)  %>% 
      unnest_wider(armdata)

df <- as.data.frame(do.call(rbind, lapply(armdata, as.data.frame)))

f <- unlist(unlist(unlist(unlist(armdata))))
f1 <- unlist(armdata, recursive = FALSE)
f2 <- unlist(f1, recursive = FALSE)
f3 <- unlist(f2, recursive = FALSE)

f <- armdata[[16]][[5]][[7]]
f

f2[1][[1]][2,]
f3[1]
f3[2]


write.csv(f1,"armdata_fixed.csv", row.names = FALSE)


f1[1]
length(f1[1])
armdata[[2]][[2]][[2]][2,]




