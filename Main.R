###### Packages, if not downloaded run install.packages()
# if needed



############### data load
load(file = "armdata.RData")
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

for (expnr in 1:16) { 
  for (persnr in 1:10) {
    for (repnr in 1:10) {
      for (x in 1:100) {
        for (y in 1:3) {
          mat <- armdata[[expnr]][[persnr]][[repnr]]
          if (is.na(mat[x, y])) {
            li = c()
            for (g in 1:10){
              li = c(li,armdata[[expnr]][[persnr]][[g]][x,y])
            }
            if (expnr == 1) {
              armdata[[expnr]][[persnr]][[repnr]][x, y] <-  mean(li, na.rm = TRUE)
            } else {
              armdata[[expnr]][[persnr]][[repnr]][x, y] <-  mean(li, na.rm = TRUE)
            }
          }
        }
      }
    }
  }
}

save(armdata, file = "fixedarmdata.RData")              
                   
