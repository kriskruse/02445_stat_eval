#### This files finds and replaces the Na values in the data with the 
#### avg of the values in that spot, for the 10 repetitions

# Load the file
load(file = "armdata.RData")

# We loop over all the repetitions
for (expnr in 1:16) { 
  for (persnr in 1:10) {
    for (repnr in 1:10) {
      for (x in 1:100) {
        for (y in 1:3) {
          mat <- armdata[[expnr]][[persnr]][[repnr]]
          # Check if any of the x,y,z contains NA, 
          # if it does loop over them and replace with mean
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
# we save the data to a file so we don't have to run it again
save(armdata, file = "fixedarmdata.RData")   