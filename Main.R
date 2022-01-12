###### Packages, if not downloaded run install.packages()
# if needed



############### data load
load(file = "fixedarmdata.RData")
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

   
                   
