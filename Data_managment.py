import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np
pandas2ri.activate()


def main():
    # Load the data form the rdata file into python, and bind it to a variable
    robjects.r['load']("fixedarmdata.RData")
    matrix = robjects.r["armdata"]

    # convert the data to a numpy array, so we can use the numpy functionality
    armdata = np.array(matrix)

    # Locating the indexes of the na data points. index read as (experiment, person, repetition, row, col)
    # Keep in mind that in python we use, 0 indexing which is not the case when viewing data in R
    t = np.argwhere(np.isnan(armdata))

    namelist = []
    personlist = []
    repetitionlist = []
    for i in range(16):
        for l in range(10):
            for j in range(10):
                namelist.append("exp" + str(i + 1))
                repetitionlist.append(("rep" + str(j + 1)))
                personlist.append("person" + str(l + 1))
    df = pd.DataFrame([])
    df = df.assign(experiment=namelist)
    df = df.assign(person=personlist)
    df = df.assign(repetition=repetitionlist)

    x, y, z = [], [], []
    for exp in range(16):
        for pers in range(10):
            for rep in range(10):
                x.append(armdata[exp][pers][rep][:, 0])
                y.append(armdata[exp][pers][rep][:, 1])
                z.append(armdata[exp][pers][rep][:, 2])
    df = df.assign(x=x)
    df = df.assign(y=y)
    df = df.assign(z=z)


    #print(df)
    #print(df.loc[0, "z"])
    df.to_csv('Dataframefile.csv')


# For good python style
if __name__ == "__main__":
    main()
