import rpy2.robjects as robjects
import numpy as np


def main():
    # Load the data form the rdata file into python, and bind it to a variable
    robjects.r['load']("armdata.RData")
    matrix = robjects.r["armdata"]

    # convert the data to a numpy array, so we can use the numpy functionality
    armdata = np.array(matrix)

    # Locating the indexes of the na data points. index read as (experiment, person, repetition, row, col)
    # Keep in mind that in python we use, 0 indexing which is not the case when viewing data in R
    t = np.argwhere(np.isnan(armdata))
    print(t)


# For good python style
if __name__ == "__main__":
    main()
