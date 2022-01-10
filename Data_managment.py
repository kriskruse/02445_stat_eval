import pandas as pd
import rpy2.robjects as robjects
import numpy as np
import pyreadr as pr

robjects.r['load']("armdata.RData")
matrix = robjects.r["armdata"]

armdata = np.array(matrix)

