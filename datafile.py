import numpy as np
import pandas as pd

armdata = pd.read_csv('armdata_fixed.csv')
my_data = armdata.values
#x = my_data[0:100,]
#y = my_data[100:200 ,]
#z = my_data[200:300 ,]
before_data = np.array(armdata).reshape(16, 10, 10, 300)

data = np.zeros((16, 10, 10, 100, 3)) 
for (epr, per, rep) in [(e, p, r) for e in range(16) for p in range(10) for r in range(10)]:
    data[epr, per, rep, :, 0] = before_data[epr, per, rep,   0:100]
    data[epr, per, rep, :, 1] = before_data[epr, per, rep, 100:200]
    data[epr, per, rep, :, 2] = before_data[epr, per, rep, 200:300]
    




