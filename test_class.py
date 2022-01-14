import numpy as np
import matplotlib.pyplot as plt
import pickle

from cross_validator import CrossValidator


from NNvalid import NeuralNetworkClass


X = 
Y = 

#Define tester
def test(models, name):
    #Cross validate
    result = cv.cross_validate(X, y, models, loss_fn)
    
    #Return
    return result


nn_params = [i for i in range(2, 16)]
nn_models = [NeuralNetworkClass(p, "classification") for p in nn_params]

results = test(nn_models, "nn")