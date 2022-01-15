from validation_model import ValidationModel
from nn_model import NNModel
import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader , TensorDataset
from sklearn.model_selection import StratifiedKFold


class NeuralNetworkClass(ValidationModel):
    def __init__(self, n_hidden_layers) -> None:
        super().__init__()
        

        #Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        self.epochs = 150000
        


    def train_predict(self, train_features, train_labels, test_features):
        
        #n_hidden_layers, n_in, n_out, purpose, shape
        feat_t = Tensor(train_features)
        lab_t = Tensor(train_labels)
        
        dataset = TensorDataset(feat_t, lab_t)
        data_batcher = DataLoader(dataset, batch_size = 72, shuffle=True)
        
      
        
        model = NNModel(self.n_hidden_layers,
                        feat_t.shape[1],
                        16)
        
        loss_func = nn.CrossEntropyLoss()
       


        learning_rate = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        
        for epoch in range(self.epochs):
            for x, y in iter(data_batcher):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                loss = loss_func(y_pred, y.long())
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            if epoch % 500 == 0:
                print(epoch, loss.item())


        
        #Wrapping data in a tensor 
        feat_test= Tensor(test_features)
        
        #Throwing it into our model
        test_pred = model(feat_test)

        #Transforming it into numpy arrays
        test_pred = test_pred.detach().numpy().squeeze().argmax(axis = 1)
        
        #If classifying, one-hot-encode predictions
        #y_pred = np.zeros_like(test_pred)
        #y_pred_idx = np.argmax(test_pred, axis=1)
        #y_pred[np.arange(len(test_pred)), y_pred_idx] = 1
        #
        #test_pred = y_pred

        #Predict
        #Discard model
        return test_pred
        