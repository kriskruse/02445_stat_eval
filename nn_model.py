import torch.nn as nn
import torch.nn.functional as F



class NNModel(nn.Module):
    def __init__(self, n_nodes_per_layer, n_in, n_out) -> None:
        super(NNModel, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_nodes_per_layer = n_nodes_per_layer

        if n_nodes_per_layer < 2:
            raise ValueError("Number of hidden layers must be above 1")



        act_inner_fn = lambda: nn.Tanh()

        layers = [nn.Linear(self.n_in, n_nodes_per_layer),
                act_inner_fn(),
                nn.Linear(n_nodes_per_layer,n_nodes_per_layer),
                act_inner_fn(),
                nn.Linear(n_nodes_per_layer,self.n_out), 
                nn.Softmax(dim=1)]



        #Create model
        self.nn_model_sequence = nn.Sequential(*layers)


    def forward(self, x):
        return self.nn_model_sequence(x)





