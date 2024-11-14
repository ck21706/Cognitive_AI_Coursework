import torch
import torch.nn as nn
from torch.nn import BatchNorm1d

def random_stable_matrix(rows, cols):
    """
    generates a random matrix with eigenvalues <= 1
    """
    A = torch.randn(rows, cols)
    A = A / torch.max(torch.abs(torch.linalg.eigvals(A)))
    return A


class Light_GRU(nn.Module):
    """
    implementation of the light GRU model from the paper "A Light Gated Recurrent Unit for Language Modeling" by Roy et al. (2019)
    """ 
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # randomly initiallizing the weights
        self.Wzx = nn.Parameter(random_stable_matrix(hidden_size, input_size))
        self.Whx = nn.Parameter(random_stable_matrix(hidden_size, input_size))
        self.Uz = nn.Parameter(random_stable_matrix(hidden_size, hidden_size))
        self.Uh = nn.Parameter(random_stable_matrix(hidden_size, hidden_size))
        # learnable batch norms
        self.bn_z = BatchNorm1d(hidden_size)
        self.bn_h = BatchNorm1d(hidden_size)
        # defining nonlinearities
        self.gate_act = nn.Sigmoid()
        self.hidden_act = nn.ReLU()

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return torch.zeros(batch_size, self.hidden_size).to(input.device)
               
    def recurrent_step(self, input, hidden):
        """light-GRU timestep"""
        z_t = self.gate_act(self.bn_z(self.Wzx @ input) + self.Uz @ hidden)
        h_ungated = self.hidden_act(self.bn_h(self.Whx @ input) + self.Uh @ (hidden))
        h_t = z_t * hidden + (1 - z_t) * h_ungated
        return h_t

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        
        output = []
        steps = input.shape[0]
        for i in range(steps):
            hidden = self.recurrent_step(input[i], hidden)
            output.append(hidden)
        
        return torch.stack(output, dim=0), hidden


class ENU_Light_GRU(Light_GRU):
    """
    implementation of light GRU model with embedded neuron units, represeneted by a series of independent
    two layer NNs, to replace the existing artificial neuron hiddent layer
    """
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        

        
