import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn.init import xavier_normal_, kaiming_normal_


def get_stable_matrix(A):
    """
    Rescales the input matrix so that all eigenvalues <= 1
    """
    A = A / torch.max(torch.abs(torch.linalg.eigvals(A)))
    return A


def create_block_diagonal_matrix(rows, columns):    
    """
    Creates a block diagonal matrix with the given number of rows and columns, 
    where each block is a vector of ones of size columns // rows. The number of columns
    should be divisible by the number of rows. For example create_block_diagonal_matrix(8, 8)
    would return an 8x8 identity matrix.
    """
    block_size = columns // rows
    assert block_size * rows == columns, "The number of columns must be divisible by the number of rows"

    blocks = []
    for i in range(rows):
        row = torch.zeros(columns)
        row[i*block_size:(i+1)*block_size] = 1
        blocks.append(row)
    
    return torch.stack(blocks)

def create_ei_mask(hidden_size, e_size):
    """
    creates a mask for the reccurent hidden layer connections in the EI GRU model. The mask ensures that the hidden layer
    is composed of excitatory and inhibitory units only, with the excitatory units in the first e_size columns and the inhibitory
    units in the remaining columns.
    """
    mask = torch.ones(hidden_size, hidden_size) - torch.eye(hidden_size)
    # Making inhibitory columns negative
    mask[:, e_size:] *= -1
    # making a parameter in case we want to access it later and to explicitly state that it is not a learnable parameter
    mask = nn.Parameter(mask, requires_grad=False)
    
    return mask


class Light_GRU(nn.Module):
    """
    implementation of the light GRU model from the paper "A Light Gated Recurrent Unit for Language Modeling" by Roy et al. (2019)
    """ 
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # randomly initiallizing the feedforward weights using xavier and kaiming initializations for sigmoid and relu respectively
        self.Wzx = nn.Parameter(xavier_normal_(torch.empty(hidden_size, input_size), gain=1))
        self.Whx = nn.Parameter(kaiming_normal_(torch.empty(hidden_size, input_size), mode='fan_in', nonlinearity='relu'))
        # initializing the recurrent weights, this time ensuring stability by rescaling each matrix according to its eigenvalues
        self.Uz = nn.Parameter(get_stable_matrix(xavier_normal_(torch.empty(hidden_size, hidden_size), gain=1)))
        self.Uh = nn.Parameter(get_stable_matrix(kaiming_normal_(torch.empty(hidden_size, hidden_size), mode='fan_in', nonlinearity='relu')))
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
        z_t = self.gate_act(self.bn_z(input @ self.Wzx.T) +  hidden @ self.Uz.T)
        h_ungated = self.hidden_act(self.bn_h(input @ self.Whx.T) + hidden @ self.Uh.T)
        h_t = z_t * hidden + (1 - z_t) * h_ungated
        return h_t

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(input)
        
        timeseries = []
        steps = input.shape[0]
        for i in range(steps):
            hidden = self.recurrent_step(input[i], hidden)
            timeseries.append(hidden)
        
        timeseries = torch.stack(timeseries, dim=0)
        hidden_final = hidden
        
        return timeseries, hidden_final


class ENU_Light_GRU(Light_GRU):
    """
    implementation of light GRU model with embedded neuron units, represented by a series of independent
    two layer NNs, to replace the existing artificial neuron hidden layer. In this implementation each neuron
    unit has a homogeneous structure, i.e. the same number of input features and hidden units. Each unit produces 
    a single output neuron, which is then combined with the outputs from the other units to form the hidden state vector.
    """
    def __init__(self, input_size, hidden_size, unit_input_size):
        """
        args:
        -----
        input_size: int - size of the input vector
        hidden_size: int - size of the hidden state vector
        unit_input_size: int - number of input features associated with each embedded neuron unit units 
                        have a homogeneous structure for now
        """
        super().__init__(input_size, hidden_size)

        self.unit_input_size = unit_input_size
        # unit activation function
        self.unit_act = nn.ReLU()
        # unit input biases
        self.b_unit = nn.Parameter(torch.zeros(hidden_size*unit_input_size))
        # weights for the input to the units. Slightly different from the parent model
        self.Whx = nn.Parameter(kaiming_normal_(torch.empty(unit_input_size*hidden_size, input_size), mode='fan_in', nonlinearity='relu'))
        # weights between the unit inputs and the hidden states, block diagonal structure since units are independent
        self.Wh_in = nn.Parameter(create_block_diagonal_matrix(hidden_size, unit_input_size*hidden_size))
        # mask for the block diagonal matrix to ensure that initially zero weights remain zero during training
        self.Wh_in_mask = nn.Parameter((self.Wh_in != 0).float(), requires_grad=False)

        
    def recurrent_step(self, input, hidden):
        """light GRU timestep with embedded neuron units"""
        z_t = self.gate_act(self.bn_z(input @ self.Wzx.T) + hidden @ self.Uz.T)
        h_in = self.unit_act(input @ self.Whx.T + self.b_unit) # layer forming the input to the units
        h_ungated = self.hidden_act(self.bn_h(h_in @ (self.Wh_in * self.Wh_in_mask).T) + hidden @ self.Uh.T) # masking ensures units remain independent
        h_t = z_t * hidden + (1 - z_t) * h_ungated
        return h_t
  

class EI_Light_GRU(Light_GRU):
    """
    Light GRU model with E-I linear transformation in the hidden layer
    """
    def __init__(self, input_size, hidden_size, e_prop):
        """
        Args:
        ------
        input_size: int, size of the input vector.
        hidden_size: int, size of the hidden state vector.
        e_prop: float between 0 and 1, the proportion of excitatory units.
        """
        super().__init__(input_size, hidden_size)
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size) # Number of excitatory units
        self.i_size = hidden_size - self.e_size # Number of inhibitory units

        # creating the mask for the reccurrent hidden layer connections, which ensures that the ANs are either excitatory or inhibitory
        self.mask = create_ei_mask(hidden_size, self.e_size)
        print("EI mask check", self.mask)

    def recurrent_step(self, input, hidden):
        """light GRU timestep with E-I linear transformation"""
        z_t = self.gate_act(self.bn_z(input @ self.Wzx.T) +  hidden @ self.Uz.T)
        h_ungated = self.hidden_act(self.bn_h(input @ self.Whx.T) + hidden @ (self.Uh*self.mask).T)
        h_t = z_t * hidden + (1 - z_t) * h_ungated
        return h_t


class GRU_Net(nn.Module):
    """
    GRU based model supporting different reccurent GRU architectures and with output layer for classification tasks
    """
    def __init__(self, input_size, hidden_size, output_size, model_type='light', unit_input_size=None, e_prop=0.75):
        """
        args:
        -----
        input_size: int - size of the input vector
        hidden_size: int - size of the hidden state vector
        output_size: int - number of output classes
        
        optional:
        ---------
        model_type: str - type of GRU model to use, either 'light' or 'enu_light'
        unit_input_size: int - number of input features associated with each embedded neuron unit. 
                        Used if the model type is 'enu_light'
        e_prop: float - proportion of excitatory units in the hidden layer. Used if the model type is 'ei_light'
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.unit_input_size = unit_input_size

        if model_type == 'light':
            self.gru = Light_GRU(input_size, hidden_size)
        elif model_type == 'enu_light':
            self.gru = ENU_Light_GRU(input_size, hidden_size, unit_input_size)
        elif model_type == 'ei_light':
            self.gru = EI_Light_GRU(input_size, hidden_size, e_prop)
        else:
            raise ValueError("Invalid model type")
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden=None, return_hidden=False):
        hidden_timeseries, _ = self.gru(input, hidden)
        logits = self.fc(hidden_timeseries)
        
        if return_hidden:
            return logits, hidden_timeseries
        else:
            return logits
    
    def predict(self, input, hidden=None):
        """
        returns predicted class for each timestep and batch in the input sequence of shape input = (seq_len, batch_size, input_size)
        """
        with torch.no_grad():
            logits = self.forward(input, hidden)
            pred_probs = torch.nn.functional.softmax(logits, dim=2)
            pred_labels = torch.argmax(pred_probs, dim=2)
            
        return pred_labels
    

class FixedPoint_GRU_Net_Wrapper(torch.nn.Module):
    def __init__(self, gru_net, batch_first=False):
        super().__init__()
        self.gru_net = gru_net
        self.batch_first = batch_first  # Ensure this matches your RNN's setting

    def forward(self, input, hidden):
        # Squeeze the extra dimension from hidden state
        # Hidden shape transforms from [1, batch_size, hidden_size] to [batch_size, hidden_size]
        if hidden is not None:
            hidden = hidden.squeeze()

        # EI-RNN expects inputs of shape [seq_len, batch_size, input_size]
        # Since we have seq_len=1, input shape is already correct

        # Forward pass through your EI-RNN
        output, hidden = self.gru_net(input, hidden, return_hidden=True)

        # Unsqueeze hidden to match FixedPointFinder's expectation
        # Hidden shape transforms from [batch_size, hidden_size] to [1, batch_size, hidden_size]
        # hidden = hidden.unsqueeze(0)

        # Return None for output as per FixedPointFinder's requirement
        return None, hidden





if __name__ == "__main__":
    test = create_block_diagonal_matrix(4, 8)
    print(test)
    mask = (test != 0).float()
    print(mask)
    