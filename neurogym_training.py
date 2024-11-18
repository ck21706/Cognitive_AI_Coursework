import neurogym as ngym
import numpy as np
import torch
import json
from adapted_GRUs import GRU_Net

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# for keeping track of model parameters
CONFIG = {}


def initialize_dataset(batch_size=32, task = 'DualDelayMatchSample-v0'):
    # Environment
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=batch_size)
    env = dataset.env

    # task specific parameters
    CONFIG['output_size'] = env.action_space.n
    CONFIG['input_size'] = env.observation_space.shape[0]
    
    # getting labelled data for supervised learning
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).to(DEVICE)
    labels = torch.from_numpy(labels).to(DEVICE)

    return inputs, labels


def initialize_model(hidden_size, model_type='light', unit_input_size=None):
    CONFIG['hidden_size'] = hidden_size
    CONFIG['model_type'] = model_type
    CONFIG['unit_input_size'] = unit_input_size
    
    net = GRU_Net(CONFIG['input_size'], hidden_size, CONFIG['output_size'], model_type=model_type, unit_input_size=unit_input_size)
    net = net.to(DEVICE)
    
    return net


def 








if __name__ == "__main__":
    pass