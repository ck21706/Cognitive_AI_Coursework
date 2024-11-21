import neurogym as ngym
import numpy as np
import torch
import torch.nn as nn
import json
import os
from adapted_GRUs import GRU_Net
from typing import List, Tuple

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# for keeping track of model parameters
CONFIG = {}
RUN_FOLDER = 'runs'


def initialize_dataset(batch_size=32, task='DualDelayMatchSample-v0', seq_len=85):
    # Environment. seq_len=85 results in each sample of dataset producing two trails
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=batch_size, seq_len=85)
    env = dataset.env

    # task specific parameters
    CONFIG['output_size'] = env.action_space.n
    CONFIG['input_size'] = env.observation_space.shape[0]
    
    return dataset


def tensor_dataset_sample(dataset):
    """
    Returns a sample of the dataset in tensor form, for supervised learning
    """
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).to(DEVICE)
    labels = torch.from_numpy(labels).to(DEVICE)
    return inputs, labels


def initialize_model(hidden_size, model_type='light', unit_input_size=None, e_prop=0.75):
    CONFIG['hidden_size'] = hidden_size
    CONFIG['model_type'] = model_type
    CONFIG['unit_input_size'] = unit_input_size
    
    net = GRU_Net(CONFIG['input_size'], hidden_size, CONFIG['output_size'], model_type=model_type, 
                  unit_input_size=unit_input_size, e_prop=e_prop)
    net = net.to(DEVICE)
    
    return net


def save_dict(dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(dict, f)


def train_run(net: nn.Module, run_dir_name: str, dataset, num_epochs: int = 100, 
              learning_rate: float = 0.001, betas: Tuple = (0.9, 0.999), l2_activation_coef: float = 0):
    """
    Trains GRU based net on inputs and labels for num_epochs epochs, using BPTT with cross entropy loss and Adam optimizer. 
    final model parameters, config file and training logs are saved in the run/run_dir_name directory.

    args:
    -----
    net: nn.Module - GRU based model to be trained
    run_dir_name: str - name of the run directory to save model parameters and logs
    dataset: ngym.Dataset - ngym dataset used to generate training data at each epoch
    
    optional:
    ---------
    num_epochs: int - number of epochs to train the model for. Default is 100
    learning_rate: float - learning rate for the Adam optimizer. Default is 0.001
    betas: Tuple - betas for the Adam optimizer. Default is (0.9, 0.999)

    returns:
    --------
    None
    """
    # adding hyperparameters to config
    CONFIG['betas'] = betas
    CONFIG['l2_activation_coef'] = l2_activation_coef

    # loss function and optimizer
    if l2_activation_coef == 0:
        criterion = nn.CrossEntropyLoss()
        return_hidden = False
    else:
        return_hidden = True
        # cross entropy with l2 penalty on hidden layer activations
        def criterion(output_logits, labels, hidden_activations):
            cross_entropy = nn.CrossEntropyLoss()
            l2_penalty = l2_activation_coef * torch.mean(torch.linalg.vector_norm(hidden_activations, ord=2, dim=1))
            return cross_entropy(output_logits, labels) + l2_penalty
            

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=betas)
    # using exponential learning rate scheduler for better stability once the loss is small.
    gamma = 0.01**(1/num_epochs) # setting gamma so that the final learning rate is 1% of the initial learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # training log
    training_log = {'train_loss': [],  
                    'train_accuracy': [],  
                    'learning_rate': [],
                    'epoch': []}

    best_model = None
    best_accuracy = 0
    
    # training loop using BPTT
    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch+1)
        
        # generating new trails for each epoch
        inputs, labels = tensor_dataset_sample(dataset)
        labels = labels.flatten() # flattening for classification

        # training phase
        optimizer.zero_grad() # zeroing gradients before each forward pass to avoid accumulation
        
        # forward pass
        if return_hidden:
            output_logits, hidden_activations = net(inputs, return_hidden=return_hidden)
            hidden_activations = hidden_activations.view(-1, CONFIG['hidden_size']) # reshaping to seq_len x batch_size, hidden_size for l2 penalty
        else:
            output_logits = net(inputs)

        output_logits = output_logits.view(-1, CONFIG['output_size']) # reshaping to seq_len x batch_size, output_size for cross entropy loss
        
        # calculating loss and backpropagating
        if return_hidden:
            loss = criterion(output_logits, labels, hidden_activations)
        else:
            loss = criterion(output_logits, labels)
        
        loss.backward()
        optimizer.step()
        
        training_log['learning_rate'].append(scheduler.get_last_lr()[0])
        scheduler.step() # decaying the learning rate by gamma
        
        # logging training loss
        training_log['train_loss'].append(loss.item())
        # calculating training accuracy and logging
        train_probs = nn.functional.softmax(output_logits, dim=1)
        train_preds = torch.argmax(train_probs, dim=1)
        train_accuracy = torch.sum(train_preds == labels).item() / len(labels)
        training_log['train_accuracy'].append(train_accuracy)

        # validation phase - We actually don't need validation data for this task as the data is low dimensional and generated by a fixed environment,

        # saving best model
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            best_model = net.state_dict()

        print('--------------------------------')
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Training Accuracy: {train_accuracy}')
        print('--------------------------------\n')

    # saving model parameters
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
    run_dir = os.path.join(RUN_FOLDER, run_dir_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    torch.save(best_model, os.path.join(run_dir, 'model.pth'))
    # saving config and training log
    save_dict(CONFIG, os.path.join(run_dir, 'config.json'))
    save_dict(training_log, os.path.join(run_dir, 'training_log.json'))
    
    print(f"Model saved in {run_dir}")

# training runs

def train_light_gru():
    dataset = initialize_dataset(batch_size=32, task='DualDelayMatchSample-v0')
    net = initialize_model(hidden_size=64, model_type='light')
    train_run(net, 'light_GRU_run2', dataset, num_epochs=1000, learning_rate=0.01, betas=(0.9, 0.999))

def train_enu_light_gru():
    dataset = initialize_dataset(batch_size=32, task='DualDelayMatchSample-v0')
    net = initialize_model(hidden_size=64, model_type='enu_light', unit_input_size=2*CONFIG['input_size'])
    train_run(net, 'enu_light_GRU_run2', dataset, num_epochs=1000, learning_rate=0.01, betas=(0.9, 0.999))

def train_light_gru_l2():
    dataset = initialize_dataset(batch_size=32, task='DualDelayMatchSample-v0')
    net = initialize_model(hidden_size=64, model_type='light')
    train_run(net, 'light_GRU_l2_reg_run2', dataset, num_epochs=1000, learning_rate=0.01, betas=(0.9, 0.999), l2_activation_coef=0.01)

def train_ie_light_gru():
    dataset = initialize_dataset(batch_size=32, task='DualDelayMatchSample-v0')
    net = initialize_model(hidden_size=64, model_type='ei_light', e_prop=0.75)
    train_run(net, 'ei_light_GRU_with_l2reg', dataset, num_epochs=1000, learning_rate=0.01, betas=(0.9, 0.999), l2_activation_coef=0.01)

if __name__ == "__main__":
    train_ie_light_gru()