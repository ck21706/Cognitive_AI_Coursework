from adapted_GRUs import GRU_Net
import numpy as np
import neurogym as ngym
import torch
import matplotlib.pyplot as plt
import json
from typing import List

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualise_inputs(inputs: torch.Tensor, batch_num=0):
    """
    Visualise the input data for a given batch
    """
    input_data = inputs[:, batch_num, :].cpu().detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.matshow(input_data.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Input data for batch {}'.format(batch_num))
    plt.xlabel('Time steps')
    plt.ylabel('Input features')
    plt.show()

def visualise_training_log(filepaths: List[str], run_names: List[str]):
    """
    visualise the training log data for the json files corresponding to entries in filepaths.
    Data will be labelled according to run_names
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    for filepath, name in zip(filepaths, run_names):
        with open(filepath, 'r') as f:
            training_log = json.load(f)
        
        ax[0].plot(training_log['train_loss'], label=name)
        ax[1].plot(training_log['train_accuracy'], label=name)
    
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training Loss')
    ax[0].legend()
    
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Training Accuracy')
    ax[1].legend()
    
    plt.show()
    
    return fig, ax


# Environment
task = 'DualDelayMatchSample-v0'
batch_size = 32
dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=batch_size, seq_len=85)
env = dataset.env

# task specific parameters
output_size = env.action_space.n
input_size = env.observation_space.shape[0]
hidden_size = 64

#getting labelled data
inputs, labels = dataset()
inputs = torch.from_numpy(inputs).to(DEVICE)
labels = torch.from_numpy(labels).to(DEVICE)

# initializing the model
net = GRU_Net(input_size, hidden_size, output_size, model_type='light')
net = net.to(DEVICE)

for param in net.parameters():
    print(param.device)

test_preds = net(inputs)

if __name__ == "__main__":
    # visualise_inputs(inputs)
    # print("actions", output_size)
    # print("observations", input_size)
    # print("inputs:", inputs.shape, "labels:", labels.shape)
    # print("inputs 1-10 :", inputs[0:10], "labels:", labels[0:10])
    # print("test_preds:", test_preds.shape)
    # print("test_preds 1-10:", test_preds[0:10])
    visualise_training_log([r"runs\light_GRU_run1\training_log.json", r"runs\enu_light_GRU_run1\training_log.json", r"runs\light_GRU_l2_reg\training_log.json"], ['light GRU', 'ENU light GRU', 'light GRU with l2 reg'])


