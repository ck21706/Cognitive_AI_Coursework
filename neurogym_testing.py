from adapted_GRUs import GRU_Net
from neurogym_training import tensor_dataset_sample
import numpy as np
import neurogym as ngym
import torch
import matplotlib.pyplot as plt
import json
from typing import List
import os

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualise_task_data(inputs: torch.Tensor, batch_num=0, ax=None, fs=16):
    """
    Visualise the input data for a given batch
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    

    input_data = inputs[:, batch_num, :].cpu().detach().numpy()
    ax.matshow(input_data.T, aspect='auto', cmap='viridis')
    ax.set_title('Input data for batch {}'.format(batch_num))
    ax.set_xlabel('Time steps', fontsize=fs)
    ax.set_ylabel('Input features', fontsize=fs)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(['Fixation', 'cue1', 'cue2', 'cue3', 'cue4', 'compare1', 'compare2'], fontsize=fs)
    
    return ax

def visualise_training_log(filepaths: List[str], run_names: List[str], plot_to_epoch=None, log_scale=False, fs=16):
    """
    visualise the training log data for the json files corresponding to entries in filepaths.
    Data will be labelled according to run_names
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    for filepath, name in zip(filepaths, run_names):
        with open(filepath, 'r') as f:
            training_log = json.load(f)
        
        ax[0].plot(training_log['train_loss'][:plot_to_epoch], label=name, linewidth=2)
        ax[1].plot(training_log['train_accuracy'][:plot_to_epoch], label=name, linewidth=2)
    
    ax[0].set_xlabel('Epoch', fontsize=fs)
    ax[0].set_ylabel('Training Loss', fontsize=fs)
    ax[0].legend(fontsize=fs)
    
    ax[1].set_xlabel('Epoch', fontsize=fs)
    ax[1].set_ylabel('Training Accuracy', fontsize=fs)
    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)


    if log_scale:
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
    
    plt.show()
    
    return fig, ax


def load_net(net, model_path):
    """
    Load the model parameters from a dictionary
    """
    model_dict = torch.load(model_path)
    net.load_state_dict(model_dict)
    return net

def load_config(config_path):
    """
    Load the configuration dictionary from a json file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def initialize_model_from_config(model_dir, **kwargs):
    config = load_config(os.path.join(model_dir, 'config.json'))
    net = GRU_Net(config['input_size'], config['hidden_size'], config['output_size'], 
                    model_type=config['model_type'], unit_input_size=config["unit_input_size"], 
                    e_prop=0.75, **kwargs)
    net = load_net(net, os.path.join(model_dir, 'model.pth'))
    net = net.to(DEVICE)
    
    return net, config['model_type']


def visualise_task_performance(model_dirs: List[str], seq_len=100, fs=16, **kwargs):
    """
    Visualise the performance of one or more trained models on the DualDelayMatchSample task
    """
    # loading config and dataset
    nets = []
    names = []
    for model_dir in model_dirs:
        net, name = initialize_model_from_config(model_dir, **kwargs)
        net.eval()
        nets.append(net)
        names.append(name)

    # generating test data
    dataset = ngym.Dataset("DualDelayMatchSample-v0", env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # getting predictions
    model_preds = []
    for net in nets:
        pred_labels = net.predict(inputs)
        # for plotting
        pred_labels = pred_labels.cpu().detach().numpy()
        model_preds.append(pred_labels)    
    
    # plotting
    labels = labels.cpu().detach().numpy()
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    # visualising inputs in first axis
    visualise_task_data(inputs, ax=axs[0], fs=fs)
    # plotting response in second axis
    t = np.arange(len(pred_labels))
    # plotting the response of each of the models
    for pred, name in zip(model_preds, names):
        axs[1].plot(t, pred, label=f"{name} pred", marker='x')
    # plotting ground truth
    axs[1].plot(t, labels, label='GT response', marker='.')
    # formatting
    axs[1].legend(fontsize=fs)
    axs[1].set_xlabel('Time steps', fontsize=fs)
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(['No action', 'Accept', 'Reject'], fontsize=fs)
    axs[1].set_xlim([0, len(pred_labels)])

    axs[0].set_title(f'model performance on task', fontsize=fs)

    plt.show()


def visualise_hidden_activations(model_dir, seq_len=75, fs=16, **kwargs):
    # loading model
    net, name = initialize_model_from_config(model_dir, **kwargs)
    net.eval()
    
    # generating test data
    dataset = ngym.Dataset("DualDelayMatchSample-v0", env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # getting hidden activations and corresponding predictions ready for plotting
    _, hidden_activations = net(inputs, return_hidden=True)
    hidden_activations = hidden_activations.cpu().detach().numpy()
    pred_label = net.predict(inputs).cpu().detach().numpy()
    hidden_activations = hidden_activations.squeeze()

    # plotting histogram of hidden activation magnitudes for the entire trial
    print(hidden_activations.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(hidden_activations.flatten(), bins=50)
    ax.set_title('Histogram of hidden activations', fontsize=fs)
    ax.set_xlabel('Activation magnitude', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)

    plt.show(block=False)

    # plotting the maximum activation for each unit throughout the trial
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    max_activations = np.max(hidden_activations, axis=0)
    ax.bar(np.arange(len(max_activations)), max_activations)
    ax.set_title('Maximum activation of each unit throughout the trial', fontsize=fs)
    ax.set_xlabel('Unit', fontsize=fs)
    ax.set_ylabel('Activation magnitude', fontsize=fs)

    plt.show(block=True)




if __name__ == "__main__":
    # visualise_inputs(inputs)
    # print("actions", output_size)
    # print("observations", input_size)
    # print("inputs:", inputs.shape, "labels:", labels.shape)
    # print("inputs 1-10 :", inputs[0:10], "labels:", labels[0:10])
    # print("test_preds:", test_preds.shape)
    # print("test_preds 1-10:", test_preds[0:10])
    # visualise_training_log([r"runs\light_GRU_run1\training_log.json", r"runs\enu_light_GRU_run1\training_log.json", r"runs\light_GRU_l2_reg\training_log.json", r"runs\ei_light_GRU_run1\training_log.json"], 
    #                        ['light GRU', 'ENU light GRU', 'light GRU with l2 reg', 'EI light GRU'])
    # visualise_training_log([r"runs\light_GRU_run1\training_log.json",r"runs\light_GRU_run2\training_log.json"], 
    #                        ["low learning rate", "high learning rate"])
    # visualise_training_log([r"runs\light_GRU_run2\training_log.json", r"runs\enu_light_GRU_run2\training_log.json", r"runs\ei_light_GRU_with_l2reg\training_log.json", r"runs/ei_light_GRU_run2/training_log.json"],
    #                        ['light GRU', 'ENU light GRU', 'EI light GRU with l2 reg', 'EI light GRU'],
    #                        plot_to_epoch = 500, log_scale=False)
    # visualise_task_performance([r"runs\light_GRU_run2", r"runs\enu_light_GRU_run2", r"runs\ei_light_GRU_run2"], seq_len=125)
    visualise_hidden_activations(r"runs\light_GRU_run2", seq_len=300, fs=16)

    #
