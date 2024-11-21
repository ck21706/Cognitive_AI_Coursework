from adapted_GRUs import GRU_Net, FixedPoint_GRU_Net_Wrapper
from neurogym_training import tensor_dataset_sample
import numpy as np
import neurogym as ngym
import torch
import matplotlib.pyplot as plt
import json
from typing import List
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

# fixed point finder local import
import sys
sys.path.append(os.path.join(os.getcwd(), 'fixed-point-finder'))
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from plot_utils import plot_fps


# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualise_task_data(inputs: torch.Tensor, batch_num=0, ax=None, fs=16, task="DualDelayMatchSample-v0"):
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
    
    if task == "DualDelayMatchSample-v0":
        ax.set_yticklabels(['Fixation', 'cue1', 'cue2', 'cue3', 'cue4', 'compare1', 'compare2'], fontsize=fs)
    elif task == "ContextDecisionMaking-v0":
        ax.set_yticklabels(['fixation', 'cont1_stim1', 'cont1_stim2', 'cont2_stim1', 'cont2_stim2', 'context1', 'context2'], fontsize=fs)

    return ax

def visualise_training_log(model_dirs: List[str], run_names: List[str], plot_to_epoch=None, log_scale=False, fs=16):
    """
    visualise the training log data for the training_log.json files inside training run directories specified in model_dirs.
    Data will be labelled according to run_names
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    for model_dir, name in zip(model_dirs, run_names):
        with open(os.path.join(model_dir, "training_log.json"), 'r') as f:
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


def visualise_task_performance(model_dirs: List[str], seq_len=100, fs=16, task="DualDelayMatchSample-v0", **kwargs):
    """
    REDUNDANT:
    I used this function to develop many of the different visualisations together, but I haven't tested it
    since and its probably non functional now.

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
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
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
    visualise_task_data(inputs, ax=axs[0], fs=fs, task=task)
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
    axs[1].set_xlim([0, len(pred_labels)])
    axs[0].set_title(f'model performance on task', fontsize=fs)

    if task == "DualDelayMatchSample-v0":
        axs[1].set_yticklabels(['No action', 'Accept', 'Reject'], fontsize=fs)
    elif task == "ContextDecisionMaking-v0":
        axs[1].set_yticklabels(['No action', 'choice1', 'choice2'], fontsize=fs)

    plt.show()


def get_hidden_activations(net, inputs):
    """
    Gets the hidden activation and formatted class predictions for a given input timeseries, assuming it contains only a single batch
    """
    _, hidden_activations = net(inputs, return_hidden=True)
    hidden_activations = hidden_activations.cpu().detach().numpy()
    pred_label = net.predict(inputs).cpu().detach().numpy()
    pred_label = pred_label.squeeze()
    hidden_activations = hidden_activations.squeeze() # squeezing to get shape (seq_len, hidden_size)
    
    return hidden_activations, pred_label

def get_binary_condition_selectivity(cond1_mat, cond2_mat):
    """
    general function for calculating selectivity of units, given two matrices corresponding to hidden activations at timesteps that
    satisfy condition 1 and hidden activations at timesteps that satisfy condition 2. 
    Both matrices should be 2D with second dimension corresponding to units 
    """
    mean_cond1 = np.mean(cond1_mat, axis=0)
    mean_cond2 = np.mean(cond2_mat, axis=0)
    std_cond1 = np.std(cond1_mat, axis=0)
    std_cond2 = np.std(cond2_mat, axis=0)

    # calculating selectivity: positive values indicate a neuron is selective for cond 1, negative values indicate selectivity for cond 2
    selectivity = (mean_cond1 - mean_cond2) / np.sqrt((std_cond1**2 + std_cond2**2)/2 + 1e-8) # to avoid division by zero

    return selectivity


def visualise_hidden_activations(model_dir, seq_len=75, fs=16, **kwargs):
    # loading model
    net, name = initialize_model_from_config(model_dir, **kwargs)
    net.eval()
    
    # generating test data
    dataset = ngym.Dataset("DualDelayMatchSample-v0", env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # getting hidden activations and corresponding predictions ready for plotting
    hidden_activations, pred_label = get_hidden_activations(net, inputs)
    
    print("prediction and hidden activation shape:", pred_label.shape, hidden_activations.shape)

    # plotting histogram of hidden activation magnitudes for the entire trial
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(hidden_activations.flatten(), bins=50)
    ax.set_title('Histogram of hidden activations', fontsize=fs)
    ax.set_xlabel('Activation magnitude', fontsize=fs)
    ax.set_ylabel('Frequency', fontsize=fs)
    ax.set_yscale('log')

    plt.show(block=False)

    # plotting the maximum activation for each unit throughout the trial
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    max_activations = np.max(hidden_activations, axis=0)
    ax.bar(np.arange(len(max_activations)), max_activations)
    ax.set_title('Maximum activation of each unit throughout the trial', fontsize=fs)
    ax.set_xlabel('Unit', fontsize=fs)
    ax.set_ylabel('Activation magnitude', fontsize=fs)

    plt.show(block=False)

    # determining the action/inaction selctivity of each unit

    # there are three actions, two of which are active (accept/reject), the third is passive (do nothing). 
    # so we can group the actions into active and passive
    hidden_active = hidden_activations[np.where(pred_label != 0)[0], :]
    hidden_passive = hidden_activations[np.where(pred_label == 0)[0], :]
    
    active_inactive_selectivity = get_binary_condition_selectivity(hidden_active, hidden_passive)
    
    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].bar(np.arange(len(active_inactive_selectivity)), active_inactive_selectivity)
    axs[0].set_title('Action(positive)/Inaction(negative) selectivity of each unit', fontsize=fs)
    axs[0].set_ylabel('Selectivity', fontsize=fs)

    # we can repeat this, but this time looking at the reject and accept action selectivity
    hidden_accept = hidden_activations[np.where(pred_label == 2)[0], :]
    hidden_reject = hidden_activations[np.where(pred_label == 1)[0], :]

    accept_reject_selectivity = get_binary_condition_selectivity(hidden_accept, hidden_reject)

    # plotting
    colors = ['red' if s > 0 else 'grey' for s in active_inactive_selectivity]
    axs[1].bar(np.arange(len(accept_reject_selectivity)), accept_reject_selectivity, color=colors)
    axs[1].set_title('Accept/Reject selectivity of each unit', fontsize=fs)
    axs[1].set_xlabel('Unit', fontsize=fs)
    axs[1].set_ylabel('Selectivity', fontsize=fs)
    # Add legend to explain colors
    red_patch = mpatches.Patch(color='red', label='Action selective')
    grey_patch = mpatches.Patch(color='grey', label='Inaction selective')
    axs[1].legend(handles=[red_patch, grey_patch], fontsize=fs)
    
    plt.tight_layout()
    plt.show(block=False)

    # now lets plot the timeseries activity of the most selective unit

    most_selective_unit = np.argmax(np.abs(active_inactive_selectivity))

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].plot(hidden_activations[:, most_selective_unit])
    axs[0].set_title(f'Timeseries activity of most selective unit {most_selective_unit}', fontsize=fs)
    axs[0].set_ylabel('Activation magnitude', fontsize=fs)
    axs[0].set_xlim([0, len(hidden_activations)])

    visualise_task_data(inputs, ax=axs[1], fs=fs)
    axs[1].set_title('', fontsize=fs)
    axs[1].set_ylabel('', fontsize=fs)

    plt.show(block=False)


def visualise_selectivity(model_dir, seq_len=75, fs=16, task="DualDelayMatchSample-v0", **kwargs):
    # loading model
    net, name = initialize_model_from_config(model_dir, **kwargs)
    net.eval()
    
    # generating test data
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, _ = tensor_dataset_sample(dataset)

    # getting hidden activations and corresponding predictions ready for plotting
    _, hidden_activations = net(inputs, return_hidden=True)
    hidden_activations = hidden_activations.cpu().detach().numpy().squeeze() # shape (seq_len, batch_size, hidden_size)
    pred_label = net.predict(inputs).cpu().detach().numpy().squeeze() # shape (seq_len, batch_size)

    # there are three actions, two of which are active (accept/reject), the third is passive (do nothing). 
    # so we can group the actions into active and passive
    hidden_active = hidden_activations[np.where(pred_label != 0)[0], :]
    hidden_passive = hidden_activations[np.where(pred_label == 0)[0], :]
    
    active_inactive_selectivity = get_binary_condition_selectivity(hidden_active, hidden_passive)
    
    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].bar(np.arange(len(active_inactive_selectivity)), active_inactive_selectivity)
    axs[0].set_ylabel('Selectivity', fontsize=fs)
    axs[0].set_title('Action(positive)/Inaction(negative) selectivity of each unit', fontsize=fs)

    # we can repeat this, but this time looking at the reject and accept action selectivity
    hidden_accept = hidden_activations[np.where(pred_label == 2)[0], :]
    hidden_reject = hidden_activations[np.where(pred_label == 1)[0], :]

    accept_reject_selectivity = get_binary_condition_selectivity(hidden_accept, hidden_reject)

    # plotting
    colors = ['red' if s > 0 else 'grey' for s in active_inactive_selectivity]
    axs[1].bar(np.arange(len(accept_reject_selectivity)), accept_reject_selectivity, color=colors)
    axs[1].set_xlabel('Unit', fontsize=fs)
    axs[1].set_ylabel('Selectivity', fontsize=fs)
    # Add legend to explain colors
    red_patch = mpatches.Patch(color='red', label='Action selective')
    grey_patch = mpatches.Patch(color='grey', label='Inaction selective')
    axs[1].legend(handles=[red_patch, grey_patch], fontsize=fs)

    if task == "DualDelayMatchSample-v0":
        axs[1].set_title('Accept/Reject selectivity of each unit', fontsize=fs)
    elif task == "ContextDecisionMaking-v0":
        axs[1].set_title('choice1/choice2 selectivity of each unit', fontsize=fs)
    
    plt.tight_layout()
    plt.show()
    
def monte_carlo_selectivity_histogram(model_dir, simulations=32, seq_len=1000, fs=16, task="DualDelayMatchSample-v0", **kwargs):
    """
    runs a monte carlo simulation to estimate the distribution of maximum absolute selectivity values for significant units
    in the hidden layer. 
    """
    # loading model
    net, name = initialize_model_from_config(model_dir, **kwargs)
    net.eval()
    
    # generating test data
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=simulations, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # getting hidden activations and corresponding predictions ready for plotting
    _, hidden_activations = net(inputs, return_hidden=True)
    hidden_activations = hidden_activations.cpu().detach().numpy().squeeze() # shape (seq_len, batch_size, hidden_size)
    pred_labels = net.predict(inputs).cpu().detach().numpy().squeeze() # shape (seq_len, batch_size)

    # getting the maximum activation for each unit throughout the trial
    max_activations = np.max(hidden_activations, axis=0) # shape (batch_size, hidden_size)
    
    max_mag_select = []
    for i in range(simulations):
        # finding units that have significant activation during any of the trails
        significant_units = np.where(max_activations[i, :] > 0.5)[0]
        significant_activations = hidden_activations[:, i, significant_units].squeeze() # shape (seq_len, num_significant_units)
        pred_label = pred_labels[:, i] # shape (seq_len,) 
        
        # calculating the active/inactive selectivity of these units
        active_hidden = significant_activations[np.where(pred_label != 0)[0], :]
        passive_hidden = significant_activations[np.where(pred_label == 0)[0], :]
        active_inactive_selectivity = get_binary_condition_selectivity(active_hidden, passive_hidden)
        
        # calculating the accept/reject selectivity of these units
        accept_hidden = significant_activations[np.where(pred_label == 2)[0], :]
        reject_hidden = significant_activations[np.where(pred_label == 1)[0], :]
        accept_reject_selectivity = get_binary_condition_selectivity(accept_hidden, reject_hidden)

        # calculating the max magnitude of selctivity across the two conditions in each unit
        stacked_selectivity = np.stack([active_inactive_selectivity, accept_reject_selectivity], axis=0)
        max_mag_selectivity = np.max(np.abs(stacked_selectivity), axis=0)
        max_mag_select.append(max_mag_selectivity)
    
    # combining all monte carlo trials to get a long list of max selectivity values
    max_mag_selectivity = np.concatenate(max_mag_select, axis=0)
        
    # plotting estimated probability distribution of selectivity magnitudes
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(max_mag_selectivity, bins=50, density=True)
    ax.set_title(f'Histogram of maximum absolute selectivity of significant units\n trails={simulations}, seq_len={seq_len}', fontsize=fs)
    ax.set_xlabel('max abs selectivity', fontsize=fs)
    ax.set_ylabel('Density', fontsize=fs)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    plt.show(block=True)


def combined_max_activation_stem_plot(model_dirs: List[str], seq_len):
    """
    Plots the maximum activation of each hidden layer unit throughout the trial for each model in model_dirs as a stem plot
    """
    # generating test data
    dataset = ngym.Dataset("DualDelayMatchSample-v0", env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    model_activations = []
    model_names = []
    model_preds = []
    for model_dir in model_dirs:
        # loading model
        net, name = initialize_model_from_config(model_dir)
        net.eval()
        # getting hidden activations and corresponding predictions ready for plotting
        hidden_activations, pred_label = get_hidden_activations(net, inputs)
        # saving for later
        model_activations.append(hidden_activations)
        model_names.append(name)
        model_preds.append(pred_label)

    # calculating the max activation for each unit in each model (assuming that models have identical numbers of units)
    max_activations = [np.max(hidden_activations, axis=0) for hidden_activations in model_activations]

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    units = np.arange(len(max_activations[0]))
    color = ['r', 'b', 'g']
    for i, max_activation in enumerate(max_activations):
        ax.stem(units, max_activation, label=model_names[i], use_line_collection=True, linefmt=color[i], markerfmt=color[i]+'o', basefmt='none')
    ax.set_title('Maximum activation of each unit throughout the trial', fontsize=16)
    ax.set_xlabel('Unit', fontsize=16)
    ax.set_ylabel('max activation', fontsize=16)
    ax.legend(fontsize=16)
    plt.show()

def combined_timeseries_of_most_sensitive_units(model_dirs:List[str], seq_len, task="DualDelayMatchSample-v0"):
    # generating test data
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    model_activations = []
    model_names = []
    model_preds = []
    for model_dir in model_dirs:
        # loading model
        net, name = initialize_model_from_config(model_dir)
        net.eval()
        # getting hidden activations and corresponding predictions ready for plotting
        hidden_activations, pred_label = get_hidden_activations(net, inputs)
        # saving for later
        model_activations.append(hidden_activations)
        model_names.append(name)
        model_preds.append(pred_label)

    # calculating the sensitivity of each unit in each model (assuming that models have identical numbers of units)
    top_selective_units = []
    for pred_label, hidden_activations in zip(model_preds, model_activations):
        hidden_active = hidden_activations[np.where(pred_label != 0)[0], :]
        hidden_passive = hidden_activations[np.where(pred_label == 0)[0], :]
        
        active_inactive_selectivity = get_binary_condition_selectivity(hidden_active, hidden_passive)
        most_selective_unit = np.argmax(np.abs(active_inactive_selectivity))
        top_selective_units.append(most_selective_unit)

    
    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    
    # visualising activation timeseries of most selective unit in each model
    for i, activations in enumerate(model_activations):
        axs[0].plot(activations[:, top_selective_units[i]], label=model_names[i])
    
    axs[0].set_title(f'Timeseries activity of most action selective unit in each model', fontsize=16)
    axs[0].set_ylabel('Activation', fontsize=16)
    axs[0].set_xlim([0, len(activations)])
    axs[0].legend(fontsize=16)

    # visualising corresponding task input data
    visualise_task_data(inputs, ax=axs[1], fs=16, task=task)
    axs[1].set_title('', fontsize=16)
    axs[1].set_ylabel('', fontsize=16)

    plt.show(block=True)


def timeseries_of_select_units(model_dir: str, unit_inds:list[int], unit_labels: List[str], seq_len=75, fs=16, task="DualDelayMatchSample-v0", **kwargs):
    # generating test data
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # loading model
    net, name = initialize_model_from_config(model_dir)
    net.eval()
    # getting hidden activations and corresponding predictions ready for plotting
    hidden_activations, pred_label = get_hidden_activations(net, inputs)
    
    # finding points where the model makes predictions
    c1_preds = np.where(pred_label == 1)[0]
    c2_preds = np.where(pred_label == 2)[0]
    
    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    for i, unit_ind in enumerate(unit_inds):
        axs[0].plot(hidden_activations[:, unit_ind], label=unit_labels[i])
    
    # plotting dashed vertical lines at c1_preds and c2_preds
    label = "c1_pred"
    for c1 in c1_preds:
        axs[0].axvline(x=c1, color='red', linestyle='--', label=label, zorder=0, linewidth=4, alpha=0.5)
        label = None
    label = "c2_pred"
    for c2 in c2_preds:
        axs[0].axvline(x=c2, color='blue', linestyle='--', label=label, zorder=0, linewidth=4, alpha=0.5)
        label = None

    axs[0].set_title(f'Timeseries activity of selected units in {name}-GRU model', fontsize=fs)
    axs[0].set_ylabel('Activation', fontsize=fs)
    axs[0].set_xlim([0, len(hidden_activations)])
    axs[0].legend(fontsize=12)

    visualise_task_data(inputs, ax=axs[1], fs=fs, task=task)
    axs[1].set_title('', fontsize=fs)
    
    plt.show()


def find_fixed_points(model_dir, seq_len=100):
    """
    Struggled to get results that made sense with this method and ran out of space anyway so never made it into the report
    """
    # loading model
    net, name = initialize_model_from_config(model_dir)
    net = FixedPoint_GRU_Net_Wrapper(net)
    net.eval()
    

    # loading dataset
    dataset = ngym.Dataset("DualDelayMatchSample-v0", env_kwargs={'dt': 100}, batch_size=1, seq_len=seq_len)
    inputs, labels = tensor_dataset_sample(dataset)

    # getting hidden activations over a test trial
    _, hidden_activations = net(inputs, hidden=None)
    hidden_activations = hidden_activations.cpu().detach().numpy().squeeze() # squeezing to get shape (seq_len, hidden_size) as a np array
    
    # initialising fixed point finder
    net.to("cpu") # moving to cpu for fixed point finder
    fpf = FixedPointFinder(net)

    # finding fixed points for some random initial hidden states and zeroed input
    initial_conditions = np.random.randn(10, net.gru_net.hidden_size)
    fixed_inputs = np.zeros((10, net.gru_net.input_size))
    fps = fpf.find_fixed_points(initial_conditions, fixed_inputs)[0]
    fixed_points = fps.xstar # fixed point locations in hidden state space
    print("fixed points shape:", fixed_points.shape)
    print("hidden activations shape:", hidden_activations.shape)


    # projecting fixed points and hidden activations into 3D space using PCA
    pca = PCA(n_components=3)
    pca.fit(np.concatenate([hidden_activations, fixed_points], axis=0))
    hidden_activations_3d = pca.transform(hidden_activations)
    fixed_points_3d = pca.transform(fixed_points)

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hidden_activations_3d[:, 0], hidden_activations_3d[:, 1], hidden_activations_3d[:, 2], label='Hidden activations')
    ax.scatter(fixed_points_3d[:, 0], fixed_points_3d[:, 1], fixed_points_3d[:, 2], label='Fixed points', marker='x')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()

    plt.show()


def PCA_analysis_hidden_activations(model_dirs: List[str], train_seq_len=1000, test_seq_len=75, dims=3, task="DualDelayMatchSample-v0"):
    """
    Visualises the hidden activations of trained models corresponding to model_dirs in a 2D or 3D PCA space,
    alongside the ground truth and predicted behaviour.
    """
    # loading dataset and generating train and test data
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=train_seq_len)
    train_inputs, _ = tensor_dataset_sample(dataset)
    dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=1, seq_len=test_seq_len)
    test_inputs, test_labels = tensor_dataset_sample(dataset)
    
    test_labels = test_labels.cpu().detach().numpy().squeeze() # for plotting later

    # getting hidden activations over a test trial for each model
    pred_labels = []
    compressed_activations = []
    names = []
    pca_objs = []
    for model_dir in model_dirs:
        # loading model
        net, name = initialize_model_from_config(model_dir)
        net.eval()
        train_activations, _ = get_hidden_activations(net, train_inputs) # returns a 2D numpy array
        test_activations, pred_label = get_hidden_activations(net, test_inputs) 
        
        # performing PCA
        pca = PCA(n_components=dims)
        pca.fit(train_activations)
        compressed_hidden = pca.transform(test_activations)

        # saving for later
        pca_objs.append(pca)
        compressed_activations.append(compressed_hidden)
        names.append(name)
        pred_labels.append(pred_label)
    
    fig, axs = plt.subplots(1, len(model_dirs)+1, figsize=(4*(len(model_dirs)+1), 4))

    if task == "DualDelayMatchSample-v0":
        action_labels = ['No action', 'Accept', 'Reject']
    elif task == "ContextDecisionMaking-v0":
        action_labels = ['No action', 'choice1', 'choice2']
    
    
    # plotting trajectories in PCA space
    for i, compressed_hidden in enumerate(compressed_activations):
        compressed_hidden_inactive = compressed_hidden[np.where(pred_labels[i] == 0)[0], :]
        compressed_hidden_accept = compressed_hidden[np.where(pred_labels[i] == 1)[0], :]
        compressed_hidden_reject = compressed_hidden[np.where(pred_labels[i] == 2)[0], :]

        if dims == 3:
            axs[i].axis('off')
            axs[i] = fig.add_subplot(1, len(model_dirs)+1, i+1, projection='3d')
            axs[i].scatter(compressed_hidden_inactive[:, 0], compressed_hidden_inactive[:, 1], compressed_hidden_inactive[:, 2], label=action_labels[0], marker='o')
            axs[i].scatter(compressed_hidden_accept[:, 0], compressed_hidden_accept[:, 1], compressed_hidden_accept[:, 2], label=action_labels[1], marker='o')
            axs[i].scatter(compressed_hidden_reject[:, 0], compressed_hidden_reject[:, 1], compressed_hidden_reject[:, 2], label=action_labels[2], marker='o')
            axs[i].plot(compressed_hidden[:, 0], compressed_hidden[:, 1], compressed_hidden[:, 2], label='trajectory', color='grey', alpha=0.5)
            axs[i].set_xlabel('PC 1', fontsize=16)
            axs[i].set_ylabel('PC 2', fontsize=16)
            axs[i].set_zlabel('PC 3', fontsize=16)
        
        elif dims == 2:
            axs[i].scatter(compressed_hidden_inactive[:, 0], compressed_hidden_inactive[:, 1], label=action_labels[0], marker='o')
            axs[i].scatter(compressed_hidden_accept[:, 0], compressed_hidden_accept[:, 1], label=action_labels[1], marker='o')
            axs[i].scatter(compressed_hidden_reject[:, 0], compressed_hidden_reject[:, 1], label=action_labels[2], marker='o')
            axs[i].plot(compressed_hidden[:, 0], compressed_hidden[:, 1], label='trajectory', color='grey', alpha=0.5)
            axs[i].set_xlabel('PC 1', fontsize=16)
            axs[i].set_ylabel('PC 2', fontsize=16)
        
        axs[i].set_title(f'{names[i]}: PCs ~ {int(round(100*np.sum(pca.explained_variance_ratio_)))}% var.', fontsize=16)

    
    axs[0].legend(fontsize=12)

    # plotting predicted labels vs. ground truth
    axs[-1].plot(test_labels, label='GT', color='red')
    # plotting predicted labels for each model
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink'] # unlikely to need more than 6 models
    for i, pred_label in enumerate(pred_labels):
        axs[-1].plot(pred_label, label=names[i], color=colors[i])
    
    axs[-1].set_xlabel('Time steps', fontsize=16)
    axs[-1].set_yticks([0, 1, 2])
    axs[-1].legend(fontsize=16)
    
    axs[-1].set_title('Predicted labels vs. ground truth', fontsize=16)
    
    if task == "DualDelayMatchSample-v0":
        axs[-1].set_yticklabels(['No action', 'Accept', 'Reject'], fontsize=16)
    elif task == "ContextDecisionMaking-v0":
        axs[-1].set_yticklabels(['No action', 'choice1', 'choice2'], fontsize=16)

    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    test_models = [r"runs\light_GRU_run2", r"runs\enu_light_GRU_run2", r"runs\ei_light_GRU_with_l2reg"]
    
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
    # visualise_hidden_activations(r"runs\light_GRU_run2", seq_len=300, fs=16)
    # visualise_hidden_activations(r"runs\ei_light_GRU_run2", seq_len=300, fs=16)
    # visualise_hidden_activations(r"runs\ei_light_GRU_with_l2reg", seq_len=300, fs=16)

    # combined_max_activation_stem_plot(test_models, seq_len=125)
    # combined_timeseries_of_most_sensitive_units(test_models, seq_len=125)
    # find_fixed_points(r"runs\light_GRU_run2", seq_len=75)
    # PCA_analysis_hidden_activations(test_models, train_seq_len=1000, test_seq_len=50, dims=2)

    # monte_carlo_selectivity_histogram(r"runs\light_GRU_run2", simulations=512, seq_len=1000, fs=16)

    ### second task ###
    test_models = [r"runs\\task2_light_GRU_run1", r"runs\\task2_enulight_GRU_run1", r"runs\\task2_eilight_GRU_with_l2reg"]
    test_names = ['light GRU', 'ENU light GRU', 'EI light GRU with l2 reg']
    task = "ContextDecisionMaking-v0"

    # visualise_training_log(model_dirs=test_models, run_names=['light GRU', 'ENU light GRU', 'EI light GRU with l2 reg'], plot_to_epoch=500, log_scale=False)
    # visualise_task_performance(test_models, seq_len=125, task=task)
    # combined_max_activation_stem_plot(test_models, seq_len=1000)
    # visualise_selectivity(test_models[0], seq_len=1000, fs=16, task=task)
    # monte_carlo_selectivity_histogram(test_models[0], simulations=1024, seq_len=1000, fs=16, task=task)
    # visualise_selectivity(test_models[2], seq_len=1000, fs=16, task=task)
    # timeseries_of_select_units(test_models[0], unit_inds=[3, 17, 14], 
    #                            unit_labels=["act. selective", "c1 selective", "c2 selective"],
    #                            task=task, seq_len=125, fs=16)
    # combined_timeseries_of_most_sensitive_units(test_models, seq_len=125, task=task)
    PCA_analysis_hidden_activations(test_models, train_seq_len=1000, test_seq_len=200, dims=3, task=task)