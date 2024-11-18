from adapted_GRUs import GRU_Net
import numpy as np
import neurogym as ngym
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Environment
task = 'DualDelayMatchSample-v0'
batch_size = 32
dataset = ngym.Dataset(task, env_kwargs={'dt': 100}, batch_size=batch_size)
env = dataset.env

# task specific parameters
output_size = env.action_space.n
input_size = env.observation_space.shape[0]
hidden_size = 64

#getting labelled data
inputs, labels = dataset()
inputs = torch.from_numpy(inputs).to(device)
labels = torch.from_numpy(labels).to(device)

# initializing the model
net = GRU_Net(input_size, hidden_size, output_size, model_type='light')
net = net.to(device)

for param in net.parameters():
    print(param.device)

test_preds = net(inputs)

if __name__ == "__main__":
    print("actions", output_size)
    print("observations", input_size)
    print("inputs:", inputs.shape, "labels:", labels.shape)
    print("inputs 1-10 :", inputs[0:10], "labels:", labels[0:10])
    print("test_preds:", test_preds.shape)
    print("test_preds 1-10:", test_preds[0:10])



