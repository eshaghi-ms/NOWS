import numpy as np

# General Setting;
gpu_number = 'cuda:1'
# gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
# nTrain = 298 * 1000
# nTest = 2 * 1000
batch_size = 100
learning_rate = 0.0001  # 0.0005  # 0.001
weight_decay = 1e-4
epochs = 1000
# iterations = epochs * (nTrain // batch_size)
modes = 16
width = 32
width_q = width
width_h = width
n_layers = 6  # 4
n_layers_q = 2
n_layers_h = 4

# Discretization
s = 64
T_in = 1
T_out = 1

# Training Setting
normalized = False

# Database
parent_dir = '/scratch/zore8312/PycharmProjects/VINO_combined/Dynamic/data/'
# matlab_dataset = 'smoke_N1000_res64_T300_dt0.5.npz'

# Plotting
index = 10
domain = [-1., 1.]
time_steps = [0]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = True

