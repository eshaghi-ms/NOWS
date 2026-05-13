#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:13:35 2023

@author: Mohammad Sadegh

Neural Operator example for Darcy 2D problem

Problem statement:
    Solve the equation \nabla \\cdot (a(x, y) \\nabla u(x, y)) = f(x, y) for x, y \\in (0, 1) with Dirichlet boundary
    conditions u(0,y) = 0, u(x,0) = 0,
    where f(x,y) is the forcing function and kept fixed f(x) = 1
    and a(x,y) is diffusion coefficient
    The network learns the operator mapping the diffusion coefficient to the solution (a ↦ u)

"""
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import time
import torch
import torch.nn.functional as F
import argparse
import statistics
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt

from utils.generate_data_darcy import generate_inputs
from utils.solvers import Darcy2D_solve, Darcy2D_itsolve, Darcy2D_pyagm_solve, Darcy2D_petsc_solve
from utils.postprocessing import (plot_pred1, plot_pred2, plot_loss_trend, plot_two_sets_trajectories,
                                  time_comparison, timing_boxplot, timing_boxplot_multi, timing_violin_multi)
from utils.fno_2d import FNO2d
from utils.fno_utils import count_params, LpLoss, train_fno

plt.rcParams['font.family'] = "DejaVu Serif"
# plt.rcParams['font.size'] = 14

################################################################
#  Parse arguments
################################################################
parser = argparse.ArgumentParser(description="Darcy 2D Solver with VINO")

parser.add_argument("--s", type=int, default=2 ** 6, help="Grid resolution")
parser.add_argument("--modes", type=int, default=12, help="Fourier modes")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

parser.add_argument("--nTrain", type=int, default=1000, help="Number of training samples")
parser.add_argument("--nTest", type=int, default=100, help="Number of test samples")
parser.add_argument("--L", type=float, default=1.0, help="Domain length")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--step_size", type=int, default=10, help="Step size for scheduler")
parser.add_argument("--gamma", type=float, default=0.5, help="Gamma for scheduler")
parser.add_argument("--width", type=int, default=32, help="Width of FNO layers")
parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
parser.add_argument("--use_data", type=int, default=1, help="Whether to use dataset or regenerate")
parser.add_argument("--load_model", type=int, default=1, help="Load pretrained model")
parser.add_argument("--training", type=int, default=1, help="Enable/disable training")

args = parser.parse_args()

################################################################
#  Initial settings
################################################################
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print("this is the device name:")
print(device)


class FNO2dSpecific(FNO2d):
    def forward(self, _x, **kwargs):
        grid = self.get_grid(_x.shape, _x.device)
        _x = super().forward(_x)
        m = grid[..., 0:1] * grid[..., 1:2] * (grid[..., 0:1] - 1) * (grid[..., 1:2] - 1)
        _x *= m
        return _x


# Calculate F at nodes based on grid elements
def ElementsToNodes(elementValues):
    temp = F.pad(elementValues, [0, 0, 0, 1, 0, 1], mode='replicate')
    return (temp[:, 0:-1, 0:-1, :] + temp[:, 1:, 0:-1, :] + temp[:, 0:-1, 1:, :] + temp[:, 1:, 1:, :]) / 4


################################################################
#  configurations from args
################################################################
nTrain = args.nTrain
nTest = args.nTest
L = args.L
inputRange = [-1, 1]
s = args.s
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma
modes = args.modes
width = args.width
patience = args.patience
use_data = bool(args.use_data)
load_model = bool(args.load_model)
training = bool(args.training)

if s // 2 + 1 < modes:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

print(f"nTrain = {nTrain}")
print(f"nTest = {nTest}")
print(f"L = {L}")
print(f"s = {s}")
print(f"batch_size = {batch_size}")
print(f"learning_rate = {learning_rate}")
print(f"epochs = {epochs}")
print(f"step_size = {step_size}")
print(f"gamma = {gamma}")
print(f"modes = {modes}")
print(f"width = {width}")
print(f"patience = {patience}")
print(f"use_data = {use_data}")
print(f"load_model = {load_model}")
print(f"training = {training}")

#################################################################
# generate the data
#################################################################
t_start = time.time()

alpha = 2.
tau = 3.

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'darcy2d'
dataset_dir = os.path.join(base_dir, "data")
dataset_filename = os.path.join(dataset_dir, f"{dataset_name}_nTrain{nTrain}_nTest{nTest}_s{s}.pt")
os.makedirs(dataset_dir, exist_ok=True)

if os.path.exists(dataset_filename):
    print("Found saved dataset at", dataset_filename)
    loaded_data = torch.load(dataset_filename, map_location=device)
    F_train = loaded_data['F_train']
    U_train = loaded_data['U_train']
    F_test = loaded_data['F_test']
    U_test = loaded_data['U_test']
else:
    F_train = torch.from_numpy(generate_inputs(nTrain, s, alpha, tau)).float()
    U_train = torch.from_numpy(Darcy2D_solve(F_train.numpy())).to(device).float()

    F_test = torch.from_numpy(generate_inputs(nTest, s, alpha, tau)).float()
    U_test = torch.from_numpy(Darcy2D_solve(F_test.numpy())).to(device).float()

    F_train = F_train.reshape(nTrain, s, s, 1)
    U_train = U_train.reshape(nTrain, s, s, 1)

    F_test = F_test.reshape(nTest, s, s, 1)
    U_test = U_test.reshape(nTest, s, s, 1)
    torch.save({'F_train': F_train, 'U_train': U_train,
                'F_test': F_test, 'U_test': U_test}, dataset_filename)

F_train = F_train[:nTrain, :, :, :]
U_train = U_train[:nTrain, :, :, :]
F_test = F_test[:nTest, :, :, :]
U_test = U_test[:nTest, :, :, :]

# Calculate F at nodes based on grid elements
F_train_Nodes = ElementsToNodes(F_train)
F_test_Nodes = ElementsToNodes(F_test)

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train_Nodes, U_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test_Nodes, U_test),
                                          batch_size=batch_size, shuffle=False)

#################################################################
# initialize model
#################################################################
model_dir = os.path.join(base_dir, "models")
model_name = (
    f"FNO_Darcy2D_nTrain{nTrain}_nTest{nTest}_S{s}_batch{batch_size}"
    f"_epochs{epochs}_gamma{gamma}_modes{modes}_width{width}_patience{patience}.pt"
)
model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

model = FNO2dSpecific(modes, modes, width, inputRange).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
################################################################
# training
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
myLoss = LpLoss(d=1, size_average=False)

t1 = default_timer()
print("epoch,      t2-t1,     train_mse,      train_l2,      test_l2")
train_l2_log = []
test_l2_log = []
if os.path.exists(model_path) and load_model:
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['model']
    train_l2_log = checkpoint.get('train_l2_log', [])
    test_l2_log = checkpoint.get('test_l2_log', [])
else:
    print("No pre-trained model loaded. Initializing a new model.")

if training:
    train_mse, train_l2, test_l2 = train_fno(model, train_loader, test_loader, myLoss, optimizer,
                                             scheduler, device, epochs, batch_size)
    train_l2_log = np.concatenate((train_l2_log, train_l2))
    test_l2_log = np.concatenate((test_l2_log, test_l2))
    print(f"Saving model and logs to {model_path}")
    torch.save({
        'model': model,
        'train_l2_log': train_l2_log,
        'test_l2_log': test_l2_log
    }, model_path)

print("Training time: ", default_timer() - t1)

# plot the convergence of the losses
losses = [train_l2_log, test_l2_log]
labels = ['Train Loss', 'Test Loss']
plot_loss_trend(losses, labels, ylims=((0, 0.3), (0, 1.5)))

################################################################
# evaluation for training database
################################################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train_Nodes, U_train),
                                           batch_size=1, shuffle=False)
index = 0
pred_vino = torch.zeros(U_train.squeeze().shape)
inp_train_nodes = torch.zeros(nTrain, s, s)
sol_train = torch.zeros(U_train.shape)
train_l2_set = []

with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        inp_train_nodes[index] = x.reshape(1, s, s)
        sol_train[index] = y
        out = model(x).view(s, s)
        pred_vino[index] = out
        train_l2 = myLoss(out.view(1, -1), y.view(1, -1)).item()
        train_l2_set.append(train_l2)
        print(index, train_l2)
        index = index + 1

train_l2_set = torch.tensor(train_l2_set)
train_l2_avg = torch.mean(train_l2_set)
train_l2_std = torch.std(train_l2_set)
train_l2_argmax = torch.argmax(train_l2_set).item()
print("################################################################")
print("evaluation for training database")
print("################################################################")
print("The average training error is", train_l2_avg.item())
print("Std. deviation of training error is", train_l2_std.item())
print("Min training error is", torch.min(train_l2_set).item())
print("Max training error is", torch.max(train_l2_set).item())
print("Index of maximum error is", train_l2_argmax)

################################################################
# evaluation for testing database
################################################################
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test_Nodes, U_test),
                                          batch_size=1, shuffle=False)
index = 0
pred_vino = torch.zeros(U_test.squeeze().shape)
inp_test_node = torch.zeros(nTest, s, s)
sol_test = torch.zeros(U_test.shape)
test_l2_set = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        inp_test_node[index] = x.reshape(1, s, s)
        sol_test[index] = y
        out = model(x).view(s, s)
        pred_vino[index] = out
        test_l2 = myLoss(out.view(1, -1), y.view(1, -1)).item()
        test_l2_set.append(test_l2)
        print(index, test_l2)
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("################################################################")
print("evaluation for testing database")
print("################################################################")
print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())
print("Index of maximum error is", torch.argmax(test_l2_set).item())
print("Index of minimum error is", torch.argmin(test_l2_set).item())
print("################################################################")

################################################################
# Plotting a random function from the test data generated by GRF
################################################################
index = 0  # random.randrange(0, nTest)
x_test_plot = np.linspace(0., L, s).astype('float32')
y_test_plot = np.linspace(0., L, s).astype('float32')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)

coefficient = inp_test_node[index, :].squeeze()
u_exact = sol_test[index, :].squeeze()
u_pred = pred_vino[index, :]
plot_pred1(x_plot_grid, y_plot_grid, coefficient, 'GRF')
plot_pred2(x_plot_grid, y_plot_grid, u_exact, u_pred, 'Test with GRF', 'Ex1')
print("################################################################")


################################################################
# combine with iterative solver
################################################################
def error_calculator(ref, pred):
    return [np.linalg.norm(ref[i, :].flatten() - pred[i, :].flatten()) /
            np.linalg.norm(ref[i, :].flatten()) for i in range(nTest)]


def print_errors(u_no_vino, u_with_vino, y_test2):
    errs_no_vino = [np.linalg.norm(y_test2[i, :, :].flatten() - u_no_vino[i, :, :].flatten()) /
                    np.linalg.norm(y_test2[i, :, :].flatten()) for i in range(nTest)]
    errs_with_vino = [np.linalg.norm(y_test2[i, :, :].flatten() - u_with_vino[i, :].flatten()) /
                      np.linalg.norm(y_test2[i, :, :].flatten()) for i in range(nTest)]
    print("Average error without VINO =", statistics.mean(errs_no_vino))
    print("Average error with VINO =", statistics.mean(errs_with_vino))
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")


inp_test_node = inp_test_node.numpy()
inp_test = F_test.view(-1, s, s).cpu().numpy()
sol_test = sol_test.numpy()
pred_vino = pred_vino.numpy()

# print("With direct solver")
u_no_vino = sol_test

# ------------------------------------------------------------
# ------------------------------------------------------------
print("With scipy")
solver_name = 'CG'
fig_size = (6, 2.5)
u_cg, time_cg, tr_cg = Darcy2D_itsolve(inp_test, rtol=1e-5)
u_nows_cg, time_nows_cg, tr_nows_cg = Darcy2D_itsolve(inp_test, pred_vino, rtol=1e-5)

errs = error_calculator(u_cg, u_no_vino)
print("Average error between u_no_vino_scipy and u_no_vino =", statistics.mean(errs))
print("It shows whether both solvers solve one problem!")
print_errors(u_cg, u_nows_cg, u_no_vino)

time_comparison(time_cg, tr_cg, time_nows_cg, tr_nows_cg, solver_name)
plot_two_sets_trajectories(tr_nows_cg, tr_cg, label1='CG-NOWS', label2='CG', fig_size=fig_size)
_, _, _ = timing_violin_multi(tr_nows_cg, tr_cg, label1="CG-NOWS", label2="CG", fig_size=fig_size)

# ------------------------------------------------------------
# ------------------------------------------------------------
print("With pyagm")
solver_name = 'pyAMG'

u_pyAMG, time_pyAMG, tr_pyAMG = Darcy2D_pyagm_solve(inp_test, rtol=1e-5)
u_nows_pyAMG, time_nows_pyAMG, tr_nows_pyAMG = Darcy2D_pyagm_solve(inp_test, pred_vino, rtol=1e-5)

errs = error_calculator(u_pyAMG, u_no_vino)
print("Average error between u_no_vino_pyagm and u_no_vino =", statistics.mean(errs))
print("It shows whether both solvers solve one problem!")
print_errors(u_pyAMG, u_nows_pyAMG, u_no_vino)

time_comparison(time_pyAMG, tr_pyAMG, time_nows_pyAMG, tr_nows_pyAMG, solver_name)
plot_two_sets_trajectories(tr_nows_pyAMG, tr_pyAMG, label1='pyAMG-NOWS', label2='pyAMG', fig_size=fig_size)
_, _, _ = timing_violin_multi(tr_nows_pyAMG, tr_pyAMG, label1="pyAMG-NOWS", label2="pyAMG", fig_size=fig_size)

# ------------------------------------------------------------
# ------------------------------------------------------------
print("With PETSc")
solver_name = 'PETSc'

u_PETSc, time_PETSc, tr_PETSc = Darcy2D_petsc_solve(inp_test, rtol=1e-6)
u_nows_PETSc, time_nows_PETSc, tr_nows_PETSc = Darcy2D_petsc_solve(inp_test, pred_vino, rtol=1e-6)

errs = error_calculator(u_PETSc, u_no_vino)
print("Average error between u_no_vino_petsc and u_no_vino =", statistics.mean(errs))
print("It shows whether both solvers solve one problem!")
print_errors(u_PETSc, u_nows_PETSc, u_no_vino)

time_comparison(time_PETSc, tr_PETSc, time_nows_PETSc, tr_nows_PETSc, solver_name)
plot_two_sets_trajectories(tr_nows_PETSc, tr_PETSc, label1='PETSc-NOWS', label2='PETSc', fig_size=fig_size)
_, _, _ = timing_violin_multi(tr_nows_PETSc, tr_PETSc, label1="PETSc-NOWS", label2="PETSc", fig_size=fig_size)
