"""
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *

torch.manual_seed(10)
np.random.seed(10)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, out_size=1):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width)  # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, out_size, self.width * 2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


time_dependent = True
################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

if time_dependent:
    sub = 1
    h = 2 ** 13 // sub
    s = 2 ** 10
else:
    sub = 2 ** 3  # subsampling rate
    h = 2 ** 13 // sub  # total grid size divided by the subsampling rate
    s = h


batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs * (ntrain // batch_size)
time_steps = 201

modes = 16
width = 64

load_model = True
training = False
################################################################
# read data
################################################################
# Data is of the shape (number of samples, grid size)
if time_dependent:
    dataloader = MatReader('data/burgers_v1000_t200_r1024_N2048.mat')
    x_data = dataloader.read_field('input')[:, ::sub]
    y_data = dataloader.read_field('output')[:, ::sub]
    y_train = y_data[:ntrain, :]
    y_test = y_data[-ntest:, :]
else:
    dataloader = MatReader('data/burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:, ::sub]
    y_data = dataloader.read_field('u')[:, ::sub]
    y_train = y_data[:ntrain, :]
    y_test = y_data[-ntest:, :]

x_train = x_data[:ntrain, :]
x_test = x_data[-ntest:, :]

x_train = x_train.reshape(ntrain, s, 1)
# y_train = y_train.reshape(ntrain, s, time_steps)
x_test = x_test.reshape(ntest, s, 1)
# y_test = y_test.reshape(ntest, s, time_steps)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        x_test, y_test), batch_size=batch_size, shuffle=False)

#################################################################
# initialize model
#################################################################
model_dir = 'models'
model_name = f'FNO_Burger_nTrain{ntrain}_nTest{ntest}_S{s}_batch{batch_size}_epochs{epochs}_modes{modes}_width{width}.pt'
model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

model = FNO1d(modes, width, y_data.shape[1]).cuda()
print(f'Our model has {count_params(model)} parameters.')

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
myloss = LpLoss(size_average=False)

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
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1), reduction='mean')
            l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))
            l2.backward()  # use the l2 relative loss

            optimizer.step()
            scheduler.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        train_l2_log.append(train_l2)
        test_l2_log.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_mse, train_l2, test_l2)

    print(f"Saving model and logs to {model_path}")
    torch.save({
        'model': model,
        'train_l2_log': train_l2_log,
        'test_l2_log': test_l2_log
    }, model_path)

# torch.save(model, 'model/ns_fourier_burgers')
pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()
        out = model(x)
        pred[index] = out

        test_l2 += myloss(out.reshape(1, -1), y.reshape(1, -1)).item()
        print(index, test_l2)
        index = index + 1

################################################################
# evaluation
################################################################
plot_index = 39
plot_test_sample_1d(plot_index, x_test, y_test, pred, s)

visc = 1/1000
steps = 200
x_grid = np.linspace(0, 1, s, endpoint=False)
tspan = np.linspace(0, 1, steps + 1)

spectral_times = []
direct_times = []
cg_times = []
cg_exact_times = []
cg_fno_times = []

for index in range(40):
# for xxx in range(1):
#     index = 19
    print(f'index = {index}')
    # Extract input and ground truth
    u0 = x_test[index, :, 0].cpu().numpy()
    u_exact = y_test[index, :, :].cpu().numpy()
    u_exact_end = u_exact[-1]
    u_pred = pred[index, :, :].cpu().numpy()
    u_pred_end = u_pred[-1]
    print(f'diff FNO and dataset = {np.linalg.norm(u_pred - u_exact)}')
    print('-------------------------------------------------------------')
    # --- Run direct solver ---
    # start = time.time()
    # u_direct = burgers_solver_direct(u0, tspan, s, visc)
    # u_direct_end = u_direct[-1]
    # direct_times.append(time.time() - start)
    # print(f'diff direct solver and dataset = {np.linalg.norm(u_direct[1:] - u_exact[1:])}')
    # print('-------------------------------------------------------------')

    # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST #
    NEW_u_direct = NEW_burgers_solver_direct(u0, tspan, s, visc)
    print(f'NEW direct solver = {np.linalg.norm(NEW_u_direct[1:] - u_exact[1:])}')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # --- Run spectral solver ---
    start = time.time()
    u_spectral = burgers_solver_spectral(u0, tspan, s, visc)
    u_spectral_end = u_spectral[-1]
    spectral_times.append(time.time() - start)
    print(f'diff spectral solver and dataset = {np.linalg.norm(u_spectral[1:] - u_exact[1:])}')
    # print(f'diff spectral solver and direct = {np.linalg.norm(u_spectral - u_direct)}')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST #
    NEW_u_spectral = NEW_burgers_solver_spectral(u0, tspan, s, visc)
    print(f'NEW spectral solver = {np.linalg.norm(NEW_u_spectral[1:] - u_exact[1:])}')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # --- Run iterative CG solver ---
    # start = time.time()
    u_cg = burgers_solver_cg(u0, tspan, s, visc, rtol=1e-5)
    u_cg_end = u_cg[-1]
    # cg_times.append(time.time() - start)
    print(f'diff CG solver and dataset = {np.linalg.norm(u_cg[1:] - u_exact[1:])}')
    # print(f'diff CG solver and direct = {np.linalg.norm(u_cg - u_direct)}')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST # # TEST #
    start = time.time()
    NEW_u_cg = NEW_burgers_solver_cg(u0, tspan, s, visc)
    cg_times.append(time.time() - start)
    print(f'NEW cg solver = {np.linalg.norm(NEW_u_cg[1:] - u_exact[1:])}')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    # --- Run iterative CG solver initialized with iterative solution ---
    start = time.time()
    u_cg_exact = NEW_burgers_solver_cg(u0, tspan, s, visc, X0=u_cg, rtol=1e-5)
    u_cg_exact_end = u_cg_exact[-1]
    cg_exact_times.append(time.time() - start)

    # --- Run iterative CG solver initialized with fno solution ---
    start = time.time()
    u_cg_fno = NEW_burgers_solver_cg(u0, tspan, s, visc, X0=u_pred, rtol=1e-5)
    u_cg_fno_end = u_cg_fno[-1]
    cg_fno_times.append(time.time() - start)

    print(f'diff CG solver and spectral = {np.linalg.norm(NEW_u_cg - NEW_u_spectral)}')
    print(f'diff direct and spectral  = {np.linalg.norm(NEW_u_direct - NEW_u_spectral)}')

print("Average Spectral Time:", np.sum(spectral_times))
# print("Average Direct Time:", np.sum(direct_times))
print("Average CG Time:", np.sum(cg_times))
print("Average CG (initialized with exact solution) Time:", np.sum(cg_exact_times))
print("Average CG (initialized with FNO) Time:", np.sum(cg_fno_times))

# Plot all
plt.figure(figsize=(8, 3))
plt.plot(x_grid, u_exact_end, label='Exact', linewidth=2)
plt.plot(x_grid, u_cg_fno_end, '--', label='NOWS', linewidth=2)
# plt.plot(x_grid, u_spectral_end, ':', label='Spectral Solver', linewidth=2)
# plt.plot(x_grid, u_direct_end, '-.', label='Direct Solver', linewidth=2)
# plt.plot(x_grid, u_cg_end, ':', label='CG Solver', linewidth=2)
# plt.plot(x_grid, u_cg_exact_end, ':', label='CG Solver (with Exact)', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Comparison for Test Sample #{index}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot error
plt.figure(figsize=(8, 3))
# plt.plot(x_grid, u_exact_end-u_direct_end, label='Dataset - Direct', linewidth=2)
plt.plot(x_grid, u_exact_end-u_cg_fno_end, label='Dataset - NOWS', linewidth=2)
plt.xlabel('x')
plt.ylabel('error')
plt.title(f'Comparison for Test Sample #{index}')
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def plot_violin_comparison(data1, data2, labels=('CG', 'NOWS'), title='CG vs CG+MHNO Runtime Comparison'):
    """
    Plot a violin plot comparing two timing distributions with overlaid data points.

    Parameters:
        data1 (list or np.ndarray): First dataset (e.g., cg_times)
        data2 (list or np.ndarray): Second dataset (e.g., cg_fno_times)
        labels (tuple): Labels for the two datasets
        title (str): Title of the plot
    """
    # Combine data into a long-form structure for seaborn
    data = [data1, data2]

    plt.figure(figsize=(4, 6))

    # Violin plot (distribution shape + quartiles)
    sns.violinplot(data=data, inner='quartile', palette='Set2')

    # Overlay data points
    sns.swarmplot(data=data, color='k', size=3, alpha=0.7)

    plt.xticks([0, 1], labels, fontsize=12)
    plt.ylabel('Runtime (s)', fontsize=12)
    # plt.title(title, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


plot_violin_comparison(cg_times, cg_fno_times)


import pandas as pd


def compare_stats(data1, data2, labels=('CG', 'NOWS')):
    """
    Compute and print statistical information of the difference between two datasets.

    Parameters:
        data1 (list or np.ndarray): First dataset (e.g., cg_times)
        data2 (list or np.ndarray): Second dataset (e.g., cg_fno_times)
        labels (tuple): Labels for the two datasets

    Returns:
        pd.DataFrame: Table summarizing statistics
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    diff = (data1 - data2) / data1 * 100

    stats = {
        'Mean': [np.mean(data1), np.mean(data2), np.mean(diff)],
        'Median': [np.median(data1), np.median(data2), np.median(diff)],
        'Std Dev': [np.std(data1), np.std(data2), np.std(diff)],
        'Min': [np.min(data1), np.min(data2), np.min(diff)],
        'Max': [np.max(data1), np.max(data2), np.max(diff)],
    }

    df = pd.DataFrame(stats, index=[labels[0], labels[1], "Time Saving (%)"])
    df.iloc[2] = df.iloc[2].round(2)
    print(df)
    return df


stats_df = compare_stats(cg_times, cg_fno_times)