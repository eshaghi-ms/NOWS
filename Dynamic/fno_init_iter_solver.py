import os
import time
import copy
import torch
import importlib
import matplotlib
from tqdm import trange
from phi.jax.flow import *
# from phi.torch.flow import *
from phiml.math import SolveTape
from types import SimpleNamespace
from phiml.backend import set_global_precision
set_global_precision(64)
torch.set_default_dtype(torch.float64)
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# plt.ion()
################################################################
# Problem Definition
################################################################
problem = 'Smoke'
network_name = 'FNO2d'
print(f"problem = {problem}")
print(f"network = {network_name}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")
network = getattr(importlib.import_module('networks'), network_name)
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
device = torch.device(cf.gpu_number if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
################################################################
# load data and data normalization
################################################################
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), problem, "smoke_N20_res64_T150_Dt0.5_substeps1", "models")

model_name = f'{network_name}_{problem}_NotNormalized_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}.pt'
print(f"model = {model_name}")
print(f"number of epoch = {cf.epochs}")
print(f"batch size = {cf.batch_size}")
print(f"learning_rate = {cf.learning_rate}")
print(f"n_layers = {cf.n_layers}")
print(f"width_q = {cf.width_q}")
print(f"width_h = {cf.width_h}")

model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)
################################################################
# Loading pre-trained model
################################################################
print(f"Loading pre-trained model from {model_path}")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = checkpoint['model']
train_mse_log = checkpoint.get('train_mse_log', [])
train_l2_log = checkpoint.get('train_l2_log', [])
test_l2_log = checkpoint.get('test_l2_log', [])

domain = Box(x=100, y=100)
inflow = Sphere(x=50, y=11, radius=5)
inflow_rate = 0.2
Dt = 0.5
num_timestep = 150
res = 256
max_it = 1000
################################################################
# Implementation W/O FNO
################################################################
metrics_nf = SimpleNamespace(diff=0.0, conv=0, nconv=0, diffc=0, st=0.0, iter=0)


# @jit_compile
def step(v, s, p, dt):
    global metrics_nf

    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt

    p_before = copy.deepcopy(p) if p is not None else None
    max_iters = max_it
    hard_cap = 5000
    t0 = time.time()
    while True:
        with SolveTape() as solve_tape:
            solve = Solve('CG',
                          rel_tol=1e-3,
                          x0=p,
                          max_iterations=max_iters)
            try:
                v, p = fluid.make_incompressible(v, (), solve)
                if p_before is not None:
                    diff_norm = np.linalg.norm(p.values - p_before.values)
                    # print(f"Norm of pressure difference: {diff_norm:.6f}")
                    metrics_nf.diff += diff_norm
                    metrics_nf.diffc += 1
                metrics_nf.conv += 1

                metrics_nf.iter += solve_tape[0].iterations
                break  # success!
            except (NotConverged, Diverged):
                if max_iters >= hard_cap:
                    print(f"Solver still not converging at {max_iters} iters → giving up")
                    metrics_nf.nconv += 1
                    break
                print(f"CG failed at {max_iters} iters; retrying with {max_iters + max_it}")
                max_iters += max_it
    metrics_nf.st += time.time() - t0
    return v, s, p


v0 = StaggeredGrid(0, 0, domain, x=res, y=res)
smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)

t0 = time.time()
v_trj, s_trj, p_trj = iterate(step, batch(time=num_timestep), v0, smoke0, None, dt=Dt, range=trange, substeps=1)
t1 = time.time() - t0

print("Without FNO")
print("total time = ", t1)
print("solver time = ", metrics_nf.st)
print("convergence number = ", metrics_nf.conv)
print("not convergence number = ", metrics_nf.nconv)
print("total iterations = ", metrics_nf.iter)
if metrics_nf.diffc != 0:
    print(f"average of pressure difference: {metrics_nf.diff/metrics_nf.diffc:.6f}")

anim = plot(s_trj, animate='time', frame_time=80)
gif_path = os.path.join(model_dir, f'smoke_animation_withoutFNO.gif')
writer = PillowWriter(fps=20)
anim.save(gif_path, writer=writer)
# anim.save('withoutFNO.mp4', writer='ffmpeg', fps=30)
# plt.show(block=True)
################################################################
# Implementation With FNO
################################################################
target_res = CenteredGrid(0, ZERO_GRADIENT, domain, x=res, y=res)
# initialize metrics for FNO run
metrics_f = SimpleNamespace(diff=0.0, conv=0, nconv=0, diffc=0, st=0.0, iter=0)


def step(v, s, p, dt):
    global metrics_f

    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt

    # p_before = copy.deepcopy(p) if p is not None else None

    if p is None:
        inp = torch.from_numpy(np.concatenate([
            np.expand_dims(v.at_centers().vector['x'].numpy(), axis=(0, -1)),
            np.expand_dims(v.at_centers().vector['y'].numpy(), axis=(0, -1)),
            np.zeros((1, res, res, 1)),
            # np.expand_dims(resample(s, to=target_res).numpy(), axis=(0, -1))
        ], axis=-1
        )).to(device).float().detach()
    else:
        inp = torch.from_numpy(np.concatenate([
            np.expand_dims(v.at_centers().vector['x'].numpy(), axis=(0, -1)),
            np.expand_dims(v.at_centers().vector['y'].numpy(), axis=(0, -1)),
            np.expand_dims(p.numpy(), axis=(0, -1)),
            # np.expand_dims(resample(s, to=target_res).numpy(), axis=(0, -1))
        ], axis=-1
        )).to(device).float().detach()

    with torch.no_grad():
        p0 = model(inp).detach()

    p0_tensor = tensor(np.squeeze(p0), spatial('x,y'))
    p0_field = CenteredGrid(p0_tensor, ZERO_GRADIENT, domain, x=res, y=res)

    max_iters = max_it
    hard_cap = 5000
    t0 = time.time()
    while True:
        with SolveTape() as solve_tape:
            solve = Solve('CG',
                          rel_tol=1e-3,
                          x0=p0_field,
                          max_iterations=max_iters)
            try:
                v, p = fluid.make_incompressible(v, (), solve)
                if p0_field is not None:
                    diff_val = np.linalg.norm(p.values - p0_field.values)
                    # print(f"FNO - Norm of pressure difference: {diff_val:.6f}")
                    metrics_f.diff += diff_val
                    metrics_f.diffc += 1
                metrics_f.conv += 1

                metrics_f.iter += solve_tape[0].iterations
                break  # success!
            except (NotConverged, Diverged):
                if max_iters >= hard_cap:
                    print(f"Solver still not converging at {max_iters} iters → giving up")
                    metrics_f.nconv += 1
                    break
                print(f"CG failed at {max_iters} iters; retrying with {max_iters + max_it}")
                max_iters += max_it
    metrics_f.st += time.time() - t0
    return v, s, p


v0 = StaggeredGrid(0, 0, domain, x=res, y=res)
smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)

t0 = time.time()
v_trj, s_trj, p_trj = iterate(step, batch(time=num_timestep), v0, smoke0, None, dt=Dt, range=trange, substeps=1)
t1 = time.time() - t0

print("With FNO")
print("total time = ", t1)
print("solver time = ", metrics_f.st)
print("convergence number = ", metrics_f.conv)
print("not convergence number = ", metrics_f.nconv)
print("total iterations = ", metrics_f.iter)
if metrics_f.diffc != 0:
    print(f"average of pressure difference: {metrics_f.diff/metrics_f.diffc:.6f}")

anim = plot(s_trj, animate='time', frame_time=80)
gif_path = os.path.join(model_dir, f'smoke_animation_withFNO.gif')
writer = PillowWriter(fps=20)
anim.save(gif_path, writer=writer)
# anim.save('withFNO.mp4', writer='ffmpeg', fps=30)
# plt.show(block=True)
