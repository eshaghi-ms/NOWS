import os
import time
import copy
import torch
import importlib
import numpy as np
# from phi.jax.flow import *
from phi.torch.flow import *
from tqdm import trange
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

plt.ion()

################################################################
# Problem Definition & Setup (unchanged)
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

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), problem, "models")
model_name = f'{network_name}_{problem}_deNormalized_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}.pt'
model_path = os.path.join(model_dir, model_name)
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = checkpoint['model']

domain = Box(x=100, y=100)
inflow = Sphere(x=50, y=9.5, radius=5)
inflow_rate = 0.2
Dt = 0.5
num_timestep = 100#300


################################################################
# Core experiment function
################################################################
def run_experiment(use_fno: bool):
    """Run one simulation, either with or without FNO. Returns metrics dict."""
    # initialize counters & timers
    diff_norm_total = 0.0
    convergence = 0
    non_convergence = 0
    diff_count = 0
    solver_timer = 0.0

    # pick your step() implementation
    def step_no_fno(v, s, p, dt):
        nonlocal diff_norm_total, convergence, non_convergence, diff_count, solver_timer
        s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
        buoyancy = resample(s * (0, 0.1), to=v)
        v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt

        p_before = copy.deepcopy(p) if p is not None else None
        t0 = time.time()
        try:
            _, p_new = fluid.make_incompressible(v, (), Solve('CG', 1e-3, x0=p, max_iterations=5000000))
            # measure pressure‐diff norm
            if p_before is not None:
                diff = p_new.values - p_before.values
                diff_norm = np.linalg.norm(diff)
                diff_norm_total += diff_norm
                diff_count += 1
            convergence += 1
            p = p_new
        except:
            non_convergence += 1
        solver_timer += time.time() - t0
        return v, s, p

    def step_with_fno(v, s, p, dt):
        nonlocal diff_norm_total, convergence, non_convergence, diff_count, solver_timer
        s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
        buoyancy = resample(s * (0, 0.1), to=v)
        v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt

        # prepare neural‐net input
        arrs = [
            np.expand_dims(v.at_centers().vector[ax].numpy(), axis=(0, -1))
            for ax in ['x', 'y']
        ]
        if p is None:
            arrs.append(np.zeros((1, 64, 64, 1)))
        else:
            arrs.append(np.expand_dims(p.at_centers().numpy(), axis=(0, -1)))
        arrs.append(
            np.expand_dims(resample(s.at_centers(), to=CenteredGrid(0, ZERO_GRADIENT, domain, x=64, y=64)).numpy(),
                           axis=(0, -1)))
        inp = torch.from_numpy(np.concatenate(arrs, axis=-1)).to(device).float()
        with torch.no_grad():
            p0 = model(inp).cpu().numpy().squeeze()

        p0_field = CenteredGrid(tensor(p0, spatial('x,y')), ZERO_GRADIENT, domain, x=64, y=64)
        t0 = time.time()
        try:
            _, p_new = fluid.make_incompressible(v, (), Solve('CG', 1e-3, x0=p0_field, max_iterations=5000000))
            diff = p_new.values - p0_field.values
            diff_norm_total += np.linalg.norm(diff)
            diff_count += 1
            convergence += 1
            p = p_new
        except:
            non_convergence += 1
        solver_timer += time.time() - t0
        return v, s, p

    # choose step
    step_fn = step_with_fno if use_fno else step_no_fno

    # initial fields
    v0 = StaggeredGrid(0, 0, domain, x=64, y=64)
    smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)

    t_start = time.time()
    iterate(step_fn, batch(time=num_timestep),
            v0, smoke0, None, dt=Dt,
            range=trange, substeps=1)
    total_time = time.time() - t_start

    return {
        'total_time': total_time,
        'solver_time': solver_timer,
        'converged': convergence,
        'not_converged': non_convergence,
        'avg_diff': (diff_norm_total / diff_count) if diff_count > 0 else 0.0
    }


################################################################
# Run 100 trials and collect stats
################################################################
import statistics


def summarize(runs, label):
    print(f"\n=== Statistics for {label} over {len(runs)} runs ===")
    for key in runs[0].keys():
        vals = [r[key] for r in runs]
        print(f"{key:15s}: mean={statistics.mean(vals):.4f}, "
              f"min={min(vals):.4f}, max={max(vals):.4f}, std={statistics.pstdev(vals):.4f}")


# run experiments
def safe_run(use_fno):
    try:
        return run_experiment(use_fno)
    except Exception as e:
        print(f"Run failed with error: {e}. Retrying once...")
        try:
            return run_experiment(use_fno)
        except Exception as e2:
            print(f"Run failed with error: {e2}. Retrying once...")
            try:
                return run_experiment(use_fno)
            except Exception as e3:
                print(f"Second attempt failed. Skipping run. Error: {e3}")
                return None  # Or some placeholder


N = 2
no_fno_runs = [r for r in (safe_run(False) for _ in range(N)) if r is not None]
with_fno_runs = [r for r in (safe_run(True) for _ in range(N)) if r is not None]

# print summaries
summarize(no_fno_runs, "Without FNO")
summarize(with_fno_runs, "With    FNO")
