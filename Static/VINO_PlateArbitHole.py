"""
Energy-based Physics-Informed Fourier Neural Operator for a 2D elasticity problem for a plate with random voids

Problem statement:
    \\Omega = (0,5)x(0,5)
    Fixed BC: x = 0
    Traction \\tau = 1 at y=5 in the horizontal direction
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import jax
import time
import torch
import pickle
import statistics
import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split

from utils.fno_2d import FNO2D
from utils.database_makers import PlateHole_dataset
from configs.config_PlateArbitVoid import model_data
from utils.postprocessing import plot_loss_trend, plot_PlateArbitVoid
from utils.pde_solvers import PlateHole_IGA_solver, PlateWithHole_solver, PlateWithHole_IGA_solver
from utils.postprocessing import plot_field_2d
from utils.fno_utils import (
    count_params, VinoPlateHoleLoss, train_fno, model_evaluation, collate_fn,
    save_training_state, load_training_state, compute_validation_loss
)

# jax.default_device = jax.devices("cpu")
torch.manual_seed(42)
np.random.seed(42)
print("JAX devices:", jax.devices())


################################################################
#  configurations
################################################################
class FNO2d_PlateHole(FNO2D):
    model_data: dict = None

    def __call__(self, x_: jnp.ndarray) -> jnp.ndarray:
        x_ = super().__call__(x_)
        x_ = x_.at[:, :, 0, :].set(0)
        return x_

    def get_grid(self, x_):
        grid = super().get_grid(x_)
        gridy = grid[..., 0:1] * self.model_data['width']
        gridx = grid[..., 1:2] * self.model_data['length']
        return jnp.concatenate((gridy, gridx), -1)


class PlateHoleDEM_dataset(PlateHole_dataset):
    def set_data(self):
        self.x = self.material
        self.y = self.disp2D
        if self.model_data['normalized']:
            self.make_normal()


# Extract model data
plate_length = model_data["beam"]["length"]
plate_width = model_data["beam"]["width"]
num_pts_x = model_data["beam"]["numPtsU"]
num_pts_y = model_data["beam"]["numPtsV"]
model_data["nrg"] = random.PRNGKey(0)

if model_data["data_type"] == "float64":
    jax.config.update("jax_enable_x64", True)
model_save_path = './model/model_params_' + model_data["filename"] + '.pkl'
losses_save_path = './model/losses_' + model_data["filename"] + '.pkl'

# Validate model parameters
assert model_data["fno"]["mode1"] <= model_data["beam"]["numPtsU"] // 2 + 1
assert model_data["fno"]["mode2"] <= model_data["beam"]["numPtsV"] // 2 + 1
assert model_data["beam"]["numPtsU"] % 2 == 0
assert model_data["beam"]["numPtsV"] % 2 == 0

#################################################################
# Generate dataset
#################################################################
t_start = time.time()
dataset = PlateHoleDEM_dataset(model_data)
train_dataset, test_dataset, rest_dataset = random_split(
    dataset, [model_data["n_train"], model_data["n_test"], model_data["n_dataset"] - model_data["n_data"]])
normalizers = [dataset.normalizer_x, dataset.normalizer_y] if model_data["normalized"] is True else None

# Convert dataset to JAX arrays
X_train, Y_train = zip(*train_dataset)
X_test, Y_test = zip(*test_dataset)
X_train, Y_train = jnp.array(X_train), jnp.array(Y_train)
X_test, Y_test = jnp.array(X_test), jnp.array(Y_test)

# Making dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=model_data["batch_size"], shuffle=True, collate_fn=collate_fn, drop_last=True)
test_loader = DataLoader(
    test_dataset, batch_size=model_data["batch_size"], shuffle=True, collate_fn=collate_fn)

t_data_gen = time.time()
print(f"Time taken for data generation: {t_data_gen - t_start}")
################################################################
# Training model
################################################################
# Initialize model
model = FNO2d_PlateHole(
    modes1=model_data["fno"]["mode1"],
    modes2=model_data["fno"]["mode2"],
    width=model_data["fno"]["width"],
    depth=model_data["fno"]["depth"],
    channels_last_proj=model_data["fno"]["channels_last_proj"],
    padding=model_data["fno"]["padding"],
    out_channels=Y_train.shape[-1],
    model_data=model_data["beam"]
)

# Define loss function
loss_fn = VinoPlateHoleLoss(model, model_data, normalizers, d=1, p=1, size_average=False)

# Load or initialize model parameters
loaded_model_params, train_losses, test_losses = load_training_state(model_save_path, losses_save_path)
if model_data["load_model"] and loaded_model_params is not None:
    model_params = loaded_model_params
    #best_loss = compute_validation_loss(model_params, train_loader, loss_fn)
    #print("Best loss test data from previous model: ", best_loss)
    best_loss = np.inf
else:
    _x, _ = train_dataset[0]
    _x = jnp.expand_dims(_x, axis=0)
    _, model_params = model.init_with_output(model_data["nrg"], _x)
    del _x
    best_loss, train_losses, test_losses = float('inf'), [], []

n_params = count_params(model_params)
print(f'\nOur model has {n_params} parameters.')

print("Training (ADAM)...")
t0 = time.time()

# Train the model
if model_data["training"]:
    if model_data["fno"]["scheduled"]:
        start_lr = 5e-3  # 1e-2
        end_lr = 1e-5
        steps_lr = 20  # 10
        schedule = optax.linear_schedule(init_value=start_lr, end_value=end_lr, transition_steps=steps_lr)
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(model_params)

        model_params, train_losses, test_losses = train_fno_scheduled(
            model, train_loader, test_loader, loss_fn, opt_state, optimizer, model_params)
    else:
        init_fun, update_fun, get_params = optimizers.adam(model_data["fno"]["learning_rate"])
        opt_state = init_fun(model_params)
        model_params, train_losses, test_losses = (
            train_fno(model, train_loader, test_loader, loss_fn, get_params, update_fun, opt_state,
                      model_params, best_loss, train_losses, test_losses))

    save_training_state(model_params, train_losses, test_losses, model_save_path, losses_save_path)

print(f"Training (ADAM) time: {time.time() - t0}")

# plot the convergence of the losses
folder = "VINO_Plate"
plot_loss_trend(train_losses, test_losses, folder)
################################################################
# evaluation
################################################################
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

print("Evaluating Train Data")
# model_evaluation(model, model_data, model_params, loss_fn, train_loader, normalizers, mask=True)
print("Evaluating Test Data")
# model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers, mask=True)

field_names = ['disp_x', 'disp_y']
print("___________________________________________________________________")
# Train Dataset Evaluation
# plot_PlateArbitVoid(train_dataset, 56, "Train", model, model_params, model_data, field_names, folder)
# Test Dataset Evaluation
# plot_PlateArbitVoid(test_dataset, 96, "Test", model, model_params, model_data, field_names, folder)
print("___________________________________________________________________")


################################################################
# combine with iterative solver
################################################################
def print_error(y_pred, y_data, hole_outside, msg):
    valid_y_data = y_data[hole_outside]
    valid_y_pred = y_pred[hole_outside]

    errs = np.linalg.norm(valid_y_data - valid_y_pred) / np.linalg.norm(valid_y_data)
    print(msg, errs)


def evaluate_all_errors(test_dataset, u_cg, u_nows_cg):
    """
    Evaluate and print average relative errors for all samples in test_dataset
    comparing solver outputs with and without VINO.

    Parameters
    ----------
    test_dataset : list
        Each entry is (x_data, y_data)
    u_cg : list
        List of displacement fields from CG solver without VINO
    u_nows_cg : list
        List of displacement fields from CG solver initialized with VINO
    """
    errs_no_vino = []
    errs_with_vino = []

    for idx in range(len(test_dataset)):
        x_data, y_data = test_dataset[idx]
        x_data, y_data = jnp.expand_dims(x_data, axis=0), jnp.expand_dims(y_data, axis=0)

        mask_hole = np.array(x_data[0, :, :, 0]).astype(bool)
        mask_nodes = ~mask_hole

        # Compute error without VINO
        valid_y_data = y_data[0][mask_nodes]
        valid_y_pred_no_vino = u_cg[idx][mask_nodes]
        err_no_vino = np.linalg.norm(valid_y_data - valid_y_pred_no_vino) / np.linalg.norm(valid_y_data)
        errs_no_vino.append(err_no_vino)

        # Compute error with VINO
        valid_y_pred_with_vino = u_nows_cg[idx][mask_nodes]
        err_with_vino = np.linalg.norm(valid_y_data - valid_y_pred_with_vino) / np.linalg.norm(valid_y_data)
        errs_with_vino.append(err_with_vino)

        # print(f"Index {idx}: Error without VINO = {err_no_vino:.6e}, with VINO = {err_with_vino:.6e}")

    # Average errors
    print("---------------------------------------------------------------------------------")
    print("Average error without VINO =", statistics.mean(errs_no_vino))
    print("Average error with VINO =", statistics.mean(errs_with_vino))
    print("---------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------")

    return errs_no_vino, errs_with_vino


x_min, x_max = 0, plate_length
y_min, y_max = 0, plate_width
mesh_size = plate_length / num_pts_x

index = 0
# x_data, y_data = dataset[index]
x_data, y_data = test_dataset[index]
x_data, y_data = jnp.expand_dims(x_data, axis=0), jnp.expand_dims(y_data, axis=0)
y_pred = model.apply(model_params, x_data)
mask_hole = np.array(x_data[0, :, :, 0]).astype(bool)
mask_nodes = ~mask_hole
rtol = 1e-6

print("----------------Network Prediction-------------------")
print_error(y_pred[0], y_data[0], mask_nodes, "Error between network prediction and database (IGA) = ")
y_pred = y_pred.at[0].set(y_pred[0].at[mask_hole].set(y_data[0][mask_hole]))

# print("----------------IGA Solver-------------------")
# y_IGA, _ = PlateWithHole_IGA_solver(mask_nodes, model_data['beam'])
# print_errors(y_IGA, y_data[0], mask_nodes, "Error between IGA solver and database = ")

print("----------------cg Solver-------------------")
# print("Without VINO")
# t1 = time.time()
# y_cg, _, solver_time = PlateWithHole_IGA_solver(mask_nodes, model_data['beam'], solver="cg", rtol=rtol)
# print_error(y_cg, y_data[0], mask_nodes, "Error between cg solver and database = ")
# print("solver time = ", solver_time)
# print("time = ", time.time() - t1)
#
# print("With VINO")
# t1 = time.time()
# y_cg_nows, _, solver_time = PlateWithHole_IGA_solver(mask_nodes, model_data['beam'], solver="cg", u0=y_pred[0], rtol=rtol)
# print_error(y_cg_nows, y_data[0], mask_nodes, "Error between cg solver and database = ")
# print("solver time = ", solver_time)
# print("time = ", time.time() - t1)

print("-----------------FIGURE-----------------")
# print("Without VINO")
# y_cg, _, solver_time, tr_hist = PlateWithHole_IGA_solver(mask_nodes, model_data['beam'], solver="cg", rtol=rtol, return_history=True)
# print_error(y_cg, y_data[0], mask_nodes, "Error between cg solver and database = ")
# print("length = ", len(tr_hist))
# print("solver time = ", solver_time)
#
# print("With VINO")
# y_cg_nows, _, solver_time, tr_hist = PlateWithHole_IGA_solver(mask_nodes, model_data['beam'], solver="cg", u0=y_pred[0], rtol=rtol, return_history=True)
# print_error(y_cg_nows, y_data[0], mask_nodes, "Error between cg solver and database = ")
# print("length = ", len(tr_hist))
# print("solver time = ", solver_time)


def run_dataset(dataset, model_data, model=None, model_params=None, use_vino=False, rtol=rtol):
    """
    Runs the iterative solver on all samples in the dataset and returns results.

    Parameters
    ----------
    dataset : list or array
        Each entry is (x_data, y_data).
    model_data : dict
        Contains 'beam' parameters for PlateWithHole_IGA_solver.
    model : optional
        Trained model for providing initial guess (VINO).
    model_params : optional
        Parameters for the model.
    use_vino : bool
        If True, use model prediction as initial guess u0.
    rtol : float
        Relative tolerance for CG.

    Returns
    -------
    results : list of tuples
        Each tuple = (y_cg, solver_time, tr_hist)
    """
    total_time_history = []
    time_residual_hist = []
    disp_hist = []
    solver_time_hist = []

    for idx in range(len(dataset)):
        print("index: ", idx)
        x_data, y_data = dataset[idx]
        x_data, y_data = jnp.expand_dims(x_data, axis=0), jnp.expand_dims(y_data, axis=0)
        mask_hole = np.array(x_data[0, :, :, 0]).astype(bool)
        mask_nodes = ~mask_hole

        # Initial guess from VINO if enabled
        u0 = None
        if use_vino and model is not None and model_params is not None:
            y_pred = model.apply(model_params, x_data)
            y_pred = y_pred.at[0].set(y_pred[0].at[mask_hole].set(y_data[0][mask_hole]))
            u0 = y_pred[0]

        # Call solver
        t0 = time.perf_counter()
        outputs = PlateWithHole_IGA_solver(
            mask_nodes,
            model_data['beam'],
            solver="cg",
            u0=u0,
            rtol=rtol,
            return_history=True
        )
        # Depending on return_space, structure changes.
        # Assume here: (ux, uy, sigma_xx, sigma_yy, sigma_xy, tr_hist)
        disp_grid, sol, solver_time, tr_hist = outputs
        print("solver time = ", solver_time)

        total_time_history.append(time.perf_counter() - t0)
        time_residual_hist.append(tr_hist)
        disp_hist.append(disp_grid)
        solver_time_hist.append(solver_time)

    return disp_hist, total_time_history, time_residual_hist, solver_time_hist


# Make results folder if it does not exist
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Length of dataset for filename
ds_len = model_data["ds_len"]
test_dataset = [test_dataset[i] for i in range(ds_len)]
ds_len = len(test_dataset)
print(ds_len)

# Filenames with folder prefix
fname_no_vino = os.path.join(results_dir, f"results_without_vino_{ds_len}_{rtol}.pkl")
fname_with_vino = os.path.join(results_dir, f"results_with_vino_{ds_len}_{rtol}.pkl")

# Without VINO
print("Without VINO")
if os.path.exists(fname_no_vino):
    with open(fname_no_vino, "rb") as f:
        u_cg, time_cg, tr_cg, solver_t_cg = pickle.load(f)
    print(f"Loaded cached results from {fname_no_vino}")
else:
    u_cg, time_cg, tr_cg, solver_t_cg = run_dataset(test_dataset, model_data, rtol=rtol)
    with open(fname_no_vino, "wb") as f:
        pickle.dump((u_cg, time_cg, tr_cg, solver_t_cg), f)
    print(f"Saved results to {fname_no_vino}")

# With VINO
print("With VINO")
if os.path.exists(fname_with_vino):
    with open(fname_with_vino, "rb") as f:
        u_nows_cg, time_nows_cg, tr_nows_cg, solver_t_nows_cg = pickle.load(f)
    print(f"Loaded cached results from {fname_with_vino}")
else:
    u_nows_cg, time_nows_cg, tr_nows_cg, solver_t_nows_cg = run_dataset(
        test_dataset, model_data, model=model, model_params=model_params, use_vino=True, rtol=rtol
    )
    with open(fname_with_vino, "wb") as f:
        pickle.dump((u_nows_cg, time_nows_cg, tr_nows_cg, solver_t_nows_cg), f)
    print(f"Saved results to {fname_with_vino}")


def time_comparison(time_cg, tr_cg, time_nows_cg, tr_nows_cg, solver_name):
    """
    Compare total and solver times between baseline CG and CG initialized with VINO.

    Parameters
    ----------
    time_cg : list[float]
        Total runtime per sample for CG solver (without VINO)
    tr_cg : list[list[(float, float)]]
        Residual history for each sample (list of (time, residual) pairs)
    time_nows_cg : list[float]
        Total runtime per sample for CG solver (with VINO)
    tr_nows_cg : list[list[(float, float)]]
        Residual history for each sample (list of (time, residual) pairs)
    solver_name : str
        Name of the solver (e.g., "CG")
    """
    total_time_cg = np.sum(time_cg)
    total_time_nows_cg = np.sum(time_nows_cg)

    solver_time_cg = np.sum([tr[-1][0] for tr in tr_cg if len(tr) > 0])
    solver_time_nows_cg = np.sum([tr[-1][0] for tr in tr_nows_cg if len(tr) > 0])

    total_saving_pct = ((total_time_cg - total_time_nows_cg) / total_time_cg * 100) if total_time_cg > 0 else 0
    solver_saving_pct = ((solver_time_cg - solver_time_nows_cg) / solver_time_cg * 100) if solver_time_cg > 0 else 0

    # Print results
    print(f"===== {solver_name} Time Comparison =====")
    print(f"Total time (no VINO): {total_time_cg:.4f} s")
    print(f"Solver time (no VINO): {solver_time_cg:.4f} s")
    print(f"Total time (with VINO): {total_time_nows_cg:.4f} s "
          f"(saving {total_saving_pct:.2f}%)")
    print(f"Solver time (with VINO): {solver_time_nows_cg:.4f} s "
          f"(saving {solver_saving_pct:.2f}%)")
    print("=========================================")


def plot_two_sets_trajectories(trajs1, trajs2,
                               color1="#1f77b4", color2="#ff7f0e",
                               label1="Without VINO", label2="With VINO",
                               fig_size=(6, 4),
                               save_dir="plots"):
    """
    Plot two sets of solver trajectories (e.g. residual histories)
    from run_dataset outputs for visual comparison.
    """

    os.makedirs(save_dir, exist_ok=True)
    fig, ax_iter = plt.subplots(figsize=fig_size)

    # --- Plot trajectories for the first set (e.g. CG baseline)
    for traj in trajs1:
        if traj is None or len(traj) == 0:
            continue
        traj = np.array(traj)  # convert list → array
        if traj.ndim != 2 or traj.shape[1] < 2:
            continue
        n = traj.shape[0]
        iterations = np.arange(n)
        ax_iter.semilogy(iterations, traj[:, 1], color=color1, linewidth=1, alpha=0.25)
        ax_iter.plot(iterations[0], traj[0, 1], marker="o", color=color1, markersize=2, alpha=0.5)
        ax_iter.plot(iterations[-1], traj[-1, 1], marker="o", color=color1, markersize=2, alpha=0.5)

    # --- Plot trajectories for the second set (e.g. CG + VINO)
    for traj in trajs2:
        if traj is None or len(traj) == 0:
            continue
        traj = np.array(traj)
        if traj.ndim != 2 or traj.shape[1] < 2:
            continue
        n = traj.shape[0]
        iterations = np.arange(n)
        ax_iter.semilogy(iterations, traj[:, 1], color=color2, linewidth=1, alpha=0.25)
        ax_iter.plot(iterations[0], traj[0, 1], marker="o", color=color2, markersize=2, alpha=0.5)
        ax_iter.plot(iterations[-1], traj[-1, 1], marker="o", color=color2, markersize=2, alpha=0.5)

    # --- Formatting
    ax_iter.set_xlabel("Iteration")
    ax_iter.set_ylabel("Residual (log scale)")
    ax_iter.plot([], [], color=color1, label=label1)
    ax_iter.plot([], [], color=color2, label=label2)
    ax_iter.legend(loc="upper right")
    ax_iter.set_xlim(left=0, right=30000)
    ax_iter.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8)
    ax_iter.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()

    # --- Save
    save_path = os.path.join(save_dir, f"trajectories_{label1.replace(' ', '_')}_{label2.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=600)
    plt.show()

    print(f"Saved trajectory comparison plot to: {save_path}")


def timing_violin_multi(trajs1, trajs2,
                        label1="CG (With VINO)", label2="CG (No VINO)",
                        tols=None, unit="s",
                        color1="#1f77b4", color2="#ff7f0e",
                        fig_size=(6, 4),
                        save_dir="plots"):
    """
    Compare solver timings vs residual tolerances using violin plots.

    Parameters
    ----------
    trajs1, trajs2 : list of np.ndarray
        Each element is an array of shape (n_points, 2) -> [time, residual].
    label1, label2 : str
        Labels for the two compared solvers.
    tols : list of float
        Residual tolerances for comparison (default: [1e0, ..., 1e-5]).
    unit : str
        Time unit ('s' or 'ms').
    color1, color2 : str
        Colors for violin plots.
    fig_size : tuple
        Figure size.
    save_dir : str
        Directory to save plot.
    """

    os.makedirs(save_dir, exist_ok=True)

    if tols is None:
        tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    # --- Helper: find time to reach given tolerance
    def find_time(traj, tol):
        if traj is None or len(traj) == 0:
            return np.nan
        traj = np.array(traj)  # <-- convert list of tuples to ndarray safely
        if traj.ndim != 2 or traj.shape[1] < 2:
            return np.nan
        mask = traj[:, 1] <= tol
        return traj[mask, 0][0] if np.any(mask) else np.nan

    # --- Collect timings for each tolerance
    results1, results2 = [], []
    for tol in tols:
        times1 = [find_time(tr, tol) for tr in trajs1 if tr is not None]
        times2 = [find_time(tr, tol) for tr in trajs2 if tr is not None]
        times1 = [t for t in times1 if not np.isnan(t)]
        times2 = [t for t in times2 if not np.isnan(t)]
        results1.append(np.array(times1))
        results2.append(np.array(times2))

    # --- Convert units
    if unit == "ms":
        results1 = [r * 1000 for r in results1]
        results2 = [r * 1000 for r in results2]

    # --- Compute percentage savings (medians)
    perc_savings = []
    for r1, r2 in zip(results1, results2):
        if len(r1) > 0 and len(r2) > 0:
            med1, med2 = np.median(r1), np.median(r2)
            perc = 100 * (med1 - med2) / med1 if med1 > 0 else np.nan
        else:
            perc = np.nan
        perc_savings.append(perc)

    # --- Plot setup
    fig, ax = plt.subplots(figsize=fig_size)
    positions = np.arange(len(tols)) * 2
    offset = 0.25

    # --- Violin plots (Set 1)
    for i, data in enumerate(results1):
        if len(data) > 0:
            vp = ax.violinplot(data, positions=[positions[i] - offset],
                               vert=False, widths=0.8, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color1)
                pc.set_alpha(0.6)

    # --- Violin plots (Set 2)
    for i, data in enumerate(results2):
        if len(data) > 0:
            vp = ax.violinplot(data, positions=[positions[i] + offset],
                               vert=False, widths=0.8, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color2)
                pc.set_alpha(0.6)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"$10^{{{int(np.log10(tol))}}}$" for tol in tols])
    ax.set_xlabel(f"Time to reach tolerance [{unit}]")
    ax.set_ylabel("Residual tolerance")
    ax.grid(True, which="both", ls=":")

    # Limit x-axis based on data spread
    all_data = np.concatenate(
        [np.concatenate(r) for r in [results1, results2] if len(r) > 0 and any(len(x) > 0 for x in r)]
    ) if any(len(r) > 0 for r in results1 + results2) else np.array([])
    if len(all_data) > 0:
        xmax = np.percentile(all_data, 99)
        ax.set_xlim(left=0, right=xmax)

    ax.invert_yaxis()

    # --- Right y-axis: percentage savings
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(positions)
    ax_right.set_yticklabels([f"{p:.1f}" if not np.isnan(p) else "NA" for p in perc_savings])
    ax_right.set_ylabel(f"Time saved by {label1} (%)")

    # --- Legend
    ax.plot([], [], color=color1, lw=8, alpha=0.8, label=label1)
    ax.plot([], [], color=color2, lw=8, alpha=0.8, label=label2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    # --- Save
    save_path = os.path.join(save_dir, f"violin_{label1.replace(' ', '_')}_{label2.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=600)
    plt.show()

    print(f"Saved violin timing comparison plot to: {save_path}")

    # --- Optional summary print
    print("\nMedian time savings by tolerance:")
    for tol, perc in zip(tols, perc_savings):
        print(f"  tol={tol:.1e}: {perc:.2f}%")

    return results1, results2, perc_savings


solver_name = 'CG'
time_comparison(time_cg, tr_cg, time_nows_cg, tr_nows_cg, solver_name)
errs_no_vino, errs_with_vino = evaluate_all_errors(test_dataset, u_cg, u_nows_cg)
plot_two_sets_trajectories(tr_cg, tr_nows_cg, label1="CG (No VINO)", label2="CG (With VINO)")
timing_violin_multi(tr_cg, tr_nows_cg, label1="CG + VINO", label2="CG Baseline")

print("Finish")
