import os
import argparse
import time
import random
import scipy.io
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from phi.jax.flow import *
import jax


def parse_args():
    parser = argparse.ArgumentParser(description="Generate smoke simulation datasets with configurable parameters.")
    parser.add_argument('--seed', type=int, default=26, help='Random seed')
    parser.add_argument('--length', type=float, default=100.0, help='Domain length')
    parser.add_argument('--width', type=float, default=100.0, help='Domain width')
    parser.add_argument('--inflow_rate', type=float, default=0.2, help='Smoke inflow rate')
    # parser.add_argument('--T', type=int, default=300, help='Number of timesteps')
    parser.add_argument('--T', type=int, default=150, help='Number of timesteps')
    parser.add_argument('--Dt', type=float, default=0.5, help='Time step size')
    parser.add_argument('--res', type=int, default=256, help='Grid resolution')
    parser.add_argument('--margin', type=float, default=0.48, help='Influx margin fraction')
    parser.add_argument('--N_data', type=int, default=15, help='Number of simulation samples')
    parser.add_argument('--if_plot', action='store_true', help='Plot intermediate frames')
    parser.add_argument('--substeps', type=int, default=1, help='Number of substeps per timestep')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    jax.config.update("jax_debug_nans", True)

    # Unpack parameters
    length, width = args.length, args.width
    inflow_rate = args.inflow_rate
    T, Dt = args.T, args.Dt
    res = args.res
    margin = args.margin
    N_data = args.N_data
    if_plot = args.if_plot
    substeps = args.substeps

    # Preallocate arrays
    shape = (N_data, T + 1, res, res)
    velocity_x_c = np.zeros(shape)
    velocity_y_c = np.zeros(shape)
    pressure_c = np.zeros(shape)
    smoke_c = np.zeros(shape)

    # Simulation loop
    for i in range(N_data):
        print(f"Simulating sample {i + 1}/{N_data}")
        inflow_x = length * (random.random() * (1 - 2 * margin) + margin)
        inflow_y = width * (random.random() * (0.5 - margin) + 0.1)
        inflow_radius = 5

        domain = Box(x=length, y=width)
        inflow = Sphere(x=inflow_x, y=inflow_y, radius=inflow_radius)

        # @jit_compile
        def step(v, s, p, dt):
            s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
            buoyancy = resample(s * (0, 0.1), to=v)
            v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt

            # Try CG; on failure double max_iterations and retry
            tol = 1e-3
            max_iters = 5000
            hard_cap = 50000
            iter_plus = 2000

            while True:
                solve = Solve('CG',
                              rel_tol=tol,
                              x0=p,
                              max_iterations=max_iters)
                try:
                    v, p = fluid.make_incompressible(v, (), solve)
                    break  # success!
                except (NotConverged, Diverged):
                    if max_iters >= hard_cap:
                        warnings.warn(f"Solver still not converging at {max_iters} iters → giving up")
                        break
                    print(f"CG failed at {max_iters} iters; retrying with {max_iters + iter_plus}")
                    max_iters += iter_plus
            return v, s, p

        v0 = StaggeredGrid(0, 0, domain, x=res, y=res)
        smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)
        target_res = CenteredGrid(0, ZERO_GRADIENT, domain, x=res, y=res)

        v_trj, s_trj, p_trj = iterate(
            step, batch(time=T), v0, smoke0, None,
            dt=Dt, range=trange, substeps=substeps
        )

        # Extract fields
        v_center = v_trj.at_centers()
        v_x = v_center.vector['x'].numpy()
        v_y = v_center.vector['y'].numpy()
        s_num = resample(s_trj, to=target_res).numpy()
        p_num = p_trj.numpy()

        # Store
        velocity_x_c[i] = v_x
        velocity_y_c[i] = v_y
        smoke_c[i] = s_num
        pressure_c[i] = np.concatenate((np.zeros((1, res, res)), p_num), axis=0)

        # Optional plotting
        if if_plot:
            for j in range(0, T, T // 5):
                frame = np.flipud(s_num[j].T)
                plt.figure()
                plt.imshow(frame, cmap='Greys')
                plt.colorbar()
                plt.close()

    # Save results
    save_name = f"smoke_N{N_data}_res{res}_T{T}_Dt{Dt}_substeps{substeps}.npz"
    save_path = os.path.join(os.path.dirname(__file__), save_name)
    np.savez(save_path,
             velocity_x_c=velocity_x_c,
             velocity_y_c=velocity_y_c,
             pressure_c=pressure_c,
             smoke_c=smoke_c)
    print(f"Data saved to {save_path}")


if __name__ == "__main__":
    main()
