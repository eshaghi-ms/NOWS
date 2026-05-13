#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting and post-processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


def plot_pred(x, u_exact, u_pred, title):
    fig_font = "DejaVu Serif"
    fig_size = (5, 3)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact, 'b', label='Ground Truth', linewidth=1.5)
    plt.plot(x, u_pred, '-.r', label='Prediction', linewidth=1.5)
    plt.legend()
    plt.title(title)
    title = title.replace('\\', '').replace('$', '')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/' + title + '.png', dpi=300)
    plt.show()
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)

    error_title = f"Relative $L^2$ error = {rel_l2_error:.4f}"
    print(error_title)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact - u_pred, 'b', label='Ground Truth', linewidth=1.5)
    # plt.legend()
    plt.title(error_title)
    # plt.savefig('error_' + title + '.png', dpi=300)
    plt.show()


def plot_pred1(x, y, f, title):
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, f, levels=2, cmap='Purples')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Input ' + title)
    plt.savefig('plots/' + 'Input_' + title + '.png', dpi=600)
    plt.show()


def plot_pred2(x, y, u_exact, u_pred, title, saved_title):
    saved_title = saved_title.replace('\\', '').replace('$', '')
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, u_pred, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots/")

    plt.savefig(folder + saved_title + '_ApproximateSolution.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Exact solution - ' + title)
    plt.savefig(folder + saved_title + '_ExactSolution.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact - u_pred, levels=500, cmap='bwr')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Error - ' + title)
    plt.savefig(folder + saved_title + '_Error.png', dpi=600)
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)
    plt.show()


def plot_field_2d(F, x_pts, y_pts, title, folder=None, file=None):
    """
    Plots a 2D field stored in a 1D tensor F

    Parameters
    ----------
    F : (1D tensor of size num_pts_v*num_pts_u)
        fields values at each point
    x_pts : (2D tensor of size num_pts_v x num_pts_u)
        x-coordinates of each field point to be plotted
    y_pts : (2d tensor of size num_pts_v x num_pts_u)
        y-coordinates of each field points to be plotted
    title : (string)
        title of the plot
    folder : (None or string)
        directory where to save the plot
    file : (None or string)
        file name for the plot

    Returns
    -------
    None.

    """
    plt.contourf(x_pts, y_pts, F, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title)
    plt.axis('equal')
    if folder is not None:
        full_name = folder + '/' + file
        plt.savefig(full_name)
    plt.show()


def plot_loss_trend(losses, labels, ylims=None):
    """
    Plot two loss trends on a shared x-axis but separate y-axes.

    Parameters:
        losses : list of arrays/lists [loss1, loss2]
        labels : list of str [label1, label2]
        ylims  : tuple ((ymin1, ymax1), (ymin2, ymax2)) or None
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots")

    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # First loss (left y-axis)
    ax1.plot(losses[0], label=labels[0], color="tab:blue")
    ax1.set_ylabel(labels[0], color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if ylims is not None and ylims[0] is not None:
        ax1.set_ylim(ylims[0])

    # Second loss (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(losses[1], label=labels[1], color="tab:red")
    ax2.set_ylabel(labels[1], color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    if ylims is not None and ylims[1] is not None:
        ax2.set_ylim(ylims[1])

    # Legend combining both axes
    lines, labels_ = ax1.get_legend_handles_labels()
    lines2, labels2_ = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_ + labels2_, loc="best")

    os.makedirs(folder, exist_ok=True)
    plot_name = os.path.join(folder, "LossTrend")
    plt.savefig(plot_name + '.png', dpi=600, bbox_inches='tight')
    plt.show()


def plot_two_sets_trajectories(trajs1, trajs2,
                               color1="#1f77b4", color2="#ff7f0e",
                               label1="Set 1", label2="Set 2", fig_size=(6, 4)):
    """
    Plot two sets of trajectories in one figure with dual x-axes:
    bottom = iteration index, top = time.

    Parameters
    ----------
    trajs1, trajs2 : list
        Each element is an array of shape (n_points, 2),
        where column 0 = time, column 1 = state variable.
    color1, color2 : str
        Colors for the two sets.
    label1, label2 : str
        Labels for the two sets.
    """
    fig, ax_iter = plt.subplots(figsize=fig_size)

    # Plot first set
    for traj in trajs1:
        n = traj.shape[0]
        iterations = np.arange(n)
        ax_iter.semilogy(iterations, traj[:, 1], color=color1, linewidth=1, alpha=0.2)
        ax_iter.plot(iterations[0], traj[0, 1], marker="o", color=color1, markersize=2, alpha=0.5)
        ax_iter.plot(iterations[-1], traj[-1, 1], marker="o", color=color1, markersize=2, alpha=0.5)

    # Plot second set
    for traj in trajs2:
        n = traj.shape[0]
        iterations = np.arange(n)
        ax_iter.semilogy(iterations, traj[:, 1], color=color2, linewidth=1, alpha=0.2)
        ax_iter.plot(iterations[0], traj[0, 1], marker="o", color=color2, markersize=2, alpha=0.5)
        ax_iter.plot(iterations[-1], traj[-1, 1], marker="o", color=color2, markersize=2, alpha=0.5)

    # ax_iter.set_xlabel("Iteration")
    # ax_iter.set_ylabel("Relative tolerance")

    # Create legend with only one entry per set
    ax_iter.plot([], [], color=color1, label=label1)
    ax_iter.plot([], [], color=color2, label=label2)
    # ax_iter.legend(loc="upper right")

    # plt.title(f"Trajectories Comparison")
    ax_iter.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8)
    ax_iter.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.6)
    plt.tight_layout()

    ax_iter.set_xlim(left=0)
    # ax_iter.set_ylim(top=5, bottom=5e-6)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots")

    plt.savefig(os.path.join(folder, f"trajectories_{label1}_{label2}.png"), dpi=1200)
    plt.show()


def plot_residual_vs_iteration(tr_cg, tr_nows_cg, solver_time_cg, solver_time_nows_cg,
                               label_cg='CG', label_nows='NOWS-CG', figsize=(6, 4)):
    """
    Plot residual vs iteration curves with mean ± std and dual x-axes:
    bottom = iteration, top = cumulative solver time.

    Parameters
    ----------
    tr_cg : list of np.ndarray
        Trajectories from CG solver, each array shape (n_points, 2)
    tr_nows_cg : list of np.ndarray
        Trajectories from NOWS-CG solver, each array shape (n_points, 2)
    solver_time_cg, solver_time_nows_cg : float
        Total solver times for each set
    label_cg, label_nows : str
        Labels for the legend
    figsize : tuple
        Figure size
    """

    # ------------------------------------------------------------
    # 1) Build cumulative residual-vs-time curves
    # ------------------------------------------------------------
    def build_concatenated_curves(tr_list):
        times = []
        resids = []
        t_offset = 0.0
        for tr in tr_list:
            if len(tr) == 0:
                continue
            t = tr[:, 0] + t_offset
            r = tr[:, 1]
            times.append(t)
            resids.append(r)
            t_offset += tr[-1, 0]
        if not times:
            return np.array([]), np.array([])
        return np.concatenate(times), np.concatenate(resids)

    global_time_cg, global_resid_cg = build_concatenated_curves(tr_cg)
    global_time_nows, global_resid_nows = build_concatenated_curves(tr_nows_cg)

    t_max = max(solver_time_cg, solver_time_nows_cg)
    t_grid = np.linspace(0, t_max, 500)
    resid_cg_interp = np.interp(t_grid, global_time_cg, global_resid_cg)
    resid_nows_interp = np.interp(t_grid, global_time_nows, global_resid_nows)

    # ------------------------------------------------------------
    # 2) Build mean ± std curves for iteration axis
    # ------------------------------------------------------------
    def build_iter_stats(list_of_arrays):
        max_len = max(len(arr) for arr in list_of_arrays)
        padded = np.full((len(list_of_arrays), max_len), np.nan)
        for i, arr in enumerate(list_of_arrays):
            padded[i, :len(arr)] = arr
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        return mean, std

    iter_mean_cg, iter_std_cg = build_iter_stats([tr[:, 1] for tr in tr_cg])
    iter_mean_nows, iter_std_nows = build_iter_stats([tr[:, 1] for tr in tr_nows_cg])

    # ------------------------------------------------------------
    # 3) Plot figure with dual x-axes
    # ------------------------------------------------------------
    fig, ax_iter = plt.subplots(figsize=figsize)

    # Bottom x-axis: iteration
    ax_iter.semilogy(iter_mean_cg, color='C0', label=label_cg)
    ax_iter.fill_between(range(len(iter_mean_cg)),
                         iter_mean_cg - iter_std_cg,
                         iter_mean_cg + iter_std_cg,
                         color='C0', alpha=.25)

    ax_iter.semilogy(iter_mean_nows, color='C1', label=label_nows)
    ax_iter.fill_between(range(len(iter_mean_nows)),
                         iter_mean_nows - iter_std_nows,
                         iter_mean_nows + iter_std_nows,
                         color='C1', alpha=.25)

    ax_iter.set_xlabel('Iteration')
    ax_iter.set_ylabel('‖r‖₂')
    ax_iter.grid(True, which="major", axis="y", linestyle="-", linewidth=2.2)
    ax_iter.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.6)
    ax_iter.legend()

    # Top x-axis: cumulative solver time
    ax_time = ax_iter.twiny()
    # invisible dummy plots to link to top axis
    ax_time.semilogy(t_grid, resid_cg_interp, color='C0', lw=0)
    ax_time.semilogy(t_grid, resid_nows_interp, color='C1', lw=0)
    ax_time.set_xlim(0, t_max)
    ax_time.set_xlabel('Cumulative solver time (s)')

    plt.tight_layout()
    plt.show()


def time_comparison(time_cg, tr_cg, time_nows_cg, tr_nows_cg, solver_name):
    solver_time_cg = sum(tr[-1, 0] for tr in tr_cg if len(tr) > 0)
    solver_time_nows_cg = sum(tr[-1, 0] for tr in tr_nows_cg if len(tr) > 0)

    total_saving_pct = ((time_cg - time_nows_cg) / time_cg * 100) if time_cg > 0 else 0
    solver_saving_pct = ((solver_time_cg - solver_time_nows_cg) / solver_time_cg * 100) if solver_time_cg > 0 else 0

    print(f"Total time for {solver_name} = {time_cg}")
    print(f"Solver time for {solver_name} = {solver_time_cg}")
    print(f"Total time for {solver_name} initialized with NOWS = {time_nows_cg} "
          f"(saving {total_saving_pct:.2f}%)")
    print(f"Solver time for {solver_name} initialized with NOWS = {solver_time_nows_cg} "
          f"(saving {solver_saving_pct:.2f}%)")



def timing_boxplot(trajs1, trajs2, label1="CG-NOWS", label2="CG", tol=1e-3, unit="ms"):
    """
    Compare solver timings to reach a target residual tolerance using a boxplot.

    Parameters
    ----------
    trajs1, trajs2 : list of arrays
        Each element is shape (n_points, 2): [time, residual].
    label1, label2 : str
        Labels for the two methods.
    tol : float
        Residual tolerance (stop criterion).
    unit : str
        "s" for seconds, "ms" for milliseconds.
    """
    timings1, timings2 = [], []

    # Helper to find first crossing time
    def find_time(traj, tol):
        mask = traj[:, 1] <= tol
        if np.any(mask):
            return traj[mask, 0][0]
        else:
            return np.nan  # did not converge within trajectory

    for tr in trajs1:
        t = find_time(tr, tol)
        if not np.isnan(t):
            timings1.append(t)
    for tr in trajs2:
        t = find_time(tr, tol)
        if not np.isnan(t):
            timings2.append(t)

    timings1 = np.array(timings1)
    timings2 = np.array(timings2)

    if unit == "ms":
        timings1 *= 1000
        timings2 *= 1000

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([timings1, timings2], labels=[label1, label2])
    ax.set_ylabel(f"Time to reach residual ≤ {tol} [{unit}]")
    ax.set_title("Timing Comparison")
    ax.grid(True, axis="y", ls=":")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots/")
    os.makedirs(folder, exist_ok=True)

    plt.savefig(os.path.join(folder, f"boxplot_tol_{tol:.0e}.png"), dpi=1200)
    plt.show()

    return timings1, timings2


def timing_boxplot_multi(trajs1, trajs2, label1="CG-NOWS", label2="CG",
                         tols=None, unit="ms",
                         color1="#1f77b4", color2="#ff7f0e"):
    """
    Compare solver timings with residual tolerance on y-axis
    and horizontal boxplots for time distributions. Also shows
    percentage of time saved by trajs1 vs trajs2 on right y-axis.

    Parameters
    ----------
    trajs1, trajs2 : list of arrays
        Each element is shape (n_points, 2): [time, residual].
    label1, label2 : str
        Labels for the two methods.
    tols : list of float
        Residual tolerances to test.
    unit : str
        "s" for seconds, "ms" for milliseconds.
    color1, color2 : str
        Colors for the two methods.
    """
    if tols is None:
        tols = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    results1, results2 = [], []

    def find_time(traj, tol):
        mask = traj[:, 1] <= tol
        if np.any(mask):
            return traj[mask, 0][0]
        else:
            return np.nan

    # Collect timings for each tolerance
    for tol in tols:
        times1, times2 = [], []
        for tr in trajs1:
            t = find_time(tr, tol)
            if not np.isnan(t):
                times1.append(t)
        for tr in trajs2:
            t = find_time(tr, tol)
            if not np.isnan(t):
                times2.append(t)
        results1.append(np.array(times1))
        results2.append(np.array(times2))

    # Unit conversion
    if unit == "ms":
        results1 = [r * 1000 for r in results1]
        results2 = [r * 1000 for r in results2]

    # Compute percentage savings (using medians)
    perc_savings = []
    for r1, r2 in zip(results1, results2):
        if len(r1) > 0 and len(r2) > 0:
            med1, med2 = np.median(r1), np.median(r2)
            perc = 100 * (med2 - med1) / med2 if med2 > 0 else np.nan
        else:
            perc = np.nan
        perc_savings.append(perc)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(tols)) * 2  # spacing on y-axis
    box_height = 0.5

    b1 = ax.boxplot(results1, positions=positions - 0.25,
                    widths=box_height, vert=False, patch_artist=True,
                    boxprops=dict(facecolor=color1, alpha=0.6),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color=color1),
                    capprops=dict(color=color1),
                    flierprops=dict(markerfacecolor=color1, markersize=3, alpha=0.4))

    b2 = ax.boxplot(results2, positions=positions + 0.25,
                    widths=box_height, vert=False, patch_artist=True,
                    boxprops=dict(facecolor=color2, alpha=0.6),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color=color2),
                    capprops=dict(color=color2),
                    flierprops=dict(markerfacecolor=color2, markersize=3, alpha=0.4))

    ax.set_yticks(positions)
    ax.set_yticklabels([f"$10^{{{int(np.log10(tol))}}}$" for tol in tols])
    ax.set_ylabel("Residual tolerance")
    ax.set_xlabel(f"Time to reach tolerance [{unit}]")
    # ax.set_title("Timing Comparison vs Accuracy")
    ax.grid(True, which="both", ls=':')

    ax.invert_yaxis()

    # Right y-axis for % savings
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(positions)
    ax_right.set_yticklabels([f"{p:.1f}%" if not np.isnan(p) else "NA" for p in perc_savings])
    ax_right.set_ylabel(f"Time saved by {label1}")

    # Legend
    ax.legend([b1["boxes"][0], b2["boxes"][0]], [label1, label2], loc="upper right")

    plt.tight_layout()

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots/")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"boxplot_multi_horizontal.png"), dpi=1200)
    plt.show()

    return results1, results2, perc_savings


def timing_violin_multi(trajs1, trajs2, label1="CG-NOWS", label2="CG",
                        tols=None, unit="ms",
                        color1="#1f77b4", color2="#ff7f0e", fig_size=(6, 4)):
    """
    Compare solver timings with residual tolerance on y-axis
    using horizontal violin plots for time distributions. Also shows
    percentage of time saved by trajs1 vs trajs2 on right y-axis.
    """
    if tols is None:
        tols = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    results1, results2 = [], []

    def find_time(traj, tol):
        mask = traj[:, 1] <= tol
        if np.any(mask):
            return traj[mask, 0][0]
        else:
            return np.nan

    # Collect timings for each tolerance
    for tol in tols:
        times1, times2 = [], []
        for tr in trajs1:
            t = find_time(tr, tol)
            if not np.isnan(t):
                times1.append(t)
        for tr in trajs2:
            t = find_time(tr, tol)
            if not np.isnan(t):
                times2.append(t)
        results1.append(np.array(times1))
        results2.append(np.array(times2))

    # Unit conversion
    if unit == "ms":
        results1 = [r * 1000 for r in results1]
        results2 = [r * 1000 for r in results2]

    # Compute percentage savings (using medians)
    perc_savings = []
    for r1, r2 in zip(results1, results2):
        if len(r1) > 0 and len(r2) > 0:
            med1, med2 = np.median(r1), np.median(r2)
            perc = 100 * (med2 - med1) / med2 if med2 > 0 else np.nan
        else:
            perc = np.nan
        perc_savings.append(perc)

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)
    positions = np.arange(len(tols)) * 2  # spacing on y-axis
    offset = 0.25

    # Plot violins for trajs1
    for i, data in enumerate(results1):
        if len(data) > 0:
            vp = ax.violinplot(data, positions=[positions[i] - offset],
                               vert=False, widths=0.8, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color1)
                pc.set_alpha(0.6)

    # Plot violins for trajs2
    for i, data in enumerate(results2):
        if len(data) > 0:
            vp = ax.violinplot(data, positions=[positions[i] + offset],
                               vert=False, widths=0.8, showextrema=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color2)
                pc.set_alpha(0.6)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"$10^{{{int(np.log10(tol))}}}$" for tol in tols])
    # ax.set_ylabel("Relative tolerance")
    # ax.set_xlabel(f"Time to reach tolerance [{unit}]")
    # ax.set_title("Timing Comparison vs Accuracy (Violin Plot)")
    ax.grid(True, which="both", ls=':')

    all_data = np.concatenate(
        [np.concatenate(r) for r in [results1, results2] if len(r) > 0 and any(len(x) > 0 for x in r)])
    if len(all_data) > 0:
        xmax = np.percentile(all_data, 99.5)
        ax.set_xlim(left=0, right=xmax)

    ax.invert_yaxis()

    # Right y-axis for % savings
    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(positions)
    ax_right.set_yticklabels([f"{p:.1f}" if not np.isnan(p) else "NA" for p in perc_savings])
    # ax_right.set_ylabel(f"Time saved by NOWS (%)")

    # Legend (fake handles since violinplot doesn’t return them)
    ax.plot([], [], color=color1, lw=8, alpha=0.8, label=label1)
    ax.plot([], [], color=color2, lw=8, alpha=0.8, label=label2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    # Save
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots/")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"violin_{label1}_{label2}.png"), dpi=1200)
    plt.show()

    return results1, results2, perc_savings


def timing_violin_several(trajs_list, tols=None, unit="ms", fig_size=(8, 4.95)):
    """
    Compare solver timings with residual tolerance on y-axis
    using horizontal violin plots for time distributions.
    Now distinguishes pairs using hatching instead of light colors.

    Parameters
    ----------
    trajs_list : list of tuples
        [(label, trajs, color, hatch), ...] where
          - label (str): solver name
          - trajs (list of np.arrays): each array shape (steps, 2) [time, residual]
          - color (str): matplotlib color
          - hatch (str): matplotlib hatch pattern ("" for solid)
    tols : list of float, optional
        Residual tolerances to measure times. Default = [1e-1, 1e-2, 1e-3, 1e-4]
    unit : str, optional
        "ms" or "s". Default "ms"
    fig_size : tuple
        Figure size.
    """
    from matplotlib.patches import Patch

    if tols is None:
        tols = [1e-1, 1e-2, 1e-3, 1e-4]

    def find_time(traj, tol):
        mask = traj[:, 1] <= tol
        return traj[mask, 0][0] if np.any(mask) else np.nan

    # Collect timings
    results = {label: [] for label, _, _, _ in trajs_list}
    for tol in tols:
        for label, trajs, _, _ in trajs_list:
            times = []
            for tr in trajs:
                t = find_time(tr, tol)
                if not np.isnan(t):
                    times.append(t)
            results[label].append(np.array(times))

    # Unit conversion
    if unit == "ms":
        for label in results:
            results[label] = [r * 1000 for r in results[label]]

    # Plot
    fig, ax = plt.subplots(figsize=fig_size)
    positions = np.arange(len(tols)) * 4  # spacing
    offsets = np.linspace(-1.6, 1.6, len(trajs_list)//2)  # spread violins
    offsets = [x for x in offsets for _ in range(2)]

    for offset, (label, _, color, hatch) in zip(offsets, trajs_list):
        for i, data in enumerate(results[label]):
            if len(data) > 0:
                vp = ax.violinplot(
                    data,
                    positions=[positions[i] + offset],
                    vert=False,
                    widths=0.7,
                    showextrema=False
                )
                for pc in vp['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    if hatch:
                        pc.set_hatch(hatch)

    # Draw dashed lines between residual tolerances
    for pos in positions[:-1]:
        ax.axhline(y=pos + 2, color='black', linestyle='--', linewidth=0.8)

    # y-axis
    ax.set_yticks(positions)
    ax.set_yticklabels([f"$10^{{{int(np.log10(tol))}}}$" for tol in tols])
    ax.grid(True, which="both", ls=':')

    # x-axis scaling
    all_data = np.concatenate(
        [np.concatenate(r) for r in results.values()
         if len(r) > 0 and any(len(x) > 0 for x in r)]
    )
    if len(all_data) > 0:
        xmax = np.percentile(all_data, 99.9)
        ax.set_xlim(left=0, right=xmax)

    ax.invert_yaxis()
    ax.set_xlabel(f"Time to reach tolerance [{unit}]")

    # Legend with hatch
    from matplotlib.patches import Patch
    # Separate handles for NOWS and non-NOWS
    handles_nows = []
    handles_non_nows = []
    for label, _, color, hatch in trajs_list:
        patch = Patch(facecolor=color, hatch=hatch, label=label)
        if hatch:  # NOWS has hatch
            handles_nows.append(patch)
        else:
            handles_non_nows.append(patch)

    # Combine handles for two columns
    # ax.legend(
    #     handles=handles_non_nows + handles_nows,
    #     loc="upper right",
    #     ncol=2,
    #     columnspacing=1.2,
    #     handletextpad=0.5
    # )

    plt.tight_layout()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots/")
    os.makedirs(folder, exist_ok=True)
    # plt.savefig(os.path.join(folder, f"violin_several.png"), dpi=1200)
    plt.show()

    return results


def plot_multiple_trajectories(traj_sets, fig_size=(6, 4)):
    """
    Plot multiple sets of trajectories in one figure with dual x-axes:
    bottom = iteration index, top = time.

    Parameters
    ----------
    traj_sets : list of tuples
        Each tuple: (label, traj_list, color, is_nows)
        - label : str, label for the legend
        - traj_list : list of np.arrays of shape (n_points, 2)
        - color : str, matplotlib color
        - is_nows : bool, True for NOWS (solid line), False otherwise (dashed line)
    fig_size : tuple
        Figure size
    """
    fig, ax_iter = plt.subplots(figsize=fig_size)

    # alphas = np.linspace(0.7, 0.3, len(traj_sets))  # spread violins
    max_trajs = max(len(trajs) for _, trajs, _, _ in traj_sets)

    for traj_idx in range(max_trajs):
        for label, trajs, color, is_nows in traj_sets:
            if traj_idx >= len(trajs):
                continue  # skip if this set has fewer trajectories

            linestyle = '-' if is_nows else (0, (50, 50))
            end_marker = 'x' if is_nows else 'o'
            alphas = 0.7 if is_nows else 1.0

            traj = trajs[traj_idx]
            n = traj.shape[0]
            iterations = np.arange(n)

            ax_iter.semilogy(
                iterations, traj[:, 1],
                color=color, linestyle=linestyle, linewidth=0.3, alpha=alphas
            )
            ax_iter.plot(
                iterations[-1], traj[-1, 1],
                marker=end_marker, color=color,
                markersize=3, markeredgewidth=0.2
            )

    for idx, (label, trajs, color, is_nows) in enumerate(traj_sets):
        linestyle = '-' if is_nows else '--'
        end_marker = 'x' if is_nows else 'o'
        alphas = 0.9 if is_nows else 0.9
        # Add fake handle for legend
        ax_iter.plot([], [], color=color, linestyle=linestyle, linewidth=1, label=label)

    # ax_iter.set_xlabel("Iteration")
    # ax_iter.set_ylabel("Residual / Error")
    ax_iter.grid(True, which="major", axis="y", linestyle="-", linewidth=0.7)
    ax_iter.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.3)
    # ax_iter.set_xlim(left=0)
    ax_iter.set_ylim(1e-5, 15)
    ax_iter.set_xlim(left=0, right=1100)
    # ax_iter.legend(loc="upper right")
    ax_iter.set_xticks(np.linspace(0, 1100, 12))
    ax_iter.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    plt.tight_layout()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"trajectories_multiple.png"), dpi=1200)
    plt.show()


def plot_multiple_trajectories_broken(traj_sets, fig_size=(8, 4)):
    """
    Plot multiple sets of trajectories with a broken x-axis.
    Shows 0-50 and 100-200 iterations, skipping 51-99.
    """
    import matplotlib.ticker as mtick
    from matplotlib.ticker import MultipleLocator

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=fig_size)

    xl0, xl1, xl2, xl3 = 0, 149, 225, 550

    # Define axis limits
    ax1.set_xlim(xl0, xl1)
    ax2.set_xlim(xl2, xl3)
    ax1.set_ylim(1e-6, 10)
    ax2.set_ylim(1e-6, 10)

    ax1.set_xticks(range(xl0, xl1 + 1, 25))
    ax2.set_xticks(range(xl2, xl3 + 1, 50))

    alphas = np.linspace(0.01, 0.9, len(traj_sets))  # spread violins

    # Plot all trajectories on both axes
    for idx, (label, trajs, color, is_nows) in enumerate(traj_sets):
        linestyle = '-' if is_nows else '-.'
        end_marker = 'x' if is_nows else 'o'
        for traj in trajs:
            n = traj.shape[0]
            iterations = np.arange(n)
            # Plot only the parts that fall within ax1 and ax2 ranges
            mask1 = (iterations >= xl0) & (iterations <= xl1)
            mask2 = (iterations >= xl2) & (iterations <= xl3)

            if mask1.any():
                ax1.semilogy(iterations[mask1], traj[mask1, 1],
                             color=color, linestyle=linestyle, linewidth=1, alpha=alphas[idx])
                # ax1.plot(iterations[mask1][0], traj[mask1, 1][0], marker="o", color=color, markersize=5)
                # ax1.plot(iterations[mask1][-1], traj[mask1, 1][-1], marker=end_marker, color=color, markersize=5)

            if mask2.any():
                ax2.semilogy(iterations[mask2], traj[mask2, 1],
                             color=color, linestyle=linestyle, linewidth=1, alpha=0.3)
                # ax2.plot(iterations[mask2][0], traj[mask2, 1][0], marker="o", color=color, markersize=50)
                # ax2.plot(iterations[mask2][-1], traj[mask2, 1][-1], marker=end_marker, color=color, markersize=2)

        # Add fake handle for legend on ax2
        ax2.plot([], [], color=color, linestyle=linestyle, linewidth=1, label=label)

    # Remove spines between axes for broken x-axis look
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # Add diagonal lines to indicate break
    d = 0.0075  # size of diagonal lines
    slope_factor = 2.0  # >1 makes angle steeper
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d * slope_factor, +d * slope_factor), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d * slope_factor, 1 + d * slope_factor), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d * slope_factor, +d * slope_factor), **kwargs)
    ax2.plot((-d, +d), (1 - d * slope_factor, 1 + d * slope_factor), **kwargs)

    # ax2.set_xlabel("Iteration")
    # ax1.set_ylabel("Residual / Error")
    ax1.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8)
    ax2.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8)
    # ax1.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.6)

    # ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025)

    # Save figure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, "plots")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f"trajectories_multiple_broken.png"), dpi=1200)
    plt.show()

