import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract smoke trajectories from .npz file and convert to .mp4")
    parser.add_argument('--input', type=str, default='smoke_N20_res64_T150_Dt0.5_substeps1.npz',
                        help='Input .npz file path')
    parser.add_argument('--output_dir', type=str, default='smoke_N20_res64_T150_Dt0.5_substeps1',
                        help='Output directory for .mp4 files')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for video')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for video quality')
    parser.add_argument('--cmap', type=str, default='Greys', help='Colormap for smoke visualization')
    parser.add_argument('--show_velocity', action='store_true', help='Include velocity field overlay')
    return parser.parse_args()


def load_data(filepath):
    """Load data from .npz file"""
    print(f"Loading data from {filepath}")
    data = np.load(filepath)

    # Extract arrays
    velocity_x_c = data['velocity_x_c']
    velocity_y_c = data['velocity_y_c']
    pressure_c = data['pressure_c']
    # pressure = data['pressure']
    smoke_c = data['smoke_c']

    print(f"Data shapes:")
    print(f"  Smoke density: {smoke_c.shape}")
    print(f"  Velocity X: {velocity_x_c.shape}")
    print(f"  Velocity Y: {velocity_y_c.shape}")
    print(f"  Pressure (centered): {pressure_c.shape}")
    # print(f"  Pressure: {pressure.shape}")

    return smoke_c, velocity_x_c, velocity_y_c, pressure_c


def create_smoke_animation(smoke_data, velocity_x=None, velocity_y=None,
                           output_path='smoke_animation.mp4', fps=30, dpi=100,
                           cmap='Greys', show_velocity=False):
    """Create animation from smoke trajectory data"""
    n_frames = smoke_data.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize plot
    im = ax.imshow(np.flipud(smoke_data[0].T), cmap=cmap, vmin=0, vmax=smoke_data.max())
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title = ax.set_title('Frame 0')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Smoke Density')

    # Prepare velocity field if requested
    if show_velocity and velocity_x is not None and velocity_y is not None:
        # Create grid for quiver plot
        y, x = np.mgrid[0:smoke_data.shape[1], 0:smoke_data.shape[2]]
        # Subsample for clarity
        skip = 4
        quiver = ax.quiver(x[::skip, ::skip], y[::skip, ::skip],
                           velocity_x[0, ::skip, ::skip].T,
                           velocity_y[0, ::skip, ::skip].T,
                           color='red', alpha=0.5, scale=50)

    def animate(frame):
        # Update smoke density
        im.set_array(np.flipud(smoke_data[frame].T))
        title.set_text(f'Frame {frame}/{n_frames - 1}')

        # Update velocity field if shown
        if show_velocity and velocity_x is not None and velocity_y is not None:
            quiver.set_UVC(velocity_x[frame, ::skip, ::skip].T,
                           velocity_y[frame, ::skip, ::skip].T)

        return [im, title] + ([quiver] if show_velocity else [])

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000 / fps, blit=True)

    # Save as MP4
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Smoke Simulation'), bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Animation saved to {output_path}")


def create_pressure_animation(pressure_data, output_path='pressure_animation.mp4',
                              fps=30, dpi=100, cmap='coolwarm'):
    """Create animation from pressure field data"""
    n_frames = pressure_data.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize plot
    im = ax.imshow(np.flipud(pressure_data[0].T), cmap=cmap,
                   vmin=pressure_data.min(), vmax=pressure_data.max())
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title = ax.set_title('Frame 0')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pressure')

    def animate(frame):
        # Update pressure field
        im.set_array(np.flipud(pressure_data[frame].T))
        title.set_text(f'Frame {frame}/{n_frames - 1}')
        return [im, title]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000 / fps, blit=True)

    # Save as MP4
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Pressure Field'), bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Pressure animation saved to {output_path}")


def create_velocity_component_animation(velocity_data, component_name='velocity',
                                        output_path='velocity_animation.mp4',
                                        fps=30, dpi=100, cmap='RdBu'):
    """Create animation from velocity component data"""
    n_frames = velocity_data.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize plot
    vmin, vmax = velocity_data.min(), velocity_data.max()
    # Center colormap around zero
    vmax_abs = max(abs(vmin), abs(vmax))
    im = ax.imshow(np.flipud(velocity_data[0].T), cmap=cmap,
                   vmin=-vmax_abs, vmax=vmax_abs)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    title = ax.set_title(f'{component_name} - Frame 0')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(component_name)

    def animate(frame):
        # Update velocity field
        im.set_array(np.flipud(velocity_data[frame].T))
        title.set_text(f'{component_name} - Frame {frame}/{n_frames - 1}')
        return [im, title]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000 / fps, blit=True)

    # Save as MP4
    writer = FFMpegWriter(fps=fps, metadata=dict(artist=f'{component_name} Field'), bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"{component_name} animation saved to {output_path}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    smoke_c, velocity_x_c, velocity_y_c, pressure_c = load_data(args.input)

    # Get number of simulations
    n_simulations = smoke_c.shape[0]

    print(f"\nExtracting {n_simulations} simulations...")

    # Process each simulation
    for i in range(n_simulations):
        print(f"\nProcessing simulation {i + 1}/{n_simulations}")

        # Extract trajectory for this simulation
        smoke_traj = smoke_c[i]
        vx_traj = velocity_x_c[i]
        vy_traj = velocity_y_c[i]
        pressure_traj = pressure_c[i]

        # Create output filenames
        smoke_filename = f"smoke_simulation_{i + 1:03d}.mp4"
        pressure_filename = f"pressure_simulation_{i + 1:03d}.mp4"
        vx_filename = f"velocity_x_simulation_{i + 1:03d}.mp4"
        vy_filename = f"velocity_y_simulation_{i + 1:03d}.mp4"

        smoke_path = os.path.join(args.output_dir, smoke_filename)
        pressure_path = os.path.join(args.output_dir, pressure_filename)
        vx_path = os.path.join(args.output_dir, vx_filename)
        vy_path = os.path.join(args.output_dir, vy_filename)

        # Create smoke animation
        print(f"  Creating smoke density animation...")
        create_smoke_animation(
            smoke_traj,
            velocity_x=vx_traj if args.show_velocity else None,
            velocity_y=vy_traj if args.show_velocity else None,
            output_path=smoke_path,
            fps=args.fps,
            dpi=args.dpi,
            cmap=args.cmap,
            show_velocity=args.show_velocity
        )

        # Create pressure animation
        print(f"  Creating pressure field animation...")
        create_pressure_animation(
            pressure_traj,
            output_path=pressure_path,
            fps=args.fps,
            dpi=args.dpi,
            cmap='coolwarm'
        )

        # Create velocity X component animation
        print(f"  Creating velocity X component animation...")
        create_velocity_component_animation(
            vx_traj,
            component_name='Velocity X',
            output_path=vx_path,
            fps=args.fps,
            dpi=args.dpi,
            cmap='RdBu'
        )

        # Create velocity Y component animation
        print(f"  Creating velocity Y component animation...")
        create_velocity_component_animation(
            vy_traj,
            component_name='Velocity Y',
            output_path=vy_path,
            fps=args.fps,
            dpi=args.dpi,
            cmap='RdBu'
        )

    print(f"\nAll animations saved to {args.output_dir}/")
    print(f"  - {n_simulations} smoke density animations")
    print(f"  - {n_simulations} pressure field animations")
    print(f"  - {n_simulations} velocity X component animations")
    print(f"  - {n_simulations} velocity Y component animations")


if __name__ == "__main__":
    main()