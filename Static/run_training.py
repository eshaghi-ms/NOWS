import subprocess
import sys
import os

# --- CONFIGURATION ---
use_bsub = False  # False = run locally, True = submit via bsub
python_exec = '/home/zore8312/miniforge3/envs/jax/bin/python'
main_folder = '/scratch/zore8312/PycharmProjects/VINO_combined'
file_folder = 'Static'
train_script = 'VINO_PlateArbitHole.py'   # Entry script that uses config_PlateArbitVoid.py

# List of experiments: (label, params dict)
experiments = [
    # (
    #     "plate_data4400_epoch_1000_128x128",
    #     {"--n_train": 4000, "--n_test": 400, "--batch_size": 20, "--num_epoch": 1000, "--learning_rate": 1e-3,
    #      "--training": 1, "--load_model": 0, "--num_pts_x": 128, "--num_pts_y": 128, "--padding": 0},
    # ),
    (
        "plate_data1200_epoch_1000_200x000",
        {"--ds_len": 2},
    ),
]

for label, params in experiments:
    # Build parameter string
    param_str = " ".join(f"{k} {v}" for k, v in params.items())
    inner_cmd = (
        f"{python_exec} {os.path.join(main_folder, file_folder, train_script)} {param_str}"
    )

    if use_bsub:
        submit_cmd = (
            f"bsub -q BatchGPU "
            f'-J "{train_script}_{label}" '
            f"-o {train_script}_{label}.%J.out "
            f"-e {train_script}_{label}.%J.err "
            f"-n 4 "                           # request 4 CPU cores
            f"-M 40000 "                       # request 32 GB memory
            f"-W 300:00 "                       # walltime 300 hours
            f'-gpu "num=1:mode=exclusive_process" '
            f"\"{inner_cmd}\""
        )

        print(f"\n>>> Submitting via bsub: {label}")
        print(submit_cmd)
        try:
            result = subprocess.run(
                submit_cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(result.stdout.strip())
            if result.stderr:
                print("STDERR:", result.stderr.strip())
        except subprocess.CalledProcessError as e:
            print(f"!!! Submission failed (code {e.returncode})")
            print("STDOUT:", e.stdout.strip())
            print("STDERR:", e.stderr.strip())

    else:
        print(f"\n>>> Running locally: {label}")
        try:
            proc = subprocess.Popen(
                inner_cmd,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            proc.communicate()
            if proc.returncode != 0:
                print(f"!!! Command failed (code {proc.returncode})")
        except Exception as e:
            print(f"!!! Execution exception: {e}")
