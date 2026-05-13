import subprocess
import sys
import os

# --- CONFIGURATION ---
use_bsub = False  # False = run locally, True = submit via bsub
python_exec = '/home/zore8312/miniforge3/envs/torch/bin/python'
main_folder = '/scratch/zore8312/PycharmProjects/VINO_combined'
file_folder = 'Conceptual'
# train_script = 'VINO_Poisson2D.py'
train_script = 'VINO_Darcy2D.py'
# train_script = 'FNO_Darcy2D.py'

# List of experiments: (label, params dict)
experiments = [

    (
        "s256_m32_phy",
        {"--s": 2 ** 8, "--modes": 32, "--batch_size": 50},
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
