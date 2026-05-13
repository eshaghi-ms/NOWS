import subprocess
import sys
import os

# --- CONFIGURATION ---
use_bsub = True  # False = run locally, True = submit via bsub
python_folder = '/home/zore8312/miniforge3/envs/jax/bin/python'
main_folder = '/scratch/zore8312/PycharmProjects/VINO_combined'
file_folder = 'Dynamic/data'
smoke_script = 'database_smoke.py'

# Different experiment configs: (label, extra_args)
experiments = [
    ('Dt0.5_res64_N30_T300', '--Dt 0.5 --res 64  --N_data 100 --T 300'),
    ('Dt0.5_res64_N30_T300', '--Dt 1.0 --res 64  --N_data 100 --T 300'),
    ('Dt0.5_res64_N30_T300', '--Dt 1.5 --res 64  --N_data 100 --T 300'),
]

for label, args in experiments:
    inner_cmd = f"{python_folder} {os.path.join(main_folder, file_folder, smoke_script)} {args}"

    if use_bsub:
        # build LSF submission
        submit_cmd = (
            f"bsub -q BatchGPU "
            f'-J "{smoke_script}_{label}" '
            f"-o {smoke_script}_{label}.%J.out "
            f"-e {smoke_script}_{label}.%J.err "
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
        # direct local execution
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
