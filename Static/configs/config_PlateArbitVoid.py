import os
import numpy as np
import argparse

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Plate with arbitrary voids config")

# Dataset/training
parser.add_argument("--n_train", type=int, default=1000, help="Number of training samples")
parser.add_argument("--n_test", type=int, default=200, help="Number of test samples")
parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
parser.add_argument("--normalized", type=int, default=0, help="Enable data normalization")
parser.add_argument("--load_model", type=int, default=1, help="Load pretrained model")
parser.add_argument("--training", type=int, default=0, help="Enable/disable training")
parser.add_argument("--ds_len", type=int, default=50, help="Number of test dataset for evaluation in post processing")

# Plate/beam geometry
parser.add_argument("--plate_length", type=float, default=5.0, help="Plate length")
parser.add_argument("--plate_width", type=float, default=5.0, help="Plate width")
parser.add_argument("--num_pts_x", type=int, default=200, help="Number of points in x-direction")
parser.add_argument("--num_pts_y", type=int, default=200, help="Number of points in y-direction")

# Material
parser.add_argument("--E", type=float, default=100.0, help="Young's modulus")
parser.add_argument("--nu", type=float, default=0.25, help="Poisson's ratio")

# FNO parameters
parser.add_argument("--mode1", type=int, default=8, help="FNO mode1")
parser.add_argument("--mode2", type=int, default=8, help="FNO mode2")
parser.add_argument("--width", type=int, default=32, help="FNO width")
parser.add_argument("--depth", type=int, default=8, help="FNO depth")
parser.add_argument("--channels_last_proj", type=int, default=128, help="FNO projection channels")
parser.add_argument("--padding", type=int, default=56, help="FNO padding")

args, _ = parser.parse_known_args()

# -----------------------------
# Assign arguments to variables
# -----------------------------
normalized = bool(args.normalized)
load_model = bool(args.load_model)
training = bool(args.training)

plate_length = args.plate_length
plate_width = args.plate_width
num_pts_x = args.num_pts_x
num_pts_y = args.num_pts_y

model_data = {
    "n_train": args.n_train,
    "n_test": args.n_test,
    "n_data": args.n_train + args.n_test,
    "n_dataset": args.n_train + args.n_test,
    "batch_size": args.batch_size,
    "batch_size_BFGS": 50,
    "num_epoch": args.num_epoch,
    "num_epoch_LBFGS": 100,
    "data_type": "float32",
    "normalized": normalized,
    "training": training,
    "load_model": load_model,
    "ds_len": args.ds_len,
    "beam": {
        "length": plate_length,
        "width": plate_width,
        "num_refinements": 5,
        "numPtsU": num_pts_x,
        "numPtsV": num_pts_y,
        "traction": np.ones_like(np.linspace(0, plate_width, num_pts_y)),
        "E": args.E,
        "nu": args.nu,
        "state": "plane stress",
    },
    "fno": {
        "mode1": args.mode1,
        "mode2": args.mode2,
        "width": args.width,
        "depth": args.depth,
        "channels_last_proj": args.channels_last_proj,
        "padding": args.padding,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "scheduled": False,
    },
    "GRF": {"alpha": 10.0, "tau": 7.0},
    "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./data/")),
}

# Define directory and filename
model_data["filename"] = (
    f"PlateHole_LxW_{plate_length}x{plate_width}_s"
    f"{model_data['beam']['numPtsU']}x{model_data['beam']['numPtsV']}_n"
    f"{model_data['n_dataset']}"
)
model_data["path"] = os.path.join(model_data["dir"], model_data["filename"] + ".npz")
