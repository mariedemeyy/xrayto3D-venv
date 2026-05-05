# use MLFlow as I am more comfortable with it

import argparse
import sys

import torch
import os 

from monai.data import CacheDataset, ThreadDataLoader  # use MONAI caching + thread loader
from monai.metrics.meandice import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import SurfaceDistanceMetric

from monai.networks.nets.attentionunet import AttentionUnet

from monai.losses.dice import DiceCELoss

from XrayTo3DShape import BaseDataset, get_kasten_transforms

import mlflow
from torch.cuda.amp import autocast, GradScaler

EXPERIMENT_NAME = "test"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# IMPORTANT: Enable system metrics monitoring 
# pip install psutil
# pip install nvidia-ml-py / pip install pyrsmi
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

def parse_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("trainpaths")
    parser.add_argument("valpaths")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--depth", choices=["small", "medium", "large"], default="medium")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    try:
        num_cpus = os.cpu_count()
        torch.set_num_threads(num_cpus)
        torch.set_num_interop_threads(num_cpus)
    except RuntimeError:
        # If threads already initialized, just skip instead of crashing
        print("[Warning] Could not change thread settings; continuing with defaults.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0))

    args = parse_training_arguments()
    print(sys.argv)
    print(args) 

    # Save arguments as variables
    train_paths = args.trainpaths
    val_paths = args.valpaths
    batch_size = args.batch_size
    lr = args.lr
    depth = args.depth

    seed = 42

    transform = get_kasten_transforms(size=128, resolution=2.734375)

    # Get train and val from two different CSV files 
    train_ds = CacheDataset(data=train_paths, transform=transform, cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_paths, transform=transform, cache_rate=1.0, num_workers=0)


    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if depth == "small": depth = (8, 16, 32)
    elif depth == "medium": depth = (8, 16, 32, 64)
    elif depth == "large": depth == (16, 32, 64, 128)

    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("depth", depth)

        mlflow.log_input()

        config_attunet = {
            "in_channels": 2,
            "out_channels": 1,
            "channels": depth,
            "strides": (2, 2, 2),
        }
        model = AttentionUnet(spatial_dims=3, **config_attunet).to(device)
        print("Model device:", next(model.parameters()).device)

        optimizer = torch.optim.AdamW(model.parameters(), lr)

        # mixed precision scaler
        scaler = GradScaler()








