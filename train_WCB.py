# use MLFlow as I am more comfortable with it

import argparse
import sys

import torch
import os 
import time
import pandas as pd

from monai.data import CacheDataset, ThreadDataLoader  # use MONAI caching + thread loader
from monai.metrics.meandice import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import SurfaceDistanceMetric

from monai.networks.nets.attentionunet import AttentionUnet
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete, Activations

from monai.losses.dice import DiceCELoss

from XrayTo3DShape import BaseDataset, get_kasten_transforms

import mlflow
from torch.cuda.amp import autocast, GradScaler

EXPERIMENT_NAME = "test"
# mlflow.set_tracking_uri("http://127.0.0.1:500")
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable system metric logging
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)


dice_metric_evaluator = DiceMetric(include_background=False)
hd95_metric_evaluator = HausdorffDistanceMetric(include_background=False, percentile=95)
asd_metric_evaluator = SurfaceDistanceMetric(include_background=False)

loss_function = DiceCELoss(sigmoid=True)


def parse_training_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("train_csv")
    parser.add_argument("val_csv")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint_freq", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--depth", choices=["small", "medium", "large", "extra_large"], default="medium")


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
    data_path = args.path
    train_csv = args.train_csv
    val_csv = args.val_csv
    num_epochs = args.epochs
    checkpoint_freq = args.checkpoint_freq
    batch_size = args.batch_size
    lr = args.lr
    depth = args.depth

    seed = 42

    transform = get_kasten_transforms(size=128, resolution=2.734375)

    # Get train and val from two different CSV files 
    train_csv_path = os.path.join(data_path, train_csv)
    val_csv_path = os.path.join(data_path, val_csv)
    train_path =  pd.read_csv(train_csv_path, index_col=0).to_numpy()
    val_path =  pd.read_csv(val_csv_path, index_col=0).to_numpy()
    train_paths = [{"ap": os.path.join(data_path, ap), "lat": os.path.join(data_path, lat), "seg": os.path.join(data_path, seg)} for ap, lat, seg in train_path]
    val_paths = [{"ap": os.path.join(data_path, ap), "lat": os.path.join(data_path, lat), "seg": os.path.join(data_path, seg)} for ap, lat, seg in val_path]

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
    elif depth == "large": depth = (16, 32, 64, 128)
    # elif depth == "extra_large": depth = (8, 16, 32, 64, 128)

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

        best_acc = -1.0
        best_ckpt_path = None
        # TODO: skip writing to checkpoint directory for now (do later)
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                ap = batch["ap"].to(device, non_blocking=True)    # B,1,H,W

                # print(type(ap))
                # print(f"ap size = {ap.shape}")
                lat = batch["lat"].to(device, non_blocking=True)  # B,1,H,W
                # print(f"lat size = {lat.shape}")
                seg = batch["seg"].to(device, non_blocking=True)  # B,1,D,H,W
                # print(f"seg size = {seg.shape}")

                D = seg.shape[2]
                ap3d = ap.unsqueeze(2).expand(-1, -1, D, -1, -1)
                lat3d = lat.unsqueeze(4).expand(-1, -1, -1, -1, D) # .permute(0, 1, 4, 2, 3)
                input_volume = torch.cat((ap3d, lat3d), dim=1)

                with autocast():
                    pred = model(input_volume)
                    loss = loss_function(pred, seg)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * input_volume.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            dice_metric_evaluator.reset()

            # accumulate val loss
            val_running_loss = 0.0
            with torch.no_grad():
                eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
                for batch in val_loader:
                    ap = batch["ap"].to(device, non_blocking=True)
                    lat = batch["lat"].to(device, non_blocking=True)
                    seg = batch["seg"].to(device, non_blocking=True)
                    D = seg.shape[2]
                    ap3d = ap.unsqueeze(2).expand(-1, -1, D, -1, -1)
                    lat3d = lat.unsqueeze(4).expand(-1, -1, -1, -1, D) # .permute(0, 1, 4, 2, 3)
                    input_volume = torch.cat((ap3d, lat3d), dim=1)
                    with autocast():
                        logits = model(input_volume)
                        # compute val loss on logits (same loss used in training)
                        batch_loss = loss_function(logits, seg)
                    val_running_loss += batch_loss.item() * input_volume.size(0)
                    pred = eval_transform(logits)
                    dice_metric_evaluator(y_pred=pred, y=seg)
                    hd95_metric_evaluator(y_pred=pred, y=seg)
                    asd_metric_evaluator(y_pred=pred, y=seg)

            val_loss = val_running_loss / len(val_loader.dataset)
            acc = dice_metric_evaluator.aggregate().item()
            hd95 = hd95_metric_evaluator.aggregate().item()
            asd = asd_metric_evaluator.aggregate().item()

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("dice", acc, step=epoch)
            mlflow.log_metric("hd95", hd95, step=epoch)
            mlflow.log_metric("asd", asd, step=epoch)


            dice_metric_evaluator.reset()
            hd95_metric_evaluator.reset()
            asd_metric_evaluator.reset()

            epoch_time = time.perf_counter() - epoch_start
            if (epoch) % checkpoint_freq == 0:
                # Save full checkpoint dict the same way as manual version
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "train_loss": train_loss,
                    "val_dice": acc,
                    "config": {"attunet": config_attunet, "lr": lr, "batch_size": batch_size},
                    "device": str(device),
                }
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name=f"checkpoint-epoch{epoch}",
                    pip_requirements=[
                        "torch==2.2.1+cu121",
                        "torchaudio==2.2.1+cu121",
                        "torchvision==0.17.1+cu121"
                    ],
                )
                # Also log the full checkpoint (includes optimizer, scaler etc.)
                ckpt_path = f"checkpoint-epoch{epoch}.pth"
                torch.save(ckpt, ckpt_path)
                mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                os.remove(ckpt_path)  # clean up local file after logging
                print(f"Saved checkpoint: epoch {epoch}")

            print(f"epoch {epoch} train_loss {train_loss:.4f} dice {acc:.4f} time {epoch_time:.2f}s")

            if acc > best_acc:
                best_acc = acc
                best_ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "train_loss": train_loss,
                    "val_dice": acc,
                    "config": {"attunet": config_attunet, "lr": lr, "batch_size": batch_size},
                    "device": str(device),
                }
                best_ckpt_path = f"best-epoch.pth"
                torch.save(best_ckpt, best_ckpt_path)
                mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")
                os.remove(best_ckpt_path)
                print(f"Saved new best checkpoint.")



