import argparse
import os
import time
from datetime import datetime
import shutil
import csv

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import SurfaceDistanceMetric
from monai.networks.nets.attentionunet import AttentionUnet
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete, Activations
from monai.data import CacheDataset, ThreadDataLoader  # use MONAI caching + thread loader

import wandb
from XrayTo3DShape import BaseDataset, get_kasten_transforms
from torch.cuda.amp import autocast, GradScaler

NUM_EPOCHS = 100
WANDB_ON = False
TEST_ZERO_INPUT = False
BATCH_SIZE = 4
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PRE_FETCH_FACTOR = 2
PIN_MEMORY = True
TEST_SIZE = 0.1  # fraction for test split
VAL_SIZE = 0.1  # fraction for validation split
CHECKPOINT_DIR = "checkpoints"  # <-- add this

dice_metric_evaluator = DiceMetric(include_background=False)
hd95_metric_evaluator = HausdorffDistanceMetric(include_background=False, percentile=95)
asd_metric_evaluator = SurfaceDistanceMetric(include_background=False)

loss_function = DiceCELoss(sigmoid=True)

def main(lr, depth, run_timestamp, run_dir):
    # print(torch.cuda.is_available())
    # device = 'cpu'
    try:
        num_cpus = os.cpu_count()
        torch.set_num_threads(num_cpus)
        torch.set_num_interop_threads(num_cpus)
    except RuntimeError:
        # If threads already initialized, just skip instead of crashing
        print("[Warning] Could not change thread settings; continuing with defaults.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # device handled by setting default device
    # torch.set_default_device("cuda")

    if WANDB_ON:
        wandb.init(project="pipeline-test-01", name="attentionUnet-01")

    paths_location = "/home/mdemey/projects/pelvis/pelvis-net/data/DRR_10"
    paths_location_csv = os.path.join(paths_location, "drr_dataset_train.csv")

    paths = pd.read_csv(paths_location_csv, index_col=0).to_numpy()
    paths = [{"ap": os.path.join(paths_location, ap), "lat": os.path.join(paths_location, lat), "seg": os.path.join(paths_location, seg)} for ap, lat, seg in paths]

    # split into train / other
    train_paths, other_paths = train_test_split(paths, test_size=TEST_SIZE + VAL_SIZE, random_state=42) # save 20% of data for val/test
    val_paths, test_paths = train_test_split(other_paths, test_size=0.5, random_state=42) # split val/test data in half
    
    # create datasets / loaders
    transform = get_kasten_transforms(size=128, resolution=2.734375)
    print(type(transform))
    print(transform)

    # Cache once (first epoch/build), then fast
    print(train_paths)
    train_ds = CacheDataset(data=train_paths, transform=transform, cache_rate=1.0, num_workers=0)
    val_ds = CacheDataset(data=val_paths, transform=transform, cache_rate=1.0, num_workers=0)
    test_ds = CacheDataset(data=test_paths, transform=transform, cache_rate=1.0, num_workers=0)
    
    # print("Train dataset size:", len(train_ds))
    # print("Validation dataset size:", len(val_ds))
    # print("Test dataset size:", len(test_ds))

    # Threaded loader avoids Windows spawn overhead
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    test_loader = ThreadDataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    config_attunet = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": depth,
        "strides": (2, 2, 2),
    }
    model = AttentionUnet(spatial_dims=3, **config_attunet).to(device)

    print("GPU available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Model device:", next(model.parameters()).device)

    optimizer = torch.optim.AdamW(model.parameters(), lr)

    # mixed precision scaler
    scaler = GradScaler()

    # ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # create a run-specific subdirectory for checkpoints
    os.makedirs(run_dir, exist_ok=True)

    # set tags
    if lr == 1e-2: lr_tag = "0100"
    elif lr == 1e-3: lr_tag = "0010"
    elif lr == 1e-4: lr_tag = "0001"

    if depth == (8, 16, 32): depth_tag = "small"
    elif depth == (8, 16, 32, 64): depth_tag = "medium"
    elif depth == (16, 32, 64, 128): depth_tag = "large"

    # CSV log for per-epoch scalars
    csv_path = os.path.join(run_dir, f"training_log_{lr_tag}_{depth_tag}.csv")
    # write header (overwrite any existing file for this run)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "dice", "hd95", "asd"])

    # track best validation score and path
    best_acc = -1.0
    best_ckpt_path = None

    for epoch in range(NUM_EPOCHS):
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

        dice_metric_evaluator.reset()
        hd95_metric_evaluator.reset()
        asd_metric_evaluator.reset()

        epoch_time = time.perf_counter() - epoch_start

        # save checkpoint every 10 epochs
        if (epoch) % 10 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": train_loss,
                "val_dice": acc,
                "config": {"attunet": config_attunet, "lr": lr, "batch_size": BATCH_SIZE},
                "device": str(device),
            }
            ckpt_path = os.path.join(
                run_dir, f"attunet_epoch{epoch+1:04d}_dice{acc:.4f}_{run_timestamp}.pth"
            )
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # update and save best model (by validation dice)
        if acc > best_acc:
            best_acc = acc
            best_ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": train_loss,
                "val_dice": acc,
                "config": {"attunet": config_attunet, "lr": lr, "batch_size": BATCH_SIZE},
                "device": str(device),
            }
            best_ckpt_path = os.path.join(run_dir, f"attunet_best_epoch{epoch+1:04d}_dice{acc:.4f}_{run_timestamp}.pth")
            torch.save(best_ckpt, best_ckpt_path)
            # also keep a stable filename for easy loading
            stable_best_path = os.path.join(run_dir, f"attunet_best_{lr_tag}_{depth_tag}.pth")
            shutil.copyfile(best_ckpt_path, stable_best_path)
            print(f"Saved new best checkpoint: {best_ckpt_path} -> {stable_best_path}")

        if WANDB_ON:
            wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": acc, "epoch_time_s": epoch_time})
        print(f"epoch {epoch} train_loss {train_loss:.4f} dice {acc:.4f} time {epoch_time:.2f}s")
        # log to CSV: epoch (1-based), current lr, train_loss,val_loss, dice
        current_lr = optimizer.param_groups[0]["lr"]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, float(current_lr), float(train_loss), float(val_loss), float(acc), float(hd95), float(asd)])
    return test_loader

from XrayTo3DShape import AttentionUnet, get_model

if __name__ == "__main__":
    
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(CHECKPOINT_DIR, run_timestamp)
    learning_rates = [1e-3]
    depths = [(16, 32, 64, 128)]

    for lr in learning_rates: 
        print(f"------------- Testing Learning Rate : {lr:.4f} -------------")
        for depth in depths: 
            print(f"------------- Attention UNet Depth : {depth} -------------")
            test_loader = main(lr, depth, run_timestamp, run_dir)


    csv_files = [f for f in os.listdir(run_dir) if f.endswith(".csv") and "training_log" in f]

    results = []

    for file in csv_files:
        path = os.path.join(run_dir, file)
        df = pd.read_csv(path)
        
        # Extract parameters from filename
        parts = file.split("_")
        lr = parts[2]  # e.g., 0001, 0010, 0100
        arch = parts[3].replace(".csv", "")  # e.g., small, medium, large
        
        # Find epoch with lowest val_loss
        min_loss_idx = df["val_loss"].idxmin()
        min_loss_row = df.loc[min_loss_idx]
        
        # Find epoch with highest Dice score
        max_dice_idx = df["dice"].idxmax()
        max_dice_row = df.loc[max_dice_idx]
        
        results.append({
            "file": file,
            "architecture": arch,
            "lr": float(lr) / 1000,  # convert 0001 -> 0.001 etc.
            "best_val_loss": min_loss_row["val_loss"],
            "epoch_val_loss": int(min_loss_row["epoch"]),
            "best_dice": max_dice_row["dice"],
            "epoch_dice": int(max_dice_row["epoch"]),
        })

    # Convert to DataFrame
    summary = pd.DataFrame(results)

    # Find the overall best runs
    best_loss_row = summary.loc[summary["best_val_loss"].idxmin()]
    best_dice_row = summary.loc[summary["best_dice"].idxmax()]

    print("===== SUMMARY OF ALL RUNS =====")
    print(summary.sort_values(by=["architecture", "lr"]).to_string(index=False))
    print("\n===== BEST MODELS =====")
    print(f"Lowest Validation Loss:\n {best_loss_row.to_dict()}\n")
    print(f"Highest Dice Score:\n {best_dice_row.to_dict()}")

    # for the file with the best loss, run the checkpoint on the test data and get the performance metrics!

    # for the file with the best loss, run the checkpoint on the test data and get the performance metrics!
    print("\n===== TESTING BEST MODEL (by lowest validation loss) =====")

    best_model_path = os.path.join(run_dir, best_loss_row["file"].replace("training_log", "attunet_best").replace(".csv", ".pth"))
    if not os.path.exists(best_model_path):
        print(f"[Warning] Best model checkpoint not found at {best_model_path}. Trying to infer location...")
        best_model_path = os.path.join(run_dir, f"attunet_best_{best_loss_row['lr']:.4f}_{best_loss_row['architecture']}.pth")

    print(f"Loading best model from: {best_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(best_model_path, map_location=device)
    state = checkpoint.get("state_dict") or checkpoint.get("model_state") or checkpoint
    att_cfg = (checkpoint.get("config", {}) or {}).get("attunet")
    if att_cfg:
        model = AttentionUnet(spatial_dims=3, **att_cfg).to(device)
    else:
        # match your training input: 2 channels (AP+LAT), 1 output
        model = get_model(model_name=AttentionUnet.__name__, image_size=128).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint; missing:", missing, "unexpected:", unexpected)
    model.eval()

    dice_metric_evaluator.reset()
    hd95_metric_evaluator.reset()
    asd_metric_evaluator.reset()

    with torch.no_grad():
        eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        test_running_loss = 0.0
        for batch in test_loader:
            ap = batch["ap"].to(device, non_blocking=True)
            lat = batch["lat"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            D = seg.shape[2]
            ap3d = ap.unsqueeze(2).expand(-1, -1, D, -1, -1)
            lat3d = lat.unsqueeze(4).expand(-1, -1, -1, -1, D)
            input_volume = torch.cat((ap3d, lat3d), dim=1)

            with autocast():
                logits = model(input_volume)
                batch_loss = loss_function(logits, seg)

            test_running_loss += batch_loss.item() * input_volume.size(0)
            pred = eval_transform(logits)
            dice_metric_evaluator(y_pred=pred, y=seg)
            hd95_metric_evaluator(y_pred=pred, y=seg)
            asd_metric_evaluator(y_pred=pred, y=seg)

    test_loss = test_running_loss / len(test_loader.dataset)
    test_dice = dice_metric_evaluator.aggregate().item()
    test_hd95 = hd95_metric_evaluator.aggregate().item()
    test_asd = asd_metric_evaluator.aggregate().item()

    print("\n===== TEST RESULTS =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Dice Score: {test_dice:.4f}")
    print(f"Hausdorff (95th percentile): {test_hd95:.4f}")
    print(f"Average Surface Distance: {test_asd:.4f}")