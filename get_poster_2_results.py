# ex. python3 training/get_poster_2_results.py P7
from monai.metrics.meandice import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import SurfaceDistanceMetric

from monai.data import MetaTensor
import torch
import nibabel as nib
import numpy as np

import argparse

RESOLUTION = 2.734375

parser = argparse.ArgumentParser()
parser.add_argument("sample_num", choices=["P1", "P5", "P6", "P7"])
args = parser.parse_args()
sample_num = args.sample_num

gt_path = f"data/recon/inputs/0623/outputs/{sample_num}/{sample_num}_ground_truth.nii.gz"
pred_path = f"data/recon/inputs/0623/outputs/{sample_num}/{sample_num}_prediction.nii.gz"
def load_binary_volume(path: str) -> np.ndarray:
    """Load a NIfTI file and return a binarized (0/1) float32 numpy array."""
    vol = nib.load(path).get_fdata()
    vol = (vol > 0.5).astype(np.float32)
    return vol

def to_metatensor(volume: np.ndarray, resolution: float) -> MetaTensor:
    """Convert a (D,H,W) numpy array into a (1,1,D,H,W) MONAI MetaTensor with pixdim meta."""
    tensor = torch.from_numpy(volume)[None, None]  # add batch + channel dims
    return MetaTensor(tensor, meta={"pixdim": (resolution, resolution, resolution)})

dice_metric = DiceMetric(include_background=False)
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
asd_metric = SurfaceDistanceMetric(include_background=False)

gt_vol = load_binary_volume(gt_path)
pred_vol = load_binary_volume(pred_path)
if gt_vol.shape != pred_vol.shape:
    print("Inputs have the incorrect shape")

gt_mt = to_metatensor(gt_vol, RESOLUTION)
pred_mt = to_metatensor(pred_vol, RESOLUTION)

dice_metric(y_pred=pred_mt, y=gt_mt)
hd95_metric(y_pred=pred_mt, y=gt_mt)
asd_metric(y_pred=pred_mt, y=gt_mt)

dice = dice_metric.aggregate().item()
hd95 = hd95_metric.aggregate().item()
asd = asd_metric.aggregate().item()

dice_metric.reset()
hd95_metric.reset()
asd_metric.reset()

print(f"DICE Similarity Coefficient on Test Split: {dice:.4f}")
print(f"95th Percentile Hausdorff Distance on Test Split (in mm): {hd95*RESOLUTION:.4f}")
print(f"Average Surface Distance on Test Split (in mm): {asd*RESOLUTION:.4f}")