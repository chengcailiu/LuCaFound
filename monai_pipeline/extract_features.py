import os
import sys
import argparse
import torch
import numpy as np
import SimpleITK as sitk

# Import MONAI components
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RepeatChanneld,
    EnsureTyped,
    ToTensord
)
from monai.data import Dataset, DataLoader

# Assuming model.py is in the current directory
from model import ModelforExtractFea


def get_monai_pipeline():
    """
    Define MONAI preprocessing pipeline
    Equivalent to the original code's operations:
    1. clip(-1500, 500)
    2. + 1500 (shift to 0-2000)
    3. Normalize to [0, 1]
    4. Repeat to 3 channels (repeat(1, 3, ...))
    """
    return Compose([
        # 1. Load data (keys=["image"] corresponds to dictionary key)
        LoadImaged(keys=["image"], reader="ITKReader"),

        # 2. Ensure channel dimension is first (C, D, H, W)
        EnsureChannelFirstd(keys=["image"]),

        # 3. CT value clipping and normalization
        # Original logic: x = (clip(x, -1500, 500) + 1500) / 2000
        # MONAI ScaleIntensityRanged does this directly:
        # a_min/max are clipping range, b_min/max are target mapping range
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1500, a_max=500,
            b_min=0.0, b_max=1.0,
            clip=True
        ),

        # 4. Repeat channels (1, D, H, W) -> (3, D, H, W)
        RepeatChanneld(keys=["image"], repeats=3),

        # 5. Convert to Tensor and ensure type
        EnsureTyped(keys=["image"], data_type="tensor"),
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='./weights/model.pt')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--save_feature', type=bool, default=True, help='save feature')
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"Path not found: {args.img_path}")
        sys.exit(1)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # --- MONAI data loading section ---
    # Build data dictionary
    data_dict = {"image": args.img_path}
    transforms = get_monai_pipeline()

    # Execute preprocessing
    processed_data = transforms(data_dict)
    # MONAI processing adds a batch dimension [1, 3, D, H, W]
    img_tensor = processed_data["image"].unsqueeze(0).to(device)

    # --- Model inference section (unchanged) ---
    if not os.path.exists(args.pretrained):
        raise ValueError(f"Model weight not found at {args.pretrained}")

    model = ModelforExtractFea(args=args).to(device)
    model.eval()  # Recommended to add eval mode

    with torch.no_grad():
        feature = model(img_tensor)

    print(f"Feature shape: {feature.shape}")