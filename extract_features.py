from model import ModelforExtractFea
import SimpleITK as sitk
import torch
import numpy as np
import argparse
import os
import sys

def data_process(path):
    img = sitk.ReadImage(path)
    x = sitk.GetArrayFromImage(img)
    x = np.clip(x, -1500, 500) + 1500
    max_val = x.max()
    x = x / max(max_val, 1e-8)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = x.repeat(1, 3, 1, 1, 1)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='./weights/model.pt', help='pretrained model path')
    parser.add_argument('--cuda', type=str, default='0', help='cuda device id (e.g. 0, 1)')
    parser.add_argument('--img_path', type=str, default='', help='DICOM directory or NIfTI file')
    args = parser.parse_args()
    
    if not os.path.exists(args.img_path):
        print("Please use --img_path to set a available path of your CT image")
        sys.exit(1)
    
    device = torch.device(f'cuda:{args.cuda}' if (torch.cuda.is_available() and args.cuda.isdigit()) else 'cpu')
    
    img = data_process(args.img_path).to(device)
    
    if os.path.exists(args.pretrained):
        pass
    else:
        raise ValueError(f"Error: Pretrained model file {args.pretrained} not found! Please use --pretrained to set a available path of model's weight")
        
    model = ModelforExtractFea(args=args).to(device)

    
    with torch.no_grad():
        feature = model(img)
    
    print(f"Feature shape: {feature.shape}")