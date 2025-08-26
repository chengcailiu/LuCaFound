from model import ModelforExtractFea
import SimpleITK as sitk
import torch
import numpy as np
import argparse

def data_process(path):
    
    # load nii
    img = sitk.ReadImage(path)
    x = sitk.GetArrayFromImage(img)
    # standardize
    x = torch.clamp(x, -1500, 500) + 1500
    x = x / x.max()
    # tensor
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## pretrained model
    parser.add_argument('--pretrained', type=str, default='/data1/lcc/log/CLIP/downstream_task/20241219mae2/10上传github/model.pt', help='pretrained model path')
    parser.add_argument('--save_feature', type=bool, default=True, help='save feature')
    parser.add_argument('--cuda', type=str, default='0', help='cuda')
    parser.add_argument('--img_path', type=str, default='', help='DICOM directory or NIfTI file')
    args = parser.parse_args()

    img = data_process(args.img_path)
    model = ModelforExtractFea(args=args)
    feature = model(img)
    print(feature.shape)
    

