import os
import time
import SimpleITK as sitk
import cv2
import numpy as np
import SimpleITK as sitk
import os
import csv
from tqdm import tqdm
import SimpleITK as sitk
import os
import json
import numpy as np
from lungmask import LMInferer
from tqdm import tqdm
import pandas as pd
import argparse


def get_boundingbox(mask):
    shape = mask.shape

    xyz_min = np.array([np.min(np.where(mask != 0)[0]), np.min(np.where(mask != 0)[1]), np.min(np.where(mask != 0)[2])])
    xyz_max = np.array([np.max(np.where(mask != 0)[0]), np.max(np.where(mask != 0)[1]), np.max(np.where(mask != 0)[2])])

    return (xyz_min, xyz_max)

def resize3D(img, size, interpolation):

    img = img.astype(np.float32)
    x, y, z = img.shape
    pointx, pointy, pointz = size
    resized_img1 = np.zeros((pointx, pointy, z))
    for z in range(img.shape[2]):
        resized_img1[:, :, z] = cv2.resize(img[:, :, z], (size[1], size[0]), interpolation=interpolation)

    resized_img = np.zeros((pointx, pointy, pointz))

    for z in range(resized_img.shape[0]):
        resized_img[z, :, :] = cv2.resize(resized_img1[z, :, :], (size[2], size[1]), interpolation=interpolation)

    return resized_img

def get_new_img_and_mask(img, mask, target_size=None):#img,
    if target_size is None:
        # x, y, z
        target_size = [48, 256, 256]
    boundingbox = get_boundingbox(mask)

    aix_len_1 = boundingbox[1][0] - boundingbox[0][0]
    aix_len_2 = boundingbox[1][1] - boundingbox[0][1]
    aix_len_3 = boundingbox[1][2] - boundingbox[0][2]
    bbox_center = np.array([boundingbox[0][0] + aix_len_1 / 2, boundingbox[0][1] + aix_len_2 / 2, boundingbox[0][2] + aix_len_3 / 2])

    new_bbox_min = np.array([bbox_center[0] - aix_len_1 / 2, bbox_center[1] - aix_len_2 / 2, bbox_center[2] - aix_len_3 / 2], dtype=np.int16)
    new_bbox_max = np.array([bbox_center[0] + aix_len_1 / 2, bbox_center[1] + aix_len_2 / 2, bbox_center[2] + aix_len_3 / 2], dtype=np.int16)
    new_bbox_min = np.where(new_bbox_min < 0, 0, new_bbox_min)
    new_bbox_max = np.where(new_bbox_max > mask.shape, mask.shape, new_bbox_max)

    new_mask = mask[new_bbox_min[0]: new_bbox_max[0]+1, new_bbox_min[1]: new_bbox_max[1]+1, new_bbox_min[2]: new_bbox_max[2]+1]
    new_image = img[new_bbox_min[0]: new_bbox_max[0]+1, new_bbox_min[1]: new_bbox_max[1]+1, new_bbox_min[2]: new_bbox_max[2]+1]

    resize_mask = resize3D(new_mask, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    resize_image = resize3D(new_image, target_size, interpolation=cv2.INTER_CUBIC).astype(np.int16)

    return resize_image, resize_mask


if __name__ == '__main__':
    
    args = parser.parse_args()
    parser = argparse.ArgumentParser()
    ### cuda
    parser.add_argument('--cuda', type=str, default='0')
    ### dicom dir / nii path
    parser.add_argument('--img_path', type=str, default='')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    inferer = LMInferer(batch_size=100)

    ### see rpptdir is dir or nii file

    if os.path.isdir(args.img_path):
        filedir = args.img_path
        ### read dicom
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filedir)
        print(f"Found {len(series_ids)} series in directory: {filedir}")
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in directory: {filedir}")
        for i,imgseries_id in enumerate(series_ids):
            imgseries = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filedir, series_ids[i])
            imgseriesreader = sitk.ImageSeriesReader()
            imgseriesreader.SetFileNames(imgseries)
            img = imgseriesreader.Execute()
            print(f"Image shape: {img.GetSize()}")

            temp_data = img
            temp_mask = inferer.apply(temp_data)
            temp_mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)
            mask = temp_mask_np
            img = sitk.GetArrayFromImage(img)
            img_array, mask_array = get_new_img_and_mask(img, mask, target_size=[48, 256, 256])
            nii_img = sitk.GetImageFromArray(img_array)
            nii_mask = sitk.GetImageFromArray(mask_array)
            os.makedirs(f'./processed/', exist_ok=True)    
            sitk.WriteImage(nii_img, f'./processed/{imgseries_id}_img.nii.gz')
            sitk.WriteImage(nii_mask, f'./processed/{imgseries_id}_mask.nii.gz')
    elif os.path.isfile(args.img_path):
        try: 
            img = sitk.ReadImage(args.img_path)
        except Exception as e:
            print(e)
            exit('Read dicom error')
        print(f"Image shape: {img.GetSize()}")
        temp_data = img
        temp_mask = inferer.apply(temp_data)
        temp_mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)
        mask = temp_mask_np
        img = sitk.GetArrayFromImage(img)
        img_array, mask_array = get_new_img_and_mask(img, mask, target_size=[48, 256, 256])
        nii_img = sitk.GetImageFromArray(img_array)
        nii_mask = sitk.GetImageFromArray(mask_array)
        os.makedirs(f'./processed/', exist_ok=True)    
        sitk.WriteImage(nii_img, f'./processed/{imgseries_id}_img.nii.gz')
        sitk.WriteImage(nii_mask, f'./processed/{imgseries_id}_mask.nii.gz')
    else:
        print('Please input dicom dir or nii file in args.img_path')
        exit()
    print(f'Processed {imgseries_id} done, saved to ./processed/{imgseries_id}_img.nii.gz and ./processed/{imgseries_id}_mask.nii.gz')







