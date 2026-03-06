import os
import time
import SimpleITK as sitk
import cv2
import numpy as np
import csv
from tqdm import tqdm
import json
from lungmask import LMInferer
import pandas as pd
import argparse
import torch
import warnings
warnings.simplefilter("ignore")


def get_boundingbox(mask):
    """
    Get the minimum and maximum coordinates of non-zero elements in a 3D mask.

    Args:
        mask: 3D numpy array representing the mask

    Returns:
        tuple: (xyz_min, xyz_max) where each is a numpy array of [x, y, z] coordinates
    """
    shape = mask.shape

    xyz_min = np.array([np.min(np.where(mask != 0)[0]), np.min(np.where(mask != 0)[1]), np.min(np.where(mask != 0)[2])])
    xyz_max = np.array([np.max(np.where(mask != 0)[0]), np.max(np.where(mask != 0)[1]), np.max(np.where(mask != 0)[2])])

    return (xyz_min, xyz_max)


def resize3D(img, size, interpolation):
    """
    Resize a 3D volume to the specified size.

    Args:
        img: 3D numpy array to resize
        size: tuple/list of target dimensions (x, y, z)
        interpolation: OpenCV interpolation method

    Returns:
        numpy array: Resized 3D volume
    """
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


def get_new_img_and_mask(img, mask, target_size=None):
    """
    Crop image and mask to lung region and resize to target size.

    Args:
        img: 3D numpy array of the original image
        mask: 3D numpy array of the lung mask
        target_size: tuple/list of target dimensions (x, y, z)

    Returns:
        tuple: (resize_image, resize_mask) both resized to target size
    """
    if target_size is None:
        # x, y, z
        target_size = [48, 256, 256]

    # Get bounding box of lung mask
    boundingbox = get_boundingbox(mask)

    # Calculate dimensions of bounding box
    aix_len_1 = boundingbox[1][0] - boundingbox[0][0]
    aix_len_2 = boundingbox[1][1] - boundingbox[0][1]
    aix_len_3 = boundingbox[1][2] - boundingbox[0][2]

    # Calculate center of bounding box
    bbox_center = np.array([boundingbox[0][0] + aix_len_1 / 2,
                            boundingbox[0][1] + aix_len_2 / 2,
                            boundingbox[0][2] + aix_len_3 / 2])

    # Calculate new bounding box centered on lung region
    new_bbox_min = np.array([bbox_center[0] - aix_len_1 / 2,
                             bbox_center[1] - aix_len_2 / 2,
                             bbox_center[2] - aix_len_3 / 2], dtype=np.int16)
    new_bbox_max = np.array([bbox_center[0] + aix_len_1 / 2,
                             bbox_center[1] + aix_len_2 / 2,
                             bbox_center[2] + aix_len_3 / 2], dtype=np.int16)

    # Ensure bounding box stays within image boundaries
    new_bbox_min = np.where(new_bbox_min < 0, 0, new_bbox_min)
    new_bbox_max = np.where(new_bbox_max > mask.shape, mask.shape, new_bbox_max)

    # Crop image and mask to bounding box
    new_mask = mask[new_bbox_min[0]: new_bbox_max[0] + 1,
    new_bbox_min[1]: new_bbox_max[1] + 1,
    new_bbox_min[2]: new_bbox_max[2] + 1]
    new_image = img[new_bbox_min[0]: new_bbox_max[0] + 1,
    new_bbox_min[1]: new_bbox_max[1] + 1,
    new_bbox_min[2]: new_bbox_max[2] + 1]

    # Resize to target size
    resize_mask = resize3D(new_mask, target_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    resize_image = resize3D(new_image, target_size, interpolation=cv2.INTER_CUBIC).astype(np.int16)

    return resize_image, resize_mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process medical images for lung segmentation and cropping")

    # CUDA device configuration
    parser.add_argument('--cuda', type=str, default='',
                        help='CUDA device ID to use (e.g., "0", "1"). Leave empty for CPU.')

    # Input path (DICOM directory or NIfTI file)
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to DICOM directory or NIfTI file')

    # Output directory
    parser.add_argument('--output_dir', type=str, default='./processed',
                        help='Output directory for processed images and masks (default: ./processed)')

    # Target size for resizing
    parser.add_argument('--target_size', type=str, default='48,256,256',
                        help='Target size for resizing in format "depth,height,width" (default: 48,256,256)')

    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # Initialize lungmask inferer
    inferer = LMInferer(batch_size=1)

    # Parse target size
    try:
        target_size = [int(dim) for dim in args.target_size.split(',')]
        if len(target_size) != 3:
            raise ValueError("Target size must have exactly 3 dimensions")
    except ValueError as e:
        print(f"Error parsing target size: {e}")
        print("Using default size [48, 256, 256]")
        target_size = [48, 256, 256]

    print(f"Target size set to: {target_size}")
    print(f"Output directory: {args.output_dir}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input path is a directory or file
    if os.path.isdir(args.img_path):
        filedir = args.img_path

        # Read DICOM series
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filedir)
        print(f"Found {len(series_ids)} series in directory: {filedir}")

        if not series_ids:
            raise RuntimeError(f"No DICOM series found in directory: {filedir}")

        # Process each DICOM series
        for i, imgseries_id in enumerate(series_ids):
            imgseries = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filedir, series_ids[i])
            imgseriesreader = sitk.ImageSeriesReader()
            imgseriesreader.SetFileNames(imgseries)
            img = imgseriesreader.Execute()
            print(f"Processing series {imgseries_id}, image shape: {img.GetSize()}")

            # Apply lungmask segmentation
            temp_data = img
            temp_mask = inferer.apply(temp_data)
            temp_mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)
            mask = temp_mask_np

            # Convert to numpy array
            img_np = sitk.GetArrayFromImage(img)

            # Crop and resize
            img_array, mask_array = get_new_img_and_mask(img_np, mask, target_size=target_size)

            # Convert back to SimpleITK images
            nii_img = sitk.GetImageFromArray(img_array)
            nii_mask = sitk.GetImageFromArray(mask_array)
            
            spacing = nii_img.GetSpacing()
            nii_img.SetSpacing((spacing[0], spacing[1], 3))
            nii_mask.SetSpacing((spacing[0], spacing[1], 3))

            # Save processed images
            sitk.WriteImage(nii_img, os.path.join(args.output_dir, f'{imgseries_id}_img.nii.gz'))
            sitk.WriteImage(nii_mask, os.path.join(args.output_dir, f'{imgseries_id}_mask.nii.gz'))

            print(f'Processed series {imgseries_id} completed.')

    elif os.path.isfile(args.img_path):
        # Process single NIfTI file
        try:
            img = sitk.ReadImage(args.img_path)
        except Exception as e:
            print(f"Error reading file: {e}")
            exit('File reading error')

        # Extract image ID from filename
        if '.nii.gz' in args.img_path:
            imgseries_id = os.path.basename(args.img_path).split('.nii.gz')[0]
        else:
            imgseries_id = os.path.splitext(os.path.basename(args.img_path))[0]

        print(f"Processing file {imgseries_id}, image shape: {img.GetSize()}")

        # Apply lungmask segmentation
        temp_data = img
        temp_mask = inferer.apply(temp_data)
        temp_mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)
        mask = temp_mask_np

        # Convert to numpy array
        img_np = sitk.GetArrayFromImage(img)

        # Crop and resize
        img_array, mask_array = get_new_img_and_mask(img_np, mask, target_size=target_size)

        # Convert back to SimpleITK images
        nii_img = sitk.GetImageFromArray(img_array)
        nii_mask = sitk.GetImageFromArray(mask_array)

        # Save processed images
        sitk.WriteImage(nii_img, os.path.join(args.output_dir, f'{imgseries_id}_img.nii.gz'))
        sitk.WriteImage(nii_mask, os.path.join(args.output_dir, f'{imgseries_id}_mask.nii.gz'))

        print(
            f'Processed {imgseries_id} done, saved to {args.output_dir}/{imgseries_id}_img.nii.gz and {args.output_dir}/{imgseries_id}_mask.nii.gz')

    else:
        print('Invalid input path. Please provide a valid DICOM directory or NIfTI file path.')
        exit()
