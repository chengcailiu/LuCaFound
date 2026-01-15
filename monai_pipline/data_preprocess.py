import os
import argparse
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
import warnings

warnings.filterwarnings("ignore")
from monai.transforms import (
    Compose,
    Resized,
    CropForegroundd,
    CastToTyped,
    EnsureChannelFirstd,
)


def monai_process_and_save(img_obj, mask_np, output_id, output_dir="./processed", target_size=[48, 256, 256]):
    """
    Process and save images and masks

    Parameters:
    - img_obj: SimpleITK image object
    - mask_np: Mask numpy array
    - output_id: Output file identifier
    - output_dir: Output directory path
    - target_size: Target size [depth, height, width]
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data dictionary
    data = {
        "image": sitk.GetArrayFromImage(img_obj),
        "mask": mask_np
    }

    # Save original image information for reference
    original_spacing = img_obj.GetSpacing()
    original_direction = img_obj.GetDirection()
    original_origin = img_obj.GetOrigin()

    print(f"Original image size: {data['image'].shape}")
    print(f"Original image spacing: {original_spacing}")

    # Define processing pipeline
    process_pipeline = Compose([
        # Add channel dimension
        EnsureChannelFirstd(keys=["image", "mask"], channel_dim="no_channel"),

        # Crop foreground
        CropForegroundd(keys=["image", "mask"], source_key="mask"),

        # Resize
        Resized(
            keys=["image", "mask"],
            spatial_size=target_size,
            mode=("trilinear", "nearest")
        ),

        CastToTyped(keys=["image"], dtype=np.int16),
        CastToTyped(keys=["mask"], dtype=np.uint8),
    ])

    # Execute processing
    processed_data = process_pipeline(data)

    # Get processed image and mask
    processed_image = processed_data["image"][0].numpy()  # Remove channel dimension
    processed_mask = processed_data["mask"][0].numpy()  # Remove channel dimension

    print(f"Processed image size: {processed_image.shape}")
    print(f"Processed mask size: {processed_mask.shape}")

    # Create SimpleITK image objects
    image_itk = sitk.GetImageFromArray(processed_image)
    mask_itk = sitk.GetImageFromArray(processed_mask)

    # Set spacing (adjusted due to size change)
    # Assume isotropic voxels, set to [1.0, 1.0, 1.0]
    # Alternatively, calculate new spacing based on original spacing and scaling ratio
    new_spacing = [1.0, 1.0, 1.0]
    image_itk.SetSpacing(new_spacing)
    mask_itk.SetSpacing(new_spacing)

    # Set direction (use identity matrix as direction information may not be applicable after processing)
    image_itk.SetDirection(np.eye(3).flatten())
    mask_itk.SetDirection(np.eye(3).flatten())

    # Build output paths
    image_output_path = os.path.join(output_dir, f"{output_id}_img.nii.gz")
    mask_output_path = os.path.join(output_dir, f"{output_id}_mask.nii.gz")

    # Save image and mask
    sitk.WriteImage(image_itk, image_output_path)
    sitk.WriteImage(mask_itk, mask_output_path)

    print(f"Image saved to: {image_output_path}")
    print(f"Mask saved to: {mask_output_path}")

    return image_output_path, mask_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process medical images and extract lung segmentation')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device ID')
    parser.add_argument('--img_path', type=str, required=True, help='Input image path or DICOM directory')
    parser.add_argument('--output_dir', type=str, default='./processed', help='Output directory path')
    parser.add_argument('--target_size', type=str, default='48,256,256', help='Target size, format: depth,height,width')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # Parse target size
    target_size = [int(x) for x in args.target_size.split(',')]
    if len(target_size) != 3:
        raise ValueError("target_size must be three integers, format: depth,height,width")

    print(f"Target size set to: {target_size}")
    print(f"Output directory: {args.output_dir}")

    # Initialize lungmask inferer
    inferer = LMInferer(batch_size=1)

    # Logic processing: determine if it's a directory or file
    if os.path.isdir(args.img_path):
        # DICOM directory processing
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(args.img_path)
        if not series_ids:
            print(f"No DICOM series found in directory: {args.img_path}")
            exit(1)

        for series_id in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(args.img_path, series_id)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            img = reader.Execute()

            print(f"Processing series: {series_id}")
            print(f"Original image size: {img.GetSize()}")

            # Apply lungmask for segmentation
            temp_mask = inferer.apply(img)
            mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)

            # Process and save
            monai_process_and_save(img, mask_np, series_id, args.output_dir, target_size)
            print(f'Processing series {series_id} completed.\n')

    elif os.path.isfile(args.img_path):
        # Single file processing (e.g., .nii, .nii.gz)
        img = sitk.ReadImage(args.img_path)
        img_id = os.path.basename(args.img_path).split('.')[0]  # Remove extension

        print(f"Processing file: {args.img_path}")
        print(f"Original image size: {img.GetSize()}")

        # Apply lungmask for segmentation
        temp_mask = inferer.apply(img)
        mask_np = np.where(temp_mask > 0, 1, 0).astype(np.uint8)

        # Process and save
        monai_process_and_save(img, mask_np, img_id, args.output_dir, target_size)
        print(f'Processing file {img_id} completed.')
    else:
        print(f"Path does not exist: {args.img_path}")
        exit(1)