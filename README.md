# A Disease-specific Vision–Language Foundation Model for Comprehensive Clinical Assessment in Lung Cancer

## Data
Public datasets used in this study are listed as follows and available from their original repositories. Other datasets are subject to access restrictions, but may be available upon reasonable request for academic research purposes from the corresponding or first author (shuo_wang@buaa.edu.cn).

| Dataset    | URL                                                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| DLCSD24    | [https://zenodo.org/records/10782891](https://zenodo.org/records/10782891)                                                                           |
| LUNA16     | [https://luna16.grand-challenge.org](https://luna16.grand-challenge.org/)                                                                            |
| TCIA       | [https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics) |
| LUNG1      | [https://www.cancerimagingarchive.net/collection/nsclc-radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics)                   |
| UCSF-PDGM  | [https://www.cancerimagingarchive.net/collection/ucsf-pdgm](https://www.cancerimagingarchive.net/collection/ucsf-pdgm)                               |
| LUNG-PET   | [https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx)                     |
| DEEPLESION | [https://nihcc.app.box.com/v/DeepLesion](https://nihcc.app.box.com/v/DeepLesion)                                                                     |
| CT-RATE    | [https://huggingface.co/datasets/ibrahimhamamci/CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)                                     |

## Code

This section explains how to set up the environment, preprocess data, define/load the model, extract features, and fine-tune on downstream tasks.  

### 1 Environment Setup

We recommend using **conda** (see `environment.yml`):  

```bash
# Create environment
conda env create -f environment.yml
conda activate lucafound

# Install local package
pip install -e .
```

---

### 2 Data Preprocessing (`data_preprocess.py`)

`data_preprocess.py` automatically handles CT preprocessing. Only the image path (DICOM folder or NIfTI file) and CUDA device need to be specified:  

```bash
python data_preprocess.py   --img_path /path/to/CT.nii.gz   --cuda 0
```

- `--img_path`: path to input CT (NIfTI `.nii/.nii.gz` or DICOM folder)  
- `--cuda`: GPU id (e.g., `0`), defaults to CPU if not specified  

The output includes the lung ROI and lung mask images.

---
