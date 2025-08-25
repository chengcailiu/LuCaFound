# A Disease-specific Vision–Language Foundation Model for Comprehensive Clinical Assessment in Lung Cancer

**News**: Initial open-source release, supporting **efficient lung CT feature extraction** and **further fine-tuning**.  
**Weights**: Pretrained encoder weights are released here → [model.pt](https://github.com/chengcailiu/LuCaFound/releases/download/weight/model.pt).  

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

### 3 Model Definition & Weight Loading (`model.py`)

`model.py` provides the encoder definition and weight loading.
We recommend downloading the pretrained weights from [model.pt](https://github.com/chengcailiu/LuCaFound/releases/download/weight/model.pt) to `./weights/`. 

```python
from model import ModelforExtractFea
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=str, default='./weights/model.pt', help='pretrained model path')
parser.add_argument('--save_feature', type=bool, default=True, help='save feature')
args = parser.parse_args()
model = ModelforExtractFea(args=args)
```

---

### 4 Feature Extraction Example

```python
import torch
from model import ModelforExtractFea
from data_preprocess import preprocess_img  # assumed API

# 1) Preprocess image
img_tensor = preprocess_img("/path/to/CT.nii.gz", cuda=0)  # shape: [1,1,D,H,W]

# 2) Load model and weights
model = ModelforExtractFea(args=args)
state_dict = torch.load("./weights/model.pt", map_location="cuda:0")
model.load_state_dict(state_dict, strict=False)
model.eval().cuda()

# 3) Extract features
with torch.no_grad():
    features = model(img_tensor.cuda())   # output: [1, 1024]
print("Feature shape:", features.shape)
```

The extracted **1024-d embedding** can be directly used for:  
- Downstream classification/regression tasks (e.g., histology, EGFR prediction)  
- Multi-modal fusion with clinical/textual features  
- Transfer learning with fine-tuning  

---
