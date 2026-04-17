# LuCaFound Pretraining
This README provides instructions for the pretraining stage of LuCaFound.

Official code release for the paper:

**LuCaFound: A Disease-Specific Vision-Language Foundation Model for Comprehensive Clinical Assessment in Lung Cancer**


## What Is Included

- the main pretraining entrypoint
- model definitions and dataset logic
- an example dataset manifest
- a one-command smoke test with synthetic data
- single-process debugging support and distributed-training compatibility

## Repository Structure

```text
pretrain/
|-- train_CLIP.py                  # Training entrypoint
|-- swinCLIP_20cls.py              # Vision-language model definition
|-- swin3d.py                      # 3D Swin image encoder wrapper
|-- multi_dataset_20clsOritext.py  # Dataset and preprocessing logic
|-- dist_utils.py                  # Distributed utilities
|-- dig_abnormal.py               # LLM-based preprocessing for 20-class texts
|-- bert-base-uncased/             # Local BERT checkpoint directory (optional)
|-- requirements.txt               # Python dependencies
|-- example_caption_data.json      # Example dataset manifest format
|-- smoke_test.py                  # One-command synthetic pipeline check
```

## Environment

Recommended Python version: `3.10` or newer.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you do not have a local `bert-base-uncased/` folder, the code defaults to the Hugging Face model name `bert-base-uncased` and will download it automatically when needed.

## Data Preparation

The recommended preprocessing order is:

1. Prepare original report text files.
2. Run `dig_abnormal.py` to convert original reports into 20-class finding text files.
3. Build the JSON manifest with `image`, `label_text`, and `report_text`.

### Step 1: Data Preprocessing (`LuCaFound/data_preprocess.py` or `./data_preprocess.py`)

You can preprocess your CT image data based on the CT data preprocessing code from the parent directory.


### Step 2: Generate 20-Class Finding Text (`dig_abnormal.py`)

`dig_abnormal.py` uses an LLM API to convert each original report into a fixed-order 20-class finding summary. This generated file is used as `label_text` during pretraining.

Example:

```bash
python dig_abnormal.py \
  --input_dir /path/to/raw_reports \
  --output_dir /path/to/generated_label_texts \
  --model deepseek-chat \
  --skip_existing
```

API configuration:

- set `OPENAI_API_KEY` in the environment, or pass `--api_key`
- optionally set `OPENAI_BASE_URL` for OpenAI-compatible providers

Expected input and output naming:

- input: original report text such as `case_0001_text_ori.txt`
- output: generated 20-class text such as `case_0001_text_anomal.txt`

### Step 3: Build the Training Manifest (`example_caption_data.json`)

Training expects:

1. A data root directory containing CT volumes and report text files.
2. A JSON manifest describing train/validation/test splits.

See `example_caption_data.json` for the expected format.

Each sample should provide:

- `image`: relative path to the CT volume file under `data_root`
- `label_text`: relative path to the 20-class findings text under `data_root`
- `report_text`: relative path to the original report text under `data_root`

Supported image formats in the current dataset loader:

- `.npy`
- `.nii`
- `.nii.gz`
- `.pt`

## Training (`train_CLIP.py`)

Example command:

```bash
python train_CLIP.py \
  --data_root /path/to/data \
  --cap_data_path /path/to/caption_data.json \
  --output_dir ./outputs/lucafound_pretrain \
  --language_model_name_or_path bert-base-uncased \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 10
```

By default, training now assumes the intended LuCaFound pretraining setup: original report text plus 20-class supervision labels.

### Local Debugging

The training code now supports single-process local debugging out of the box. If distributed training is not initialized, the contrastive loss automatically falls back to local batch computation.

### Multi-GPU Training

For multi-GPU runs, launch distributed training with your preferred launcher and keep `gather_loss=True` so image-text contrastive pairs are aggregated across processes.

Important arguments:

- `--data_root`: root directory for images and text files
- `--cap_data_path`: JSON manifest for dataset splits
- `--output_dir`: directory to save checkpoints and tokenizer
- `--language_model_name_or_path`: local BERT directory or Hugging Face model name
- `--img_size`: input CT size in `depth,height,width` format
- `--ifclsoridata`: enabled by default; keeps the 20-class auxiliary supervision active

## Smoke Test (`smoke_test.py`)

You can run a minimal end-to-end check with synthetic data:

```bash
python smoke_test.py
```

This script will:

- generate synthetic CT volumes and report files
- validate dataset loading and batch collation
- run a single model forward pass
- run a one-step CPU training job through `train_CLIP.py`

Artifacts are written under `.smoke_test_artifacts/` by default.

The smoke test follows the same logic as the main pretraining task:

- `label_text` provides the 20-class supervision labels
- `report_text` provides the original report text for text encoding

## Contact

If you use this repository for academic research, please cite the corresponding paper when it is available. If you have any questions or need assistance, please contact out to us (shuo_wang@buaa.edu.cn or chengcailiu@buaa.edu.cn).
