import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import BertTokenizer

from multi_dataset_20clsOritext import ITRDataset
from swinCLIP_20cls import swinCLIP, swinCLIPConfig
from train_CLIP import DataCollator


LABEL_KEYS = [
    "bronchiectasis",
    "incomplete thymic involution",
    "localized pleural thickening",
    "low-density shadow in thyroid",
    "thickening of adrenal gland",
    "liver calcification",
    "thyroid nodules",
    "pericardial effusion",
    "old lesions in lungs",
    "aortic and coronary artery calcification",
    "fatty liver",
    "pulmonary emphysema",
    "pulmonary bullae",
    "calcification of mediastinal lymph nodes",
    "pulmonary infection or inflammation",
    "liver cysts",
    "gallstones",
    "renal cysts or stones",
    "pulmonary glass-ground nodules",
    "pulmonary micronodules",
]


def build_synthetic_dataset(work_dir: Path, shape: tuple[int, int, int]) -> tuple[Path, Path]:
    data_dir = work_dir / "synthetic_data"
    img_dir = data_dir / "images"
    text_dir = data_dir / "reports"
    img_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for index in range(3):
        arr = rng.normal(loc=-600, scale=250, size=shape).astype("float32")
        arr = np.clip(arr, -1500, 500)
        np.save(img_dir / f"case_{index:04d}.npy", arr)

        labels = []
        for label_index, key in enumerate(LABEL_KEYS):
            state = "positive" if (index + label_index) % 2 == 0 else "negative"
            labels.append(f"{key}-{state}")

        anomal_text = ". ".join(labels) + "."
        original_text = (
            "Synthetic chest CT report for smoke testing. "
            "No patient data is included. "
            "This file only validates data loading and tokenization."
        )

        (text_dir / f"case_{index:04d}_text_anomal.txt").write_text(anomal_text, encoding="utf-8")
        (text_dir / f"case_{index:04d}_text_ori.txt").write_text(original_text, encoding="utf-8")

    manifest = {
        "train": [
            {
                "image": "images/case_0000.npy",
                "label_text": "reports/case_0000_text_anomal.txt",
                "report_text": "reports/case_0000_text_ori.txt",
            },
            {
                "image": "images/case_0001.npy",
                "label_text": "reports/case_0001_text_anomal.txt",
                "report_text": "reports/case_0001_text_ori.txt",
            },
        ],
        "validation": [
            {
                "image": "images/case_0002.npy",
                "label_text": "reports/case_0002_text_anomal.txt",
                "report_text": "reports/case_0002_text_ori.txt",
            },
        ],
        "test": [
            {
                "image": "images/case_0002.npy",
                "label_text": "reports/case_0002_text_anomal.txt",
                "report_text": "reports/case_0002_text_ori.txt",
            },
        ],
    }
    manifest_path = work_dir / "synthetic_caption_data.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_dir, manifest_path


def run_dataset_and_forward_checks(repo_root: Path, data_dir: Path, manifest_path: Path, img_size: list[int]) -> None:
    language_model_name_or_path = str(repo_root / "bert-base-uncased")
    if not (repo_root / "bert-base-uncased").exists():
        language_model_name_or_path = "bert-base-uncased"

    args = SimpleNamespace(
        version="v0",
        vision_model_name="swin_clip",
        language_model_name_or_path=language_model_name_or_path,
        language_model_type="None",
        gather_loss=False,
        local_loss=False,
        in_channels=3,
        hidden_size=768,
        spatial_dims=3,
        if20clsloss=True,
        ifclsoridata=True,
        data_root=str(data_dir),
        cap_data_path=str(manifest_path),
        max_length=64,
        testdatakey="validation",
        img_size=img_size,
    )

    tokenizer = BertTokenizer.from_pretrained(args.language_model_name_or_path)
    dataset = ITRDataset(args, tokenizer, mode="train")
    collator = DataCollator(gather_all=False, data_args=args)

    item0 = dataset[0]
    assert tuple(item0["image"].shape) == (3, *img_size), f"Unexpected image shape: {item0['image'].shape}"
    assert tuple(item0["cls_gold"].shape) == (20,), f"Unexpected cls_gold shape: {item0['cls_gold'].shape}"

    batch = collator([dataset[0], dataset[1]])
    assert tuple(batch["images"].shape) == (2, 3, *img_size), f"Unexpected batch image shape: {batch['images'].shape}"
    assert tuple(batch["cls_gold"].shape) == (2, 20), f"Unexpected batch label shape: {batch['cls_gold'].shape}"

    config = swinCLIPConfig.from_dict(vars(args))
    model = swinCLIP(config, args=args).eval()
    with torch.no_grad():
        out = model(**batch)

    assert "loss" in out, "Model forward did not return loss."
    assert tuple(out["logits"].shape) == (2, 2), f"Unexpected logits shape: {out['logits'].shape}"
    print("DATASET_AND_FORWARD_OK")


def run_one_step_training(repo_root: Path, data_dir: Path, manifest_path: Path, img_size: list[int]) -> None:
    output_dir = repo_root / "synthetic_run"
    language_model_name_or_path = str(repo_root / "bert-base-uncased")
    if not (repo_root / "bert-base-uncased").exists():
        language_model_name_or_path = "bert-base-uncased"
    command = [
        sys.executable,
        "train_CLIP.py",
        "--data_root",
        str(data_dir),
        "--cap_data_path",
        str(manifest_path),
        "--output_dir",
        str(output_dir),
        "--language_model_name_or_path",
        language_model_name_or_path,
        "--img_size",
        ",".join(str(dim) for dim in img_size),
        "--ifclsoridata",
        "--no_gather_loss",
        "--use_cpu",
        "--no_bf16",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--max_steps",
        "1",
        "--logging_steps",
        "1",
        "--eval_strategy",
        "no",
        "--save_strategy",
        "no",
        "--dataloader_num_workers",
        "0",
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    assert output_dir.exists(), "Training output directory was not created."
    print("ONE_STEP_TRAIN_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal smoke test for the LuCaFound training pipeline.")
    parser.add_argument("--work_dir", type=str, default=".smoke_test_artifacts", help="Directory used for synthetic data and outputs.")
    parser.add_argument("--img_size", type=str, default="16,32,32", help="Synthetic CT size as depth,height,width.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    work_dir = (repo_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    img_size = [int(dim) for dim in args.img_size.split(",")]

    data_dir, manifest_path = build_synthetic_dataset(work_dir, tuple(img_size))
    run_dataset_and_forward_checks(repo_root, data_dir, manifest_path, img_size)
    run_one_step_training(repo_root, data_dir, manifest_path, img_size)
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
