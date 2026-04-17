import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import tqdm
import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta
import SimpleITK as sitk
import cv2


LABEL_KEYS = [
    'bronchiectasis',
    'incomplete thymic involution',
    'localized pleural thickening',
    'low-density shadow in thyroid',
    'thickening of adrenal gland',
    'liver calcification',
    'thyroid nodules',
    'pericardial effusion',
    'old lesions in lungs',
    'aortic and coronary artery calcification',
    'fatty liver',
    'pulmonary emphysema',
    'pulmonary bullae',
    'calcification of mediastinal lymph nodes',
    'pulmonary infection or inflammation',
    'liver cysts',
    'gallstones',
    'renal cysts or stones',
    'pulmonary glass-ground nodules',
    'pulmonary micronodules',
]

def resize3D(img, size, interpolation=cv2.INTER_CUBIC):

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





def one_hot_encode(tensor):
    num_classes = int(tensor.max() + 1)
    # 创建一个全零的张量，形状为 (tensor.shape[0], num_classes)
    one_hot = torch.zeros(tensor.shape[0], num_classes)
    # 使用 scatter_ 函数将相应位置置为 1
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot




class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", use_weighted_attention_mask=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.windows = [[-1500,500]]
        self.use_weighted_attention_mask = use_weighted_attention_mask
        self.args.img_size = args.img_size

        self.cnt = 0

        with open(args.cap_data_path, 'r', encoding='utf-8') as file:
            self.json_file = json.load(file)
        
        self.data_list = self.json_file[mode]

    

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                
                mtf.ToTensor(dtype=torch.float),
            ]
        )



        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list
        elif "validation" in mode:
            self.transform = val_transform
            self.data_list = self.data_list
        elif "val" in mode:
            self.transform = val_transform
            self.data_list = self.data_list
        elif 'test' in mode:
            self.data_list = self.data_list
            self.transform = val_transform
        else:
            raise ValueError(f"{mode} is not a valid mode")

   
    def resolve_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.data_root, path_value)

    def get_ori_text(self, textpath):
        candidates = []

        if textpath.endswith("_text_anomal.txt"):
            candidates.extend([
                textpath.replace("_text_anomal.txt", "_text_ori.txt"),
                textpath.replace("_text_anomal.txt", "_ori.txt"),
                textpath.replace("_text_anomal.txt", "_text.txt"),
                # Backward compatibility for older exported filenames.
                textpath.replace("_text_anomal.txt", "_text_text_ori.txt"),
            ])

        if "_anomal" in textpath:
            candidates.extend([
                textpath.replace("_anomal", "_ori"),
                textpath.replace("_anomal", "_text_ori"),
                textpath.replace("_anomal", "_text"),
            ])

        seen = set()
        ordered_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                ordered_candidates.append(candidate)
                seen.add(candidate)

        for candidate in ordered_candidates:
            if os.path.exists(candidate):
                return candidate

        raise ValueError(
            f"Original text file not found for {textpath}. Tried: {ordered_candidates}"
        )

    def get_label_and_report_paths(self, data):
        label_text_rel = data.get("label_text", data.get("text"))
        if not label_text_rel:
            raise ValueError("Each sample must contain `label_text` or backward-compatible `text`.")

        label_text_abs_path = self.resolve_path(label_text_rel)

        report_text_rel = data.get("report_text", data.get("ori_text"))
        if report_text_rel:
            report_text_abs_path = self.resolve_path(report_text_rel)
        else:
            report_text_abs_path = self.get_ori_text(label_text_abs_path)

        return label_text_abs_path, report_text_abs_path
        

    def __len__(self):
        return len(self.data_list)


    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')
        

        random.shuffle(sentences)

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)

        if count_tokens(truncated_text) > max_tokens:
            truncated_text = self.truncate_text(input_text, max_tokens)

        return truncated_text


    def text_2_cls_label(self, text):
        text = str(text).lower()
        label = []
        for key in LABEL_KEYS:
            if key+'-positive' in text.lower():
                label.append(1)
            elif key+'-negative' in text.lower():
                label.append(0)
            else:
                raise ValueError(f"{key} is not found in {text}")
        label = torch.tensor(label).float()
        return label


    def __getitem__(self, idx):
        max_attempts = 100

        for _ in range(max_attempts):

            try:
                
                data = self.data_list[idx]
                image_path = data["image"]


                image_abs_path = os.path.join(self.data_root, image_path)
                window = self.windows[0]
                windowmin = window[0]
                windowmax = window[1]
                
                if image_abs_path.endswith('.npy'):
                    image = np.load(image_abs_path)
                    image = np.clip(image, windowmin, windowmax) - windowmin
                    max = image.max()
                    if max == 0.:
                        print(image_abs_path)
                        raise ValueError("The image is all zero !")
                    image = image / max
                    if image.shape != self.args.img_size:
                        image = resize3D(image, self.args.img_size)
                    image = np.array(image)[np.newaxis, ...]

                elif image_abs_path.endswith('.nii.gz') or image_abs_path.endswith('.nii'):
                    image = sitk.ReadImage(image_abs_path)
                    image = sitk.GetArrayFromImage(image)
                    image = np.clip(image, windowmin, windowmax) - windowmin
                    max = image.max()
                    if max == 0.:
                        print(image_abs_path)
                        raise ValueError("The image is all zero !")
                    image = image / max
                    if image.shape != self.args.img_size:
                        image = resize3D(image, self.args.img_size)
                    image = image[np.newaxis, ...]

                elif image_abs_path.endswith('.pt'):
                    image = torch.load(image_abs_path)
                    image = np.clip(image, windowmin, windowmax) - windowmin
                    max = image.max()
                    if max == 0.:
                        print(image_abs_path)
                        raise ValueError("The image is all zero !")
                    image = image / max
                    if image.shape != self.args.img_size:
                        image = resize3D(image, self.args.img_size)
                    image = np.array(image)[np.newaxis, ...]
                else:
                    raise ValueError("The image format is not supported !: {}".format(image_abs_path))
                
                
                image = self.transform(image)
                
                label_text_abs_path, ori_text_abs_path = self.get_label_and_report_paths(data)

                with open(label_text_abs_path, 'r', encoding='utf-8') as text_file:
                    raw_text = text_file.read()
                
                with open(ori_text_abs_path, 'r', encoding='utf-8') as ori_text_file:
                    ori_raw_text = ori_text_file.read()

                text = self.truncate_text(ori_raw_text, self.args.max_length)
                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]
                trainortest = self.mode
                if self.args.in_channels == 3:
                    image = image.repeat(3,1,1,1)
                
                if self.args.ifclsoridata:
                    cls_gold = self.text_2_cls_label(raw_text)
                    ret = {
                        'image': image,
                        'text': text,
                        'input_id': input_id,
                        'attention_mask': attention_mask,
                        'cls_gold': cls_gold,
                        'trainortest': trainortest,
                        'image_path': image_abs_path,
                        'text_path': label_text_abs_path,
                        'ori_text_path': ori_text_abs_path
                    }


                else:
                    ret = {
                        'image': image,
                        'text': text,
                        'input_id': input_id,
                        'attention_mask': attention_mask,
                        'trainortest': trainortest,
                    }
                    
                    
                return ret

            except Exception as e:
                
                print(f"Error in __getitem__ at index {idx}: {e}, {self.data_list[idx]}")
                idx = random.randint(0, len(self.data_list) - 1)

        raise RuntimeError(f"Failed to load a valid sample after {max_attempts} attempts.")
