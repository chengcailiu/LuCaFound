import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
import sys
from dist_utils import gather_features
import sys
from swin3d import Swin3DforPretrain
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
import time
import os
import json
import h5py


class swinCLIPConfig(PretrainedConfig):
    model_type = "swin_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "bert-base-uncased",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path

        self.hidden_size = hidden_size
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        super().__init__(**kwargs)




class swinCLIP(PreTrainedModel):
    config_class = swinCLIPConfig

    def __init__(self, config, args=None):
        super().__init__(config)
        
        self.vision_encoder = Swin3DforPretrain(pretrain_mode='clip')

        self.args = args

        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)
        self.mm_vision_proj = nn.Linear(1024, config.hidden_size)
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss
        self.cls_linear = nn.Linear(config.hidden_size, 20)

    def should_gather_loss(self):
        return self.gather_loss and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1



    def encode_image(self, image):
        image_feats = self.vision_encoder(image)
        image_feats = self.mm_vision_proj(image_feats)
        image_feats = F.normalize(image_feats, dim=-1)
        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)
        return text_feats


    def forward(self, images, input_ids, attention_mask, labels, trainortest, cls_gold, image_paths,text_paths,ori_text_paths, **kwargs):

        if self.args.if20clsloss == False:
            image_features = self.encode_image(images)
            text_features = self.encode_text(input_ids, attention_mask)[:, 0]
            print(image_features.shape, text_features.shape)
                    
            if self.should_gather_loss():
                all_image_features, all_text_features = gather_features(image_features, text_features)
                

                if self.local_loss:
                    logits_per_image = self.logit_scale * image_features @ all_text_features.T
                    logits_per_text = self.logit_scale * text_features @ all_image_features.T
                else:
                    logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
            else:
                logits_per_image = self.logit_scale * image_features @ text_features.T
                logits_per_text = self.logit_scale * text_features @ image_features.T


            loss = ( F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels) ) / 2.


            ret = {
                "loss": loss,
                "logits": (logits_per_image + logits_per_text) / 2.0,
                "labels": labels,
                "cls_gold": cls_gold,
            }




        else:
            image_features = self.encode_image(images)
            text_features = self.encode_text(input_ids, attention_mask)[:, 0]
            image_cls = self.cls_linear(image_features)
            pred = F.sigmoid(image_cls)
            loss_cls = F.binary_cross_entropy_with_logits(image_cls, cls_gold) * 4.

            
            if self.should_gather_loss():
                all_image_features, all_text_features = gather_features(image_features, text_features)
                
                if self.local_loss:
                    logits_per_image = self.logit_scale * image_features @ all_text_features.T
                    logits_per_text = self.logit_scale * text_features @ all_image_features.T
                else:
                    logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
            else:
                logits_per_image = self.logit_scale * image_features @ text_features.T
                logits_per_text = self.logit_scale * text_features @ image_features.T

            loss = (
                            F.cross_entropy(logits_per_image, labels) +
                            F.cross_entropy(logits_per_text, labels) +
                            loss_cls
                    ) / 3

            print('logits_per_image:', logits_per_image.shape)
            print('logits_per_text:', logits_per_text.shape)

            ret = {
                "loss": loss,
                "logits": (logits_per_image + logits_per_text) / 2.0,
                "labels": labels,
                "cls_gold": cls_gold,
                "image_cls": image_cls,
            }


            
        return ret

AutoConfig.register("swin_clip", swinCLIPConfig)
AutoModel.register(swinCLIPConfig, swinCLIP)
