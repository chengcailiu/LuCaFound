from typing import Optional
import torch
import sys
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
import sys
from transformers import BertTokenizer
import torch
from safetensors.torch import load_file
import os
import tensorboard
import torch.distributed as dist
from transformers.trainer_utils import IntervalStrategy
from swinCLIP_20cls import swinCLIPConfig, swinCLIP
from multi_dataset_20clsOritext import ITRDataset
from dist_utils import get_world_size

@dataclass
class ModelArguments:
    
    version: Optional[str] = field(default="v0")
    vision_model_name: str = field(default="swin_clip")
    language_model_name_or_path: str = field(default="bert-base-uncased")
    language_model_type: str = field(default="None")
    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)
    in_channels: int = field(default=3)
    hidden_size: int = field(default=768)
    spatial_dims: int = field(default=3)
    if20clsloss: bool = field(default=True)

@dataclass
class DataArguments:
    
    data_root: str = field(default="", metadata={"help": "Root directory for all data."})
    cap_data_path: str = field(default="", metadata={"help": "Path to caption data."})
    max_length: int = field(default=512)
    ifclsoridata: bool = field(default=True, metadata={"help": "Use 20-class label text to derive multi-label supervision together with the original report text."})
    testdatakey: str = field(default="test")
    img_size: str = field(default="48,256,256")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    remove_unused_columns: bool = False

    ddp_backend: Optional[str] = None
    ddp_find_unused_parameters: bool = True
    fp16: bool = False
    bf16: bool = True
    output_dir: str = "./outputs"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32

    evaluation_strategy: IntervalStrategy = IntervalStrategy.STEPS
    eval_steps: int = 20
    save_strategy: IntervalStrategy = IntervalStrategy.STEPS
    save_steps: int = 1000

    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20

    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    resume_from_checkpoint: Optional[str] = None
    max_grad_norm: float = 1.0


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        preds = torch.argmax(logits[0], dim=-1)
        preds2 = torch.argmax(logits[1], dim=-1)
        return preds
    else:
        print("no tuple")
        print("logits.shape", logits.shape)
        preds = torch.argmax(logits, dim=-1)



@dataclass
class DataCollator:
    def __init__(self, gather_all, data_args=None):
        self.gather_all = gather_all
        self.data_args = data_args
    

    def __call__(self, batch: list) -> dict:

        
        if self.data_args.ifclsoridata:
            images, texts, input_ids, attention_mask, cls_gold ,trainortest,image_paths,text_paths,ori_text_paths = tuple(
                [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask','cls_gold', 'trainortest','image_path','text_path','ori_text_path'))
            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            # print("input_ids.shape", input_ids.shape)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
            cls_gold = torch.cat([_.unsqueeze(0) for _ in cls_gold], dim=0)
            trainortest = [_ for _ in trainortest]
            image_paths = [_ for _ in image_paths]
            text_paths = [_ for _ in text_paths]
            ori_text_paths = [_ for _ in ori_text_paths]


            batch_size = images.shape[0]
            if self.gather_all:
                world_size = get_world_size()
                batch_size *= world_size
            
            labels = torch.arange(batch_size, device=images.device, dtype=torch.long)
            return_dict = dict(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cls_gold=cls_gold,
                trainortest=trainortest,
                image_paths=image_paths,
                text_paths=text_paths,
                ori_text_paths=ori_text_paths,
                
            )


        else:

            images, texts, input_ids, attention_mask,trainortest = tuple(
                [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask','trainortest'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            # print("input_ids.shape", input_ids.shape)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
            trainortest = [_ for _ in trainortest]

            batch_size = images.shape[0]
            if self.gather_all:
                world_size = get_world_size()
                batch_size *= world_size

            labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                trainortest=trainortest,

            )

        return return_dict




def main():


    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.ifclsoridata = data_args.ifclsoridata
    data_args.in_channels = model_args.in_channels
    data_args.img_size = [int(i) for i in data_args.img_size.split(',')]
    model_args.img_size = data_args.img_size
    data_args.vision_model_name = model_args.vision_model_name
    data_args.language_model_type = model_args.language_model_type

    if model_args.vision_model_name == 'swin_clip':
        config = swinCLIPConfig.from_dict(vars(model_args))
        model = swinCLIP(config,args=model_args)
    else:
        raise ValueError('please set right model')


    try:
        tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)
    except:
        tokenizer = model.tokenizer

    
    train_dataset = ITRDataset(data_args, tokenizer, mode='train') 
    eval_dataset = ITRDataset(data_args, tokenizer, mode=data_args.testdatakey)
        

    if model_args.gather_loss and not model_args.local_loss and get_world_size() > 1:
        gather_all = True
    else:
        gather_all = False


    data_collator = DataCollator(gather_all, data_args)


    trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics,
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                    )
    print(len(train_dataset), len(eval_dataset),"train_dataset, eval_dataset")

    

    
    if training_args.resume_from_checkpoint:
        print(f"resuming from checkpoint:{training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        print("resumed training")
    else:
        trainer.train()

    trainer.save_state()
    model.config.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))


if __name__ == "__main__":

    main()
