import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.video import Swin3D_B_Weights
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import tempfile
import urllib.request

def load_safetensors_from_path_or_url(path_or_url: str):

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        # 下载到临时文件
        tmpdir = tempfile.mkdtemp()
        filename = os.path.basename(path_or_url)
        local_path = os.path.join(tmpdir, filename)
        print(f"Downloading pretrained weights from {path_or_url} ...")
        urllib.request.urlretrieve(path_or_url, local_path)
        print(f"Downloaded to {local_path}")
        return torch.load(local_path, map_location="cpu")
    else:
        # 本地文件
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Pretrained weight file not found: {path_or_url}")
        return torch.load(path_or_url, map_location="cpu")


class ModelforPretrain(nn.Module):
    def __init__(self, args=None):
        super(ModelforPretrain, self).__init__()
        weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        self.model =  torchvision.models.video.swin3d_b(num_classes=400, weights=weights)
        self.model.head = nn.Identity()
        self.args = args
            
        
        if args.pretrained:
                
            ckpt0 = load_safetensors_from_path_or_url(self.args.pretrained)
                
            self.model.load_state_dict(ckpt0, strict=True)

    def forward(self, x):
      
        x = self.model(x)
        
        return x




class ModelforExtractFea(nn.Module):
    def __init__(self, num_classes=2, args=None):
        super(ModelforExtractFea, self).__init__()
        self.args = args
        self.model =  torchvision.models.video.swin3d_b(num_classes=400)
        ## 去掉model中的head这个key
        self.model.head = nn.Identity()
        self.head = nn.Linear(in_features=400, out_features=num_classes, bias=True)
        

        if args.pretrained:

            ckpt0 = load_safetensors_from_path_or_url(self.args.pretrained)
            self.model.load_state_dict(ckpt0, strict=True)
            print(f"load pretrained model done from {self.args.pretrained}")



    def forward(self, x):
        
        x = self.model(x)

        if self.args.save_feature:
            fea = x.detach().clone()
            return fea
        else:
            x = self.head(x)
            return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## pretrained model
    parser.add_argument('--pretrained', type=str, default='/data1/lcc/log/CLIP/downstream_task/20241219mae2/10上传github/model.pt', help='pretrained model path')
    parser.add_argument('--save_feature', type=bool, default=True, help='save feature')
    args = parser.parse_args()


    # model = ModelforExtractFea(args=args)
    model = ModelforPretrain(args=args)
    x = torch.randn(1, 3, 48, 256, 256)
    out = model(x)
    print(out.shape)



