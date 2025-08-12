import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models.video import Swin3D_B_Weights
from safetensors.torch import load_file
import warnings

class ModelforPretrain(nn.Module):
    def __init__(self, args=None):
        super(ModelforPretrain, self).__init__()
        weights=Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
        self.model =  torchvision.models.video.swin3d_b(num_classes=400, weights=weights)
        self.model.head = nn.Identity()
            
        if args:
            self.args = args
            if args.my_pretrained:
                print("load pretrained model")
                new_dict = {}
                ckpt0 = load_file(self.args.my_pretrained)
                for k,v in ckpt0.items():
                    if 'vision_encoder.model.' in k:
                        k = k.replace('vision_encoder.model.','')
                        new_dict[k] = v
                    else:
                        pass
                
                newdict2 = {}
                for k,v in self.model.state_dict().items():
                    if k in new_dict.keys():
                        newdict2[k] = new_dict[k]
                    else:
                        newdict2[k] = v
                        print(k,"not in self.model.state_dict().keys()")
                self.model.load_state_dict(newdict2, strict=True)

    def forward(self, x):
      
        x = self.model(x)
        
        return x
