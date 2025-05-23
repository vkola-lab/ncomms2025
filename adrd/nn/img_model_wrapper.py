import torch
# from .. import nn
# from .. import model
import numpy as np
from icecream import ic
from monai.networks.nets.swin_unetr import SwinUNETR
from typing import Any
# from .conv_layers import ConvClassifierNorm

def normalization(planes, norm='bn', eps=1e-4):
    if norm == 'bn':
        m = torch.nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = torch.nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = torch.nn.InstanceNorm3d(planes, eps=eps)
    # elif norm == 'sync_bn':
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv3d(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = torch.nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = torch.nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = torch.nn.LeakyReLU(negative_slope=relufactor, inplace=True)
    
    def forward(self, x, last=False):
        x = self.conv(x)
        x = torch.clamp(x, min=-1e4, max=1e4)  # Clamp to avoid extreme values
        assert not torch.isnan(x).any(), "NaN detected after convolution & clamping"
        if not last:
            x = self.norm(x)
            assert not torch.isnan(x).any(), "NaN detected after normalization"
        x = self.activation(x)
        assert not torch.isnan(x).any(), "NaN detected after activation"
        return x

class ImagingModelWrapper(torch.nn.Module):
    def __init__(
            self,
            arch: str = 'ViTAutoEnc',
            tgt_modalities: dict | None = {},
            img_size: int | None = 128,
            patch_size: int | None = 16,
            ckpt_path: str | None = None,
            train_backbone: bool = False,
            out_dim: int = 128,
            layers: int | None = 1,
            device: str = 'cpu',
            fusion_stage: str = 'middle',
            ):
        super(ImagingModelWrapper, self).__init__()

        self.arch = arch
        self.tgt_modalities = tgt_modalities
        self.img_size = img_size
        self.patch_size = patch_size
        self.train_backbone = train_backbone
        self.ckpt_path = ckpt_path
        self.device = device
        self.out_dim = out_dim
        self.layers = layers
        self.fusion_stage = fusion_stage
        
        
        if "swinunetr" in self.arch.lower():
            if "emb" not in self.arch.lower():
                ckpt_path = '/projectnb/ivc-ml/dlteif/pretrained_models/model_swinvit.pt'
                ckpt = torch.load(ckpt_path, map_location='cpu')
                self.img_model = SwinUNETR(
                    in_channels=1,
                    out_channels=1,
                    img_size=128,
                    feature_size=48,
                    use_checkpoint=True,
                )
                ckpt["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in ckpt["state_dict"].items()}
                ic(ckpt["state_dict"].keys())
                self.img_model.load_from(ckpt)
            self.dim = 768

        elif "vit" in self.arch.lower():    
            if "emb" not in self.arch.lower():
                # Initialize image model
                self.img_model = nn.__dict__[self.arch](
                    in_channels = 1,
                    img_size = self.img_size,
                    patch_size = self.patch_size,
                )

                if self.ckpt_path:
                    self.img_model.load(self.ckpt_path, map_location=self.device)
                self.dim = self.img_model.hidden_size
            else:
                self.dim = 768

        if "vit" in self.arch.lower() or "swinunetr" in self.arch.lower():    
            dim = self.dim
            if self.fusion_stage == 'middle':
                downsample = torch.nn.ModuleList()
                # print('Number of layers: ', self.layers)
                for i in range(self.layers):
                    if i == self.layers - 1:
                        dim_out = self.out_dim
                        # dim_out = 1
                        # print(layers)
                        ks = 2
                        stride = 2
                    else:
                        dim_out = dim // 2
                        ks = 2
                        stride = 2

                    downsample.append(
                        torch.nn.Conv1d(in_channels=dim, out_channels=dim_out, kernel_size=ks, stride=stride)
                    )
                    
                    dim = dim_out
                    
                    downsample.append(
                        torch.nn.BatchNorm1d(dim)
                    )
                    downsample.append(
                        torch.nn.ReLU()
                    )
                    
                self.downsample = torch.nn.Sequential(*downsample)
                # self.linear = torch.nn.Linear(1, self.out_dim)
                
                # downsample = torch.nn.ModuleList()
                # dim = self.dim

                # for i in range(self.layers):
                #     if i == self.layers - 1:
                #         dim_out = self.out_dim
                #     else:
                #         dim_out = dim // 2
                    
                #     # Using Conv3D for spatial downsampling
                #     downsample.append(torch.nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=2, stride=1, padding=1, padding_mode='zeros', bias=True))
                #     downsample.append(torch.nn.BatchNorm3d(dim_out))
                #     downsample.append(torch.nn.ReLU())
                #     dim = dim_out
                    
                # self.downsample = torch.nn.Sequential(*downsample)
                # self.pool = torch.nn.AdaptiveAvgPool3d(1)
                
            elif self.fusion_stage == 'late':
                self.downsample = torch.nn.Identity()
            else:
                pass
            
            # print('Downsample layers: ', self.downsample)
                
        elif "densenet" in self.arch.lower():
            if "emb" not in self.arch.lower():
                self.img_model = model.ImagingModel.from_ckpt(self.ckpt_path, device=self.device, img_backend=self.arch, load_from_ckpt=True).net_
            
            self.downsample = torch.nn.Linear(3900, self.out_dim)

        # randomly initialize weights for downsample block
        for p in self.downsample.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            p.requires_grad = True
            
        if "emb" not in self.arch.lower():
            # freeze imaging model parameters
            if "densenet" in self.arch.lower():
                for n, p in self.img_model.features.named_parameters():
                    if not self.train_backbone:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                for n, p in self.img_model.tgt.named_parameters():
                    p.requires_grad = False
            else:
                for n, p in self.img_model.named_parameters():
                    # print(n, p.requires_grad)
                    if not self.train_backbone:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
    
    def forward(self, x):
        # print("--------ImagingModelWrapper forward--------")
        if "emb" not in self.arch.lower():
            if "swinunetr" in self.arch.lower():
                # print(x.size())
                out = self.img_model(x)
                # print(out.size())
                out = self.downsample(out)
                # print(out.size())
                out = torch.mean(out, dim=-1)
                # print(out.size())
            elif "vit" in self.arch.lower():
                out = self.img_model(x, return_emb=True)
                ic(out.size())
                out = self.downsample(out)
                out = torch.mean(out, dim=-1)
            elif "densenet" in self.arch.lower():
                out = torch.nn.Sequential(*list(self.img_model.features.children()))(x)
                # print(out.size())
                out = torch.flatten(out, 1)
                out = self.downsample(out)
        else:
            # print(x.size())
            if "swinunetr" in self.arch.lower():
                x = torch.squeeze(x, dim=1)
                x = x.view(x.size(0),self.dim, -1)
            # print('x: ', x.size())    
            out = self.downsample(x)
            # print('out: ', out.size())
            # out = out.view(out.size(0), out.size(1), -1)
            # print('out: ', out.size())
            if self.fusion_stage == 'middle':
                if "vit" in self.arch.lower() or "swinunetr" in self.arch.lower():
                    out = torch.mean(out, dim=-1)
                else:
                    out = torch.squeeze(out)
            elif self.fusion_stage == 'late':
                pass 
            # print('out: ', out.size())
            # out = self.linear(out)
            # print('out: ', out.size())
            
            # print('out: ', out.size())
            
            
        # x = torch.squeeze(x, dim=1)
        # # print('x: ', x.size())    
        # out = self.downsample(x)
        # # print('out: ', out.size())
        # out = self.pool(out)
        # # print('out: ', out.size())
        # out = out.view(out.size(0), out.size(1))
        # # print('out: ', out.size())

        return out


if __name__ == '__main__':
    ''' for testing purpose only '''
    model = ImagingModelWrapper(arch='SwinUNETREMB', layers=2, out_dim=256)
    data = torch.randn(10, 1, 768, 8, 8, 8)  # Example data
    out = model(data)