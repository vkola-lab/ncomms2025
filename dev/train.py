# import sys
# sys.path.append('..')
import pandas as pd
import torch
import json
import argparse
import os
import itertools
import monai
import random
import wandb
import nibabel as nib

from sklearn.model_selection import KFold, StratifiedKFold
from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from tqdm import tqdm
from collections import defaultdict
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.disable()
from torchvision import transforms
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Resized,
)

def parser():
    parser = argparse.ArgumentParser("Transformer pipeline", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--train_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--vld_path', default='/home/skowshik/ADRD_repo/adrd_tool/data/nacc_new/new_nacc_processed_revised_labels.csv', type=str,
        help='Please specify path to the validation data.')
    parser.add_argument('--cnf_file', default='/home/skowshik/ADRD_repo/adrd_tool/dev/data/toml_files/default_nacc_revised_labels.toml', type=str,
        help='Please specify path to the configuration file.')
    parser.add_argument('--img_mode', type=int, choices=[-1, 0, 1, 2])
    parser.add_argument('--img_net', type=str, choices=['ViTAutoEnc', 'ViTEMB', 'DenseNet', 'DenseNetEMB', 'SwinUNETR', 'SwinUNETREMB', 'NonImg'])
    parser.add_argument('--imgnet_ckpt', type=str, help="Path to Imaging model checkpoint")
    parser.add_argument('--fusion_stage', type=str, default="middle", help="Fusion stage of the image embeddings")
    parser.add_argument('--train_imgnet', action="store_true", help="Set to True to train imaging model along transformer.")
    parser.add_argument("--img_size", type=str, help="input size to the imaging model")
    parser.add_argument("--imgnet_layers", type=int, default=2, help="Number of layers of the downsampling block.")
    parser.add_argument("--patch_size", type=int, help="patch size")
    parser.add_argument('--ckpt_path', default='/home/skowshik/ADRD_repo/adrd_tool/dev/ckpt/revised_labels/ckpt.pt', type=str,
        help='Please specify the ckpt path')
    parser.add_argument('--load_from_ckpt', action="store_true", help="Set to True to load model from checkpoint.")
    parser.add_argument('--save_intermediate_ckpts', action="store_true", help="Set to True to save intermediate model checkpoints.")
    parser.add_argument('--wandb', action="store_true", help="Set to True to init wandb logging.")
    parser.add_argument('--balanced_sampling', action="store_true", help="Set to True for balanced sampling.")
    parser.add_argument('--ranking_loss', action="store_true", help="Set to True to apply ranking loss.")
    parser.add_argument('--parallel', action='store_true', default=False, help='Set True for DP training.')
    parser.add_argument('--d_model', default=64, type=int,
        help='Please specify the dimention of the feature embedding')
    parser.add_argument('--nhead', default=1, type=int,
        help='Please specify the number of transformer heads')
    parser.add_argument('--num_epochs', default=256, type=int,
        help='Please specify the number of epochs')
    parser.add_argument('--batch_size', default=128, type=int,
        help='Please specify the batch size')
    parser.add_argument('--lr', default=1e-4, type=float,
        help='Please specify the learning rate')
    parser.add_argument('--gamma', default=2, type=float,
        help='Please specify the gamma value for the focal loss')
    parser.add_argument('--weight_decay', default=0.0, type=float,
        help='Please specify the weight decay (optional)')
    parser.add_argument('--n_splits', default=0, type=int,
        help='Please specify k for k fold cross validation')
    parser.add_argument('--all_data', action="store_true", help="Set to True to use all data for training.")
    parser.add_argument('--stage', default=1, type=int, choices=[1, 2])
    parser.add_argument('--lambda_coeff', default=0.025, type=float, help="Ranking loss scaling factor")
    parser.add_argument('--margin', default=0.25, type=float, help="Ranking loss margin")
    parser.add_argument('--fine_tune', action="store_true", help="Set to True to search for hyperparameters.")
    parser.add_argument('--early_stop_threshold', default=15, type=int,
        help='Please specify the early stopping threshold')
    parser.add_argument('--transfer_epoch', default=15, type=int,
        help='Please specify the epochs after which transferred weights atart training')
    parser.add_argument('--wandb_project', type=str, help="The name of the wandb project")
    parser.add_argument('--device', default='cuda', type=str,
        help='Please specify cuda or cpu.')
    args = parser.parse_args()
    return args

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

args = parser()
img_size = eval(args.img_size)
print(f"Image backbone: {args.img_net}")
# print(f"Embedding path: {args.emb_path}")
if args.img_net == 'None':
    args.img_net = None
    

cuda_devices = [0]
data_parallel = False
_dataloader_num_workers = 4

save_path = '/'.join(args.ckpt_path.split('/')[:-1])
if not os.path.exists(save_path):
    os.makedirs(save_path)

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.8, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                # Resized(keys=["image"], spatial_size=img_size),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        try:
            image_data = data["image"]
            check = nib.load(image_data).get_fdata()
            if len(check.shape) > 3:
                return None
            return self.transforms(data)
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
trn_filter_transform = FilterImages(dat_type='trn')
vld_filter_transform = FilterImages(dat_type='vld')

stripped = '_stripped_MNI'
print("Loading training dataset ... ")
dat_trn = CSVDataset(dat_file=args.train_path, cnf_file=args.cnf_file, img_mode=args.img_mode, transforms=FilterImages('trn'), stripped=stripped)
print("Done.\nLoading Validation dataset ...")
dat_vld = CSVDataset(dat_file=args.vld_path, cnf_file=args.cnf_file, img_mode=args.img_mode, transforms=FilterImages('vld'), stripped=stripped)


label_fractions = dat_trn.label_fractions
# print(label_fractions)

label_distribution = {}

ckpt_path = args.ckpt_path

mdl = ADRDModel(
    src_modalities = dat_trn.feature_modalities,
    tgt_modalities = dat_trn.label_modalities,
    label_fractions = label_fractions,
    d_model = args.d_model,
    nhead = args.nhead,
    num_encoder_layers = 1,
    num_epochs = args.num_epochs,
    batch_size = args.batch_size, 
    batch_size_multiplier = 1,
    lr = args.lr,
    weight_decay = args.weight_decay,
    gamma = args.gamma,
    criterion = 'AUC (PR)',
    device = args.device,
    cuda_devices = cuda_devices,
    img_net = args.img_net,
    imgnet_layers = args.imgnet_layers,
    img_size = img_size,
    fusion_stage= args.fusion_stage,
    imgnet_ckpt = args.imgnet_ckpt,
    patch_size = args.patch_size,
    ckpt_path = ckpt_path,
    train_imgnet = args.train_imgnet,
    load_from_ckpt = args.load_from_ckpt,
    save_intermediate_ckpts = args.save_intermediate_ckpts,
    data_parallel = data_parallel,
    verbose = 4,
    wandb_ = args.wandb,
    label_distribution = label_distribution,
    ranking_loss = args.ranking_loss,
    _amp_enabled = False,
    _dataloader_num_workers = _dataloader_num_workers,
    stage = args.stage,
    lambda_coeff = args.lambda_coeff,
    margin = args.margin,
    fine_tune=args.fine_tune,
    early_stop_threshold=args.early_stop_threshold,
    transfer_epoch=args.transfer_epoch,
    wandb_project=args.wandb_project
)

# print(dat_trn.labels)

if args.img_mode == 0 or args.img_mode == 2:
    mdl.fit(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels, img_train_trans=trn_filter_transform, img_vld_trans=vld_filter_transform, img_mode=args.img_mode)
else:
    mdl.fit(dat_trn.features, dat_vld.features, dat_trn.labels, dat_vld.labels, img_train_trans=None, img_vld_trans=None, img_mode=args.img_mode)



