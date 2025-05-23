# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import datetime
import monai
import adrd.utils.misc

from tqdm import tqdm
from matplotlib.pyplot import figure
from torchvision import transforms
from icecream import ic
ic.disable()
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
from sklearn.metrics import roc_auc_score, roc_curve, auc

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics


# %%
from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', size=18)
plt.rcParams['font.family'] = 'Arial'

# Save model generated probabilities
def save_predictions(dat_tst, scores_proba, scores, save_path=None, filename=None, if_save=True):
    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
    # mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in dat_file.columns}
    
    scores_proba_ = {f'{k}_prob': [smp[k] if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in dat_file.columns}
    scores_ = {f'{k}_logit': [smp[k] if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in dat_file.columns}
    
    
    # scores_proba_ = {f'{k}_prob': [round(smp[k], 3) for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in dat_file.columns}
    # scores_ = {f'{k}_logit': [round(smp[k], 3) for i, smp in enumerate(scores)] for k in scores[0] if k in dat_file.columns}
    
    
    ids = dat_file['ID']
    cohort = dat_file['COHORT']

    y_true_df = pd.DataFrame(y_true_)
    scores_df = pd.DataFrame(scores_)
    scores_proba_df = pd.DataFrame(scores_proba_)
    if 'cdr_CDRGLOB' in dat_file:
        cdr = dat_file['cdr_CDRGLOB']
        cdr_df = pd.DataFrame(cdr)
        
    id_df = pd.DataFrame(ids)
    cohort_df = pd.DataFrame(cohort)
    if 'cdr_CDRGLOB' in dat_file:
        df = pd.concat([id_df, y_true_df, scores_proba_df, cdr_df, cohort_df], axis=1)
        
    if if_save:
        df.to_csv(save_path + filename, index=False)
        
    return df

def generate_performance_report(dat_tst, y_pred, scores_proba):
    y_true = [{k:int(v) if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
    y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
    scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
    mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

    met = {}
    for k in dat_tst.labels[0].keys():
        print('Performance metrics of {}'.format(k))
        metrics = get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
        print_metrics(metrics)

        met[k] = metrics
        print(k)
        print(met[k]['Confusion Matrix'])
        met[k].pop('Confusion Matrix')
        # met[k].pop('NPV')

    return met
    
def roc_auc_scores(y_true, y_pred, features):

    tpr = dict()
    fpr = dict()
    auc_scores = dict()
    thresholds = dict()
        
    for i, fea in enumerate(features):
        y_true_ = np.array(y_true[:, i])
        y_pred_ = np.array(y_pred[:, i])
        mask = np.array([1 if not np.isnan(k) else 0 for k in y_true_])
        masked_y_true = y_true_[np.where(mask == 1)]
        masked_y_pred = y_pred_[np.where(mask == 1)]
        print(len(masked_y_true), len(masked_y_pred))
        print(round(roc_auc_score(masked_y_true, masked_y_pred), 4))
        fpr[fea], tpr[fea], thresholds[fea] = roc_curve(y_true=masked_y_true, y_score=masked_y_pred, pos_label=1, drop_intermediate=True)
        auc_scores[fea] = auc(fpr[fea], tpr[fea])

    return fpr, tpr, auc_scores, thresholds

def generate_predictions_for_data_file(dat_file, vld_file, cnf_file, img_mode,
 labels, tst_filter_transform=None):
    # initialize datasets
    seed = 0
    print('Done.\nLoading testing dataset ...')
    dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, img_mode=img_mode, stripped='_stripped_MNI')

    print('Done.\nLoading validation dataset ...')
    dat_vld = CSVDataset(dat_file=vld_file, cnf_file=cnf_file, img_mode=1, stripped='_stripped_MNI')
    print('Done.')
    
    # generate model predictions for validation set
    scores_vld, scores_proba_vld, y_pred_vld, _ = mdl.predict(dat_vld.features, _batch_size=128, img_transform=None)

    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_vld.labels]
    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in vld_file.columns}
    scores_proba_ = {f'{k}_prob': [smp[k] if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba_vld)] for k in scores_proba_vld[0] if k in vld_file.columns}
    y_true_df = pd.DataFrame(y_true_)
    scores_proba_df = pd.DataFrame(scores_proba_)
    df = pd.concat([y_true_df, scores_proba_df], axis=1)
    y_true_ar =  np.array(df[[f'{lab}_label' for lab in labels]])
    scores_proba_ar = np.array(df[[f'{lab}_prob' for lab in labels]])

    # generate fpr, tpr and thresholds for calidation set to calculate Youden index
    fpr, tpr, auc_scores, thresholds = roc_auc_scores(y_true=y_true_ar, y_pred=scores_proba_ar, features=labels)
    print('Done.')
    
    # generate model predictions
    print('Generating model predictions')
    
    scores, scores_proba, y_pred, outputs = mdl.predict(dat_tst.features, _batch_size=128, fpr=fpr, tpr=tpr, thresholds=thresholds, img_transform=tst_filter_transform)
    
    # save model predictions
    save_predictions(dat_tst, scores_proba, scores, save_path=save_path, filename=f'{fname}.csv', if_save=if_save)
    print('Done.')
    
    print('Generating performance reports')
    met = generate_performance_report(dat_tst, y_pred, scores_proba)
    
    return outputs, met
    
    
def generate_predictions_for_case(case_dict):
    return  mdl.predict(x=[test_case], _batch_size=1, img_transform=None)

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
                CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.7, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
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
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(128*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        image_data = data["image"]
        try:
            return self.transforms(data)
        except Exception as e:
            print(f"Error processing image: {image_data}{e}")
            return None


#%%
if __name__ == '__main__':
    fname = 'name_of_file' # the name of file you want to save the model predictions to
    save_path = f'path/to/model/predictions' # the path where the model predictions will be saved
    # dat_file = 'path/to/test/data' # the test data path
    # dat_file = 'path/to/vld/data' # the validation data path to caculate Youden Index
    dat_file = "/projectnb/vkolagrp/varuna/mri_pet/adrd_tool/data_varuna/data/0225/ADNI_HABS_NACCTEST_HARMONIZED.csv"
    vld_file = "/projectnb/vkolagrp/skowshik/pet_project/mri_pet/adrd_tool/data_varuna/data/0225/val_0225_new_harmonization.csv"

    # Uncomment this for stage 1
    # cnf_file = f'./data/toml_files/stage_1.toml' # the path configuration file
    # ckpt_path = '../ckpt/model_stage_1.ckpt'
    # labels = ['amy_label', 'tau_label']

    # Uncomment this for stage 2
    cnf_file = f'./data/toml_files/stage_2.toml' # the path configuration file
    ckpt_path = '../ckpt/model_stage_2.ckpt' # the path to the model checkpoint
    labels = ['tau_medtemp_label', 'tau_lattemp_label','tau_medpar_label', 'tau_latpar_label', 'tau_front_label', 'tau_occ_label']

    dat_file = pd.read_csv(dat_file)
    vld_file = pd.read_csv(vld_file)
    if "tau_front_label" in dat_file:
        dat_file = dat_file[~dat_file['tau_front_label'].isna()].reset_index(drop=True)
    
    if "tau_front_label" in vld_file:
        vld_file = vld_file[~vld_file['tau_front_label'].isna()].reset_index(drop=True)
        
    print(dat_file)
    
    save_path = "./"
    fname = "test2"
    if_save = False
    

    # uncommment this to run without image embeddings
    # img_net="NonImg"
    # img_mode=-1

    img_net="SwinUNETREMB"
    img_mode=1

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # load saved Transformer
    device = 'cuda:0'
    mdl = ADRDModel.from_ckpt(ckpt_path, device=device) 
    print("All keys matched")
    print(f"Epoch: {torch.load(ckpt_path)['epoch']}")

    #%%
    if img_mode in [0,2]:
        tst_filter_transform = FilterImages(dat_type='tst')
    else:
        tst_filter_transform = None
        
    df_pred, met = generate_predictions_for_data_file(dat_file, vld_file, cnf_file, img_mode, labels, tst_filter_transform)
    
    #%%
    met_df = pd.DataFrame(met)
    met_df = met_df[labels]
    print(met_df.round(2))


# %%
