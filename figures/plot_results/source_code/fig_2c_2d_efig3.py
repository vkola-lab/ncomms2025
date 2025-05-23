
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
import toml
import scipy
import pickle

from tqdm import tqdm
import json
# from adrd.data import _conf
import adrd.utils.misc
import torch
import monai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, balanced_accuracy_score, average_precision_score, multilabel_confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay, precision_score, recall_score, PrecisionRecallDisplay, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, StratifiedKFold
from icecream import ic
ic.disable()

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12

def plot_heatmap(df_dict, figname, plot_type):
    vmin = 0
    vmax = 1

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})

    # Group names
    if plot_type == "add":
        group_names = ['History', '+ Neurological/Physical', '+ MRI volumes', '+ FAQ', '+ Neuropsych Battery', '+ CDR', '+ Plasma', '+ APOE-ε4']
    elif plot_type == "del":
        group_names = ['All',
        'All - History',
        'All - Neurological/Physical',
        'All - MRI',
        'All - FAQ',
        'All - Neuropsych Battery',
        'All - CDR',
        'All - Plasma',
        'All - APOE-ε4']

    sns.set_palette("colorblind")
    # Plotting amyloid heatmap
    sns.heatmap(df_dict['amy'], ax=axes[0], annot=True, cmap='magma', fmt=".2f", cbar=False, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
    axes[0].set_title('Amyloid', fontname='Arial', fontsize=12)
    axes[0].set_yticklabels(group_names, rotation=0)  # Set group names
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')

    # Plotting tau heatmap
    sns.heatmap(df_dict['tau'], ax=axes[1], annot=True, cmap='magma', fmt=".2f", cbar_kws={'label': ''}, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
    axes[1].set_title('Tau', fontname='Arial', fontsize=12)
    axes[1].set_yticklabels(group_names, rotation=0)  # Set group names
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    
def get_amy_tau_df(met_dict):
    rows = []
    for group, labels in met_dict.items():
        for label, metrics in labels.items():
            row = {'Group': group, 'Label': label, 'AUROC': metrics['AUC (ROC)'], 'AUPR': metrics['AUC (PR)']}
            rows.append(row)

    df = pd.DataFrame(rows)

    # amy
    amy = df[df['Label'] == 'amy_label']
    amy.drop(columns=['Label'], inplace = True)
    amy.set_index('Group', inplace=True)
    #tau 
    tau = df[df['Label'] == 'tau_label']
    tau.drop(columns=['Label'], inplace = True)
    tau.set_index('Group', inplace=True)
    
    df_dict = {'amy': amy, 'tau': tau}
    return df_dict


def plot(config):
    # WITH MRIs
    print("Plotting figure 2c")
    with open(config['source_data']['fig2c'], 'rb') as handle:
        met_list_2c = pickle.load(handle)
        
    df_dict_2c = get_amy_tau_df(met_list_2c)
    plot_heatmap(df_dict_2c, figname=config['output']['fig2c'], plot_type="add")
    
    print("Plotting figure 2d")
    with open(config['source_data']['fig2d'], 'rb') as handle:
        met_list_2d = pickle.load(handle)
        
    df_dict_2d = get_amy_tau_df(met_list_2d)
    plot_heatmap(df_dict_2d, figname=config['output']['fig2c'], plot_type="del")
    
    # WITHOUT MRIs
    print("Plotting figure efig3a")
    with open(config['source_data']['efig3a'], 'rb') as handle:
        met_list_efig3a = pickle.load(handle)
        
    df_dict_efig3a = get_amy_tau_df(met_list_efig3a)
    plot_heatmap(df_dict_efig3a, figname=config['output']['efig3a'], plot_type="add")
    
    # CATBOOST
    print("Plotting figure efig3b")
    amy_add = pd.read_csv(config['source_data']['efig3b_amy'])
    amy_add['AUROC'] = amy_add['AUC-ROC']
    amy_add = amy_add[['AUROC', 'AUPR']]
    tau_add = pd.read_csv(config['source_data']['efig3b_tau'])[['AUC-ROC', 'AUPR']]
    tau_add['AUROC'] = tau_add['AUC-ROC']
    tau_add = tau_add[['AUROC', 'AUPR']]

    df_dict_efig3b = {'amy': amy_add, 'tau': tau_add}
    
    plot_heatmap(df_dict_efig3b, figname=config['output']['efig3b'], plot_type="add")
    




        
    
