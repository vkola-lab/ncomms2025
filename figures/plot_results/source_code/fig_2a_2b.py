# Make ROC and PR curves for Amyloid and Tau labels
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import pickle
from tqdm import tqdm
from matplotlib.pyplot import figure
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve, roc_auc_score
     
from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', size=7)
plt.rcParams['font.family'] = 'Arial'

# AUC ROC

def roc_auc_scores(y_true, y_pred, features):
    # n_classes = y_true.shape[1]

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
        # print(len(masked_y_true), len(masked_y_pred))
        # print(round(roc_auc_score(masked_y_true, masked_y_pred), 4))
        fpr[fea], tpr[fea], thresholds[fea] = roc_curve(y_true=masked_y_true, y_score=masked_y_pred, pos_label=1, drop_intermediate=True)
        auc_scores[fea] = auc(fpr[fea], tpr[fea])

    return fpr, tpr, auc_scores, thresholds

def generate_roc(y_true, y_pred, features, figsize=(2.3, 2.3), figname='Average_ROC_curves'):
    fpr, tpr, auc_scores, _ = roc_auc_scores(y_true=y_true, y_pred=y_pred, features=features)
    lw = 1
    
    plt.figure(figsize=figsize)
    custom_labels = {'amy_label': 'Amyloid', 'tau_label': 'Tau'}
    
    # Initialize a DataFrame to store all the data for plotting
    all_data = pd.DataFrame()

    # Generate interpolated data for each feature
    for feature in features:
        fpr_value = np.linspace(0, 1, 100)
        interp_tpr = np.interp(fpr_value, fpr[feature], tpr[feature])
        temp_df = pd.DataFrame({
            'Specificity': 1 - fpr_value,
            'Sensitivity': interp_tpr,
            'Feature': '{0}: {1:0.2f}'.format(custom_labels[feature], auc_scores[feature])
        })
        all_data = pd.concat([all_data, temp_df])

    # Use seaborn to plot using 'Feature' as hue
    sns.lineplot(data=all_data, x='Specificity', y='Sensitivity', hue='Feature', palette='colorblind', linewidth=lw)

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)  # Diagonal line indicating random chance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('Average ROC Curves')
    plt.legend(title='', loc='lower right', fontsize=7)
    plt.tight_layout()
    
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()

# P-R curve

def precision_recall(y_true, y_pred, features):
    # Compute the precision-recall curve and average precision for each class
    # n_classes = y_true.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, fea in enumerate(features):
        y_true_ = np.array(y_true[:, i])
        y_pred_ = np.array(y_pred[:, i])
        mask = np.array([1 if not np.isnan(k) else 0 for k in y_true_])
        masked_y_true = y_true_[np.where(mask == 1)]
        masked_y_pred = y_pred_[np.where(mask == 1)]
        
        # print(len(masked_y_true), len(masked_y_pred))
        # print(round(average_precision_score(masked_y_true, masked_y_pred), 4))
        precision[fea], recall[fea], _ = precision_recall_curve(masked_y_true, masked_y_pred)
        precision[fea], recall[fea] = precision[fea][::-1], recall[fea][::-1]
        average_precision[fea] = average_precision_score(masked_y_true, masked_y_pred)

    return precision, recall, average_precision


def generate_pr(y_true, y_pred, features, figsize=(2.3, 2.3), figname='Average_PR_curves'):
    precision, recall, average_precision = precision_recall(y_true=y_true, y_pred=y_pred, features=features)
    lw = 1
    
    plt.figure(figsize=figsize)
    custom_labels = {'amy_label': 'Amyloid', 'tau_label': 'Tau'}
    
    # Initialize a DataFrame to store all the data for plotting
    all_data = pd.DataFrame()

    # Generate interpolated data for each feature
    for feature in features:
        mean_recall = np.linspace(0, 1, 100)
        interp_precision = np.interp(mean_recall, recall[feature], precision[feature])
        temp_df = pd.DataFrame({
            'Recall': mean_recall,
            'Precision': interp_precision,
            'Feature': '{0}: {1:0.2f}'.format(custom_labels[feature], average_precision[feature])  # This column will be used for hue in the plot
        })
        all_data = pd.concat([all_data, temp_df])

    # Use seaborn to plot using 'Feature' as hue
    sns.lineplot(data=all_data, x='Recall', y='Precision', hue='Feature', palette='colorblind', linewidth=lw)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Average Precision-Recall Curves')
    plt.legend(title='', loc='lower left', fontsize=7)
    plt.tight_layout()
    
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    # plt.show()
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')


def plot(config):
    print("Plotting figures 2a and 2b")
    model_preds = pd.read_csv(config['source_data']['fig2a_2b'])
    labels = ['amy_label', 'tau_label']
    y_true_ =  np.array(model_preds[[f'{lab}_label' for lab in labels]])
    scores_proba_ = np.array(model_preds[[f'{lab}_prob' for lab in labels]])

    generate_roc(y_true_, scores_proba_, labels, figname=config['output']['fig2a'])
    generate_pr(y_true_, scores_proba_, labels, figname=config['output']['fig2b'])






