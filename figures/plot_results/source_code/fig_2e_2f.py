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
     
# AUC ROC

def roc_auc_scores(y_true, y_pred, features):
    # n_classes = y_true.shape[1]

    tpr = dict()
    fpr = dict()
    auc_scores = dict()
    thresholds = dict()
        
    for i, fea in enumerate(features):
        fpr[fea], tpr[fea], thresholds[fea] = roc_curve(y_true=y_true[:, i], y_score=y_pred[:, i], pos_label=1, drop_intermediate=False)
        auc_scores[fea] = auc(fpr[fea], tpr[fea])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true.ravel(), y_pred.ravel()
    )
    auc_scores["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[fea] for fea in features]))
    mean_tpr = np.zeros_like(all_fpr)
    for i, fea in enumerate(features):
        mean_tpr += np.interp(all_fpr, fpr[fea], tpr[fea])
    mean_tpr /= len(features)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    auc_scores["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute weighted-average ROC curve and ROC area
    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    # print(len(weights))
    weighted_tpr = np.zeros_like(all_fpr)
    for i, fea in enumerate(features):
        weighted_tpr += weights[i] * np.interp(all_fpr, fpr[fea], tpr[fea])
    fpr["weighted"] = all_fpr
    tpr["weighted"] = weighted_tpr
    auc_scores["weighted"] = auc(fpr["weighted"] , tpr["weighted"])  

    return fpr, tpr, auc_scores, thresholds

def generate_roc(y_true, y_pred, features, figsize=(2.3, 2.3), figname='Average_ROC_curves'):
    fpr, tpr, auc_scores, _ = roc_auc_scores(y_true=y_true, y_pred=y_pred, features=features)
    # n_classes = y_true.shape[1]
    
    lw = 0.5

    # Average ROC curves
    colors_ = [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
    
    sensitivity_comb = np.array(list(tpr['micro']) + list(tpr['macro']) + list(tpr['weighted']))
    specificity_comb = np.array(list(1 - fpr['micro']) + list(1 - fpr['macro']) + list(1 - fpr['weighted']))
    plt_style = np.array(['micro (AUC = {0:0.2f})'.format(auc_scores["micro"])] * len(tpr['micro']) 
                         + ['macro (AUC = {0:0.2f})'.format(auc_scores["macro"])] * len(tpr['macro']) 
                         + ['weighted (AUC = {0:0.2f})'.format(auc_scores["weighted"])] * len(tpr['weighted']))
    df_roc = pd.DataFrame({'Specificity': specificity_comb, 'Sensitivity': sensitivity_comb, 'AUC ROC': plt_style})
    # print(df_roc)
    
    fig = plt.figure(figsize=figsize, dpi=300)
    sns.lineplot(data=df_roc, x='Specificity', y='Sensitivity', style='AUC ROC', hue='AUC ROC', palette=colors_, linewidth=lw)
    
    # plt.setp(plt.spines.values(), color='w')
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('')
    plt.legend(loc='lower left', fontsize=6) #, prop=legend_properties)
    plt.tight_layout()
    # plt.show()
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')

# P-R curve

def precision_recall(y_true, y_pred, features):
    # Compute the precision-recall curve and average precision for each class
    # n_classes = y_true.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i, fea in enumerate(features):
        precision[fea], recall[fea], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        precision[fea], recall[fea] = precision[fea][::-1], recall[fea][::-1]
        average_precision[fea] = average_precision_score(y_true[:, i], y_pred[:, i])

    # Compute the micro-average precision-recall curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred,
                                                        average="micro")

    # Compute the macro-average precision-recall curve and average precision
    mean_recall = np.unique(np.concatenate([recall[fea] for i, fea in enumerate(features)]))
    # mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i, fea in enumerate(features):
        mean_precision += np.interp(mean_recall, recall[fea], precision[fea])
    mean_precision /= len(features)
    recall["macro"] = mean_recall
    precision["macro"] = mean_precision

    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                        average="macro")

    # Compute the weighted-average precision-recall curve and average precision

    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    # print(weights)
    weighted_precision = np.zeros_like(mean_recall)
    for i, fea in enumerate(features):
        weighted_precision += weights[i] * np.interp(mean_recall, recall[fea], precision[fea])
    recall["weighted"] = mean_recall
    precision["weighted"] = weighted_precision
    average_precision["weighted"] = average_precision_score(y_true, y_pred,
                                                                average="weighted")

    return precision, recall, average_precision


def generate_pr(y_true, y_pred, features, figsize=(2.3, 2.3), figname='Average_PR_curves'):
    precision, recall, average_precision = precision_recall(y_true=y_true, y_pred=y_pred, features=features)
    # n_classes = y_true.shape[1]
    lw = 0.5
    
    colors_ = [(30/255, 136/255, 229/255, 1.0), (255/255, 193/255, 7/255, 1.0), (216/255, 27/255, 96/255, 1.0)]
    
    precision_comb = np.array(list(precision['micro']) + list(precision['macro']) + list(precision['weighted']))
    recall_comb = np.array(list(recall['micro']) + list(recall['macro']) + list(recall['weighted']))
    plt_style = np.array(['micro (AP = {0:0.2f})'.format(average_precision["micro"])] * len(precision['micro']) 
                         + ['macro (AP = {0:0.2f})'.format(average_precision["macro"])] * len(precision['macro']) 
                         + ['weighted (AP = {0:0.2f})'.format(average_precision["weighted"])] * len(precision['weighted']))
    df_pr = pd.DataFrame({'Recall': recall_comb, 'Precision': precision_comb, 'AUC PR': plt_style})
    # print(df)
    
    fig = plt.figure(figsize=figsize, dpi=300)
    sns.lineplot(data=df_pr, x='Recall', y='Precision', style='AUC PR', hue='AUC PR', palette=colors_, linewidth=lw)
    
    
    
    plt.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
    plt.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
    plt.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)


    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('')
    plt.legend(loc='lower left', fontsize=6) #, prop=legend_properties)
    plt.tight_layout()
    plt.savefig(figname, format='pdf', dpi=300, bbox_inches='tight')
    # plt.show()

def plot(config):
    print("Plotting figures 2e and 2f")
    labels =['tau_front_label', 'tau_occ_label', 'tau_medtemp_label', 'tau_lattemp_label', 'tau_medpar_label', 'tau_latpar_label']

    df = pd.read_csv(config['source_data']['fig2e_2f'])
    y_true_ =  np.array(df[[f'{lab}_label' for lab in labels]])
    scores_proba_ =  np.array(df[[f'{lab}_prob' for lab in labels]])

    generate_roc(y_true_, scores_proba_, labels, figname=config['output']['fig2e'])
    generate_pr(y_true_, scores_proba_, labels, figname=config['output']['fig2f'])


