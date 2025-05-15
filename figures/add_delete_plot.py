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

# Load the data
with open(f'./source_data/fig2c.pickle', 'rb') as handle:
    met_list_add = pickle.load(handle)
    
with open(f'./source_data/fig2d.pickle', 'rb') as handle:
    met_list_delete = pickle.load(handle)

    
### Adding
rows = []
for group, labels in met_list_add.items():
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
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12


vmin = 0
vmax = 1

fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})

# Group names
group_names = ['History', '+ Neurological/Physical', '+ MRI volumes', '+ FAQ', '+ Neuropsych Battery', '+ CDR', '+ Plasma', '+ APOE-ε4']#, '+ CSF']

# Plotting amyloid heatmap
sns.heatmap(amy, ax=axes[0], annot=True, cmap='magma', fmt=".2f", cbar=False, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
axes[0].set_title('Amyloid', fontname='Arial', fontsize=12)
axes[0].set_yticklabels(group_names, rotation=0)  # Set group names
axes[0].set_xlabel('')
axes[0].set_ylabel('')

# Plotting tau heatmap
sns.heatmap(tau, ax=axes[1], annot=True, cmap='magma', fmt=".2f", cbar_kws={'label': ''}, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
axes[1].set_title('Tau', fontname='Arial', fontsize=12)
axes[1].set_yticklabels(group_names, rotation=0)  # Set group names
axes[1].set_xlabel('')
axes[1].set_ylabel('')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'./pdf_plots/fig2c.pdf', format='pdf', dpi=300, bbox_inches='tight')


### Deleting
rows = []
for group, labels in met_list_delete.items():
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
# del_groups = ['All',
#  'All - History',
#  'All - Neurological/Physical',
#  'All - MRI',
#  'All - FAQ',
#  'All - Neuropsych Battery',
#  'All - CDR',
#  'All - Plasma',
#  'All - APOE-ε4',
#  'All - CSF'
#  ]

# without csf
del_groups = ['All',
 'All - History',
 'All - Neurological/Physical',
 'All - MRI',
 'All - FAQ',
 'All - Neuropsych Battery',
 'All - CDR',
 'All - Plasma',
 'All - APOE-ε4']
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12

sns.set_palette("colorblind")


vmin = 0
vmax = 1

fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})


# Plotting amyloid heatmap
sns.heatmap(amy, ax=axes[0], annot=True, cmap='magma', fmt=".2f", cbar=False, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
axes[0].set_title('Amyloid', fontname='Arial', fontsize=12)
axes[0].set_yticklabels(del_groups, rotation=0)  # Set group names
axes[0].set_xlabel('')
axes[0].set_ylabel('')

# Plotting tau heatmap
sns.heatmap(tau, ax=axes[1], annot=True, cmap='magma', fmt=".2f", cbar_kws={'label': ''}, linewidths=.5, annot_kws={"size": 12}, vmin=vmin, vmax=vmax)
axes[1].set_title('Tau', fontname='Arial', fontsize=12)
axes[1].set_yticklabels(del_groups, rotation=0)  # Set group names
axes[1].set_xlabel('')
axes[1].set_ylabel('')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'./pdf_plots/fig2d.pdf', format='pdf', dpi=300, bbox_inches='tight')

# plt.show()


