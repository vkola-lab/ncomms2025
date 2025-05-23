"""
This script plots extended figure 2 - SHAP visualizations of amyloid and tau predictions
"""

import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def shap_viz(data, figname):
    plt.figure(figsize=(4, 4))
    shap.summary_plot(data, max_display=15, show=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('SHAP Value', fontsize=16, fontname='Arial')
    plt.tick_params(axis='both', labelsize=16)
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=16)
    cbar.set_ylabel('Feature value', fontsize=16, fontname='Arial')
    plt.savefig(figname, bbox_inches='tight', dpi=300)


def plot(config):
    # SHAP amyloid
    with open(config['source_data']['efig2a'], 'rb') as f:
        shap_values_exp_amy = pickle.load(f)
    shap_viz(shap_values_exp_amy, figname=config['output']['efig2a'])

    # SHAP tau
    with open(config['source_data']['efig2b'], 'rb') as f:
        shap_values_exp_tau = pickle.load(f)
    shap_viz(shap_values_exp_tau, figname=config['output']['efig2b'])


