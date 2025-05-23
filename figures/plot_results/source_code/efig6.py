
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def boxplots(data, figname):
    font_sizes = 10
    sns.set_style("white")
    melted_df = data.melt(id_vars='COHORT', value_vars=['tau_META_VILLE_SUVR', 'tau_MEDIAL_TEMPORAL_SUVR', 'tau_LATERAL_TEMPORAL_SUVR', 'tau_MEDIAL_PARIETAL_SUVR', 'tau_LATERAL_PARIETAL_SUVR', 'tau_OCCIPITAL_SUVR',  'tau_FRONTAL_SUVR'], var_name='SUVR_Type', value_name='SUVr')

    plt.figure(figsize=(8,4))
    boxplot = sns.boxplot(x='SUVR_Type', y='SUVr', hue='COHORT', data=melted_df, palette='Set2', fliersize=0.5)
    label_map = {
        'tau_META_VILLE_SUVR': 'Meta-temporal',
        'tau_MEDIAL_TEMPORAL_SUVR': 'Medial temporal',
        'tau_LATERAL_TEMPORAL_SUVR': 'Lateral temporal',
        'tau_MEDIAL_PARIETAL_SUVR': 'Medial parietal',
        'tau_LATERAL_PARIETAL_SUVR': 'Lateral parietal',
        'tau_FRONTAL_SUVR': 'Frontal',
        'tau_OCCIPITAL_SUVR': 'Occipital'
    }
    current_labels = [label.get_text() for label in boxplot.get_xticklabels()]
    boxplot.set_xticklabels([label_map.get(label, label) for label in current_labels], rotation=45)

    plt.title('')
    plt.xlabel('')
    plt.xticks(fontsize=font_sizes)
    plt.yticks(fontsize=font_sizes)

    plt.legend(title='', fontsize=8, title_fontsize=8)
    plt.tight_layout()
    # plt.savefig(figname, facecolor='white')
    plt.tight_layout()
    plt.show()


def plot(config):
    # unharmonized plot
    unharmonized_roi = pd.read_csv(config['source_data']['efig6a'])
    boxplots(unharmonized_roi, figname=config['output']['efig6a'])

    # harmonized plot
    harmonized_roi = pd.read_csv(config['source_data']['efig6b'])
    boxplots(harmonized_roi, figname=config['output']['efig6b'])
