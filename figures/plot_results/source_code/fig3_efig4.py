# This creates main figure 3 and extended figures 4.
import pandas as pd
import numpy as np
import re
import scikit_posthocs as sp
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ptitprince1 as pt
import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, shapiro, levene, kruskal, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['font.family'] = 'Arial'


def levene_test(df, group, var):
    for label in df[group].unique():
        stat, p = shapiro(df[var][df[group] == label])
        print(f'Normality test for group {label}: Stat={stat}, p-value={p}')

    stat, p = levene(*[df[var][df[group] == label] for label in df[group].unique()])
    print(f'Levene’s test for equal variances: Stat={stat}, p-value={p}')


def mann_whitney_test_tau(df, group_col, value_col):
    groups = df[group_col].unique()
    
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups for Mann-Whitney test, but found {len(groups)}")
    
    group1_data = df[df[group_col] == groups[0]][value_col]
    group2_data = df[df[group_col] == groups[1]][value_col]

    statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='less')
    
    if value_col == 'amy_CENTILOIDS':
        table_value = 'CL' 
    elif value_col == 'amy_label_prob':
        table_value = 'P(Aβ)' 
    elif value_col == 'tau_META_VILLE_SUVR':
        table_value = 'Meta-τ SUVR'
    elif value_col == 'tau_label_prob':
        table_value = 'P(τ)' 
    result_df = pd.DataFrame({
        'Measure': [table_value],
        'U Statistic': [statistic],
        'p-value': [f'{p_value:.2e}']
    })
    
    return result_df


def kruskall_duns(df, group, var):
    kruskal_result = kruskal(*[df[var][df[group] == label] for label in df[group].unique()])
    print(f'Kruskal-Wallis result: Statistic={kruskal_result.statistic}, p-value={kruskal_result.pvalue}')
    posthoc_result = sp.posthoc_dunn(df, val_col=var, group_col=group, p_adjust='holm')
    print(posthoc_result)
    return posthoc_result


def bubble_plot(df, proba_col, outcome_col, proba_name, outcome_name, figname, apply_log=False, color_col="bat_TOTAL13", color_label="ADAS13"):
    """
    Create scatter plot with regression line for amyloid or tau data
    """
    
    y_values = np.log(df[outcome_col]) if apply_log else df[outcome_col]
    
    if apply_log:
        pear_correlation, p_value = pearsonr(df[proba_col], np.log(df[outcome_col]))
    else:
        pear_correlation, p_value = pearsonr(df[proba_col], df[outcome_col])
        
        spear_correlation, spear_p_value = spearmanr(df[proba_col], df[outcome_col])
        print(f"Spearman's correlation: {spear_correlation:.2f}, P-value: {spear_p_value:.2e}")
    
    print(f"Pearson's correlation coefficient: {pear_correlation:.2f}, P-value: {p_value:.2e}")
    
    slope, intercept, r_value, p_value, std_err = linregress(df[proba_col], y_values)
    
    fontsize = 15
    
    fig = px.scatter(
        x=df[proba_col], 
        y=y_values,
        color=df[color_col],
        labels={'x': proba_name, 'y': f'log({outcome_name})' if apply_log else outcome_name, 'color': color_label}, 
        color_continuous_scale='Plasma_r'
    ) 
    
    fig.add_trace(
        go.Scattergl(
            x=df[proba_col], 
            y=intercept + slope * df[proba_col],
            mode='lines', 
            name='Regression Line',
            line=dict(color='black', width=2), 
            showlegend=False
        )
    )
    
    fig.add_annotation(
        x=0.02, y=0.95, 
        xref="paper", yref="paper",
        text=f"Pearson's r = {pear_correlation:.2f}", 
        showarrow=False,
        font=dict(family="Arial, sans-serif", size=fontsize, color="black"),
        bgcolor="white", 
        bordercolor="black", 
        borderwidth=1, 
        borderpad=4,
        align="left"
    )
    
    fig.update_layout(
        plot_bgcolor='white', 
        paper_bgcolor='white',
        xaxis=dict(
            title=proba_name,
            title_font=dict(size=fontsize, family='Arial, sans-serif', color='black'),
            linecolor='black', 
            showgrid=False,
            ticks='outside', 
            tickcolor='black',
        ),
        yaxis=dict(
            title=f'log({outcome_name})' if apply_log else outcome_name,
            title_font=dict(size=fontsize, family='Arial, sans-serif', color='black'),
            linecolor='black',
            showgrid=False,
            gridcolor='lightgrey', 
            gridwidth=0.5,
            ticks='outside',
            tickcolor='black',
        ),
        legend=dict(
            title_text=color_label,
            title_font=dict(size=fontsize, family='Arial, sans-serif', color='black'),
            font=dict(size=fontsize, family='Arial, sans-serif', color='black')
        ),
        font=dict(
            family="Arial, sans-serif",
            size=fontsize,
            color="black"
        )
    )
    
    for trace in fig.data:
        trace.showlegend = False
    
    fig_width_inch = 2
    fig_height_inch = 1
    dpi = 300
    fig_width_px = fig_width_inch * dpi
    fig_height_px = fig_height_inch * dpi
    
    # fig.show()
    pio.write_image(fig, figname, width=fig_width_px, height=fig_height_px)
    
    return fig


def rainclouds_preclin(data, x_col, y_col, figname, 
                      x_labels=None, y_label='P(Aβ)', 
                      palette='inferno_r', 
                      font_sizes=7, figsize=(3, 2.3),
                      test='Mann-Whitney', pairs=None, dpi=300):

    if x_labels is None:
        x_labels = [f'Aβ PET-', f'Aβ PET+']
    if pairs is None:
        pairs = [(0, 1)]
    fig, ax = plt.subplots(figsize=figsize)

    ax = pt.RainCloud(data=data, x=x_col, y=y_col, orient='v', alpha=0.8, palette=palette, bw=.2, ax=ax, linewidth=0.5, dodge=True, 
    width_viol=.7, width_box=0.2, point_size=1, jitter=1, cut=3)
    
    annotator = Annotator(ax, pairs, data=data, x=x_col, y=y_col, order=[0, 1])
    annotator.configure(
        test=test, 
        text_format='star',
        loc='inside',
        verbose=2,
        fontsize=font_sizes,
        line_width=0.5
    )
    annotator.apply_and_annotate()
    
    plt.xlabel('')
    plt.ylabel(y_label)
    plt.xticks([0, 1], x_labels, fontname='Arial', fontsize=font_sizes)
    ax.set_xlabel("", fontname='Arial', fontsize=font_sizes)
    ax.set_ylabel(y_label, fontname='Arial', fontsize=font_sizes)
    ax.tick_params(axis='both', labelsize=font_sizes)
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0] - 0.2, current_xlim[1])
    sns.despine()
    plt.savefig(figname, dpi=dpi, bbox_inches='tight')

    return fig, ax


def get_kruskal_dunn_pvalues(df, pairs):
    
    group_labels = sorted(list(set([value for pair in pairs for value in pair])))
    
    data = [
        list(df[df['AT_ProfileReg'] == cdr]['ATRegScore_PC1']) for cdr in group_labels
    ]
    
    for i in data:
        print(len(i))

    values = []
    for i, group_data in enumerate(data):
        values.extend([(value, group_labels[i]) for value in group_data])

    df_ = pd.DataFrame(values, columns=['Value', 'Group'])
    
    kw_result = stats.kruskal(*[group_data for label, group_data in df_.groupby('Group')['Value']])

    print("Kruskal-Wallis Test:")
    print(f"Statistically significant: {kw_result.pvalue < 0.05}", kw_result.pvalue)

    # post hoc pairwise- Dunn's
    posthoc_dunn = sp.posthoc_dunn(df_, val_col='Value', group_col='Group', p_adjust='holm')

    posthoc_dunn = posthoc_dunn[posthoc_dunn.columns[::-1]]
    p_values = {val: dict(posthoc_dunn[val]) for val in group_labels}
    
    values = [p_values[i][j] for (i,j) in pairs]
    
    return posthoc_dunn, values
    # return values

def map_values(value):
    if isinstance(value, str):
        return value
    if value < 0.0001:
        return '****        '
    elif value < 0.001:
        return '***       '
    elif value < 0.01:
        return '**         '
    elif value < 0.05:
        return '    *     '
    elif value <= 1.0:
        return 'ns'
    else:
        return str(value)
    

def get_annotate_matrix(matrix_content):
    print(matrix_content)
    row_indices = matrix_content.index
    column_indices = matrix_content.columns

    matrix_content_values = np.vectorize(map_values)(matrix_content.values)
    # print(matrix_content_values)

    matrix_with_indices = np.column_stack((matrix_content_values, row_indices))
    column_indices = np.insert(column_indices.values, 0, '', axis=0)
    matrix_with_indices = np.row_stack((matrix_with_indices, column_indices))
    print("matrix_with_indices", matrix_with_indices)
    main = []
    # main.append('p-values              ')
    for i, row in enumerate(matrix_with_indices):
        if i == len(matrix_with_indices) - 1:
            row = row[2:]
        elif i == 0:
            row = row[len(row) - i:]
        else:
            row = row[len(row) - i - 1:]
        cr = []
        for cell in row:
            cr.append('{: <5}'.format(str(cell)))
        main.append('  '.join(cr))
        
    main[-1] = main[-1] + "              "
    main = main[1:]
    return main


def annotate(df, ax, pairs, order, xy=(0.95, 0.03)):
    annot = Annotator(ax, pairs=pairs, data=df, x='AT_ProfileReg', y='ATRegScore_PC1', order=order)
    matrix_content, _ = get_kruskal_dunn_pvalues(df, pairs=pairs)
    matrix_text = get_annotate_matrix(matrix_content)
    
    matrix_text.insert(0, 'p-values{: <20}'.format(' '))
    # matrix_text.insert(0, 'p-values{: <5}'.format(' '))
    matrix_text = '\n'.join(matrix_text)
    print(matrix_text)
    ax.annotate(matrix_text, xy=xy, xycoords='axes fraction', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor=(1, 1, 1, 0.0)))


def rainclouds_stages(df, figname):
    profile_order = ['A-T-', 'A+T-', 'A+MTL+', 'A+NEO+']
    ax1_pairs = [('A-T-', 'A+T-'), ('A+T-', 'A+MTL+'), ('A+MTL+', 'A+NEO+'), ('A-T-', 'A+NEO+'), ('A-T-', 'A+MTL+'), ('A+T-', 'A+NEO+')]
    df_filtered = df[df['AT_ProfileReg'].isin(profile_order)]

    font_sizes = 11
    fig, ax = plt.subplots(figsize=(6,4))

    ax = pt.RainCloud(data = df_filtered, x = "AT_ProfileReg", y = "ATRegScore_PC1", orient='v', alpha=0.8,
    palette = "inferno_r", bw=.2, ax=ax, linewidth=0.5, dodge=True, width_viol=.7, width_box=0.2, point_size = 2, jitter=1, cut =3)

    annotate(df=df_filtered, ax=ax, pairs=ax1_pairs, order=profile_order, xy=(0.50, 0.80))

    plt.xlabel('')
    plt.ylabel('AT score')
    ax.set_xlabel("",  fontname='Arial', fontsize = font_sizes)
    ax.set_ylabel("AT score", fontname='Arial', fontsize = font_sizes)
    ax.tick_params(axis='both', labelsize=font_sizes)
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0] - 0.2, current_xlim[1])  # Add more space to the left

    sns.despine()
    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot(config):
    # Fig 3a: P(AB) over centiloids bubble plot colored by ADAS
    print("Plotting figure 3a")
    adas_data = pd.read_csv(config['source_data']['fig3a'])
    bubble_plot(adas_data, "amy_label_prob", "amy_CENTILOIDS", "P(Aβ)", "Centiloids", figname=config['output']['fig3a'], apply_log=False)

    # Fig 3b: P(tau) over meta-temporal tau colored by ADAS
    print("Plotting figure 3b")
    adas_data_tau = pd.read_csv(config['source_data']['fig3b'])
    bubble_plot(adas_data_tau, 'tau_label_prob', 'tau_META_VILLE_SUVR', 'P(τ)', 'meta-τ SUVr', figname=config['output']['fig3b'], apply_log=True)

    # Fig 3c: preclinical AD plot.
    print("Plotting figure 3c")
    preclin = pd.read_csv(config['source_data']['fig3c'])
    # levene_test(preclin, 'amy_label_label', 'amy_label_prob')
    # mann_whitney_test_tau(preclin, 'amy_label_label', 'amy_label_prob')
    rainclouds_preclin(data=preclin, x_col="amy_label_label", y_col="amy_label_prob", figname=config['output']['fig3c'])

    # Fig 3d: AT score plot.
    print("Plotting figure 3d")
    df_tau = pd.read_csv(config['source_data']['fig3d'])
    rainclouds_stages(df_tau, figname=config['output']['fig3d'])

    # Regional tau labels - Extended Figure 4
    print("Plotting extended figure 4")
    ## medtemp
    medtemp_tau = pd.read_csv(config['source_data']['efig4a'])
    bubble_plot(medtemp_tau, 'tau_medtemp_label_prob', 'tau_MEDIAL_TEMPORAL_SUVR', 'P(med-temp τ)', 'med-temp τ SUVr', figname=config['output']['efig4a'], apply_log=True)
    ## lattemp
    lattemp_tau = pd.read_csv(config['source_data']['efig4b'])
    bubble_plot(lattemp_tau, 'tau_lattemp_label_prob', 'tau_LATERAL_TEMPORAL_SUVR', 'P(lat-temp τ)', 'lat-temp τ SUVr', figname=config['output']['efig4b'], apply_log=True)
    ## medpar
    medpar_tau = pd.read_csv(config['source_data']['efig4c'])
    bubble_plot(medpar_tau, 'tau_medpar_label_prob', 'tau_MEDIAL_PARIETAL_SUVR', 'P(med-par τ)', 'med-par τ SUVr', figname=config['output']['efig4c'], apply_log=True)
    ## latpar
    latpar_tau = pd.read_csv(config['source_data']['efig4d'])
    bubble_plot(latpar_tau, 'tau_latpar_label_prob', 'tau_LATERAL_PARIETAL_SUVR', 'P(lat-par τ)', 'lat-par τ SUVr', figname=config['output']['efig4d'], apply_log=True)
    ## frontal
    frontal_tau = pd.read_csv(config['source_data']['efig4e'])
    bubble_plot(frontal_tau, 'tau_front_label_prob', 'tau_FRONTAL_SUVR', 'P(frt τ)', 'frt τ SUVr', figname=config['output']['efig4e'], apply_log=True)
    ## Occipital
    occ_tau = pd.read_csv(config['source_data']['efig4f'])
    bubble_plot(occ_tau, 'tau_occ_label_prob', 'tau_OCCIPITAL_SUVR', 'P(occ τ)', 'occ τ SUVr', figname=config['output']['efig4f'], apply_log=True)