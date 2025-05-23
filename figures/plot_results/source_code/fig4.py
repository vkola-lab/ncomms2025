import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ptitprince1 as pt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, mannwhitneyu
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'


def stats_test(df, tertile_var, probabilities):
    # noramlity test
    for tertile in df[tertile_var].unique():
        data = df[df[tertile_var] == tertile][probabilities]
        stat, p = stats.shapiro(data)
        print(f"Normality test for {tertile}: Statistics={stat:.2f}, p={p:.2e}")

    # homogeneity of variances
    grouped_data = [df[df[tertile_var] == tertile][probabilities] for tertile in df[tertile_var].unique()]
    stat, p = stats.levene(*grouped_data)
    print(f"Levene's test for homogeneity of variances: Statistics={stat:.2f}, p={p:.2e}")

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


def add_stat_significance(fig, x_start, x_end, y_start, y_end, text, orientation):
    if orientation == 'horizontal':
        # Horizontal Line
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_end, y1=y_start, line=dict(color="black", width=2))
        # Ticks
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start, y1=y_start + 0.02, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=x_end, y0=y_start, x1=x_end, y1=y_start + 0.02, line=dict(color="black", width=2))
        # Text
        fig.add_annotation(x=(x_start + x_end) / 2, y=y_start + 0.02, text=text, showarrow=False, font=dict(size=14))
    else:
        # Vertical Line
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start, y1=y_end, line=dict(color="black", width=2))
        # Ticks
        fig.add_shape(type="line", x0=x_start, y0=y_start, x1=x_start - 0.02, y1=y_start, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=x_start, y0=y_end, x1=x_start - 0.02, y1=y_end, line=dict(color="black", width=2))
        # Text
        fig.add_annotation(x=x_start - 0.03, y=(y_start + y_end) / 2, text=text, showarrow=False, font=dict(size=14), textangle=-90)


def rainclouds_tau_levels(df, figname):
    sns.set_theme(style="white", context="paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.3), sharey=True)
    font_sizes = 7

    # centiloids
    cb_palette1 = sns.color_palette("inferno_r")
    custom_pal1 = {'Low/med τ PET': cb_palette1[0], 'High τ PET': cb_palette1[2]}

    pt.RainCloud(data = df, x = "tau_level", y = "amy_CENTILOIDS", orient='h', 
    palette = custom_pal1, bw=.2, ax=ax1, move = .2, linewidth=0.5, dodge=True, 
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1, cut=3)

    ax1.plot([300, 300], [0, 1], color='black', linewidth=2) # vert line
    ax1.plot([300, 294], [0, 0], color='black', linewidth=2) # tick 1
    ax1.plot([300, 294], [1, 1], color='black', linewidth=2) # tick 2
    ax1.text(308, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    if ax1.get_yticks().size > 0:
        positions = ax1.get_yticks()
        labels = ['Low/med τ PET', 'High τ PET']
        ax1.set_yticks(positions[:len(labels)])
        ax1.set_yticklabels(labels)

    ax1.set_xlabel("Centiloids", fontname='Arial', fontsize=font_sizes)
    ax1.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax1.tick_params(axis='both', labelsize=font_sizes)

    for c in ax1.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    # # prob amy
    cb_palette = sns.color_palette("YlGnBu")
    custom_pal = {'Low/med τ PET': cb_palette[0], 'High τ PET': cb_palette[2]}

    pt.RainCloud(data = df, x = "tau_level", y = "amy_label_prob", orient='h', cut=3,
    palette = custom_pal, bw=.2, ax=ax2, move = .2, linewidth=0.5, dodge=True,
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1)

    ax2.plot([0.81, 0.81], [0, 1], color='black', linewidth=2) # vert line
    ax2.plot([0.81, 0.80], [0, 0], color='black', linewidth=2) # tick 1
    ax2.plot([0.81, 0.80], [1, 1], color='black', linewidth=2) # tick 2
    ax2.text(0.83, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    ax2.set_xlabel("P(Aβ)", fontname='Arial', fontsize=font_sizes)
    ax2.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax2.tick_params(axis='both', labelsize=font_sizes)

    # makes lines thinner
    for c in ax2.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    sns.despine()
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()


def rainclouds_cl_levels(df, figname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.3), sharey=True)
    font_sizes = 7

    # centiloids
    pt.RainCloud(data = df, x = "cl_level", y = "tau_META_VILLE_SUVR", orient='h', 
    palette = "BuPu", bw=.2, ax=ax1, move = .2, linewidth=0.5, dodge=True, 
    width_viol=.7, width_box=0.2, point_size = 1, jitter=1, cut=3)

    ax1.plot([3, 3], [0, 1], color='black', linewidth=2) # vert line
    ax1.plot([3, 2.97], [0, 0], color='black', linewidth=2) # tick 1
    ax1.plot([3, 2.97], [1, 1], color='black', linewidth=2) # tick 2
    ax1.text(3.05, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars

    ax1.set_xlabel("Meta-τ SUVR", fontname='Arial', fontsize=font_sizes)
    ax1.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax1.tick_params(axis='both', labelsize=font_sizes)

    for c in ax1.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    # probs
    cb_palette = sns.color_palette("magma_r")
    custom_pal = {'Low/med CL': cb_palette[3], 'High CL': cb_palette[5]}

    pt.RainCloud(data=df, x="cl_level", y="tau_label_prob", orient='h', cut=3,
                palette=cb_palette, bw=.2, ax=ax2, move=.2, linewidth=0.5, dodge=True,
                width_viol=.7, width_box=0.2, point_size=1, jitter=1)

    ax2.plot([1, 1], [0, 1], color='black', linewidth=2) # vert line
    ax2.plot([1, 0.98], [0, 0], color='black', linewidth=2) # tick 1
    ax2.plot([1, 0.98], [1, 1], color='black', linewidth=2) # tick 2
    ax2.text(1.025, 0.5, "****", ha='center', va='center', rotation=270, fontname='Arial', fontsize=font_sizes) # stars


    ax2.set_xlabel("P(τ)", fontname='Arial', fontsize=font_sizes)
    ax2.set_ylabel("", fontname='Arial', fontsize=font_sizes)
    ax2.tick_params(axis='both', labelsize=font_sizes)

    # makes lines thinner
    for c in ax2.get_children():
        if isinstance(c, plt.Line2D):
            c.set_linewidth(1)
        if isinstance(c, mpatches.Patch):
            c.set_linewidth(1)

    sns.despine()
    plt.tight_layout()

    plt.savefig(figname, dpi=300)
    plt.show()


def kde_plot(df, figname):
    fontsizes = 16

    cohort_markers = {'ADNI': 'circle', 'HABS': 'cross', 'NACC': 'diamond'}

    color_map = {'Aβ+, τ+': '#CC503E', 'Aβ-, τ-': '#008080'}

    fig = px.density_contour(
        df,
        x='amy_label_prob',
        y='tau_label_prob',
        color='Profile',
        marginal_x='box',
        marginal_y='box',
        color_discrete_map=color_map,
        title=''
    )

    for trace in fig.data:
        if trace.type == 'contour':
            trace.line.width = 1
            trace.showlegend = True  

    for i, trace in enumerate(fig.data):
        if trace.type in ['box', 'violin']:
            trace.showlegend = False

    fig.add_trace(go.Scatter(
        x=df['amy_label_prob'],
        y=df['tau_label_prob'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Profile'].map(color_map),
            symbol=df['COHORT'].map(cohort_markers)
        ),
        showlegend=False
    ))

    for cohort, marker in cohort_markers.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol=marker
            ),
            name=cohort,
            showlegend=True
        ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='P(Aβ)',
            title_font=dict(size=fontsizes),
            linecolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            ticks='outside',
            tickcolor='black',
            ticklen=10,
            nticks=10
        ),
        yaxis=dict(
            title='P(τ)',
            title_font=dict(size=fontsizes),
            linecolor='black',
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=0.5,
            ticks='outside',
            tickcolor='black',
            ticklen=10,
            nticks=10
        ),
        legend=dict(
            title_text="",
            yanchor="top",
            y=1.2,
            xanchor="center",
            x=0.5,
            orientation="h",
            itemsizing="constant"
        ),
        font=dict(family="Arial, sans-serif", size=fontsizes, color="black")
    )

    fig.show()

    # Export to PDF
    fig_width_inch = 3
    fig_height_inch = 1.5
    dpi = 300
    fig_width_px = fig_width_inch * dpi
    fig_height_px = fig_height_inch * dpi
    pio.write_image(fig, figname, width=fig_width_px, height=fig_height_px)


def plot(config):
    # Figure 4a
    tau_level_df = pd.read_csv(config['source_data']['fig4a'])
    rainclouds_tau_levels(tau_level_df, figname=config['output']['fig4a'])
    # Figure 4b
    cl_level_df = pd.read_csv(config['source_data']['fig4b'])
    rainclouds_cl_levels(cl_level_df, figname=config['output']['fig4b'])
    # Figure 4c
    df = pd.read_csv(figname=config['source_data']['fig4c'])           
    kde_plot(df, figname=config['output']['fig4c'])