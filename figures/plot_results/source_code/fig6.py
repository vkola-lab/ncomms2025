import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np
import statsmodels.api as sm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pandas as pd


def perform_statistical_tests(df, neuropath_var, proba_var):
    # Kruskal-Wallis test
    df = df.dropna(subset=[neuropath_var, proba_var])
    groups = df.groupby(neuropath_var)[proba_var].apply(list).values
    kruskal_result = stats.kruskal(*groups)
    
    if kruskal_result.pvalue < 0.05:
        # Perform Dunn's test if Kruskal-Wallis test is significant
        dunn_posthoc = sp.posthoc_dunn(df, val_col=proba_var, group_col=neuropath_var, p_adjust='holm')
        return dunn_posthoc, kruskal_result
    else:
        print("Kruskal-Wallis test not significant")
        return None, kruskal_result
    
def add_significance_annotations(ax, pairwise_p_values, groups, y_values, height_increment):
    """
    Add significance markers to the plot.
    
    Parameters:
        ax (Axes): The matplotlib axes to annotate.
        pairwise_p_values (DataFrame): DataFrame containing the pairwise p-values.
        groups (list): List of group labels.
        y_values (list): List of y-values for each group's annotation.
        height_increment (float): Increment to raise the annotation for overlapping lines.
    """
    if pairwise_p_values is None:
        return 0
    else:
        iter = 0
        n = len(groups)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the p-value is significant
                p_value = pairwise_p_values.loc[groups[i], groups[j]]
                if p_value < 0.05:
                    # Calculate the position for the annotation
                    x1, x2 = i, j
                    y, h = max(y_values) + height_increment * iter + 0.05, 0.03
                    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
                    ax.text((x1+x2)*.5, y+h+0.0, stars(p_value), ha='center', va='bottom', color='k', fontsize=13)
                    max_height = y+h+0.0
                    # print(max_height)
                    iter += 1
        return max(max_height, 1.0)

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "ns"

# Function to format median and IQR in LaTeX
def format_median_iqr(df, neuropath_var, proba_var, labels = None):
    table_str = ""
    for group in sorted(df[neuropath_var].unique()):
        group_name = labels["X_tics"].get(str(int(group)))
        group_data = df[df[neuropath_var] == group][proba_var]
        median = np.median(group_data)
        q1, q3 = np.percentile(group_data, [25, 75])
        table_str += f"{group_name} & {median:.2f} & [{q1:.2f}, {q3:.2f}] \\\\\n"
    return table_str

def format_p_value_latex(p_value):
    exponent = int(np.floor(np.log10(abs(p_value))))
    mantissa = p_value / (10 ** exponent)
    return rf"${mantissa:.2f} \times 10^{{{exponent}}}$"

def add_to_latex_table(kruskal_result, spearman_corr, spearman_p_value, labels):
    if labels["Y_axis"] == "P(Aβ)":
        prob_var = r"$P(\beta)$"
    elif labels["Y_axis"] == "P(τ)":
        prob_var = r"$P(\tau)$"
    else:
        raise ValueError("Invalid probability variable")
    
    name = labels.get("X_axis")
    
    # Format p-values for LaTeX
    spearman_p_value_str = format_p_value_latex(spearman_p_value)
    kruskal_p_value_str = format_p_value_latex(kruskal_result.pvalue)
    
    # Begin the table content
    table_str = ""
    
    if labels["Y_axis"] == "P(τ)":
        # no need to do the multirow in this case
        table_str += f"& {prob_var} & Spearman correlation & $r={spearman_corr:.2f}$ & {spearman_p_value_str} \\\\ \n"
        table_str += f" & & Kruskal-Wallis & $H={kruskal_result.statistic:.2f}$ & {kruskal_p_value_str} \\\\ \n"
    elif labels["Y_axis"] == "P(Aβ)":
        table_str += f"\\multirow{{4}}{{4cm}}{{{name}}} & {prob_var} & Spearman correlation & $r={spearman_corr:.2f}$ & {spearman_p_value_str} \\\\ \n"
        table_str += f" & & Kruskal-Wallis & $H={kruskal_result.statistic:.2f}$ & {kruskal_p_value_str} \\\\ \n"


    return table_str

def neuropath_stats(neuropath_var, proba_var, df_, labels = None, save_path = None, lowess = False, figsize = (4,6)):
    warnings.filterwarnings('ignore')

    fontsize = 14
    fontname = 'Nimbus Sans'

    if proba_var == 'amy_label_prob':
        labels["Y_axis"] = "P(Aβ)"
        # color_palette = 'coolwarm'
        color_palette = 'green'
    elif proba_var == 'tau_label_prob':
        labels["Y_axis"] = "P(τ)"
        # color_palette = 'PRGn'
        color_palette = 'purple'
    else:
        raise ValueError("Invalid probability variable")

    #stats--------------------------
    df = df_.dropna(subset=[neuropath_var, proba_var])
    latex_string = format_median_iqr(df, neuropath_var, proba_var, labels)

    try:
        correlation, spearman_p_value = spearmanr(df[neuropath_var], df[proba_var])
    except Exception as e:
        print(f"Error calculating Spearman correlation: {e}")
        correlation, spearman_p_value = np.nan, np.nan
        
    print(f"Spearman correlation between {neuropath_var} & {proba_var}: {correlation}, P-value: {spearman_p_value}")
    pairwise_p_values, kruskal_result = perform_statistical_tests(df, neuropath_var, proba_var)
    print(pairwise_p_values)

    # Add test results to LaTeX table
    latex_stats_string = add_to_latex_table(kruskal_result, correlation, spearman_p_value, labels)
    print(latex_stats_string)

    #plotting--------------------------

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(x=neuropath_var, y=proba_var, data=df, 
                color=color_palette, #palette=color_palette,
                ax=ax, boxprops=dict(alpha=.3),
                whiskerprops=dict(color='black', linewidth=1), showfliers=False)

    cohort_colors = ["#FFC20A", "#0C7BDC"]
    a = ax.scatter([1, 2], [3, 4], marker='o')  # Square marker
    b = ax.scatter([1, 2], [3, 4], marker='^')  # Triangle marker
    square_mk, = a.get_paths()
    triangle_up_mk, = b.get_paths()
    a.remove()  # Remove the dummy plot
    b.remove()  # Remove the dummy plot

    # Create a swarmplot with dodging
    ax = sns.swarmplot(x=neuropath_var, y=proba_var, hue="COHORT", data=df, size=6, ax=ax, dodge=True, palette=cohort_colors, alpha=0.8, edgecolor='black', linewidth=0.3)

    # Number of unique hue categories (sex)
    N_hues = 2

    # Retrieve collections created by swarmplot and adjust their marker shapes
    collections = ax.collections
    for collection in collections[::N_hues]:
        collection.set_paths([triangle_up_mk])  # Set triangle marker
    for collection in collections[1::N_hues]:
        collection.set_paths([square_mk])  # Set square marker
    
    # store the legend information - which marker is which cohort
    legend = ax.get_legend()

    ax.legend_.remove()


    if lowess:        
        text_str = f'ρ = {correlation:.2f}, p = {spearman_p_value:.2e}'
        ax.annotate(text_str, xy=(0.95, 0.05), xycoords='axes fraction', fontsize=13,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'),
                ha='right', va='bottom', fontname=fontname)
        sorted_df = df.sort_values(by=neuropath_var)
        x_mapped = sorted_df[neuropath_var].astype('category').cat.codes
        # print(x_mapped)
        lowess_results = sm.nonparametric.lowess(sorted_df[proba_var], x_mapped, frac=1)
        # Convert mapped x values back to original for plotting
        ax.plot(x_mapped, lowess_results[:, 1], ':', color='red', label='Lowess trend')



    
    y_values = df.groupby(neuropath_var)[proba_var].max().values
    max_height = add_significance_annotations(ax, pairwise_p_values, ax.get_xticks(), y_values, 0.1)
    plt.ylim(0, max_height + 0.15)
    plt.xlabel(labels["X_axis"], fontsize=fontsize + 2, fontname=fontname)
    plt.ylabel(labels["Y_axis"], fontsize=fontsize, fontname=fontname)

    current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    current_labels = [label.split('.')[0] for label in current_labels]  # Remove the decimal points
    # print("Current labels:", current_labels)  # See what labels are fetched from the plot

    new_labels = [labels["X_tics"].get(label, "Unknown Label") for label in current_labels]
    # print("New labels to set:", new_labels)  # See the mapping results

    ax.set_xticklabels(new_labels, rotation=labels["Rotation"], fontsize=fontsize, fontname=fontname)

    ytick_positions = np.linspace(0.0, 1.0, 6)
    ytick_labels = [f'{ytick:.1f}' for ytick in ytick_positions]

    # Set the positions and labels for the y-ticks
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=fontsize, fontname=fontname)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return legend, latex_string, latex_stats_string

def plot(config):
    # Load the data
    combined = pd.read_csv(config['source_data']['fig6'])
    lowess_all = True
    latex_table = "\\begin{longtable}[ht]\n\\centering\n\\begin{tabular}{lll}\nGroup & Median & IQR \\\\\n\\hline\n"
    latex_stats_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{p{4cm}llll}\nNeuropathological Variable & Model Probability & Test & Statistic & P-value \\\\\n\\hline\n"

    #Thal phase gets saved as 6a
    print("Plotting figure 6a")
    npthal_labels = {
        "X_axis": "Thal phase for amyloid plaques",
        "X_tics": {"0": "P0 (A0)", "1": "P1 (A1)", "2": "P2 (A1)", "3": "P3 (A2)", "4": "P4 (A3)", "5": "P5 (A3)"},
        "Rotation": 90,
    }

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(A\\beta)}}$ vs \\textbf{{{npthal_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPTHAL', 'amy_label_prob', combined, npthal_labels, save_path=config['output']['fig6a'] + '_amy', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\cline{2-5}\n"

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(\\tau)}}$ vs \\textbf{{{npthal_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPTHAL', 'tau_label_prob', combined, npthal_labels, save_path=config['output']['fig6a'] + '_tau', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\hline\n"

    # Braak stage for neurofibrillary degeneration should get saved as fig6b
    print("Plotting figure 6b")
    npbraak_labels = {
        "X_axis": "Braak stage for \n neurofibrillary degeneration (NFD)",
        "X_tics": {"0": "S0 (B0)", "1": "S1 (B1)", "2": "S2 (B1)", "3": "S3 (B2)", "4": "S4 (B2)", "5": "S5 (B3)", "6": "S6 (B3)", "7": "Other tauopathy"},
        "Rotation": 90,
    }

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(A\\beta)}}$ vs \\textbf{{{npbraak_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPBRAAK', 'amy_label_prob', combined[combined['NPBRAAK'] != 7], npbraak_labels, save_path=config['output']['fig6b'] + '_amy', lowess=lowess_all, figsize=(4, 8))
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\cline{2-5}\n"

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(\\tau)}}$ vs \\textbf{{{npbraak_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPBRAAK', 'tau_label_prob', combined[combined['NPBRAAK'] != 7], npbraak_labels, save_path=config['output']['fig6b'] + '_tau', lowess=lowess_all, figsize=(4, 8))
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\hline\n"

    # CERAD score is fig6c
    print("Plotting figure 6c")
    npneur_labels = {
        "X_axis": "CERAD score for density \nof neocortical neuritic plaque",
        "X_tics": {"0": "C0", "1": "C1", "2": "C2", "3": "C3"},
        "Rotation": 0,
    }

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(A\\beta)}}$ vs \\textbf{{{npneur_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPNEUR', 'amy_label_prob', combined, npneur_labels, save_path=config['output']['fig6c'] + '_amy', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\cline{2-5}\n"

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(\\tau)}}$ vs \\textbf{{{npneur_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPNEUR', 'tau_label_prob', combined, npneur_labels, save_path=config['output']['fig6c'] + '_tau', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\hline\n"

    # Cerebral amyloid angiopathy should get saved as fig6d
    print("Plotting figure 6d")
    npamy_labels = {
        "X_axis": "Cerebral amyloid angiopathy",
        "X_tics": {"0": "None", "1": "Mild", "2": "Moderate", "3": "Severe"},
        "Rotation": 90,
    }

    latex_table += f"\\multicolumn{{3}}{{l}}{{$\\bm{{P(A\\beta)}}$ vs \\textbf{{{npamy_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPAMY', 'amy_label_prob', combined, npamy_labels, save_path=config['output']['fig6d'] + '_amy', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\cline{2-5}\n"

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(\\tau)}}$ vs \\textbf{{{npamy_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPAMY', 'tau_label_prob', combined, npamy_labels, save_path=config['output']['fig6d'] + '_tau', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\hline\n"

    # CERAD for diffuse plaques is fig6e
    print("Plotting figure 6e")
    npdiff_labels = {
        "X_axis": "CERAD score for \ndensity of diffuse plaques",
        "X_tics": {"0": "No diffuse plaques", "1": "Sparse diffuse plaques", "2": "Moderate diffuse plaques", "3": "Frequent diffuse plaques"},
        "Rotation": 90,
    }

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(A\\beta)}}$ vs \\textbf{{{npdiff_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPDIFF', 'amy_label_prob', combined, npdiff_labels, save_path=config['output']['fig6e'] + '_amy', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\cline{2-5}\n"

    latex_table += f"\hline\n\\multicolumn{{3}}{{l}}{{$\\bm{{P(\\tau)}}$ vs \\textbf{{{npdiff_labels['X_axis']}}}}} \\\\\n\hline\n"
    legend, latex_medians_string, latex_stats_string = neuropath_stats('NPDIFF', 'tau_label_prob', combined, npdiff_labels, save_path=config['output']['fig6e'] + '_tau', lowess=lowess_all)
    latex_table += latex_medians_string
    latex_stats_table += latex_stats_string + "\\hline\n"


    latex_table += "\\end{longtable}\n\\caption{Statistical results for neuropathological validation}\n\\end{table}"
    latex_stats_table += "\\end{tabular}\n\\caption{Statistical results for neuropathological validation}\n\\end{table}"

    # print(latex_table)