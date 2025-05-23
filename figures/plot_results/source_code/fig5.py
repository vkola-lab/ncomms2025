import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PyComplexHeatmap import *


def get_df_row():

    stage_dict = {
                'left entorhinal': 'Stage I-II',
                'right entorhinal': 'Stage I-II',
                'left hippocampus': 'Stage I-II',
                'right hippocampus': 'Stage I-II',
                'left amygdala': 'Stage III',
                'right amygdala': 'Stage III',
                'left parahippocampal': 'Stage III',
                'right parahippocampal': 'Stage III',
                'left fusiform': 'Stage III',
                'right fusiform': 'Stage III',
                'left lingual': 'Stage III',
                'right lingual': 'Stage III',
                'left inferior temporal': 'Stage IV',
                'right inferior temporal': 'Stage IV',
                'left middle temporal': 'Stage IV',
                'right middle temporal': 'Stage IV',
                'left insula': 'Stage IV',
                'right insula': 'Stage IV',
                'left posterior cingulate': 'Stage IV',
                'right posterior cingulate': 'Stage IV',
                'left isthmus cingulate': 'Stage IV',
                'right isthmus cingulate': 'Stage IV',
                'left caudal anterior cingulate': 'Stage IV',
                'right caudal anterior cingulate': 'Stage IV',
                'left rostral anterior cingulate': 'Stage IV',
                'right rostral anterior cingulate': 'Stage IV',
                'left superior temporal': 'Stage V',
                'right superior temporal': 'Stage V',
                'left transverse temporal': 'Stage V',
                'right transverse temporal': 'Stage V',
                'left inferior parietal': 'Stage V',
                'right inferior parietal': 'Stage V',
                'left supramarginal': 'Stage V',
                'right supramarginal': 'Stage V',
                'left precuneus': 'Stage V',
                'right precuneus': 'Stage V',
                'left superior parietal': 'Stage V',
                'right superior parietal': 'Stage V',
                'left lateral occipital': 'Stage V',
                'right lateral occipital': 'Stage V',
                'left lateral orbitofrontal': 'Stage V',
                'right lateral orbitofrontal': 'Stage V',
                'left medial orbitofrontal': 'Stage V',
                'right medial orbitofrontal': 'Stage V',
                'left pars opercularis': 'Stage V',
                'right pars opercularis': 'Stage V',
                'left pars orbitalis': 'Stage V',
                'right pars orbitalis': 'Stage V',
                'left pars triangularis': 'Stage V',
                'right pars triangularis': 'Stage V',
                'left caudal middle frontal': 'Stage V',
                'right caudal middle frontal': 'Stage V',
                'left rostral middle frontal': 'Stage V',
                'right rostral middle frontal': 'Stage V',
                'left superior frontal': 'Stage V',
                'right superior frontal': 'Stage V',
                'left postcentral': 'Stage VI',
                'right postcentral': 'Stage VI',
                'left precentral': 'Stage VI',
                'right precentral': 'Stage VI',
                'left cuneus': 'Stage VI',
                'right cuneus': 'Stage VI',
                'left pericalcarine': 'Stage VI',
                'right pericalcarine': 'Stage VI'}
    
    return pd.DataFrame.from_dict(stage_dict, orient='index', columns=['stage'])


def get_df_col(community_data, stats_sig):

    df_col = community_data[['community_full','label','graph']].drop_duplicates()
    df_col['star'] = df_col['label'].replace(stats_sig)
    label_names = ['med-temp', 'lat-temp', 'med-par', 'lat-par', 'frontal', 'occipital']
    order_map = {k: i for i, k in enumerate(label_names)}
    df_col['category_sort_key'] = df_col['label'].map(order_map)
    df_col = df_col.sort_values(
        by=['category_sort_key', 'graph', 'community_full'],
        ascending=[True, True, True])
    df_col = df_col.drop('category_sort_key', axis=1)
    df_col.set_index('community_full', inplace=True)
    
    return df_col


def plot_graph_communities(community_data, df_row, df_col, cmap, figname):

    community_full = community_data[['community_full','graph']].drop_duplicates().set_index('community_full')
    approach_colors = {'SHAP-derived graph':cmap[0], 'SUVR-derived graph':cmap[1]}
    community_full['color'] = community_full['graph'].replace(approach_colors)
    comm_full_colors = community_full['color'].to_dict()
    comm_full_markers = community_data[['community_full','marker']].drop_duplicates().set_index('community_full').to_dict()['marker']
    
    stage_order = ['Stage I-II', 'Stage III', 'Stage IV', 'Stage V', 'Stage VI']
    stage_colors = {'Stage I-II': colormaps.get_cmap('Pastel2')(np.linspace(0,1,8))[7],
                    'Stage III':  colormaps.get_cmap('Pastel2')(np.linspace(0,1,8))[1],
                    'Stage IV':   colormaps.get_cmap('Pastel2')(np.linspace(0,1,8))[2],
                    'Stage V':    colormaps.get_cmap('Pastel2')(np.linspace(0,1,8))[3],
                    'Stage VI':   colormaps.get_cmap('Pastel2')(np.linspace(0,1,8))[4]}

    label_names = ['med-temp', 'lat-temp', 'med-par', 'lat-par', 'frontal', 'occipital']
    label_colors = {'med-temp':   colormaps.get_cmap('Set1')(np.linspace(0,1,9))[0],
                    'lat-temp':   colormaps.get_cmap('Set1')(np.linspace(0,1,9))[1],
                    'med-par':    colormaps.get_cmap('Set1')(np.linspace(0,1,9))[2],
                    'lat-par':    colormaps.get_cmap('Set1')(np.linspace(0,1,9))[3],
                    'frontal':    colormaps.get_cmap('Set1')(np.linspace(0,1,9))[4],
                    'occipital':  colormaps.get_cmap('Set1')(np.linspace(0,1,9))[6]}

    row_ha = HeatmapAnnotation(label=anno_simple(df_row.stage, add_text=False, legend=False, text_kws={'color':'black'},
                                                 colors=stage_colors, height=6, legend_kws={'color_text':False}),
                               label0=anno_label(df_row.stage, merge=True, colors='black', rotation=0, height=4, fontsize=10),
                               label_side='top', label_kws={'fontsize':'1', 'color':'white'}, verbose=0, axis=0,
                               hgap=0.5, plot_legend=True)

    col_ha = HeatmapAnnotation(label0=anno_label(df_col.star, merge=True, colors='black', rotation=90, height=1, arrowprops=dict(visible=False)),
                               label=anno_simple(df_col.label, colors=label_colors, height=6, legend=True,
                                                 add_text=True, text_kws={'color':'white','fontsize':8}, legend_kws={'color_text':False,'fontsize':10}),
                               graph=anno_simple(df_col.graph, colors=approach_colors, height=6, legend=True,
                                                 add_text=False, text_kws={'color':'white','fontsize':8}, legend_kws=
                                                 {'labels':['SHAP-derived graph','SUVr-derived graph'],'color_text':False,'fontsize':10}),
                               verbose=0, label_side='right', label_kws={'fontsize':'10','color':'black'},
                               hgap=0.5, plot_legend=True)

    plt.figure(figsize=(12,15))

    cm = DotClustermapPlotter(community_data, x='community_full', y='index', value='value', hue='community_full',
                              colors=comm_full_colors, marker=comm_full_markers,
                              vmax=1, vmin=0, s='value', max_s=50, grid='minor', verbose=0,
                              top_annotation=col_ha, right_annotation=row_ha,
                              col_split=df_col.label, col_split_order=label_names, col_cluster=False, col_split_gap=2,
                              row_split=df_row.stage, row_split_order=stage_order, row_cluster=False, row_split_gap=2,
                              show_colnames=False, show_rownames=True, row_names_side='left',
                              legend_hpad=2, subplot_gap=3, legend_vpad=30, legend_gap=15, spines=True, legend=False)

    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()


def plot(config):

    print("Plotting figure 5")
    community_data = pd.read_csv(config['source_data']['fig5'])
    df_row = get_df_row()
    stats_sig = {'med-temp': '**',
                 'lat-temp': '**',
                 'med-par': '*',
                 'lat-par': '**',
                 'frontal': '*',
                 'occipital': '**'}
    df_col = get_df_col(community_data, stats_sig)
    cmap = ['#E0B542', '#C25954']
    figname = config['output']['fig5']
    
    plot_graph_communities(community_data, df_row, df_col, cmap, figname)
