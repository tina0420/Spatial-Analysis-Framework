"""
@author: Tina Hsu
Showing the cell types in different layers per image and filter out the images with no or less than 10% epithelial cells
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import matplotlib.pyplot as plt
from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT, LAYER2_CELL_TYPE_COLOR_DICT, \
    E_TME_COLOR_DICT
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from matplotlib import gridspec


def plot_cell_type_prop_per_img_bar(adata, unit_key, cluster_key, color_dict, out_dir, cluster_key_order=None,
                                    figsize=(50, 10), name=None):
    # Compute the proportion of each cell type
    counts = adata.obs.pivot_table(
        index=unit_key,
        columns=cluster_key,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)
    # print(prop_df)

    # Run hierachical clustering
    Z = linkage(prop_df, method="complete", metric="euclidean")
    # reorder the images based on clustering result
    new_idx_order = leaves_list(Z)

    if cluster_key_order:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], cluster_key_order]
    else:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], :]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # For adding dendrogram to the top of the bar chart, but need modification (layout not good)
    # gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    # plt.subplots_adjust(hspace=0.05)
    #
    # # --- Top: Dendrogram
    # ax1 = fig.add_subplot(gs[0])
    # dendrogram(Z, labels=new_order_df.index, ax=ax1)
    # ax1.tick_params(bottom=False, labelbottom=False)
    #
    # # --- Bottom: Stacked Bar Chart
    # ax2 = fig.add_subplot(gs[1])
    #
    # # Plot stacked bars
    bottom = np.zeros(new_order_df.shape[0])
    for cell_type in new_order_df.columns:
        bars = ax.bar(new_order_df.index, new_order_df[cell_type],
                      bottom=bottom, color=color_dict[cell_type], label=cell_type)
        bottom += new_order_df[cell_type].values

    # Some setting for the figure
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(range(new_order_df.shape[0]))
    ax.set_xticklabels(list(new_order_df.index), rotation=90, fontsize=30, va='top')
    ax.set_xlim(-0.25, new_order_df.shape[0] - 0.75)
    ax.tick_params(axis='x', pad=2)
    plt.subplots_adjust(bottom=0.25)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='y', labelsize=35)
    ax.set_ylabel('Proportion', fontsize=35)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Cell Types", fontsize=30, title_fontsize=35,
              bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False,
              )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_proportion_bar.png", dpi=500)
    else:
        plt.savefig(out_dir / "proportion_bar.png", dpi=500)

    plt.clf()


if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/abundance")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]

    adata = FULL_ADATA.copy()

    # for layer1, esi and layer2 cell type proportions across all images
    UNIT_KEY = "unique_core_id"
    CLUSTER_KEY = "layer1"
    plot_cell_type_prop_per_img_bar(adata, UNIT_KEY, CLUSTER_KEY, E_TME_COLOR_DICT,
                                    out_dir, cluster_key_order=None, figsize=(50, 25), name="layer1_ct")

    CLUSTER_KEY = "ESI"
    CLUSTER_KEY_ORDER = ["Stromal", "Immune", "Epithelial"]
    plot_cell_type_prop_per_img_bar(adata, UNIT_KEY, CLUSTER_KEY, ESI_CELL_TYPE_COLOR_DICT,
                                    out_dir, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="esi_ct")

    CLUSTER_KEY = "layer2"
    LAYER2_CELL_TYPE_COLOR_DICT = {'Epithelial CK5+': '#f10024',
                                   'Epithelial CK5+CK7+': '#ffb6c1',
                                   'Epithelial CK7+': '#ae8ae7',
                                   'Epithelial CK8-18+': '#ef8616',
                                   'Epithelial CK_low': '#7639d6',
                                   'Epithelial H3K27me3+': '#d0be6b',
                                   'Myoepithelial': '#8e5112',
                                   'B Cell': '#178867',
                                   'T Cell': '#c3e78a',
                                   'Macrophage': '#9ac4a0',
                                   'Blood Endothelial': '#5ea3cd',
                                   'Smooth Muscle Cell': '#01588d',
                                   'Fibroblast': '#bbe4fc',
                                   'Vimentin+': '#92b6cb'}
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[::-1]
    plot_cell_type_prop_per_img_bar(adata, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    out_dir, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25),
                                    name="layer2_ct")

    import scanpy as sc

    dp = sc.pl.dotplot(adata, adata.var_names, groupby='layer2', figsize=(7, 7), dendrogram=True)
    dp.figure.tight_layout()
    plt.show()

    UNIT_KEY = "cid"
    CLUSTER_KEY = "layer2"
    CLUSTER_KEY_ORDER = ['Epithelial CK5+',
                         'Epithelial CK5+CK7+',
                         'Epithelial CK7+',
                         'Epithelial CK8-18+',
                         'Epithelial CK_low',
                         'Epithelial H3K27me3+',
                         'Myoepithelial',
                         'B Cell',
                         'T Cell',
                         'Macrophage',
                         'Blood Endothelial',
                         'Fibroblast',
                         'Smooth Muscle Cell',
                         'Vimentin+'][::-1]
    CLUSTER_KEY_ORDER = ['B Cell',
                         'T Cell',
                         'Macrophage',
                         'Blood Endothelial',
                         'Fibroblast',
                         'Smooth Muscle Cell',
                         'Vimentin+'][::-1]
    t_adata = adata_analysis[adata_analysis.obs.layer2.isin(CLUSTER_KEY_ORDER)].copy()
    plot_cell_type_prop_per_img_bar(t_adata, UNIT_KEY, CLUSTER_KEY,
                                    t_adata.uns['LAYER2_CELL_TYPE_COLOR_DICT'],
                                    out_dir, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="analysis_cid_tme")

    # Check the diversity of intra-tumour
    def compute_spatial_entropy(p_list):
        """
        Compute spatial entropy for each cell
        :param p_list: probabilities of all cell types among the neighbors of each cell
        :return: entropy value
        """
        entropy = (-1) * sum([x * np.log(x) if x > 0 else 0 for x in p_list])
        return entropy


    counts = adata_analysis.obs.pivot_table(
        index=UNIT_KEY,
        columns="layer2",
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)

    # Shannon diversity
    score_shannon = []
    score_pielou = []
    for i in range(prop_df.shape[0]):
        score = compute_spatial_entropy(prop_df.iloc[i, :])
        score_shannon.append(score)
        non_zero_count = prop_df.iloc[i, :].gt(0).sum()
        if non_zero_count > 0:
            score_pielou.append(score / np.log(non_zero_count))
        else:
            score_pielou.append(0)


