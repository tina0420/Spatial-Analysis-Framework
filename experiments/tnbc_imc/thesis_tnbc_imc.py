"""
Functions that are used in thesis plotting
"""
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from src.visualize import plot_umap, plot_umap_panel, plot_cluster_cell_count
from src.visualize import plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap, \
    plot_spatial_cluster_distribution, plot_spatial_cluster_panel, plot_edge_count
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from matplotlib import gridspec
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from itertools import combinations
import math
from tqdm import tqdm
from scipy.stats import combine_pvalues, tmean
from statsmodels.stats.multitest import multipletests
from scipy.spatial import cKDTree
from src.preprocess import read_neighbors
from src.spatial_stats import get_interacting_pairs
from src.annotate import update_labels
from src.spatial_stats import get_interacting_neighborhood_cell_idxs
from experiments.tnbc_imc.tnbc_imc_setup_analysis_e import *

"""
Visualization: Distribution of layer_2 cell type in each image.
    CLUSTER_KEY = "layer_2"
    UNIT_KEY = "unique_img_key"
    OBSERVATION_KEY = "Recurrence status"
    for pid in pd.unique(FULL_ADATA.obs[UNIT_KEY]):
        print(pid)
        p_adata = FULL_ADATA[FULL_ADATA.obs[UNIT_KEY] == pid]

        if p_adata.obs[OBSERVATION_KEY].values[0] == 0:
            TEMP_OUT_DIR = OUT_DIR / "layer_2/No recurrence"
            if pid.split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5, 5), dotsize=3.5,
                                                  color=LAYER2_CELL_TYPE_COLOR_DICT,
                                                  name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5, 5), dotsize=2.5,
                                                  color=LAYER2_CELL_TYPE_COLOR_DICT,
                                                  name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)

        elif p_adata.obs[OBSERVATION_KEY].values[0] == 1:
            TEMP_OUT_DIR = OUT_DIR / "layer_2/Recurrence"
            if pid.split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5, 5), dotsize=3.5,
                                                  color=LAYER2_CELL_TYPE_COLOR_DICT,
                                                  name=f"R_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5, 5), dotsize=2.5,
                                                  color=LAYER2_CELL_TYPE_COLOR_DICT,
                                                  name=f"R_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
        else:
            print("Wrong!")

"""

"""
Barplot showing the cell type proportions across all patients/images.
    UNIT_KEY = "CID"
    CLUSTER_KEY = "layer_1"
    CLUSTER_KEY_ORDER = ["Stromal", "Immune", "Epithelial"]
    plot_cell_type_prop_per_img_bar(FULL_ADATA, UNIT_KEY, CLUSTER_KEY, ESI_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="cid_esi")

    # All layer 2 cell types
    CLUSTER_KEY = "layer_2"
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[::-1]
    plot_cell_type_prop_per_img_bar(FULL_ADATA, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="CID_cell_type_prop")
    # roi by roi
    UNIT_KEY = "unique_img_key"
    plot_cell_type_prop_per_img_bar(FULL_ADATA, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="cid_layer2_all")

    # For epithelial, immune and stromal compartments
    sub_adata = FULL_ADATA[FULL_ADATA.obs.layer_1 == "Epithelial"]
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[:4][::-1]
    plot_cell_type_prop_per_img_bar(sub_adata, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="cid_layer2_epi")

    sub_adata = FULL_ADATA[FULL_ADATA.obs.layer_1 == "Immune"]
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[4:7][::-1]
    plot_cell_type_prop_per_img_bar(sub_adata, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25),
                                    name="cid_layer2_immune")

    sub_adata = FULL_ADATA[FULL_ADATA.obs.layer_1 == "Stromal"]
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[7:][::-1]
    plot_cell_type_prop_per_img_bar(sub_adata, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25),
                                    name="cid_layer2_stromal")

    # immune and stromal proportions in TME compartment
    sub_adata = FULL_ADATA[FULL_ADATA.obs.layer_1 != "Epithelial"]
    CLUSTER_KEY_ORDER = list(LAYER2_CELL_TYPE_COLOR_DICT.keys())[4:][::-1]
    plot_cell_type_prop_per_img_bar(sub_adata, UNIT_KEY, CLUSTER_KEY, LAYER2_CELL_TYPE_COLOR_DICT,
                                    OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25),
                                    name="cid_layer2_tme")
    
    # Plot for cell niche proportion per patient
    UNIT_KEY = "CID"
    CLUSTER_KEY = "cn_cluster_name"
    CLUSTER_KEY_ORDER = list(CELL_NICHE_COLOR_DICT.keys())[::-1]
    plot_cluster_proportion_bar(FULL_ADATA, UNIT_KEY, CLUSTER_KEY, CELL_NICHE_COLOR_DICT,
                                OUT_DIR, cluster_key_order=CLUSTER_KEY_ORDER, figsize=(50, 25), name="Cell niche")
    
"""


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

    # Plot stacked bars
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

    ax.set_xlim(-0.25, new_order_df.shape[0] - 0.75)

    ax.set_xticks([])
    ax.set_xlabel("Patient", fontsize=55, labelpad=15)
    plt.subplots_adjust(bottom=0.25)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([int(0), 0.5, int(1)])
    ax.tick_params(axis='y', labelsize=45)
    ax.set_ylabel('Proportion', fontsize=55, labelpad=15)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Cell Type", fontsize=40, title_fontsize=45,
              bbox_to_anchor=(1.005, 0.5), loc='center left', frameon=False,
              )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_proportion_bar.png", dpi=200)
    else:
        plt.savefig(out_dir / "proportion_bar.png", dpi=200)

    plt.clf()


def plot_cluster_proportion_bar(adata, unit_key, cluster_key, color_dict, out_dir, cluster_key_order=None,
                                figsize=(50, 10), name=None):
    # Compute the proportion of each cell type
    counts = adata.obs.pivot_table(
        index=unit_key,
        columns=cluster_key,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)

    # Run hierachical clustering
    Z = linkage(prop_df, method="complete", metric="euclidean")
    # reorder the images based on clustering result
    new_idx_order = leaves_list(Z)

    if cluster_key_order:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], cluster_key_order]
    else:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], :]

    # plot stacked bars
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bottom = np.zeros(new_order_df.shape[0])
    for cell_type in new_order_df.columns:
        bars = ax.bar(new_order_df.index, new_order_df[cell_type], bottom=bottom, color=color_dict[cell_type],
                      label=cell_type, width=0.9, edgecolor='none', linewidth=0)
        bottom += new_order_df[cell_type].values

    # Some setting for the figure
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlim(-0.25, new_order_df.shape[0] - 0.75)

    ax.set_xticks([])
    ax.set_xlabel("Patient", fontsize=55, labelpad=15)
    plt.subplots_adjust(bottom=0.25)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([int(0), 0.5, int(1)])
    ax.tick_params(axis='y', labelsize=45)
    ax.set_ylabel('Proportion', fontsize=55, labelpad=15)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=f"{name}", fontsize=40, title_fontsize=45,
              bbox_to_anchor=(1.005, 0.5), loc='center left', frameon=False,
              )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_proportion_bar.png", dpi=200)
    else:
        plt.savefig(out_dir / "proportion_bar.png", dpi=200)

    plt.clf()


"""
UMAP to see how distinct are the cell types.
    sc.tl.pca(FULL_ADATA, svd_solver='arpack')
    sc.pp.neighbors(FULL_ADATA, use_rep='X_pca')
    sc.tl.umap(FULL_ADATA)

    CLUSTER_KEY = "layer_2"
    plot_umap(FULL_ADATA, CLUSTER_KEY, OUT_DIR, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name="layer_2_cell_type")
    plot_umap_panel(FULL_ADATA, CLUSTER_KEY, OUT_DIR, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
                    dotsize=0.1, show_counts=False, name="layer_2_cell_type", dpi=350)
    
    CLUSTER_KEY = "cn_cluster_name"
    plot_umap_panel(FULL_ADATA, CLUSTER_KEY, OUT_DIR, color_dict=CELL_NICHE_COLOR_DICT,
                    dotsize=0.1, show_counts=False, name="cell_niche", dpi=350)
"""

"""
Cell type mean marker expression and count in the whole dataset
    CLUSTER_KEY = "layer_2"
    plot_zscore_cluster_mean_heatmap(FULL_ADATA, FULL_ADATA.var_names, CLUSTER_KEY, vmax=3, vmin=-1,
                                     out_dir=OUT_DIR, figsize=(5, 3), name=f"cell_type_range3-1")
    
    # Heatmap 
    CLUSTER_KEY = "cn_cluster_name"                        
    plot_zscore_cluster_mean_heatmap(FULL_ADATA, FULL_ADATA.var_names, CLUSTER_KEY, vmax=2.5, vmin=-1,
                                     out_dir=OUT_DIR, figsize=(5, 3), name=f"cell_niche__range2.5-1")

    
"""

"""
Cell counts in each cell type and cell niche
    CLUSTER_KEY = "layer_2"
    # Bar plot for cell type counts
    plot_cluster_cell_count(FULL_ADATA, CLUSTER_KEY, OUT_DIR, figsize=(3.5, 2), name="cell_type", dpi=250)
    
    CLUSTER_KEY = "cn_cluster_name"
    plot_cluster_cell_count(FULL_ADATA, CLUSTER_KEY, OUT_DIR, figsize=(3.5, 2), name="cell_niche", dpi=250)    
"""

"""
Marker expression distribution in each cell type /cell niche
    MARKER = "Ki67"
    MARKER = "H3K27me3"
    UNIT_KEY = "CID"
    CLUSTER_KEY = "layer_2"
    CLUSTER_KEY = "cn_cluster_name"
    color_dict = {0: 'tab:green', 1: 'tab:orange'}
    marker_intensity_by_clusters(FULL_ADATA, MARKER, CLUSTER_KEY, OUT_DIR, figsize=(7, 5), name="cell_niche")
    
    # for different observations (for cell types and cell niches individually)
    CLUSTER_KEY = "layer_2"
    UNIT_KEY = "CID"
    OBSERVATION_KEY = "Recurrence status"
    OBSERVATION_ORDER = [1, 0]
    LABEL_DICT = {1: "Recurrence", 0: "No recurrence"}
    COLOR_DICT = {0: 'tab:green', 1: 'tab:orange'}
    COLOR_DICT = {0: "lightblue", 1: "pink"}
    marker_intensity_in_clusters_across_observations(FULL_ADATA, UNIT_KEY, MARKER, CLUSTER_KEY, list(CELL_TYPE_COLOR_DICT.keys()), OBSERVATION_KEY,
                                                     OBSERVATION_ORDER, OUT_DIR, COLOR_DICT, LABEL_DICT, name="cell_type")
    
    CLUSTER_KEY = "cn_cluster_name"
    marker_intensity_in_clusters_across_observations(FULL_ADATA, UNIT_KEY, MARKER, CLUSTER_KEY, list(CELL_NICHE_COLOR_DICT.keys()), OBSERVATION_KEY,
                                                     OBSERVATION_ORDER, OUT_DIR, COLOR_DICT, LABEL_DICT, name="cell_niche")                    
                                                    
"""
idx = np.where(FULL_ADATA.var_names == "Ki67")[0][0]

labels = FULL_ADATA.obs["cn_cluster_name"].values
df = pd.DataFrame(FULL_ADATA.X[:, idx], columns=["Ki67"]).assign(label=labels)
results = {}
unique_cell_types = df['label'].unique()
for i in range(len(unique_cell_types) - 1):
    g1 = list(df[df['label'] == unique_cell_types[i]]["Ki67"])
    p_list = []
    for j in range(i + 1, len(unique_cell_types)):
        g2 = list(df[df['label'] == unique_cell_types[j]]["Ki67"])
        stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
        p_list.append(p_val)
    results[unique_cell_types[i]] = p_list


# Ki67, cell niche,  'T_Mp_Aggregation': [0.0,
#   0.0,
#   0.07912067880535087,
#   0.0,
#   0.0,
#   4.99306250715787e-31]

def marker_intensity_by_clusters(adata, target_marker, cluster_key, out_dir, figsize=(3.5, 2), name=None):
    # Get the index of the marker in var_names
    idx = np.where(adata.var_names == target_marker)[0][0]

    labels = adata.obs[cluster_key].values
    df = pd.DataFrame(adata.X[:, idx], columns=[target_marker]).assign(label=labels)
    df = df.sort_values(by="label", ascending=True)

    results = []
    unique_cell_types = df['label'].unique()

    for ct in unique_cell_types:
        # 1. Isolate the target group and the "rest" of the background
        group_target = df[df['label'] == ct][target_marker]
        group_rest = df[df['label'] != ct][target_marker]

        stat, p_val = mannwhitneyu(group_target, group_rest, alternative='two-sided')

        # 3. Store the results, including medians to see the direction of change
        results.append({
            'label': ct,
            'p_value': p_val,
            'median_expression': group_target.median(),
            'background_median': group_rest.median()
        })

    # Convert to a DataFrame for easy viewing and sorting
    results_df = pd.DataFrame(results)

    print(results_df)
    results_df.to_csv(out_dir / f"{target_marker}_intensity_{name}.csv", index=False)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # sns.violinplot(x="label", y=target_marker, data=df, ax=ax, color="lightgray")
    sns.boxplot(x="label", y=target_marker, data=df, ax=ax, color="lightgray", width=0.5)

    plt.grid(alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('', fontsize=8)
    plt.ylabel(f'{target_marker} Intensity', fontsize=10)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8, rotation=35, ha="right")

    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{target_marker}_intensity_{name}.png", dpi=300)
    else:
        plt.savefig(out_dir / "marker_intensity.png", dpi=300)
    plt.clf()


# Marker intensity in different clusters by specific clinical outcomes
def marker_intensity_in_clusters_across_observations(adata, unit_key, marker, cluster_key, cell_type_list,
                                                     observation_key,
                                                     observation_order, out_dir, color_dict, observation_dict,
                                                     name=None):
    """
    Compare marker intensity of each category (e.g. cell type) across different observations
    :param adata: Anndata
    :param marker: indicate which marker intensity to compare, type: str
    :param cluster_key: indicate the column used in obs to show comparison, type: str
    :param observation_key: indicate the column in obs with observation, type: str
    :param observation_order: indicate the order of observation , type: str
    :param out_dir: output directory, type: Path
    :param color_dict: colour setting for observations, type: dict
    :param observation_dict: the dict of observations , type: dict
    :param name: name for the output file, type: str
    :return:
    """
    # Get the index of the marker in X
    idx = np.where(adata.var_names == marker)[0][0]

    # Get two groups in the observations
    group1 = adata[(adata.obs[observation_key] == observation_order[0])]
    group2 = adata[(adata.obs[observation_key] == observation_order[1])]

    # Plot
    rows = int(len(cell_type_list) / 3) + 1

    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 1.7))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(cell_type_list) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = cell_type_list[i]
        print(ct)
        sub_group1 = list(group1.X[np.where(group1.obs[cluster_key] == ct)[0], idx])
        sub_group2 = list(group2.X[np.where(group2.obs[cluster_key] == ct)[0], idx])

        sns.violinplot(data=[sub_group1, sub_group2],
                       palette=[color_dict[observation_order[0]],
                                color_dict[observation_order[1]]], ax=ax)

        # Add lines from the median values
        ax.plot([0, 1], [np.median(sub_group1), np.median(sub_group2)], color='red', linestyle='--', linewidth=1.5,
                zorder=1)
        if len(sub_group1) > 0 and len(sub_group2) > 0:
            ax.set_yticks([0, np.round(max(max(sub_group1), max(sub_group2)), 1)])
        else:
            print(len(sub_group1), len(sub_group2))

        # Calculate p-value
        stat, pval = mannwhitneyu(sub_group1, sub_group2, alternative='two-sided')

        # Add p-value to the figure
        if pval < 0.01:
            ax.text(0.5, 0.8, "p < 0.01", ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.8, f'p = {pval:.3f}', ha='center', va='center', fontsize=12, transform=ax.transAxes)

        # Calculate row and column index based on i
        row_idx = i // 3
        col_idx = i % 3

        if row_idx < rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            # Ensure the labels are visible for the last row
            ax.tick_params(labelbottom=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(list(observation_dict.values()), size=9)

        # Logic for Y-ticks: Only show on the first column
        if col_idx > 0:
            ax.tick_params(labelleft=False)
        else:
            # Ensure labels are visible for the first column
            ax.tick_params(labelleft=True)

        ax.grid(False)

        ax.set_title(
            cell_type_list[i], weight="bold", fontsize=13)

    # Add the marker name to the y-label of the figure
    fig.supylabel(f"{marker} Intensity", fontsize=13, fontweight='bold')

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]],
                            label=observation_dict[observation_order[0]] + " (n=" + str(
                                len(set(group1.obs[unit_key]))) + ")")
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]],
                            label=observation_dict[observation_order[1]] + " (n=" + str(
                                len(set(group2.obs[unit_key]))) + ")")

    fig.legend(title=observation_key,
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=2,
               fontsize=9,
               title_fontsize=10
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_{marker}_intensity_panel.png", dpi=300)
    else:
        plt.savefig(out_dir / f"{marker}_intensity_panel.png", dpi=300)

    plt.clf()


"""
Cell type abundance across different outcomes (recurrence status and pCR)
    # Compare cell type proportions between different observations (cell type count/epi-TME counts)
    TARGET_CT_GROUP_DICT = dict(zip(['CK5+ Basal', 'CK7+ Luminal', 'CK8/18+ Luminal', 'CK+ Epithelial',
                                     'T-cell', 'B-cell', 'Macrophage', 'Endothelial', 'SMA+ Stromal', 'Stromal'],
                                    ["Epithelial", "Epithelial", "Epithelial", "Epithelial",
                                     "TME", "TME", "TME", "TME", "TME", "TME"]))
    TEMP_ADATA = FULL_ADATA.copy()
    TEMP_ADATA.obs["layer_1"] = [TARGET_CT_GROUP_DICT[ct] for ct in TEMP_ADATA.obs["layer_2"].values]
    CLUSTER_KEY = "layer_2"
    UNIT_KEY = "CID"
    OBSERVATION_KEY = "Recurrence status"
    OBSERVATION_ORDER = [1, 0]
    LABEL_DICT = {1: "Recurrence", 0: "No recurrence"}
    compare_cell_type_proportion(adata=TEMP_ADATA, unit_key=UNIT_KEY,
                                 target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                 cluster_key=CLUSTER_KEY, observation_key=OBSERVATION_KEY,
                                 observation_order=OBSERVATION_ORDER, label_dict=LABEL_DICT,
                                 out_dir=OUT_DIR, color_dict={0: 'tab:blue', 1: 'tab:pink'},
                                 name="Recurrence status_by CID")
    
"""


def compare_cell_type_proportion(adata, unit_key, target_ct_group_dict, cluster_key, observation_key, observation_order,
                                 label_dict, out_dir, color_dict, name=None, marker_focus_key=None):
    props = []
    for unit in pd.unique(adata.obs[unit_key]):
        p_adata = adata[adata.obs[unit_key] == unit].obs[["unique_img_key", "CID", "layer_1",
                                                          cluster_key, observation_key]]

        for target_ct in list(target_ct_group_dict.keys()):
            print(target_ct)
            if target_ct not in p_adata[cluster_key].values:
                prop = 0
            else:
                prop = p_adata[cluster_key].value_counts()[target_ct] / p_adata["layer_1"].value_counts()[
                    target_ct_group_dict[target_ct]]

            props.append([unit, target_ct, p_adata[observation_key].values[0], prop])

    props_df = pd.DataFrame(props, columns=[unit_key, "cell type", observation_key, "prop"])

    # Split groups based on the observation, calculate p-value and plot
    rows = int(len(target_ct_group_dict.keys()) / 3) + 1

    if observation_order:
        # Get two groups in the observations
        group1 = props_df[props_df[observation_key] == observation_order[0]]
        group2 = props_df[props_df[observation_key] == observation_order[1]]
    else:
        group1 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[0]]
        group2 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[1]]

    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 2))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(target_ct_group_dict.keys()) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = list(target_ct_group_dict.keys())[i]

        sns.boxplot(data=[list(group1[group1["cell type"] == ct]["prop"].values),
                          list(group2[group2["cell type"] == ct]["prop"].values)],
                    palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        stat, pval = mannwhitneyu(list(group1[group1["cell type"] == ct]["prop"].values),
                                  list(group2[group2["cell type"] == ct]["prop"].values),
                                  alternative='two-sided')
        # Add p-value to the figure
        p_val_loc = max(max(list(group1[group1["cell type"] == ct]["prop"].values)),
                        max(list(group2[group2["cell type"] == ct]["prop"].values)))
        if pval < 0.01:
            ax.text(0.5, p_val_loc, "p<0.01", ha='center', va='top', fontsize=11)

        else:
            ax.text(0.5, p_val_loc, f'p = {pval:.2f}', ha='center', va='top', fontsize=11)

        # Calculate row and column index based on i
        row_idx = i // 3
        col_idx = i % 3

        if row_idx < rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            # Ensure the labels are visible for the last row
            ax.tick_params(labelbottom=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([label_dict[l] for l in observation_order], size=9)

        # # Logic for Y-ticks: Only show on the first column
        # if col_idx > 0:
        #     ax.tick_params(labelleft=False)
        # else:
        #     # Ensure labels are visible for the first column
        #     ax.tick_params(labelleft=True)
        #     ax.set_yticks([0, 1.0])

        ax.grid(False)
        # ax.set_ylim(-0.05, 1.1)
        # ax.set_yticks([0, 1.0])

        ax.set_title(ct, weight="bold", fontsize=13)

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]],
                            label=label_dict[observation_order[0]] + " (n=" + str(len(set(group1[unit_key]))) + ")")
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]],
                            label=label_dict[observation_order[1]] + " (n=" + str(len(set(group2[unit_key]))) + ")")

    fig.legend(title=name.split("_")[0],
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=2,
               fontsize=9,
               title_fontsize=10
               )

    plt.tight_layout()

    if name:
        plt.savefig(OUT_DIR / f"{name}_cell_type_proportions_panel.png", dpi=300)
    else:
        plt.savefig(OUT_DIR / "cell_type_proportions_panel.png", dpi=300)

    plt.clf()


"""
Cell niche expression heatmap across different markers (colour bar for cel type is -3 and 1)
    CLUSTER_KEY = "cn_cluster_name"
    plot_zscore_cluster_mean_heatmap(FULL_ADATA, FULL_ADATA.var_names, CLUSTER_KEY, vmax=2, vmin=-1,
                                     out_dir=OUT_DIR, figsize=(7, 7), name=f"cell_niche_range2-1")
"""

"""
Cell niche distribution
    CLUSTER_KEY = "cn_cluster_name"
    UNIT_KEY = "unique_img_key"
    for pid in pd.unique(FULL_ADATA.obs[UNIT_KEY]):
        print(pid)
        p_adata = FULL_ADATA[FULL_ADATA.obs[UNIT_KEY] == pid]
    
        if p_adata.obs[OBSERVATION_KEY].values[0] == 0:
            TEMP_OUT_DIR = OUT_DIR / "cn_map/No recurrence"
            if pid.split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5.5, 5.5), dotsize=3.5,
                                                  color=CELL_NICHE_COLOR_DICT,
                                                  name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
    
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5.5, 5.5), dotsize=2.5,
                                                  color=CELL_NICHE_COLOR_DICT,
                                                  name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
    
        elif p_adata.obs[OBSERVATION_KEY].values[0] == 1:
            TEMP_OUT_DIR = OUT_DIR / "cn_map/Recurrence"
            if pid.split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5.5, 5.5), dotsize=3.5,
                                                  color=CELL_NICHE_COLOR_DICT,
                                                  name=f"R_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
    
                # plot_spatial_cluster_panel(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, dotsize=0.2, color=CELL_NICHE_COLOR_DICT,
                #                            ncol=6,
                #                            figsize=(10, 4.5), name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}")
    
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(5.5, 5.5), dotsize=2.5,
                                                  color=CELL_NICHE_COLOR_DICT,
                                                  name=f"R_{p_adata.obs['CID'].values[0]}_{pid}", dpi=350)
    
                # plot_spatial_cluster_panel(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, dotsize=0.2, color=CELL_NICHE_COLOR_DICT,
                #                            ncol=6,
                #                            figsize=(10, 4.5), name=f"NR_{p_adata.obs['CID'].values[0]}_{pid}")
    
        else:
            print("Wrong!")
"""

"""
Cell niche proportions of all cells image by image (it is similar to cell type calculation above but is divided by all cell counts in an image)
CLUSTER_KEY = "cn_cluster_name"
UNIT_KEY = "CID"
OBSERVATION_KEY = "Recurrence status"
OBSERVATION_ORDER = [1, 0]
LABEL_DICT = {1: "Recurrence", 0: "No recurrence"}
compare_cell_niche_proportion(FULL_ADATA, UNIT_KEY, CELL_NICHE_COLOR_DICT, CLUSTER_KEY, OBSERVATION_KEY,
                              OBSERVATION_ORDER,
                              LABEL_DICT, OUT_DIR, {0: 'tab:blue', 1: 'tab:pink'}, name="Recurrence status_byCID")

"""


def compare_cell_niche_proportion(adata, unit_key, target_ct_group_dict, cluster_key, observation_key,
                                  observation_order, label_dict, out_dir, color_dict, name=None):
    props = []
    for unit in pd.unique(adata.obs[unit_key]):
        p_adata = adata[adata.obs[unit_key] == unit].obs[["unique_img_key", "CID", "layer_1",
                                                          cluster_key, observation_key]]

        for target_ct in list(target_ct_group_dict.keys()):
            print(target_ct)
            if target_ct not in p_adata[cluster_key].values:
                prop = 0
            else:
                prop = p_adata[cluster_key].value_counts()[target_ct] / p_adata.shape[0]

            props.append([unit, target_ct, p_adata[observation_key].values[0], prop])

    props_df = pd.DataFrame(props, columns=[unit_key, "cell type", observation_key, "prop"])

    # Split groups based on the observation, calculate p-value and plot
    rows = int(len(target_ct_group_dict.keys()) / 3) + 1

    if observation_order:
        # Get two groups in the observations
        group1 = props_df[props_df[observation_key] == observation_order[0]]
        group2 = props_df[props_df[observation_key] == observation_order[1]]
    else:
        group1 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[0]]
        group2 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[1]]

    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 2))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(target_ct_group_dict.keys()) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = list(target_ct_group_dict.keys())[i]

        sns.boxplot(data=[list(group1[group1["cell type"] == ct]["prop"].values),
                          list(group2[group2["cell type"] == ct]["prop"].values)],
                    palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        stat, pval = mannwhitneyu(list(group1[group1["cell type"] == ct]["prop"].values),
                                  list(group2[group2["cell type"] == ct]["prop"].values),
                                  alternative='two-sided')

        # Add p-value to the figure
        p_val_loc = max(max(list(group1[group1["cell type"] == ct]["prop"].values)),
                        max(list(group2[group2["cell type"] == ct]["prop"].values)))
        if pval < 0.01:
            ax.text(0.5, p_val_loc, "p<0.01", ha='center', va='top', fontsize=12)

        else:
            ax.text(0.5, p_val_loc, f'p = {pval:.2f}', ha='center', va='top', fontsize=12)

        # Calculate row and column index based on i
        row_idx = i // 3
        col_idx = i % 3

        if row_idx < rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            # Ensure the labels are visible for the last row
            ax.tick_params(labelbottom=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([label_dict[l] for l in observation_order], size=9)

        # # Logic for Y-ticks: Only show on the first column
        # if col_idx > 0:
        #     ax.tick_params(labelleft=False)
        # else:
        #     # Ensure labels are visible for the first column
        #     ax.tick_params(labelleft=True)
        #     ax.set_yticks([0, 1.0])

        ax.grid(False)
        # ax.set_ylim(-0.05, 1.1)
        # ax.set_yticks([0, 1.0])

        ax.set_title(ct, weight="bold", fontsize=13)

    # Add the marker name to the y-label of the figure
    # fig.supylabel(f"{marker_focus_key}+ fraction", fontsize=10, fontweight='bold')

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]],
                            label=label_dict[observation_order[0]] + " (n=" + str(len(set(group1[unit_key]))) + ")")
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]],
                            label=label_dict[observation_order[1]] + " (n=" + str(len(set(group2[unit_key]))) + ")")

    fig.legend(title=name.split("_")[0],
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=1,
               fontsize=9,
               title_fontsize=10
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_niche_proportions_panel.png", dpi=250)
    else:
        plt.savefig(out_dir / "cell_niche_proportions_panel.png", dpi=250)

    plt.clf()


"""
Shannon diversity to check how diverse are the cell type and cell niches in the TME
    # CLUSTER_KEY = "layer_2"
    CLUSTER_KEY = "cn_cluster_name"
    UNIT_KEY = "CID"
    NAME = "cell_niche_in_patient"
    
    # Get the proportion df
    counts = FULL_ADATA.obs.pivot_table(
        index=UNIT_KEY,
        columns=CLUSTER_KEY,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)
    scores = [calculate_shannon_entropy(x) for x in prop_df.values.tolist()]
    
    # boxplot
    plt.figure(figsize=(3, 4))
    plt.boxplot(scores, medianprops={'linewidth': 2})
    
    plt.xticks([])
    plt.xlabel("Patient", fontsize=15)
    plt.yticks(fontsize=11)
    plt.ylabel("Entropy", fontsize=15)
    plt.tight_layout()
    
    plt.savefig(OUT_DIR / f"{NAME}_shannon_entropy.png", dpi=200)
    plt.clf()
"""


# Compare entropy scores of cell type and cell niche by mannwhitneyu
# g1 = [calculate_shannon_entropy(x) for x in prop_df.values.tolist()]
# g2 = [calculate_shannon_entropy(x) for x in prop_df.values.tolist()]
# stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
# pval = 1.1864804289638385e-09

def calculate_shannon_entropy(p_list):
    """
    Compute spatial entropy for each cell
    :param p_list: probabilities of all cell types among the neighbors of each cell
    :return: entropy value
    """
    entropy = (-1) * sum([x * np.log(x) if x > 0 else 0 for x in p_list])

    return entropy


"""
Cliff's delta to see the differences in cell type and cell niche between recurrence and non-recurrence
"""

"""
Immune Impact Score/Tumour infiltrating lymphocytes. Get to know the impact of T cells to epithelial(tumour) cells
    if "imm_tils" not in FULL_ADATA.obs.columns:
        # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
        CLUSTER_KEY = "layer_1"
        new_labels = np.zeros(FULL_ADATA.obs.shape[0], dtype=object)
        new_labels[np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Epithelial')[0]] = ['Epithelial'] * len(
            np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Epithelial')[0])
        new_labels[np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Stromal')[0]] = ['Stromal'] * len(
            np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Stromal')[0])
        new_labels[np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Immune')[0]] = FULL_ADATA.obs.layer_2.values[
            np.where(FULL_ADATA.obs[CLUSTER_KEY] == 'Immune')[0]]

        update_labels(FULL_ADATA, 'imm_tils', new_labels)

    # Check the medium radius of all the cells and get the twice number of it as the distance threshold
    CLUSTER_KEY = "unique_img_key"
    radius_list = []
    for img in pd.unique(FULL_ADATA.obs[CLUSTER_KEY]):
        p_adata = FULL_ADATA[np.where(FULL_ADATA.obs[CLUSTER_KEY] == img)[0], :]
        radius = np.sqrt(p_adata.obs.area / np.pi)
        radius_list.append(math.ceil(pd.Series(radius).value_counts().index[0]))

    THRESHOLD = np.median(radius_list) * 6  # 3 cells away
    UNIT_KEY = 'unique_img_key'
    OBSERVATION_KEY = 'Recurrence status'
    CLUSTER_KEY = 'imm_tils'
    TARGET_CELL = 'Epithelial'
    CELL_TYPE = 'T-cell'
    TILS_TYPES = ['Infiltrating', 'Bystanding', 'Ignored']

    result_dict, labels_to_add, infiltration_df, target_df = compute_tils_categories(adata=FULL_ADATA,
                                                                                     unit_key=UNIT_KEY,
                                                                                     observation_key=OBSERVATION_KEY,
                                                                                     cluster_key=CLUSTER_KEY,
                                                                                     cell_type=CELL_TYPE,
                                                                                     target_cell=TARGET_CELL,
                                                                                     tils_types=TILS_TYPES,
                                                                                     threshold=THRESHOLD,
                                                                                     out_dir=OUT_DIR)
                                                                                     
    # Get the list from the record and update to anndata
    check_dict = {}
    unit_order = []
    for i in list(FULL_ADATA.obs["unique_img_key"]):
        if i not in check_dict:
            check_dict[i] = 0
            unit_order.append(i)
        else:
            continue

    # Use the order to get the label record
    new_labels = []
    for img in unit_order:
        if FULL_ADATA[FULL_ADATA.obs[UNIT_KEY] == img].shape[0] == len(labels_to_add[img]["T-cell"]):
            new_labels += labels_to_add[img]["T-cell"]
        else:
            print(img)
            break

    # Update the labels
    CLUSTER_KEY = "T-cell_infiltrating"
    FULL_ADATA.obs[CLUSTER_KEY] = new_labels

    # FOR THE TARGET CELL
    new_labels = []
    for img in unit_order:
        if FULL_ADATA[FULL_ADATA.obs[UNIT_KEY] == img].shape[0] == len(labels_to_add[img]["Epithelial"]):
            new_labels += labels_to_add[img]["Epithelial"]
        else:
            print(img)
            break

    CLUSTER_KEY = "Epithelial_infiltrated_num"
    FULL_ADATA.obs[CLUSTER_KEY] = new_labels

    # Given the effected and non_effected labels of Epithelial
    CLUSTER_KEY = "Epithelial_infiltrated_num"
    records = []
    for i in list(FULL_ADATA.obs[CLUSTER_KEY]):
        if type(i) == str:
            records.append(i)
        else:
            if i > 0:
                records.append("Yes")
            else:
                records.append("No")

    CLUSTER_KEY = "Epithelial_infiltrated"
    FULL_ADATA.obs[CLUSTER_KEY] = records

    modify_record = []
    CLUSTER_KEY = "Epithelial_infiltrated_num"
    for i in list(FULL_ADATA.obs[CLUSTER_KEY]):
        if type(i) == str:
            modify_record.append(-1)
        else:
            modify_record.append(i)

    FULL_ADATA.obs[CLUSTER_KEY] = modify_record

    FULL_ADATA.write(IN_DIR / "TNBC_T-cell_Epithelial_interactions.h5ad")
    
    # Make patient-level df
    merge_df_by_pid = merge_df.groupby(['CID', OBSERVATION_KEY], as_index=False)[CLUSTER_KEY].mean()
    merge_df_by_pid["Group"] = ["Immune_hot" if i >= 0.3 else "Immune_cold" for i in
                                merge_df_by_pid[CLUSTER_KEY].values]
    merge_df_by_pid.columns = ['CID', 'Recurrence status', 'Affected epithelial prop', 'Group']
    # merge_df_by_pid.to_csv(IN_DIR / "IMC_immune_hot_cold.csv", index=False)
    
    # Read in the IIS csv
    merge_df_by_pid = pd.read_csv(IN_DIR / "IMC_immune_hot_cold.csv")
    
    ################################# Plot1: IIS score of all the patients in a violin plot
    CLUSTER_KEY = "Affected epithelial prop"
    plt.figure(figsize=(3, 4))
    #plt.violinplot(merge_df_by_pid[CLUSTER_KEY].values)
    sns.violinplot(merge_df_by_pid[CLUSTER_KEY].values, color="lightgray")
    
    plt.xticks([])
    plt.xlabel("Patient", fontsize=12)
    plt.yticks(fontsize=10)
    plt.ylabel("Immune Impact Score", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(OUT_DIR / "IIS_all_patient", dpi=300)
    plt.clf()
    
    ###################################### Plot2: IIS scores of patients in different clinical outcomes
    LABEL_DICT = {0: "No recurrence", 1: "Recurrence"}
    CLUSTER_KEY = "Affected epithelial prop"
    OBSERVATION_KEY = "Recurrence status"
    g1 = list(merge_df_by_pid[merge_df_by_pid[OBSERVATION_KEY] == 0][CLUSTER_KEY])
    g2 = list(merge_df_by_pid[merge_df_by_pid[OBSERVATION_KEY] == 1][CLUSTER_KEY])
    stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(5, 6))
    ax = sns.boxplot(x = OBSERVATION_KEY, y=CLUSTER_KEY, data=merge_df_by_pid, palette=["lightblue", "pink"],
    width=0.8, boxprops=dict(edgecolor='black'), whiskerprops=dict(color='black'),
    capprops=dict(color='black'), medianprops=dict(color='black'))
    
    sns.stripplot(x=OBSERVATION_KEY, y=CLUSTER_KEY, data=merge_df_by_pid, color=".3", alpha=0.5)

    # p-value    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL_DICT[0], LABEL_DICT[1]], fontsize=14)
    ax.set_xlabel("")
    
    x1, x2 = 0, 1
    y_max = merge_df_by_pid[CLUSTER_KEY].max()
    h = y_max * 0.05
    

    plt.plot([x1, x1, x2, x2], [y_max + h, y_max + 2 * h, y_max + 2 * h, y_max + h], lw=1.5, c='black')
    plt.text((x1 + x2) * 0.5, y_max + 2.5 * h, f"p = {pval:.2f}", ha='center', va='bottom', color='black',
             fontsize=14)
    
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)
    plt.ylabel("Immune Impact Score", fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Recurrence status_IIS_by_pid.png", dpi=300)
    plt.clf()
    
    ############################## Plot3: IIS spatial distribution (Yes/No Epithelial and T-cell)
    # Replace the cells to "Others" except for T-cell, Yes and No
    mapping_dict = dict(zip(['B-cell', 'Macrophage', 'No', 'Stromal', 'T-cell', 'Yes'],
                        ['Others', 'Others', 'No', 'Others', 'T-cell', 'Yes']))
    FULL_ADATA.obs.Epithelial_infiltrated = [mapping_dict[x] for x in FULL_ADATA.obs.Epithelial_infiltrated]
    FULL_ADATA.obs.Epithelial_infiltrated = FULL_ADATA.obs.Epithelial_infiltrated.astype("category")
    
    TEMP_OUT_DIR = OUT_DIR / "IIS"
    if os.path.exists(TEMP_OUT_DIR) == 0:
        os.makedirs(TEMP_OUT_DIR, exist_ok=True)
    UNIT_KEY = "unique_img_key"
    OBSERVATION_KEY = "Recurrence status"
    
    CLUSTER_KEY = "Epithelial_infiltrated"
    for i in range(target_df.shape[0]):
        print(target_df[UNIT_KEY].values[i])
        p_adata = FULL_ADATA[FULL_ADATA.obs[UNIT_KEY] == target_df[UNIT_KEY].values[i]]

        if target_df[OBSERVATION_KEY].values[i] == 0:
            TEMP_OUT_DIR = OUT_DIR / "IIS/No recurrence"

            if target_df[UNIT_KEY].values[i].split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(4.5, 5), dotsize=3.5,
                                                  color=EPI_TILS_COLOR_DICT,
                                                  name=f"NR_{target_df['CID'].values[i]}_{target_df[UNIT_KEY].values[i]}_{target_df['Yes_prop'].values[i]:.2f}",
                                                  dpi=350)
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(4.5, 5), dotsize=2.5,
                                                  color=EPI_TILS_COLOR_DICT,
                                                  name=f"NR_{target_df['CID'].values[i]}_{target_df[UNIT_KEY].values[i]}_{target_df['Yes_prop'].values[i]:.2f}",
                                                  dpi=350)

        elif target_df[OBSERVATION_KEY].values[i] == 1:
            TEMP_OUT_DIR = OUT_DIR / "IIS/Recurrence"

            if target_df[UNIT_KEY].values[i].split("_")[0] == "BOB":
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(4.5, 5), dotsize=3.5,
                                                  color=EPI_TILS_COLOR_DICT,
                                                  name=f"R_{target_df['CID'].values[i]}_{target_df[UNIT_KEY].values[i]}_{target_df['Yes_prop'].values[i]:.2f}",
                                                  dpi=350)
            else:
                plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, TEMP_OUT_DIR, figsize=(4.5, 5), dotsize=2.5,
                                                  color=EPI_TILS_COLOR_DICT,
                                                  name=f"R_{target_df['CID'].values[i]}_{target_df[UNIT_KEY].values[i]}_{target_df['Yes_prop'].values[i]:.2f}",
                                                  dpi=350)
        else:
            print("Wrong, no recurrence info!")
"""


# Run one cell type each time, e.g. T cell
def compute_tils_categories(adata, unit_key, observation_key, cluster_key, cell_type, target_cell,
                            tils_types, threshold, out_dir):
    result = {}
    infiltrating_labels = {}

    # Run through each unit, e.g. image
    for pid in np.unique(adata.obs[unit_key]):
        print(pid)
        result[pid] = {}
        infiltrating_labels[pid] = {}

        p_adata = adata[np.where(adata.obs[unit_key] == pid)[0], :]

        # Read the neighbors whose distances are in the threshold
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=threshold)

        # Start to check the interactions between the cell type and target cell
        print(cell_type)
        result[pid][cell_type] = {}

        # Count edges between the cell type and the target cell type (e.g. T Cell and Epithelial)
        interactions = get_interacting_pairs(p_adata, cluster_key=cluster_key,
                                             type_i=target_cell, type_j=cell_type)

        # Compute the number of neighbor does each immune cell have
        infiltrating = []
        bystanding = []
        ignored = []
        tils_labels = p_adata.obs[cluster_key].to_list()

        # Go through the cell type to classify
        for idx in np.where(p_adata.obs[cluster_key] == cell_type)[0]:
            # When the cell has 3 or more tumour neighbors
            if np.sum(interactions.flatten() == idx) >= 3:
                infiltrating.append(idx)
                tils_labels[idx] = 'Infiltrating'
            # When the cell has one or two tumour neighbors
            elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                bystanding.append(idx)
                tils_labels[idx] = 'Bystanding'
            # When the cell has no tumour neighbors
            else:
                ignored.append(idx)
                tils_labels[idx] = 'Ignored'

        result[pid][cell_type][tils_types[0]] = infiltrating
        result[pid][cell_type][tils_types[1]] = bystanding
        result[pid][cell_type][tils_types[2]] = ignored

        infiltrating_labels[pid][cell_type] = tils_labels

        # Go through the target cell type to record how it is effected
        result[pid][target_cell] = {}
        target_labels = p_adata.obs[cluster_key].to_list()
        effected = []
        non_effected = []

        for idx in np.where(p_adata.obs[cluster_key] == target_cell)[0]:
            if np.sum(interactions.flatten() == idx) >= 1:
                target_labels[idx] = np.sum(interactions.flatten() == idx)
                effected.append(idx)

            # When the cell has no interactions
            else:
                target_labels[idx] = np.sum(interactions.flatten() == idx)
                non_effected.append(idx)

        result[pid][target_cell]["Yes"] = effected
        result[pid][target_cell]["No"] = non_effected

        infiltrating_labels[pid][target_cell] = target_labels

    # Make df
    infiltrating_calculation = []
    for pid, info in result.items():
        observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]
        info_to_add = [pid, observation]

        total_count = (len(info[cell_type][tils_types[0]]) + len(info[cell_type][tils_types[1]]) +
                       len(info[cell_type][tils_types[2]]))
        if total_count == 0:
            info_to_add.append(0)
            info_to_add.append(0)
            info_to_add.append(0)
        else:
            info_to_add.append(len(info[cell_type][tils_types[0]]))
            info_to_add.append(len(info[cell_type][tils_types[1]]))
            info_to_add.append(len(info[cell_type][tils_types[2]]))

        infiltrating_calculation.append(info_to_add)

    # Create a new dataframe for the infiltration result
    infiltrating_df = pd.DataFrame(infiltrating_calculation,
                                   columns=[unit_key, observation_key] +
                                           [f"{cell_type}_{tils}" for tils in tils_types])

    # Make df for the target cell
    target_calculation = []
    for pid, info in result.items():
        observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]
        info_to_add = [pid, observation]

        info_to_add.append(len(result[pid][target_cell]["Yes"]))
        info_to_add.append(len(result[pid][target_cell]["No"]))

        target_calculation.append(info_to_add)
    # Create a new dataframe for the infiltration result
    target_df = pd.DataFrame(target_calculation,
                             columns=[unit_key, observation_key, "Yes", "No"])

    return result, infiltrating_labels, infiltrating_df, target_df


"""
Run Spearman correlation to check the correlation between images from the same patient
starting from the basic cell types
"""
# UNIT_KEY = "unique_img_key"
# CLUSTER_KEY = "layer_1"
# counts = FULL_ADATA.obs.pivot_table(
#     index=UNIT_KEY,
#     columns=CLUSTER_KEY,
#     aggfunc='size',
#     fill_value=0
# )
# prop_df = counts.div(counts.sum(axis=1), axis=0)
#
# # Move image_key from index to a column and add CID to the df
# prop_df["unique_img_key"] = list(prop_df.index)
# prop_df.reset_index(drop=True, inplace=True)
# mapping_dict = dict(zip(FULL_ADATA.obs.unique_img_key.values, FULL_ADATA.obs.CID.values))
# prop_df["CID"] = [mapping_dict[x] for x in prop_df["unique_img_key"].values]
#
# grouped = prop_df.groupby('CID')
#
# cell_types = ['Epithelial', 'Immune', 'Stromal']
# all_correlations = []
#
# for patient, group in grouped:
#     if len(group) > 1:
#         print(f"Processing Patient: {patient}")
#         for i, j in combinations(range(len(group)), 2):
#             vec1 = group.iloc[i][cell_types].values
#             vec2 = group.iloc[j][cell_types].values
#
#             rho, pval = spearmanr(vec1, vec2)
#             print(
#                 f"  - Correlation between {group.iloc[i]['unique_img_key']} and {group.iloc[j]['unique_img_key']}: {rho:.3f}")
#
#             if not np.isnan(rho):
#                 all_correlations.append(rho)
#
# if all_correlations:
#     print(f"\nFinal Median Correlation: {np.median(all_correlations):.3f}")
# else:
#     print("No patients with multiple images found.")

"""
Investigate the distance of tumour cells to their nearest TME cells in different clinical outcome
# Calculation and grouping
merge_df = pd.read_csv(IN_DIR / "TNBC_Epithelial_T-cell_impact.csv")
    df = FULL_ADATA.obs[['CID', 'unique_img_key', 'layer_2']]
    df["x"] = np.array(FULL_ADATA.obsm["spatial"][:, 0])
    df["y"] = np.array(FULL_ADATA.obsm["spatial"][:, 1])
    df["layer_2"] = ['Epithelial' if i in ['CK5+ Basal', 'CK7+ Luminal', 'CK8/18+ Luminal', 'CK+ Epithelial'] else "TME"
                     for i in df["layer_2"]]
    df["layer_2"] = ['Epithelial' if i in ['CK5+ Basal', 'CK7+ Luminal', 'CK8/18+ Luminal', 'CK+ Epithelial'] else i
                     for i in df["layer_2"]]

    UNIT_KEY = "unique_img_key"
    CLUSTER_KEY = "layer_2"
    epi_label = "Epithelial"
    e_m_df = analyze_interface_per_image(df, UNIT_KEY, CLUSTER_KEY, epi_label)
    e_m_all_df = analyze_specific_interfaces_per_patient(df, list(set(df[CLUSTER_KEY].unique()) - set(epi_label)),
                                                         UNIT_KEY, CLUSTER_KEY, epi_label)

    # Add CID and recurrence status to compare between their difference in median distance
    OBSERVATION_KEY = "Recurrence status"
    LABEL_DICT = {0: 'No recurrence', 1: 'Recurrence'}
    mapping_dict = dict(zip(merge_df["unique_img_key"].values, merge_df["CID"].values))
    e_m_all_df["CID"] = [mapping_dict[x] for x in e_m_all_df["unique_img_key"].values]

    # Group and average the numbers by patient and cell types
    patient_level_df = e_m_all_df.groupby(['CID', 'cell_type']).agg({'min_dist': 'mean',
                                                                     'median_dist': 'mean'}).reset_index()

    mapping_dict = dict(zip(merge_df["CID"].values, merge_df[OBSERVATION_KEY].values))
    patient_level_df[OBSERVATION_KEY] = [mapping_dict[x] for x in patient_level_df["CID"].values]
    patient_level_df.to_csv(IN_DIR / "tumour-tme_distances_2_classes.csv", index=False)
    patient_level_df.to_csv(IN_DIR / "tumour-tme_distances_all_individual_type.csv", index=False)
    
    g1 = patient_level_df[patient_level_df[OBSERVATION_KEY] == 0]
    g2 = patient_level_df[patient_level_df[OBSERVATION_KEY] == 1]
    # g1 = g1[g1.median_dist < 50]



# Visualization (overall)

    # For Epithelial-TME distances
    CLUSTER_KEY = "min_dist"
    CLUSTER_KEY = "median_dist"
    ct = "TME"

    fig, ax = plt.subplots(figsize=(5, 6))
    sns.boxplot(data=[list(g1[g1["cell_type"] == ct][CLUSTER_KEY].values),
                      list(g2[g2["cell_type"] == ct][CLUSTER_KEY].values)],
                palette=["tab:green", "tab:orange"], ax=ax)

    sns.stripplot(data=[list(g1[g1["cell_type"] == ct][CLUSTER_KEY].values),
                        list(g2[g2["cell_type"] == ct][CLUSTER_KEY].values)], color=".3", alpha=0.5)

    stat, pval = mannwhitneyu(list(g1[g1["cell_type"] == ct][CLUSTER_KEY].values),
                              list(g2[g2["cell_type"] == ct][CLUSTER_KEY].values),
                              alternative='two-sided')

    # p-value
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL_DICT[0], LABEL_DICT[1]], fontsize=14)
    ax.set_xlabel("")

    x1, x2 = 0, 1
    y_max = patient_level_df[CLUSTER_KEY].max()
    h = y_max * 0.05

    plt.plot([x1, x1, x2, x2], [y_max + h, y_max + 2 * h, y_max + 2 * h, y_max + h], lw=1.5, c='black')
    plt.text((x1 + x2) * 0.5, y_max + 2.5 * h, f"p = {pval:.2f}", ha='center', va='bottom', color='black',
             fontsize=14)

    plt.yticks(fontsize=11)
    plt.ylabel("Distance", fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"Epi-TME_2_classes_{CLUSTER_KEY}.png", dpi=300)
    plt.clf()
    
# Visualization (individual cell type)
TYPE_LIST = ['T-cell', 'B-cell', 'Macrophage', 'Endothelial', 'SMA+ Stromal', 'Stromal']
    CLUSTER_KEY = "min_dist"
    CLUSTER_KEY = "median_dist"
    rows = int(len(TYPE_LIST) / 3) + 1
    fig, axes = plt.subplots(rows, 3, figsize=(10, rows * 1.7))
    for i, ax in enumerate(fig.axes):
        if i > len(TYPE_LIST) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ct = TYPE_LIST[i]
        print(ct)
        sns.boxplot(data=[list(g1[g1["cell_type"] == ct][CLUSTER_KEY].values),
                          list(g2[g2["cell_type"] == ct][CLUSTER_KEY].values)],
                    palette=["tab:green", "tab:orange"], ax=ax)

        stat, pval = mannwhitneyu(list(g1[g1["cell_type"] == ct][CLUSTER_KEY].values),
                                  list(g2[g2["cell_type"] == ct][CLUSTER_KEY].values),
                                  alternative='two-sided')

        # Add p-value to the figure
        if pval < 0.01:
            ax.text(0.5, 0.8, "p < 0.01", ha='center', va='center', fontsize=12, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.8, f'p = {pval:.2f}', ha='center', va='center', fontsize=12, transform=ax.transAxes)

        # Calculate row and column index based on i
        row_idx = i // 3

        if row_idx < rows - 2:
            ax.tick_params(labelbottom=False)
        else:
            # Ensure the labels are visible for the last row
            ax.tick_params(labelbottom=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([LABEL_DICT[l] for l in [0, 1]], size=10)

        ax.tick_params(labelleft=True)

        ax.grid(False)
        # # ax.yaxis.set_tick_params(labelleft=False)
        # ax.set_ylim(-0.05, 1.1)
        # ax.set_yticks([0, 1.0])

        ax.set_title(ct, weight="bold", fontsize=13)

    # Add legend
    patch1 = mpatches.Patch(color="tab:green",
                            label=LABEL_DICT[0] + " (n=" + str(len(set(g1["CID"]))) + ")")
    patch2 = mpatches.Patch(color="tab:orange",
                            label=LABEL_DICT[1] + " (n=" + str(len(set(g2["CID"]))) + ")")

    fig.legend(title=OBSERVATION_KEY,
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=2,
               fontsize=9,
               title_fontsize=10
               )

    plt.tight_layout()

    plt.savefig(OUT_DIR / f"Epi-TME_{CLUSTER_KEY}_panel.png", dpi=500)
    plt.clf()


"""


def analyze_interface_per_image(df, image_key, cell_key, epi_label):
    """
    Calculates spatial interface metrics individually for each patient's tissue core.
    """
    results = []

    for img in tqdm(df[image_key].unique()):
        p_data = df[df[image_key] == img]

        # Extract local coordinates
        epi_mask = p_data[cell_key] == epi_label
        tme_mask = p_data[cell_key] != epi_label

        epi_coords = p_data[epi_mask][['x', 'y']].values
        tme_coords = p_data[tme_mask][['x', 'y']].values

        # Edge case
        if len(epi_coords) == 0 or len(tme_coords) == 0:
            print(f"No epithelial or TME for {img}")
            continue

        # Compute distance using local KDTree
        tree = cKDTree(tme_coords)
        dists, _ = tree.query(epi_coords, k=1)

        results.append({
            image_key: img,
            'min_dist': np.min(dists),
            'median_dist': np.median(dists),
        })

    return pd.DataFrame(results)


def analyze_specific_interfaces_per_patient(df, tme_cell_types, image_key, cell_key, epi_label):
    results = []

    for img in tqdm(df[image_key].unique()):
        p_data = df[df[image_key] == img]

        all_coords = p_data[['x', 'y']].values

        # Extract local coordinates
        epi_mask = p_data[cell_key] == epi_label

        epi_coords = all_coords[epi_mask]

        # Calculate distances for microenvironment cell types
        obs_metrics = {}

        for tme_type in tme_cell_types:
            tme_mask = p_data[cell_key] == tme_type
            tme_coords = all_coords[tme_mask]

            # Edge case
            if len(epi_coords) == 0 or len(tme_coords) == 0:
                # print(f"No epithelial or TME for {img}")
                continue

            # Compute distance using local KDTree
            tree = cKDTree(tme_coords)
            dists, _ = tree.query(epi_coords, k=1)

            results.append({
                image_key: img,
                'cell_type': tme_type,
                'min_dist': np.min(dists),
                'median_dist': np.median(dists),
            })

    return pd.DataFrame(results)


"""
Permutation testing
"""


def run_spatial_permutation_test(df, patient_key, niche_key, cell_key, n_perms=1000):
    """
    Assesses if the observed frequency of a specific cell type/niche is non-random.
    """
    # Proportion of each cell type within each niche per patient
    obs_counts = df.groupby([patient_key, niche_key, cell_key]).size().unstack(fill_value=0)
    obs_prop = obs_counts.div(obs_counts.sum(axis=1), axis=0)

    # Set up 3D mtx: [patient * niche, cell type]
    count_mtx = np.zeros_like(obs_prop.values, dtype=int)

    # Count how many times the random composition > the observed enrichment
    for _ in tqdm(range(n_perms)):
        # Shuffle cell types within each patient, niches remain unchanged
        shuffled_table = df.copy()
        shuffled_table[cell_key] = df.groupby(patient_key)[cell_key].transform(
            lambda x: np.random.permutation(x.values)
        )

        # Calculate permuted composition
        perm_counts = shuffled_table.groupby([patient_key, niche_key, cell_key]).size().unstack(fill_value=0)
        perm_prop = perm_counts.div(perm_counts.sum(axis=1), axis=0)
        perm_prop = perm_prop.reindex(index=obs_prop.index, columns=obs_prop.columns, fill_value=0)

        # Add 1 if the random shuffle created a higher proportion of that cell type in that niche
        count_mtx += (perm_prop.values >= obs_prop.values).astype(int)

    # Compute p-values for each cell type in niche association
    p_mtx = pd.DataFrame((count_mtx + 1) / (n_perms + 1),
                         index=obs_prop.index,
                         columns=obs_prop.columns)

    # Combine into a single p-value per niche (Fisher's Method)
    global_p_df = []

    for niche in df[niche_key].cat.categories:
        # Get the p-values for this specific niche across all cell types and patients
        if niche in p_mtx.index.get_level_values(niche_key):
            niche_p_vals = p_mtx.xs(niche, level=niche_key).values.flatten()

            # Combine all p-values for this niche to see if its overall composition is non-random
            _, p_fisher = combine_pvalues(niche_p_vals, method='fisher')
            global_p_df.append({'niche': niche, 'p_fisher': p_fisher})

    global_p_df = pd.DataFrame(global_p_df)

    # Multiple test correction
    _, p_adj, _, _ = multipletests(global_p_df['p_fisher'], method='fdr_bh')
    global_p_df['p_adj'] = p_adj

    return global_p_df


df = FULL_ADATA.obs[['CID', 'unique_img_key', 'layer_2', 'cn_cluster_name']]
patient_key = "CID"
niche_key = "cn_cluster_name"
cell_key = "layer_2"
p_df = run_spatial_permutation_test(df, patient_key, niche_key, cell_key, n_perms=1000)
p_df.to_csv(IN_DIR / "cell_niche_permutation.csv", index=False)
p_df.to_csv(OUT_DIR / "cell_niche_permutation.csv", index=False)

if __name__ == '__main__':
    FULL_ADATA

    # # Group the patients
    # mapping_dict = dict(zip(FULL_ADATA.obs[UNIT_KEY].values, FULL_ADATA.obs.CID.values))
    # infiltration_df["CID"] = [mapping_dict[x] for x in infiltration_df[UNIT_KEY].values]
    # target_df["CID"] = [mapping_dict[x] for x in target_df[UNIT_KEY].values]
    # target_df = target_df[['CID', 'unique_img_key', 'Recurrence status', 'Yes', 'No']]
    # target_df.columns = ['CID', 'unique_img_key', 'Recurrence status',
    #                      'Epithelial_infiltrating_count', 'Epithelial_no_infiltrating_count']
    #
    # target_df["Yes_prop"] = target_df['Epithelial_infiltrating_count'] / (
    #         target_df['Epithelial_infiltrating_count'] + target_df['Epithelial_no_infiltrating_count'])
    #
    # target_df.to_csv(IN_DIR / "Epithelial_infiltrating_table", index=False)
    #
    # Read in the IIS table
    target_df = pd.read_csv(IN_DIR / "Epithelial_infiltrating_table")

    # Visualization check

    # infiltration_df = infiltration_df[['CID', 'unique_img_key', 'Recurrence status', 'T-cell_Infiltrating',
    #                                    'T-cell_Bystanding', 'T-cell_Ignored']]
    # infiltration_df.columns = ['CID', 'unique_img_key', 'Recurrence status', 'T-cell_Infiltrating_count',
    #                            'T-cell_Bystanding_count', 'T-cell_Ignored_count']
    #
    # merge_df = pd.merge(target_df, infiltration_df, on=['CID', 'unique_img_key', 'Recurrence status'], how="left")
    #
    # merge_df['TILs_ratio'] = np.where(merge_df["T-cell_Infiltrating_count"] != 0,
    #                                   merge_df["T-cell_Bystanding_count"] / merge_df["T-cell_Infiltrating_count"], 0)
    #
    # # Compare the Epithelial score and T infiltrating score and use Mann-Whitney to see difference b/w re and non-re
    # CLUSTER_KEY = "Yes_prop"
    # OBSERVATION_KEY = "Recurrence status"
    # g1 = list(merge_df[merge_df[OBSERVATION_KEY] == 0][CLUSTER_KEY])
    # g2 = list(merge_df[merge_df[OBSERVATION_KEY] == 1][CLUSTER_KEY])
    # stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
    #
    # plt.figure(figsize=(7, 7))
    # sns.boxplot(merge_df[[OBSERVATION_KEY, CLUSTER_KEY]], y=CLUSTER_KEY, hue=OBSERVATION_KEY)
    # plt.show()
    #
    # # Get the average of the patients, by patients
    # merge_df_by_pid = merge_df.groupby(['CID', OBSERVATION_KEY], as_index=False)[CLUSTER_KEY].mean()
    # merge_df_by_pid["Group"] = ["Immune_hot" if i >= 0.3 else "Immune_cold" for i in
    #                             merge_df_by_pid[CLUSTER_KEY].values]
    # merge_df_by_pid.columns = ['CID', 'Recurrence status', 'Affected epithelial prop', 'Group']
    # merge_df_by_pid.to_csv(IN_DIR / "IMC_immune_hot_cold.csv", index=False)
    #
    # CLUSTER_KEY = "Yes_prop"
    # CLUSTER_KEY = "TILs_ratio"
    # OBSERVATION_KEY = "Recurrence status"
    # g1 = list(merge_df_by_pid[merge_df_by_pid[OBSERVATION_KEY] == 0][CLUSTER_KEY])
    # g2 = list(merge_df_by_pid[merge_df_by_pid[OBSERVATION_KEY] == 1][CLUSTER_KEY])
    # stat, pval = mannwhitneyu(g1, g2, alternative='two-sided')
    #
    # plt.figure(figsize=(7, 7))
    # sns.boxplot(merge_df_by_pid[[OBSERVATION_KEY, CLUSTER_KEY]], y=CLUSTER_KEY, hue=OBSERVATION_KEY)
    # plt.show()
    #
    # CLUSTER_KEY = "TILs_ratio"
    # merge_df_by_pid = merge_df.groupby(['CID', OBSERVATION_KEY], as_index=False)[CLUSTER_KEY].mean()
    #
    # merge_df.to_csv(IN_DIR / "TNBC_Epithelial_T-cell_impact.csv", index=False)
    #
    # # Check pCR
    # NAT_ADATA = FULL_ADATA[FULL_ADATA.obs['Chemo type'] == 0]
    # UNIT_KEY = "CID"
    # CLUSTER_KEY = "pCR"
    #
    # mapping_dict = dict(zip(FULL_ADATA.obs.CID.values, FULL_ADATA.obs[CLUSTER_KEY].values))
    # merge_df[CLUSTER_KEY] = [mapping_dict[x] for x in merge_df[UNIT_KEY].values]
    #
    #
    # def plot_cell_distribution_boxplot(prop_df, cell_type, observation_key, observation_key_order,
    #                                    out_dir, color_dict, figsize=(7, 4), name=None):
    #     # Compute p-value of Mann-whitney U test for the difference between two observation groups
    #     group1 = prop_df[prop_df[observation_key] == observation_key_order[0]][f"{cell_type}_prop"].values
    #     group2 = prop_df[prop_df[observation_key] == observation_key_order[1]][f"{cell_type}_prop"].values
    #
    #     stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    #
    #     # Plotting
    #     fig, ax = plt.subplots(1, 1, figsize=figsize)
    #
    #     sns.boxplot(x=observation_key, y=f"{cell_type}_prop", data=prop_df, palette=color_dict, ax=ax)
    #
    #     # Add p-value to the figure
    #     ax.text(1.05, 1, f'p = {p_value:.3f}', ha='left', va='top', fontsize=8)
    #
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #
    #     plt.title(f'{cell_type}', weight='bold', fontsize=10)
    #     plt.xlabel('', fontsize=8)
    #     plt.ylabel('Proportion', fontsize=8)
    #     plt.ylim(-0.1, 1)
    #     plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)
    #
    #     # Add point count of each box to x-ticks (right below category name)
    #     # Get the labels showing on x-axis currently in the order from left to right
    #     xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]
    #
    #     # Get the count of each group
    #     counts = prop_df[observation_key].value_counts()
    #
    #     # Build new labels in the same order as plotted
    #     new_labels = [f"{cat}\n(n={counts[cat]})" for cat in xtick_labels]
    #
    #     # Set the new labels
    #     ax.set_xticklabels(new_labels)
    #     plt.xticks(fontsize=8)
    #
    #     plt.tight_layout()
    #
    #     if name:
    #         plt.savefig(out_dir / f"{name}_cell_type_distribution_box.png", dpi=500)
    #     else:
    #         plt.savefig(out_dir / "cell_type_distribution_box.png", dpi=500)
    #
    #     plt.clf()

    # merge_df
    # # boxplot
    # plt.figure(figsize=(3, 4))
    #
    # counts = merge_df_by_pid[OBSERVATION_KEY].value_counts()
    #
    # sns.barplot(
    #     x=counts.index,
    #     y=counts.values,
    #     hue=counts.index.astype(str),  # Ensure hue matches the palette keys
    #     palette={'0': "tab:blue", '1': "tab:pink"},
    #     legend=False  # Remove redundant legend
    # )
    #
    # plt.xticks(ticks=[0, 1], labels=["No recurrence", "Recurrence"], fontsize=12)
    # plt.xlabel(" ")
    # plt.yticks(fontsize=9)
    # plt.ylabel("Patient Count", fontsize=12)
    #
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.tight_layout()
    #
    # plt.savefig(OUT_DIR / "Recurrence status_patient counts.png", dpi=300)
    # plt.clf()

    UNIT_KEY = "CID"
    CLUSTER_KEY = "cn_cluster_name"

    # for effect size checking
    counts = FULL_ADATA.obs.pivot_table(
        index=UNIT_KEY,
        columns=CLUSTER_KEY,
        aggfunc='size',
        fill_value=0
    )
prop_df = counts.div(counts.sum(axis=1), axis=0)
# Add clinical info
mapping_dict = dict(zip(FULL_ADATA.obs[UNIT_KEY].values, FULL_ADATA.obs["Recurrence status"].values))
prop_df["Recurrence status"] = [mapping_dict[x] for x in prop_df[UNIT_KEY].values]
import pingouin as pg

# Group A (n=17), Group B (n=30)
res = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
print(f"P-value: {res.pvalue}")

# For a "confident" report, get the effect size
mwu_effect = pg.mwu(group_a, group_b)
print(mwu_effect[['AUC', 'CLES']])  # Common Language Effect Size

# Extract the png files in representative filder
png_files = list(TEMP_ADATA.glob('*.png'))
unique_img_list = [str(x) for x in png_files]
unique_img_list = [x.split("\\")[-1] for x in unique_img_list]
temp_list = [x.split("_")[2] + "_" + x.split("_")[3] + "_" + x.split("_")[4] for x in unique_img_list]

# Big figure with all 49 cores
df = pd.read_csv(OUT_DIR / "df.csv")

df = pd.DataFrame(FULL_ADATA.obsm['spatial'], columns=["x", "y"])
df["unique_img_key"] = list(FULL_ADATA.obs.unique_img_key)
df["CID"] = list(FULL_ADATA.obs.CID)
df["layer_2"] = list(FULL_ADATA.obs.layer_2)
df["cn_cluster_name"] = list(FULL_ADATA.obs.cn_cluster_name)
df["Epithelial_infiltrated"] = list(FULL_ADATA.obs.Epithelial_infiltrated)
img_list = ['22-010C_ROI_069',
            '22-010C_ROI_065',
            '22-010C_ROI_030',
            '22-010C_ROI_053',
            'BOB_ROI_031',
            '22-010A_ROI_005',
            '23-005_ROI_013',
            '22-010A_ROI_030',
            '22-010A_ROI_025',
            '22-010A_ROI_023',
            '22-010C_ROI_027',
            '23-010_ROI_004',
            'BOB_ROI_040',
            '23-010_ROI_003',
            '23-010_ROI_005',
            '22-010C_ROI_063',
            '22-010A_ROI_034',
            '22-010C_ROI_032',
            '22-010C_ROI_018',
            '23-010_ROI_006',
            '22-010C_ROI_087',
            '23-005_ROI_004',
            '22-010C_ROI_040',
            '23-005_ROI_008',
            '22-010C_ROI_036',
            '22-010C_ROI_034',
            '23-005_ROI_005',
            '22-010C_ROI_083',
            '23-009_ROI_016',
            '22-010A_ROI_067',
            '22-010A_ROI_071',
            '22-010C_ROI_059',
            '22-010C_ROI_022',
            '23-010_ROI_001',
            '22-010A_ROI_007',
            'BOB_ROI_080',
            '22-010B_ROI_007',
            '23-005_ROI_003',
            '22-010C_ROI_005',
            '23-005_ROI_015',
            '22-010C_ROI_081',
            '22-010B_ROI_010',
            '22-010C_ROI_061',
            '22-010B_ROI_005',
            '22-010B_ROI_009',
            '22-010C_ROI_038',
            '23-005_ROI_002',
            '22-010B_ROI_013',
            '22-010C_ROI_075']
df = df[df["unique_img_key"].isin(img_list)]
df["y"] = df["y"] * (-1)

# add the clinical outcome of the patient
mapping_dict = dict(zip(FULL_ADATA.obs.CID.values, FULL_ADATA.obs["Recurrence status"].values))
df["Recurrence status"] = [mapping_dict[x] for x in df.CID.values]
df = df.sort_values(by=['Recurrence status'])
df.to_csv(IN_DIR / "representative_core_panel.csv", index=False)

# Plotting
unique_ids = df['CID'].unique()
color_dict = LAYER2_CELL_TYPE_COLOR_DICT
cluster_key = "layer_2"
name = "cell_type"

fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 30))
axes = axes.flatten()

# Go through the image one by one
for i, img_id in enumerate(df['CID'].unique()):
    sub_df = df[df['CID'] == img_id]

    sns.scatterplot(
        data=sub_df,
        x='x',
        y='y',
        hue=cluster_key,
        palette=color_dict,
        s=3,
        edgecolor='none',
        ax=axes[i],
        legend=False
    )
    iis_score = target_df[target_df.unique_img_key ==]['Yes_prop']
    if sub_df["Recurrence status"].values[0] == 1:
        axes[i].set_title(f"R-Patient {i + 1}-{iis_score}", fontsize=15)
    else:
        axes[i].set_title(f"NR-Patient {i + 1 - {}", fontsize=15)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        axes[i].spines['top'].set_visible(True)
        axes[i].spines['right'].set_visible(True)
        axes[i].spines['left'].set_visible(True)
        axes[i].spines['bottom'].set_visible(True)

        # Hide the empty 50th subplot
        if len(unique_ids) < len(axes):
                          for j in range(len(unique_ids), len(axes)):
        axes[j].axis('off')

        plt.tight_layout()

        plt.savefig(OUT_DIR / f"{name}_core_panel.png", dpi=200)
