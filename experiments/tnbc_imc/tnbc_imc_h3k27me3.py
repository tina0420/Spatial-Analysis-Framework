"""
H3K27me3 focus
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import read_h5ad
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT
from src.utils import *


# Function is only for two-layer cell types (the prop of layer2 ct in its layer1 population)
def compare_ct_prop_bw_observation(adata, unit_key, target_ct, target_group, observation_key, observation_order,
                                   out_dir, color_dict, figsize=(5, 7), plot_type='box', name=None):
    props = []
    for unit in pd.unique(adata.obs[unit_key]):
        p_adata = adata[adata.obs[unit_key] == unit].obs[
            ["unique_core_id", "cid", "layer1", "layer2", "relapse"]]
        if TARGET_CT not in p_adata["layer2"].values:
            prop = 0
        else:
            prop = p_adata["layer2"].value_counts()[target_ct] / p_adata["layer1"].value_counts()[target_group]

        props.append([p_adata[unit_key].values[0], p_adata.cid.values[0], p_adata[observation_key].values[0], prop])

    props_df = pd.DataFrame(props, columns=[unit_key, "cid", observation_key, "prop"])

    mapping_dict = dict(zip(props_df.cid.values, props_df[observation_key].values))

    props_df_by_pid = props_df.groupby("cid", as_index=False)[['prop']].mean()
    props_df_by_pid[observation_key] = [mapping_dict[x] for x in props_df_by_pid.cid.values]

    stat, pval = mannwhitneyu(props_df_by_pid[props_df_by_pid[observation_key] == observation_order[0]].prop.values,
                              props_df_by_pid[props_df_by_pid[observation_key] == observation_order[1]].prop.values,
                              alternative='two-sided')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(x=observation_key, y="prop", data=props_df_by_pid, palette=color_dict,
                width=0.6, ax=ax)

    # Add p-value to the figure
    if pval < 0.01:
        ax.text(1.05, 1, "p<0.01", ha='left', va='top', fontsize=12)
    else:
        ax.text(1.05, 1, f'p = {pval:.3f}', ha='left', va='top', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(TARGET_CT, weight='bold', fontsize=12)
    plt.xlabel('', fontsize=10)
    plt.ylabel('Proportion', fontsize=10)
    plt.ylim(-0.1, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

    # Build new labels for the counts of each box
    xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]

    counts = props_df_by_pid[observation_key].value_counts()
    new_labels = [f"{cat}\n(n={counts[cat]})" for cat in xtick_labels]

    # Set the new labels
    ax.set_xticklabels(new_labels)
    plt.xticks(fontsize=12)

    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{target_ct}_prop_in_{target_group}.png", dpi=300)
    else:
        plt.savefig(out_dir / "ct_prop_in_group_box.png", dpi=300)

    plt.clf()


def compare_one_marker_intensity(adata, target_marker, target_group, observation_key, observation_order,
                                 out_dir, color_dict, figsize=(5, 7), name=None):
    # Get the index of the marker in var_names
    idx = np.where(adata.var_names == target_marker)[0][0]
    # group1 = list(
    #     adata[(adata.obs.ESI == target_group) & (adata.obs[observation_key] == observation_order[0])].X[:, idx])
    # group2 = list(
    #     adata[(adata.obs.ESI == target_group) & (adata.obs[observation_key] == observation_order[1])].X[:, idx])

    group1 = list(
        adata[(adata.obs[observation_key] == observation_order[0])].X[:, idx])
    group2 = list(
        adata[(adata.obs[observation_key] == observation_order[1])].X[:, idx])

    new_labels = np.array([observation_order[0]] * len(group1) + [observation_order[1]] * len(group2))

    df = pd.DataFrame({"intensity": group1 + group2,
                       observation_key: new_labels}).sort_values(by=observation_key)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(x=observation_key, y="intensity", data=df, palette=color_dict, width=0.6, ax=ax)

    stat, pval = mannwhitneyu(group1, group2, alternative='two-sided')

    # Add p-value to the figure
    if pval < 0.01:
        ax.text(1.2, max(df.intensity), "p<0.01", ha='left', va='top', fontsize=12)
    else:
        ax.text(1.2, max(df.intensity), pval, ha='left', va='top', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(target_marker, weight='bold', fontsize=12)
    plt.xlabel('', fontsize=10)
    plt.ylabel('Intensity', fontsize=10)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=12)

    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{target_marker}_intensity_in_all_cells.png", dpi=300)
    else:
        plt.savefig(out_dir / "marker_intensity_box.png", dpi=300)
    plt.clf()


# modify by the function in src.visualize
def plot_spatial_cluster_panel_no_counts(adata, cluster_key, out_dir, color, figsize=(7, 7), dotsize=0.1, name=None):
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    rows = len(construct_range(len(pd.unique(labels)) - 1, 7))

    fig, axes = plt.subplots(rows, 7, figsize=figsize)
    for i, ax in enumerate(fig.axes):
        if i > len(pd.unique(labels)) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lbl_idxs = np.where(labels == label_categories[i])[0]
        non_lbl_idxs = np.where(labels != label_categories[i])[0]

        subset = adata.obsm["spatial"][lbl_idxs]
        inv_subset = adata.obsm["spatial"][non_lbl_idxs]

        # Plot the background
        ax.scatter(inv_subset[:, 0], -inv_subset[:, 1], s=dotsize, c="#e0e0e0")

        ax.scatter(subset[:, 0], -subset[:, 1], s=dotsize, c=color[label_categories[i]])

        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.set_title(f"{label_categories[i]} (n={len(lbl_idxs)})", weight="bold", fontsize=7)
        ax.set_title(f"{label_categories[i]}", weight="bold", fontsize=13)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_spatial_cluster_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "spatial_cluster_panel.png", dpi=500)
    plt.clf()


if __name__ == '__main__':
    # Example for real data
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    in_dir = out_dir = Path(f"data/tnbc_imc_fluidigm/22-010B")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/abundance")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]

    """
    For Epithelial H3K27me3+ cell proportion among all epithelial cells per image 
    (comparison b/w recurrence and non-recurrence)
    """
    # cell_types:
    UNIT_KEY = "unique_core_id"
    TARGET_GROUP = "TME"
    OBSERVATION = "relapse"
    for cell_type in ['B Cell', 'Blood Endothelial', 'Fibroblast', 'Macrophage', 'Smooth Muscle Cell',
                      'T Cell', 'Vimentin+']:
        TARGET_CT = cell_type
        compare_ct_prop_bw_observation(adata=FULL_ADATA,
                                       unit_key=UNIT_KEY,
                                       target_ct=TARGET_CT,
                                       target_group=TARGET_GROUP,
                                       observation_key=OBSERVATION,
                                       observation_order=["Recurrence", "No recurrence"],
                                       out_dir=out_dir,
                                       color_dict=RELAPSE_TF_COLOR_DICT,
                                       figsize=(5, 7), plot_type='box', name="final_tnbc")

    """
    Mean expression of Epithelial H3K27me3+ in the cells from recurrence and non-recurrence
    """
    TARGET_GROUP = "Epithelial"
    OBSERVATION = "relapse"
    # TARGET_MARKER = "H3K27me3"

    for mk in ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3']:
        TARGET_MARKER = mk
        compare_one_marker_intensity(adata=FULL_ADATA,
                                     target_marker=TARGET_MARKER,
                                     target_group=TARGET_GROUP,
                                     observation_key=OBSERVATION,
                                     observation_order=["Recurrence", "No recurrence"],
                                     out_dir=out_dir,
                                     color_dict=RELAPSE_TF_COLOR_DICT, figsize=(5, 7), name="final_tnbc")

    # marker intensity comparison b/w relapse and non-relapse
    RELAPSE_TF_COLOR_DICT = {'No recurrence': 'tab:blue',
                             'Recurrence': 'tab:pink'}

    # fixed colors for Yes or No of pCR
    PCR_TF_COLOR_DICT = {'No': 'tab:blue',
                         'Yes': 'tab:pink'}
    adata
    adata = adata[adata.obs.chemo_type == "NAT"]
    OBSERVATION = "pCR"
    OBSERVATION_ORDER = ["Yes", "No"]
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df[OBSERVATION] = adata.obs[OBSERVATION].values

    marker_cols = [c for c in df.columns if c != OBSERVATION]

    df_long = df.melt(
        id_vars=OBSERVATION,
        value_vars=marker_cols,
        var_name="Marker",
        value_name="Intensity"
    )
    for MARKER in marker_cols:
        fig, ax = plt.subplots(1, 1, figsize=(5, 7))
        temp_df = df_long[df_long.Marker == MARKER]
        sns.violinplot(x=OBSERVATION, y="Intensity", data=temp_df, palette=PCR_TF_COLOR_DICT, ax=ax)

        group1 = temp_df[temp_df[OBSERVATION] == OBSERVATION_ORDER[0]]["Intensity"].values
        group2 = temp_df[temp_df[OBSERVATION] == OBSERVATION_ORDER[1]]["Intensity"].values

        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # Add p-value to the figure
        if p_value < 0.01:
            ax.text(1.05, max(temp_df.Intensity), "p<0.01", ha='left', va='top', fontsize=8)
        else:
            ax.text(1.05, max(temp_df.Intensity), f'p = {p_value:.3f}', ha='left', va='top', fontsize=8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.title(MARKER, weight='bold', fontsize=12)
        plt.xlabel('', fontsize=10)
        plt.ylabel('Intensity', fontsize=10)
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=12)

        plt.tight_layout()
        plt.savefig(out_dir / f"pCR_{MARKER}_marker_intensity.png", dpi=300)
        plt.clf()

    adata = FULL_ADATA[FULL_ADATA.obs.unique_core_id == "23-010_ROI_004"]
    plot_spatial_cluster_panel_no_counts(adata, "layer2", out_dir, adata.uns['LAYER2_CELL_TYPE_COLOR_DICT'],
                                         figsize=(15, 7), dotsize=0.8, name="23-010_ROI_004")

