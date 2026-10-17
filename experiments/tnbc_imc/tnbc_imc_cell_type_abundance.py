"""
@author: Tina Hsu
Calculating cell type abundance and plotting the proportion of each cell type in different layers.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import pandas as pd
from src.annotate import update_labels
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, PCR_TF_COLOR_DICT


def compute_cell_type_proportion(adata, cluster_key, unit_key, cell_type, population_type, observation_key):
    proportions = []
    for pid in np.unique(adata.obs[unit_key]):
        # For all cells
        if population_type == 'all':
            p_obs = adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[cluster_key]
            if cell_type in p_obs.value_counts():
                prop = p_obs.value_counts()[cell_type] / len(p_obs)
            else:
                prop = 0
            proportions.append([pid,
                                adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[observation_key].values[0],
                                prop])

        # For specific cell types only, e.g. Epithelial, Immune, Stromal
        else:
            p_obs = adata[np.where((adata.obs[unit_key] == pid) &
                                   (adata.obs['esi'] == population_type))[0], :].obs[cluster_key]
            if cell_type in p_obs.value_counts():
                prop = p_obs.value_counts()[cell_type] / len(p_obs)
            else:
                prop = 0

            # List with patient, observation, cell proportion
            proportions.append([pid,
                                adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[observation_key].values[0],
                                prop])

    # Store the calculation result in adata.uns
    proportions_df = pd.DataFrame(proportions, columns=[unit_key, observation_key, f"{cell_type}_prop"])
    adata.uns[f"{cell_type}_prop"] = proportions_df

    return proportions_df


def plot_cell_distribution_boxplot(prop_df, cell_type, observation_key, observation_key_order,
                                   out_dir, color_dict, figsize=(7, 4), name=None):
    # Compute p-value of Mann-whitney U test for the difference between two observation groups
    group1 = prop_df[prop_df[observation_key] == observation_key_order[0]][f"{cell_type}_prop"].values
    group2 = prop_df[prop_df[observation_key] == observation_key_order[1]][f"{cell_type}_prop"].values

    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(x=observation_key, y=f"{cell_type}_prop", data=prop_df, palette=color_dict, ax=ax)

    # Add p-value to the figure
    ax.text(1.05, 1, f'p = {p_value:.3f}', ha='left', va='top', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f'{cell_type}', weight='bold', fontsize=10)
    plt.xlabel('', fontsize=8)
    plt.ylabel('Proportion', fontsize=8)
    plt.ylim(-0.1, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

    # Add point count of each box to x-ticks (right below category name)
    # Get the labels showing on x-axis currently in the order from left to right
    xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]

    # Get the count of each group
    counts = prop_df[observation_key].value_counts()

    # Build new labels in the same order as plotted
    new_labels = [f"{cat}\n(n={counts[cat]})" for cat in xtick_labels]

    # Set the new labels
    ax.set_xticklabels(new_labels)
    plt.xticks(fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_type_distribution_box.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_type_distribution_box.png", dpi=500)

    plt.clf()


def plot_multiple_cell_types_distributions_boxplot(adata, patient_key, observation_key, cluster_key, cell_types,
                                                   out_dir, color_dict, figsize=(10, 6), name=None):
    prop = []
    for pid in np.unique(adata.obs[patient_key]):
        p_obs = adata[adata.obs[patient_key] == pid, :].obs[cluster_key]
        observation = adata[adata.obs[patient_key] == pid, :].obs[observation_key].values[0]
        prop.append([pid, observation] +
                    [p_obs.value_counts()[ct] / np.sum(p_obs.value_counts()) for ct in cell_types])

    prop_df = pd.DataFrame(prop, columns=[patient_key, observation_key] + cell_types)

    # Rearrange the patients by observation value, e.g. non-relapse, relapse
    prop_df = prop_df.sort_values(by=observation_key, ascending=True)

    df_melt = prop_df.melt(id_vars=[observation_key],
                           value_vars=cell_types,
                           var_name='variable',
                           value_name='value')

    # Set up figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Use seaborn with a specific axis
    sns.boxplot(data=df_melt, x='variable', y='value', hue=observation_key, palette=color_dict, ax=ax)

    # Add the p-values of all pairs
    group1 = df_melt[df_melt[observation_key] == list(set(df_melt[observation_key]))[0]]
    group2 = df_melt[df_melt[observation_key] == list(set(df_melt[observation_key]))[1]]

    for i, ct in enumerate(cell_types):
        # Compute p-value
        stat, p_value = mannwhitneyu(list(group1[group1['variable'] == ct]['value'].values),
                                     list(group2[group2['variable'] == ct]['value'].values),
                                     alternative='two-sided')

        p_text = f"p = {p_value:.3f}" if p_value >= 0.001 else "p < 0.001"

        # Position of the lines
        x1, x2 = i - 0.2, i + 0.2
        y_max = max(list(group1[group1['variable'] == ct]['value'].values) +
                    list(group2[group2['variable'] == ct]['value'].values)) + 0.05
        # y_max = max(list(group2[group2['variable'] == ct]['value'].values)) + 0.05
        hight = 0.015

        # Draw bracket (|-| shape)
        ax.plot([x1, x1, x2, x2], [y_max, y_max + hight, y_max + hight, y_max], lw=1.5, c='black')

        # Add p-value text
        ax.text((x1 + x2) / 2, y_max + hight + 0.01, p_text,
                ha='center', va='bottom', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('', fontsize=10)
    plt.ylabel('Proportion', fontsize=10)
    plt.ylim(-0.1, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10)
    plt.xticks(fontsize=10)

    # Add count number of the observation showing in the legend block, right after the texts
    # Get original labels
    handles, labels = ax.get_legend_handles_labels()

    # Create new labels with count numbers
    new_labels = [f"{label} (n={prop_df[observation_key].value_counts()[label]})" for label in labels]

    # Update legend
    ax.legend(
        handles,
        new_labels,
        title='Images',
        loc='upper right',
        bbox_to_anchor=(1.12, 1.1),
        prop={'size': 10}
    )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_type_distribution_box.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_type_distribution_box.png", dpi=500)

    plt.clf()


if __name__ == '__main__':
    # Set disease type and which experiment to process
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/abundance")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    adata = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"].copy()

    # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal (Immune focus)
    CLUSTER_KEY = "ESI"
    NEW_CLUSTER_KEY = "layer2"
    new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0]] = ['Epithelial'] * len(
        np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0])
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0]] = ['Stromal'] * len(
        np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0])
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]] = adata.obs[NEW_CLUSTER_KEY].values[
        np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]]

    CLUSTER_KEY = "immune_ct"
    update_labels(adata, CLUSTER_KEY, new_labels)
    adata.write(out_dir / f"{XP}_immune_abundance.h5ad")

    # Compute immune cell proportion in each patient
    CLUSTER_KEY = 'immune_ct'
    UNIT_KEY = 'image_roi'
    OBSERVATION_KEY = 'pCR'
    OBSERVATION_KEY_ORDER = ['Yes', 'No']

    temp_adata = adata[np.where(pd.isna(adata.obs.pCR) == False)[0], :]
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/pCR")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    CELL_TYPE = 'B Cell'
    POPULATION = 'Immune'

    # denominator is the total number of the cells or total counts of all immune cells (in each image)
    cell_proportion_df = compute_cell_type_proportion(adata=temp_adata,
                                                      cluster_key=CLUSTER_KEY,
                                                      unit_key=UNIT_KEY,
                                                      cell_type=CELL_TYPE,
                                                      population_type=POPULATION,
                                                      observation_key=OBSERVATION_KEY)

    plot_cell_distribution_boxplot(prop_df=cell_proportion_df,
                                   cell_type=CELL_TYPE,
                                   observation_key=OBSERVATION_KEY,
                                   observation_key_order=OBSERVATION_KEY_ORDER,
                                   out_dir=out_dir,
                                   color_dict=PCR_TF_COLOR_DICT,
                                   figsize=(3, 4),
                                   name=f"pCR_{CELL_TYPE}_{POPULATION}")

    # """
    # Extra work
    # Try to compare the recurrence and non-recurrence between the images with similar number of cells
    # """
    # # Get the counts of each image
    # list(adata.obs[UNIT_KEY].value_counts())
    #
    # # Hierarchical clustering
    # Z = linkage(samples, method='complete')
    #
    # # 2. Cut into k clusters
    # k = 5
    # clusters = fcluster(Z, t=k, criterion='maxclust')
    #
    # # 3. Map to sample names
    # cluster_df = pd.DataFrame({
    #     "sample": new_order_df.index,
    #     "cluster": clusters
    # })

    #
    # # # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
    # # new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    # # new_labels[np.where(adata.obs.layer1 == 'Epithelial')[0]] = ['Epithelial'] * len(
    # #     np.where(adata.obs.layer1 == 'Epithelial')[0])
    # # new_labels[np.where(adata.obs.layer1 == 'Stromal')[0]] = ['Stromal'] * len(
    # #     np.where(adata.obs.layer1 == 'Stromal')[0])
    # # new_labels[np.where(adata.obs.layer1 == 'Immune')[0]] = adata.obs.layer2.values[
    # #     np.where(adata.obs.layer1 == 'Immune')[0]]
    # #
    # # update_labels(adata, 'ct_tils', new_labels)
    # #
    # # adata.write(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")
