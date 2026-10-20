"""
Exploratory Data Analysis for spatial-omics data on expression and proportion comparison.
@author: Tina Hsu
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from analysis_setup import *


def compare_intensity_panel(adata, marker, cluster_key, cell_type_list, observation_key, out_dir, color_dict,
                            observation_order=None, name=None, dpi=150):
    """
    Compare marker intensity of each category (e.g. cell type) across different observations
    :param adata: Anndata
    :param marker: indicate which marker intensity to compare, type: str
    :param cluster_key: indicate the column used in obs to show comparison, type: str
    :param observation_key: indicate the column in obs with observation, type: str
    :param out_dir: output directory, type: Path
    :param color_dict: colour setting for observations, type: dict
    :param observation_order: the order of observations showing in the figure, type: list[str]
    :param name: name for the output file, type: str
    :return:
    """
    # Get the index of the marker in X
    idx = np.where(adata.var_names == marker)[0][0]

    # Get two groups in the observations
    group1 = adata[(adata.obs[observation_key] == observation_order[0])]
    group2 = adata[(adata.obs[observation_key] == observation_order[1])]

    # Plot
    rows = int(len(cell_type_list) / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(12, rows * 1.4))
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
                       palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        # Add lines from the median values
        # ax.plot([0, 1], [np.median(sub_group1), np.median(sub_group2)], color='red', linestyle='--', linewidth=1,
        #         zorder=1)

        #ax.set_yticks([0, np.round(max(max(sub_group1), max(sub_group2)), 1)])

        if len(sub_group1) == 0 or len(sub_group2) == 0:
            print(sub_group1, sub_group2)
        else:
            # Calculate p-value
            stat, pval = mannwhitneyu(sub_group1, sub_group2, alternative='two-sided')

            # Add p-value to the figure
            if pval < 0.01:
                ax.text(0.5, 0.8, "p < 0.01", ha='center', va='center', fontsize=7, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.8, f'p = {pval:.3f}', ha='center', va='center', fontsize=7, transform=ax.transAxes)

        # Add x-ticks and legend
        ax.set_xticks([0, 1])
        ax.set_xticklabels(observation_order, size=8)

        ax.grid(False)

        ax.set_title(
            cell_type_list[i], weight="bold", fontsize=8)

    # Add the marker name to the y-label of the figure
    fig.supylabel(f"{marker} intensity", fontsize=10, fontweight='bold')

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]], label=observation_order[0])
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]], label=observation_order[1])

    fig.legend(title=name,
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=1,
               fontsize=8,
               title_fontsize=8
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_{marker}_intensity_panel.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "ct_marker_intensity_panel.png", dpi=dpi)

    plt.clf()


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
                                   (adata.obs['layer_1'] == population_type))[0], :].obs[cluster_key]
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







