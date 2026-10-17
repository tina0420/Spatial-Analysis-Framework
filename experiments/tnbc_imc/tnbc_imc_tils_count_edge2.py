"""
@author: Tina Hsu
Calculating TILs (tumour infiltration lymphocytes) in each image of TNBC IMC dataset
"""
import os
from pathlib import Path
from anndata import read_h5ad
import matplotlib.pyplot as plt
import seaborn as sns
import math
from src.preprocess import read_neighbors
from src.spatial_stats import get_interacting_pairs
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from src.annotate import update_labels
from src.spatial_stats import get_interacting_neighborhood_cell_idxs
from src.visualize import plot_neighbor_graph, plot_spatial_cluster_distribution
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, ESI_TILS_COLOR_DICT, CT_TILS_COLOR_DICT


def compute_tils_categories(adata, unit_key, cluster_key, cell_types, target_cell, threshold):
    result = {}
    infiltrating_labels = {}

    for pid in np.unique(adata.obs[unit_key]):
        print(pid)
        result[pid] = {}
        infiltrating_labels[pid] = {}

        p_adata = adata[np.where(adata.obs[unit_key] == pid)[0], :]

        # Read the neighbors whose distances are in the threshold
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=threshold)

        for cell_type in cell_types:
            if cell_type == 'Immune':
                result[pid][cell_type] = {}

                # Count edges from Epithelial and Immune (for layer 1)
                interactions = get_interacting_pairs(p_adata, cluster_key='ESI',
                                                     type_i=target_cell, type_j=cell_type)

                edges = []
                # Compute the number of neighbor does each immune cell has
                for idx in np.where(p_adata.obs['ESI'] == cell_type)[0]:
                    # add the count of the edges to the list
                    edges.append(np.sum(interactions.flatten() == idx))

                result[pid][cell_type] = edges

            else:
                result[pid][cell_type] = {}

                # Count edges between the cell type and the target cell type (e.g. T Cell and Epithelial)
                interactions = get_interacting_pairs(p_adata, cluster_key=cluster_key,
                                                     type_i=target_cell, type_j=cell_type)

                edges = []
                # Compute the number of neighbor does each immune cell have
                for idx in np.where(p_adata.obs[cluster_key] == cell_type)[0]:
                    edges.append(np.sum(interactions.flatten() == idx))

                result[pid][cell_type] = edges

    # store the edge count result to adata.uns
    adata.uns[f'tils_edge_counts_{threshold}'] = result

    return result


def plot_tils_edge_counts_line(result, adata, unit_key, observation_key, cell_types,
                               out_dir, name=None):
    # Merge the results from one
    pos = {}
    neg = {}

    merge_edges_to_df = {}
    for cell_type in cell_types:
        pos[cell_type] = []
        neg[cell_type] = []
        pos_num = 0
        neg_num = 0

        for pid, info in result.items():
            observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]

            if observation == 'Recurrence':
                pos[cell_type] = pos[cell_type] + result[pid][cell_type]
                pos_num += 1
            elif observation == 'No recurrence':
                neg[cell_type] = neg[cell_type] + result[pid][cell_type]
                neg_num += 1
            else:
                print("Wrong observation type!")

    labels_num = {'Recurrence': pos_num,
                  'No recurrence': neg_num}

    for cell_type in cell_types:
        print(cell_type)
        fig, ax = plt.subplots(figsize=(7, 5))

        df_non = pd.DataFrame({'edge': pd.Series(neg[cell_type]).value_counts().index,
                               'counts': pd.Series(neg[cell_type]).value_counts().values})

        sns.lineplot(data=df_non, x='edge', y='counts', label='No recurrence', marker='o', color='tab:blue')

        df_re = pd.DataFrame({'edge': pd.Series(pos[cell_type]).value_counts().index,
                              'counts': pd.Series(pos[cell_type]).value_counts().values})

        # Compute p-value
        stat, p_value = mannwhitneyu(pos[cell_type], neg[cell_type], alternative='two-sided')

        # Get the position of p-value (middle point for y-axis)
        # ytick_labels = [ytick.get_text() for ytick in ax.get_yticklabels()]
        yticks = ax.get_yticks()
        if p_value < 0.01:
            # Convert the p-value into scientific expression
            mantissa, exponent = f"{p_value:.2e}".split("e")
            mantissa = float(mantissa)
            exponent = int(exponent)

            # LaTeX-style string for plotting
            scientific_exp = fr"${mantissa} \times 10^{{{exponent}}}$"

            ax.text(max((max(pos[cell_type]), max(neg[cell_type]))) - 4, yticks[int(len(yticks) / 2)],
                    f"p = {p_value}",
                    ha='left', va='bottom', fontsize=10)
        else:
            ax.text(max((max(pos[cell_type]), max(neg[cell_type]))) - 4, yticks[int(len(yticks) / 2)],
                    p_value, ha='left', va='bottom', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.title(f'{cell_type}', weight='bold', fontsize=15)
        plt.xlabel('Edge', fontsize=10)
        plt.ylabel('Cell count', fontsize=10)
        plt.xticks(list(range(0, max(max(df_re.edge), max(df_non.edge)) + 1)), fontsize=8)
        plt.yticks(fontsize=8)

        # Add patient counts to the legend block
        handles, labels = ax.get_legend_handles_labels()

        # Create new labels with count numbers
        new_labels = [f"{label} (n={labels_num[label]})" for label in labels]

        # Update legend
        ax.legend(
            handles,
            new_labels,
            loc='upper right',
            prop={'size': 10}
        )
        plt.tight_layout()

        plt.savefig(out_dir / f"{name}_{cell_type}_edge_counts.png", dpi=500)

        plt.clf()


# def plot_tils_edge_counts_line(result, adata, unit_key, observation_key, cell_types,
#                                out_dir, name=None):
#     # Merge the results from one
#     pos = {}
#     neg = {}
#
#     for cell_type in cell_types:
#         pos[cell_type] = []
#         neg[cell_type] = []
#         pos_num = 0
#         neg_num = 0
#
#         for pid, info in result.items():
#             observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]
#
#             if observation == 'Recurrence':
#                 pos[cell_type] = pos[cell_type] + result[pid][cell_type]
#                 pos_num += 1
#             elif observation == 'No recurrence':
#                 neg[cell_type] = neg[cell_type] + result[pid][cell_type]
#                 neg_num += 1
#             else:
#                 print("Wrong observation type!")
#
#     labels_num = {'Recurrence': pos_num,
#                   'No recurrence': neg_num}
#
#     for cell_type in cell_types:
#         print(cell_type)
#         fig, ax = plt.subplots(figsize=(7, 5))
#
#         df_non = pd.DataFrame({'edge': pd.Series(neg[cell_type]).value_counts().index,
#                                'counts': pd.Series(neg[cell_type]).value_counts().values})
#
#         sns.lineplot(data=df_non, x='edge', y='counts', label='No recurrence', marker='o', color='tab:blue')
#
#         df_re = pd.DataFrame({'edge': pd.Series(pos[cell_type]).value_counts().index,
#                               'counts': pd.Series(pos[cell_type]).value_counts().values})
#
#         return df_re
#
#         sns.lineplot(data=df_re, x='edge', y='counts', label='Recurrence', marker='o', color='tab:pink')
#
#         # Compute p-value
#         stat, p_value = mannwhitneyu(pos[cell_type], neg[cell_type], alternative='two-sided')
#
#         # Get the position of p-value (middle point for y-axis)
#         # ytick_labels = [ytick.get_text() for ytick in ax.get_yticklabels()]
#         yticks = ax.get_yticks()
#         if p_value < 0.01:
#             # Convert the p-value into scientific expression
#             mantissa, exponent = f"{p_value:.2e}".split("e")
#             mantissa = float(mantissa)
#             exponent = int(exponent)
#
#             # LaTeX-style string for plotting
#             scientific_exp = fr"${mantissa} \times 10^{{{exponent}}}$"
#
#             ax.text(max((max(pos[cell_type]), max(neg[cell_type]))) - 4, yticks[int(len(yticks) / 2)],
#                     f"p = {p_value}",
#                     ha='left', va='bottom', fontsize=10)
#         else:
#             ax.text(max((max(pos[cell_type]), max(neg[cell_type]))) - 4, yticks[int(len(yticks) / 2)],
#                     p_value, ha='left', va='bottom', fontsize=10)
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         plt.title(f'{cell_type}', weight='bold', fontsize=15)
#         plt.xlabel('Edge', fontsize=10)
#         plt.ylabel('Cell count', fontsize=10)
#         plt.xticks(list(range(0, max(max(df_re.edge), max(df_non.edge)) + 1)), fontsize=8)
#         plt.yticks(fontsize=8)
#
#         # Add patient counts to the legend block
#         handles, labels = ax.get_legend_handles_labels()
#
#         # Create new labels with count numbers
#         new_labels = [f"{label} (n={labels_num[label]})" for label in labels]
#
#         # Update legend
#         ax.legend(
#             handles,
#             new_labels,
#             loc='upper right',
#             prop={'size': 10}
#         )
#         plt.tight_layout()
#
#         plt.savefig(out_dir / f"{name}_{cell_type}_edge_counts.png", dpi=500)
#
#         plt.clf()


def plot_tils_comparison(result, adata, unit_key, observation_key, cell_types, color_dict,
                         out_dir, figtype='box', show_palue=False, figsize=(15, 6), name=None):
    # Merge the results from one
    pos = {}
    neg = {}
    pos_num = 0
    neg_num = 0

    for cell_type in cell_types:
        pos[cell_type] = []
        neg[cell_type] = []

        for pid, info in result.items():
            observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]

            if observation == 'Recurrence':
                pos[cell_type] = pos[cell_type] + result[pid][cell_type]
                pos_num += 1

            elif observation == 'No recurrence':
                neg[cell_type] = neg[cell_type] + result[pid][cell_type]
                neg_num += 1

            else:
                print("Wrong observation type!")

    labels_num = {'Recurrence': pos_num,
                  'No recurrence': neg_num}

    for cell_type in cell_types:
        print(cell_type)

        # Prepare the df
        df = pd.DataFrame({
            'value': pos[cell_type] + neg[cell_type],
            'type': ['Recurrence'] * len(pos[cell_type]) + ['No recurrence'] * len(neg[cell_type])
        })

        fig, ax = plt.subplots(1, 1, figsize=(3, 4))

        if figtype == 'box':
            sns.boxplot(data=df, x='type', y='value', palette={'Recurrence': 'tab:pink', 'No recurrence': 'tab:blue'})
        elif figtype == 'violin':
            sns.violinplot(data=df, x='type', y='value',
                           palette={'Recurrence': 'tab:pink', 'No recurrence': 'tab:blue'})

        # Add the p-values we calculated above to the plot
        group1 = df[df['type'] == 'Recurrence']['value'].values
        group2 = df[df['type'] == 'No recurrence']['value'].values

        if show_palue:
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

            x1, x2 = 0, 1
            y, h, col = max(max(group1), max(group2)) + 0.1, 0.1, 'black'
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            plt.text((x1 + x2) * .5, y + h + 0.01, f"p = {p_value:.3e}", ha='center', va='bottom', color=col)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.title(f'{cell_type}', weight='bold', fontsize=10)
        plt.xlabel('', fontsize=8)
        plt.ylabel('Proportion', fontsize=8)
        plt.ylim(-1, 14)
        plt.yticks([0, 2, 4, 6, 8, 10, 12], fontsize=8)

        plt.xlabel('', fontsize=8)
        plt.ylabel('Edge count', fontsize=10)
        plt.xticks(fontsize=10, rotation=35)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

        xtick_labels = [xtick.get_text() for xtick in ax.get_xticklabels()]

        new_labels = [f"{cat}\n(n={labels_num[cat]})" for cat in xtick_labels]

        # Set the new labels
        ax.set_xticklabels(new_labels)
        plt.xticks(rotation=35, fontsize=8)

        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{cell_type}_edge_count_infiltration_{figtype}_{name}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"edge_count_infiltration_{figtype}.png", dpi=500)

        plt.clf()

    # # Compute p-value of Mann-whitney U test for each category between two observation groups
    # pvalues = []
    # group1_df = infiltrating_df[infiltrating_df[observation_key] == list(set(infiltrating_df[observation_key]))[0]]
    # group2_df = infiltrating_df[infiltrating_df[observation_key] == list(set(infiltrating_df[observation_key]))[1]]
    #
    # for col in [f"{ct}_{tils}" for ct in cell_types for tils in tils_types]:
    #     # If there's no this cell type in the image, then continue
    #     if (len(group1_df[col].values) == 0) | (len(group2_df[col].values) == 0):
    #         continue
    #
    #     stat, p_value = mannwhitneyu(group1_df[col].values, group2_df[col].values, alternative='two-sided')
    #
    #     pvalues.append([col, p_value])
    #
    # pvalues_df = pd.DataFrame(pvalues, columns=['type', 'pvalue'])
    # pvalues_df.to_csv(out_dir / f"pval_{threshold}.csv", index=False)
    #
    # # Plotting
    # df_melt = infiltrating_df.iloc[:, 1:].melt(id_vars=observation_key, var_name='variable', value_name='value')
    #
    # fig, ax = plt.subplots(1, 1, figsize=figsize)
    #
    # if figtype == 'box':
    #     sns.boxplot(x="variable", y="value", hue=observation_key, data=df_melt, palette=color_dict, ax=ax)
    # elif figtype == 'violin':
    #     sns.violinplot(x="variable", y="value", hue=observation_key, data=df_melt, palette=color_dict, ax=ax)
    #
    # # Add the p-values we calculated above to the plot
    # if show_palue:
    #     for i, type in enumerate([f"{ct}_{tils}" for ct in cell_types for tils in tils_types]):
    #         # Position of the lines
    #         x1, x2 = i - 0.2, i + 0.2
    #         y_max = max(list(group1_df[type].values) + list(group2_df[type].values)) + 0.05
    #
    #         hight = 0.015
    #
    #         # Draw bracket (|-| shape)
    #         ax.plot([x1, x1, x2, x2], [y_max, y_max + hight, y_max + hight, y_max], lw=1.5, c='black')
    #
    #         # Add p-value text
    #         ax.text((x1 + x2) / 2, y_max + hight + 0.01, f"p = {pvalues_df['pvalue'][i]:.2f}",
    #                 ha='center', va='bottom', fontsize=8)
    #
    # plt.xlabel('', fontsize=8)
    # plt.ylabel('Proportion', fontsize=10)
    # plt.xticks(fontsize=10, rotation=35)
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)
    #
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.ylim(-0.1, 1.2)
    #
    # # Add patient counts to the legend block
    # handles, labels = ax.get_legend_handles_labels()
    #
    # # Create new labels with count numbers
    # new_labels = [f"{label} (n={infiltrating_df[observation_key].value_counts()[label]})" for label in labels]
    #
    # # Update legend
    # ax.legend(
    #     handles,
    #     new_labels,
    #     title='Patients',
    #     loc='upper right',
    #     prop={'size': 8}
    # )
    # # ax.legend(
    # #     handles,
    # #     new_labels,
    # #     title='Patients',
    # #     loc='upper right',
    # #     bbox_to_anchor=(1, 1.1),
    # #     prop={'size': 8}
    # # )
    #
    # plt.tight_layout()
    #
    # if name:
    #     plt.savefig(out_dir / f"infiltration_{figtype}_{name}.png", dpi=500)
    # else:
    #     plt.savefig(out_dir / f"infiltration_{figtype}.png", dpi=500)
    #
    # plt.clf()


if __name__ == '__main__':
    # Method 1: Use neighborhood directly to get the interaction (>=3 -> infiltration)
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/tils")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    adata = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]
    adata = adata[adata.obs.chemo_type == "NAT"]

    if "imm_tils" not in adata.obs.columns:
        # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
        CLUSTER_KEY = "ESI"
        new_labels = np.zeros(adata.obs.shape[0], dtype=object)
        new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0]] = ['Epithelial'] * len(
            np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0])
        new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0]] = ['Stromal'] * len(
            np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0])
        new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]] = adata.obs.layer2.values[
            np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]]

        update_labels(adata, 'imm_tils', new_labels)

    # Infiltration
    CLUSTER_KEY = "unique_core_id"
    radius_list = []
    for pid in pd.unique(adata.obs[CLUSTER_KEY]):
        p_adata = adata[np.where(adata.obs[CLUSTER_KEY] == pid)[0], :]
        radius = np.sqrt(p_adata.obs.area / np.pi)
        radius_list.append(math.ceil(pd.Series(radius).value_counts().index[0]) * 2)

    # THRESHOLD = 15
    THRESHOLD = np.median(radius_list)  # 10
    UNIT_KEY = 'unique_core_id'
    OBSERVATION_KEY = 'relapse'
    CLUSTER_KEY = 'imm_tils'
    TARGET_CELL = 'Epithelial'
    CELL_TYPES = ['Immune', 'B Cell', 'T Cell', 'Macrophage']
    TILS_TYPES = ['Infiltrating', 'Bystanding', 'Ignored']

    result_dict = compute_tils_categories(adata=adata_h3k27me3,
                                          unit_key=UNIT_KEY,
                                          cluster_key=CLUSTER_KEY,
                                          cell_types=CELL_TYPES,
                                          target_cell=TARGET_CELL,
                                          threshold=THRESHOLD)

    plot_tils_edge_counts_line(result=result_dict,
                               adata=adata,
                               unit_key=UNIT_KEY,
                               observation_key=OBSERVATION_KEY,
                               cell_types=CELL_TYPES,
                               out_dir=out_dir,
                               name="tils_counts_recurrence")

    result = result_dict.copy()
    pos = {}
    neg = {}
    pos_num = 0
    neg_num = 0

    for cell_type in CELL_TYPES:
        pos[cell_type] = []
        neg[cell_type] = []

        for pid, info in result.items():
            observation = adata_h3k27me3.obs[adata_h3k27me3.obs[UNIT_KEY] == pid][OBSERVATION_KEY].values[0]

            if observation == 'Recurrence':
                pos[cell_type] = pos[cell_type] + result[pid][cell_type]
                pos_num += 1

            elif observation == 'No recurrence':
                neg[cell_type] = neg[cell_type] + result[pid][cell_type]
                neg_num += 1

            else:
                print("Wrong observation type!")

    labels_num = {'Recurrence': pos_num,
                  'No recurrence': neg_num}

    # Create a df to build edge counts, outcome
    merge_edge = []
    outcome = []
    ct = []
    for cell_type in CELL_TYPES:
        print(cell_type)
        merge_edge += pos[cell_type]
        outcome += ["Recurrence"] * len(pos[cell_type])
        ct += [cell_type] * len(pos[cell_type])

    for cell_type in CELL_TYPES:
        print(cell_type)
        merge_edge += neg[cell_type]
        outcome += ["No recurrence"] * len(neg[cell_type])
        ct += [cell_type] * len(neg[cell_type])

    df = pd.DataFrame({"Edge": merge_edge,
                       OBSERVATION_KEY: outcome,
                       "Cell Type": ct})

    RELAPSE_TF_COLOR_DICT = {'No recurrence': 'tab:blue',
                             'Recurrence': 'tab:pink'}
    PCR_TF_COLOR_DICT = {'No': 'tab:blue',
                         'Yes': 'tab:pink'}

    df = df[df.Edge > 0]
    p_vals = {}
    for ct in CELL_TYPES:
        sub_df = df[df["Cell Type"] == ct]

        stat, p_value = mannwhitneyu(sub_df[sub_df[OBSERVATION_KEY] == "Recurrence"]["Edge"].values,
                                     sub_df[sub_df[OBSERVATION_KEY] == "No recurrence"]["Edge"].values,
                                     alternative='two-sided')
        p_vals[ct] = p_value

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    sns.boxplot(data=df, x="Cell Type", y="Edge", hue=OBSERVATION_KEY,
                palette=RELAPSE_TF_COLOR_DICT, ax=ax)

    plt.xlabel("", fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel("Edges", fontsize=15)
    for i, marker in enumerate(df['Cell Type'].unique()):
        y_max = df[df['Cell Type'] == marker]['Edge'].max()
        x = i
        p_val = p_vals[marker]
        if p_val > 0.01:
            ax.text(
                x,
                y_max + 0.1,  # slightly above the max
                f"p={p_val:.3f}",
                ha='center',
                va='bottom',
                fontsize=12
            )
        else:
            ax.text(
                x,
                y_max + 0.1,  # slightly above the max
                "p<0.01",
                ha='center',
                va='bottom',
                fontsize=12
            )

    ax.legend(
        title='Recurrence status',
        loc='center left',  # anchor point of legend
        bbox_to_anchor=(1, 0.5)  # (x, y) coordinates relative to axes
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir / "Recurrence_tils_edge_counts_box_without_zero.png", dpi=300)
    plt.clf()

    # Plot the line plot (looks like two distributions)
    # for one cell type each time
    CT = "Immune"
    CLUSTER_KEY = "relapse"
    df_plot = df[df["Cell Type"] == CT]
    df_plot_1 = df_plot[df_plot[CLUSTER_KEY] == "Recurrence"]
    df_plot_2 = df_plot[df_plot[CLUSTER_KEY] == "No recurrence"]

    list(df_plot_1.Edge.value_counts())
    list(df_plot_2.Edge.value_counts())

    stat, pval = mannwhitneyu(list(df_plot_1.Edge.value_counts()),
                              list(df_plot_2.Edge.value_counts()),
                              alternative='two-sided')



    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    sns.lineplot(
        data=df,
        x='NUMBER',
        y='COUNTS',
        hue='STORES',
        palette=RELAPSE_TF_COLOR_DICT,
        linewidth=2.5
    )

    plt.xlabel("", fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel("Edges", fontsize=15)

    ax.legend(
        title='Recurrence status',
        loc='center left',  # anchor point of legend
        bbox_to_anchor=(1, 0.5)  # (x, y) coordinates relative to axes
    )
    plt.tight_layout()
    plt.show()




    #
    # # plot_tils_comparison(result=result_dict,
    # #                      adata=adata,
    # #                      unit_key=UNIT_KEY,
    # #                      observation_key=OBSERVATION_KEY,
    # #                      cell_types=CELL_TYPES,
    # #                      color_dict=RELAPSE_TF_COLOR_DICT,
    # #                      out_dir=out_dir,
    # #                      figtype='box',
    # #                      show_palue=True,
    # #                      figsize=(15, 6),
    # #                      name="tils")
