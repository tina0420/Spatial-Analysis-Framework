"""
@author: Tina Hsu
Checking the results from rsi classification across all TNBC TMAs.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from tnbc_data_info import EXPERIMENT_LIST, EXPERIMENT_MARKERS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import seaborn as sns
from src.visualize import COLOR_CODE
from collections import Counter
from sklearn.metrics import confusion_matrix
from src.annotate import update_labels

"""
Comparison between different groups
1. Individual TMA result and full dataset result
2. Original full dataset result and full dataset after batch correction
3. Individual TMA result and full dataset result batch correction
4. For the expression ranges across different TMAs, doing for each cell type (ESI)
"""


def plot_marker_exp_across_tma_boxplot(all_df, marker_list, out_dir, figsize=(), name=None):
    pass


def extract_difference_from_sources(list_1, list_2, out_dir, plot_pie=True, plot_show_diff_pie=True, title=None,
                                    name=None):
    counts_s1 = Counter(list_1)
    diff = list_1 != list_2
    diff_counts = Counter(list_1[diff])
    cell_types = sorted(set(list_1))
    basis_proportions = [counts_s1[c] / len(list_1) for c in cell_types]
    mis_proportions = [diff_counts.get(c, 0) / len(list_1) for c in cell_types]

    if plot_pie:
        plot_cell_type_proportions_diff_source_pie(cell_types=cell_types,
                                                   basis_proportions=basis_proportions,
                                                   diff_proportions=mis_proportions,
                                                   out_dir=out_dir,
                                                   show_diff=plot_show_diff_pie,
                                                   title=title,
                                                   name=name
                                                   )

    return cell_types, basis_proportions, mis_proportions


def plot_cell_type_proportions_diff_source_pie(cell_types, basis_proportions, diff_proportions, out_dir, figsize=(7, 7),
                                               color=COLOR_CODE, show_diff=False, title=None, name=None):
    # prepare basic setting
    color_dict = dict(zip(cell_types, color[:len(cell_types)]))
    # plot the basis proportions pie
    fig, ax = plt.subplots(figsize=figsize)
    start_angle = 90
    angle_list = []
    for i, category in enumerate(cell_types):
        print("_______________________________________")
        print(i, category)

        angle = 360 * basis_proportions[i]
        end_angle = start_angle + angle

        w = Wedge(center=(0, 0), r=1.0, theta1=start_angle % 360, theta2=end_angle % 360,
                  facecolor=color_dict[category],
                  edgecolor='white', label=category)
        ax.add_patch(w)

        # Compute label angle (middle of the wedge)
        mid_angle = (start_angle + end_angle) / 2
        mid_angle_rad = np.deg2rad(mid_angle)

        # Coordinates for the label
        x = 0.7 * np.cos(mid_angle_rad)
        y = 0.7 * np.sin(mid_angle_rad)

        # Add percentage label
        pct = f"{basis_proportions[i] * 100:.1f}%"
        ax.text(x, y, pct, ha='center', va='center', fontsize=12, weight='bold')

        angle_list.append([start_angle, angle])
        start_angle = end_angle

    legends = cell_types
    if show_diff:
        # add the mismatched part
        for i, category in enumerate(cell_types):
            if diff_proportions[i] > 0:
                print("mismatch")
                mismatch_angle = angle_list[i][1] * diff_proportions[i]
                kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center", fontsize=12, weight='bold')

                ax.add_patch(Wedge(center=(0, 0),
                                   r=1.0,
                                   theta1=angle_list[i][0] % 360,
                                   theta2=(angle_list[i][0] + mismatch_angle) % 360,
                                   facecolor="white",
                                   hatch='/',
                                   edgecolor='black',
                                   label='mismatch'))

                mid_angle = mismatch_angle / 2 + angle_list[i][0]
                mid_angle_rad = np.deg2rad(mid_angle)

                # Coordinates for the label
                x = np.cos(mid_angle_rad)
                y = np.sin(mid_angle_rad)

                pct = f"{diff_proportions[i] * 100:.1f}%"
                ax.annotate(pct, xy=(x, y), xytext=(np.sign(x), y),
                            horizontalalignment={-1: "right", 1: "left"}[int(np.sign(x))], **kw)
        legends = cell_types + ['mismatch']
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Proportion of each category')
    plt.legend(legends, title='Cell type')
    plt.axis('off')
    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{name}_proportion_diff_sources_pie.png", dpi=500)
    else:
        plt.savefig(out_dir / "proportion_diff_sources_pie.png", dpi=500)


def find_mismatched_per_across_tma(adata, experiment_list, diff, cluster_key,
                                   out_dir, plot_bar=True, figsize=(7, 7), name=None):
    # get the difference and count how many mismatches in each TMA
    mismatch_count = adata[np.where(diff == True)[0], :].obs[cluster_key].value_counts()
    mis_proportions = [round(count / np.sum(mismatch_count), 2) for count in mismatch_count]

    # create dataframe to plot
    diff_tma_df = pd.DataFrame(
        {'experiment': experiment_list, 'mis_count': mismatch_count, 'mis_per': mis_proportions})
    if plot_bar:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.barplot(data=diff_tma_df, x="experiment", y="mis_per", color="#1f77b4")

        plt.xticks(fontsize=8, rotation=35, ha="right")
        plt.yticks(fontsize=8)

        # plt.grid(alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel("Experiment", fontsize=8)
        plt.ylabel("Mismatched Percentage", fontsize=8)

        plt.tight_layout()
        if name:
            plt.savefig(out_dir / f"{name}_mismatched_percentages_across_xp.png", dpi=500)
        else:
            plt.savefig(out_dir / "percentages_across_xp_bar.png", dpi=500)

        plt.clf()

    return diff_tma_df


def get_confusion_matrix(true_list, pred_list, out_dir, plot_matrix=True, figsize=(7, 7), name=None):
    labels = sorted(set(true_list))
    cm = confusion_matrix(true_list, pred_list, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("sum matrix")
    print(np.sum(cm))
    print("___")
    print(np.sum(cm_df, axis=1))
    if plot_matrix:
        plt.figure(figsize=figsize)
        sns.heatmap(cm_df / np.sum(cm_df, axis=1), annot=True, cmap='Blues', xticklabels=labels,
                    yticklabels=labels)

        plt.xlabel('Full dataset')
        plt.ylabel('Each individual TMA')
        plt.title('Mismatched from the full dataset with batch correction')
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_confusion_matrix.png", dpi=500)
        plt.clf()

    return cm_df, cm_df / np.sum(cm_df, axis=1)


if __name__ == '__main__':
    # experiment setting
    TYPE = 'tnbc_imc'

    # merge all individual anndata together
    EXPERIMENT_LIST = ["XP12446", "XP13085",
                       "XP12813", "XP13083",
                       "XP13087", "XP13084",
                       "XP12814", "XP13086",
                       "XP12274r1", "XP12274r23", "XP12274r34"]
    COMMON_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'CD3e', 'CD20', 'CD45',
                      'CD68', 'CD31', 'SMA', 'Vimentin']

    in_dir = Path(f"output/{TYPE}")
    adata_list = [read_h5ad(in_dir / xp / f"{xp}_annotation.h5ad")[:, COMMON_MARKERS] for xp in EXPERIMENT_LIST]

    all_labels = []
    CLUSTER_KEY = 'esi'
    for tma in adata_list:
        print("----------------------------")
        # check if it merged correctly (length)
        print(tma.shape[0])
        labels = tma.obs[CLUSTER_KEY].to_numpy()
        print(len(labels))

        all_labels.append(labels)

    all_individual_labels = np.concatenate(all_labels)

    ##### get the full dataset (original)
    EXPERIMENT = "full_11_xp"

    in_dir = Path(f"analysis/{TYPE}/original_data_results")
    out_dir = Path(f"analysis/{TYPE}/downstream")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the xp_filter.h5ad which includes only passed markers
    adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")

    full_labels = adata.obs[CLUSTER_KEY].to_numpy()

    diff_num = np.sum(full_labels != all_individual_labels)

    # plot and calculate for the difference (full_labels and all_individual_labels)
    cell_types, proportion_s1, proportion_diff = extract_difference_from_sources(list_1=all_individual_labels,
                                                                                 list_2=full_labels,
                                                                                 out_dir=out_dir,
                                                                                 plot_pie=True,
                                                                                 plot_show_diff_pie=True,
                                                                                 title='ESI cell types identification by original full dataset',
                                                                                 name='correct_full_dataset_individual_tma')

    cm_df, cm_per = get_confusion_matrix(true_list=all_individual_labels, pred_list=full_labels, out_dir=out_dir,
                                         plot_matrix=True, figsize=(7, 7), name="Full_dataset")

    """
    Getting the full dataset (with batch correction)
    """
    EXPERIMENT = "full_11_xp_batcor"

    in_dir = Path(f"analysis/{TYPE}")
    out_dir = Path(f"analysis/{TYPE}/downstream")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the xp_filter.h5ad which includes only passed markers
    adata_batcor = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")

    # modify the name of batches, from '0', '1', '2', ... to exact experiment names
    # batch identities are '0', '1', '2', ..., transfer them into the exact names of the experiment (EXPERIMENT_LIST)
    CLUSTER_KEY = 'batch'
    xp_dixt = dict(zip([str(i) for i in range(len(EXPERIMENT_LIST))], EXPERIMENT_LIST))
    adata_batcor.obs[CLUSTER_KEY] = adata_batcor.obs[CLUSTER_KEY].map(xp_dixt)

    # get ESI labels from the full dataset with batch correction
    CLUSTER_KEY = 'esi'
    full_labels_batcor = adata_batcor.obs[CLUSTER_KEY].to_numpy()

    diff_num = np.sum(full_labels_batcor != all_individual_labels)

    # plot and calculate for the difference (full_labels and all_individual_labels)
    cell_types, proportion_s1, proportion_diff = extract_difference_from_sources(list_1=all_individual_labels,
                                                                                 list_2=full_labels_batcor,
                                                                                 out_dir=out_dir,
                                                                                 plot_pie=True,
                                                                                 plot_show_diff_pie=True,
                                                                                 title='ESI cell types identification by batch correction full dataset',
                                                                                 name='full_dataset_bc_individual_tma')

    cm_df, cm_per = get_confusion_matrix(true_list=all_individual_labels, pred_list=full_labels_batcor, out_dir=out_dir,
                                         plot_matrix=True, figsize=(7, 7), name="Full_dataset_bc")

    diff = all_individual_labels != full_labels_batcor
    CLUSTER_KEY = 'esi'
    batcor_tma_df = find_mismatched_per_across_tma(adata_batcor, EXPERIMENT_LIST, diff, CLUSTER_KEY,
                                   out_dir, plot_bar=True, figsize=(7, 7), name=f"{EXPERIMENT}")

    # check the batch correction results from leiden clustering
    # file: full_11_xp_batcor_new_harmony_non-epithelial.h5ad
    new_bc_adata = read_h5ad(in_dir / f"{EXPERIMENT}_new_harmony_non-epithelial.h5ad")

    # annotate immune and stromal cells (from z-score heatmap), cluster 11 not sure
    IMM_STR_LABELS = ['stromal', 'immune', 'immune', 'immune', 'stromal',
                      'stromal', 'immune', 'immune', 'immune', 'stromal',
                      'immune', 'stromal', 'stromal', 'stromal', 'immune',
                      'immune', 'stromal', 'stromal', 'stromal']
    label_name_dict = dict(zip(range(len(IMM_STR_LABELS)), IMM_STR_LABELS))
    CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'
    update_labels(new_bc_adata, NEW_CLUSTER_KEY, [label_name_dict[int(i)] for i in new_bc_adata.obs[CLUSTER_KEY]])
    new_bc_adata.write(in_dir / f"{EXPERIMENT}_new_harmony_non-epithelial.h5ad")

    epi_bc_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # merge epithelial and non-epithelial here in annotation file with all cells, 2038297
    bc_merge_annotation_adata = read_h5ad(in_dir / f"{EXPERIMENT}_post.h5ad")

    # i don't have the h5ad with only gmm results so borrow the information in annotation.h5ad
    refer_adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")

    new_labels = np.zeros(bc_merge_annotation_adata.shape[0], dtype=object)
    epi_labels = np.array(['epithelial'] * epi_bc_adata.shape[0])
    CLUSTER_KEY = 'esi'
    tme_labels = new_bc_adata.obs[CLUSTER_KEY].to_numpy()

    new_labels[np.where(refer_adata.obs[CLUSTER_KEY] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(refer_adata.obs[CLUSTER_KEY] != 'epithelial')[0]] = tme_labels

    update_labels(bc_merge_annotation_adata, CLUSTER_KEY, new_labels)
    # finish the ESI identification from full_11_xp_batcor!
    bc_merge_annotation_adata.write(in_dir / f"{EXPERIMENT}_new_harmony_all.h5ad")

    # now we can compare the results after batch correction with individual slide result
    bc_esi_labels = bc_merge_annotation_adata.obs[CLUSTER_KEY].to_numpy()

    diff_num = np.sum(bc_esi_labels != all_individual_labels)

    # plot and calculate for the difference (full_labels and all_individual_labels)
    cell_types, proportion_s1, proportion_diff = extract_difference_from_sources(list_1=all_individual_labels,
                                                                                 list_2=bc_esi_labels,
                                                                                 out_dir=out_dir,
                                                                                 plot_pie=True,
                                                                                 plot_show_diff_pie=True,
                                                                                 title='ESI identification differences from full dataset with batch correction',
                                                                                 name='bc_full_diff_to_individual_tma')

    cm_df, cm_per = get_confusion_matrix(true_list=all_individual_labels, pred_list=bc_esi_labels, out_dir=out_dir,
                                         plot_matrix=True, figsize=(7, 7), name="bc_full_diff_to_individual")

