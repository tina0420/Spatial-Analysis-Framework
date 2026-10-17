"""
@author: Tina Hsu
Functions for evaluating clusters.
"""
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import COLORS
from src.utils import construct_range


def davies_bouldin(exp_mtx, labels):
    """
    Average similarity measure of each cluster with its most similar cluster,
    where similarity is the ratio of within-cluster distances to between-cluster distances.

    The minimum score is zero, with lower values indicating better clustering.

    :param exp_mtx: expression, type: np.array
    :param labels: labels, type: np.array
    :return: DB index
    """
    if issparse(exp_mtx):
        exp_mtx = exp_mtx.toarray()

    return davies_bouldin_score(exp_mtx, labels)


def get_agreement_indices(labels_i, labels_j):
    """
    Get agreement and disagreement indices.
    :param labels_i: first set of labels, type: np.array with dtype: category
    :param labels_j: first set of labels, type: np.array with dtype: category
    :return: [1] indices of agreement, [2] indices of disagreement, type: list(int), list(int)
    """
    agree_idxs = []
    disagree_idxs = []

    for i in range(len(labels_i)):
        if labels_i.iloc[i] == labels_j.iloc[i]:
            agree_idxs.append(i)
        else:
            disagree_idxs.append(i)

    return agree_idxs, disagree_idxs


def plot_cluster_agreement_panel(adata, cluster_key, agree_idxs, out_dir, dotsize=1):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param agree_idxs: indices of agreement, type: list(int)
    :param out_dir: output path, type: Path
    :param dotsize: dot size, type: float
    :return: directly return if no indices are agreeing
    """
    if len(agree_idxs) == 0:
        return

    labels = adata.obs[cluster_key].values
    label_categories = sorted(list(pd.unique(labels)))
    is_digit_labels = all(str(e).isdigit() for e in labels)
    rows = len(construct_range(len(pd.unique(labels)) - 1, 5))

    fig, axes = plt.subplots(rows, 5, figsize=(7, rows * 1.4))
    for i, ax in enumerate(fig.axes):
        if i > labels.to_numpy().astype(int).max():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lbl_idxs = np.where(labels == label_categories[i])[0]
        agree_in_label_idxs = list(set(list(lbl_idxs)) & set(list(agree_idxs)))
        disagree_in_label_idxs = list(set(lbl_idxs) - set(agree_in_label_idxs))

        subset = adata.obsm["spatial"][agree_in_label_idxs]
        inv_subset = adata.obsm["spatial"][disagree_in_label_idxs]

        # Plot the background
        if len(disagree_in_label_idxs) > 0:
            ax.scatter(inv_subset[:, 0], -inv_subset[:, 1], s=dotsize, c="lightgray")

        if len(agree_in_label_idxs) > 0:
            if is_digit_labels:
                ax.scatter(subset[:, 0], -subset[:, 1], s=dotsize, c=COLORS[i])
            else:
                ax.scatter(subset[:, 0], -subset[:, 1], s=dotsize, c=COLORS[i])

        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(i, weight="bold", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "agree_panel.png", dpi=500)
    plt.clf()


def plot_cluster_agreement_summary(adata, agree_idxs, disagree_idxs, out_dir, figsize=(3.5, 2)):
    """
    :param adata: AnnData
    :param agree_idxs: indices of agreement, type: list(int)
    :param disagree_idxs: indices of disagreement, type: list(int)
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :return: directly return if no indices are agreeing
    """
    if len(agree_idxs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    target_list = [["Agree", agree_idxs, "green"], ["Disagree", disagree_idxs, "red"]]

    for i, ax in enumerate(fig.axes):
        subset = adata.obsm["spatial"][target_list[i][1]]
        inv_subset = adata.obsm["spatial"]

        ax.scatter(inv_subset[:, 0], -inv_subset[:, 1], s=0.5, c="lightgray")
        ax.scatter(subset[:, 0], -subset[:, 1], s=0.5, c=target_list[i][2])

        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(target_list[i][0], weight="bold", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "agree_summary.png", dpi=500)
    plt.clf()


def plot_cluster_disagreement_matrix(labels_i, labels_j, out_dir, figsize=(7, 7), annot=False, annot_size=8):
    """
    :param labels_i: first set of labels, type: np.array with dtype: category
    :param labels_j: first set of labels, type: np.array with dtype: category
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param annot: whether to annotate, type: bool
    :param annot_size: size of annotation, type: float
    :return:
    """
    disagreement_matrix = __get_disagreement_matrix(labels_i, labels_j)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    mask = np.triu(np.ones_like(disagreement_matrix, dtype=bool))
    for i in range(len(disagreement_matrix)):
        mask[i, i] = False

    sns.heatmap(disagreement_matrix, annot=annot,
                annot_kws={"size": annot_size}, cmap="RdPu", ax=ax, mask=mask, cbar=False)

    plt.xlabel("Cluster", fontsize=8)
    plt.ylabel("Cluster", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.title(f"Disagreement confusion matrix", weight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"agree_confusion_matrix.png", dpi=500)
    plt.clf()


def __get_disagreement_matrix(labels_i, labels_j):
    """
    Get confusion matrix.
    :param labels_i: labels, type: np.array with dtype: category
    :param labels_j: labels, type: np.array with dtype: category
    :return: confusion matrix, type: np.array
    """
    disagree_mtx = np.zeros((labels_i.to_numpy().astype(int).max() + 1, labels_i.to_numpy().astype(int).max() + 1))

    for i in range(len(labels_i)):
        if labels_i[i] != labels_j[i]:
            a = int(labels_i[i])
            b = int(labels_j[i])
            if a > b:
                disagree_mtx[a, b] = disagree_mtx[a, b] + 1
            else:
                disagree_mtx[b, a] = disagree_mtx[b, a] + 1

    return disagree_mtx
