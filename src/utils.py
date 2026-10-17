"""
@author: Tina Hsu
Helper functions.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from itertools import combinations


def construct_range(columns, set_count):
    """
    Construct range for plotting.
    :param columns: number of total columns, type: int
    :param set_count: number of columns in a range, type: int
    :return: pairs of start and end for ranges that split all columns, type: np.array
    """
    pairs = []
    acc = 0
    cond = True
    while cond:
        if acc + set_count >= columns+1:
            cond = False
        pairs.append((acc, acc+set_count))
        acc = acc + set_count

    pairs = np.array(pairs)
    pairs[-1][1] = columns+1

    return pairs


def threshold_column_correlation(exp_mtx, threshold=0.90):
    """
    Threshold correlation for columns to split to high and low columns.
    :param exp_mtx: expression, type: np.array
    :param threshold: threshold for Pearson correlation, type: float
    :return: [1] columns exceeding threshold and [2] below threshold, type: list
    """
    r_mtx = __calculate_coexpression_matrix(exp_mtx)
    high_r = r_mtx[np.where(np.abs(r_mtx[:, 2]) > threshold)[0], :-1]
    high_r = list(map(int, set(high_r.flatten())))
    low_r = list(set(range(exp_mtx.shape[1])) - set(high_r))

    return high_r, low_r


def get_coexpression_matrix(exp_mtx, col_idxs):
    """
    Filter co-expression between all columns by Pearson correlation by thresholding.
    :param exp_mtx: expression, type: np.array
    :param col_idxs: columns of interest, type: list
    :return: Pearson correlation matrix for all combinations of columns exceeding threshold, type: np.array
    """
    return __calculate_coexpression_matrix(exp_mtx[:, col_idxs])


def __calculate_coexpression_matrix(exp_mtx):
    """
    Calculate co-expression between all columns by Pearson correlation.
    :param exp_mtx: expression, type: np.array
    :return: Pearson correlation matrix for all combinations of columns, type: np.array
    """
    r_mtx = np.array([(a[0], a[1], spearmanr(exp_mtx[:, a[0]], exp_mtx[:, a[1]])[0])
                      for a in combinations(range(exp_mtx.shape[1]), 2)])

    return r_mtx


def spatial_expression_threshold_by_mean(exp_mtx, loc, mean_arr, a):
    """
    Thresholding spatial expression by mean.
    :param exp_mtx: expression, type: np.array
    :param loc: location, type: np.array
    :param mean_arr: mean expression, type: np.array
    :param a: combination of indicies, type: tuple
    :return: [1] locations exceeding threshold and [2] below threshold, type: np.array
    """
    idx0 = np.where(exp_mtx[:, a[0]] > mean_arr[a[0]])[0]
    idx1 = np.where(exp_mtx[:, a[1]] > mean_arr[a[1]])[0]
    idx_high = list(set(idx0) & set(idx1))
    idx_low = list(set(range(len(exp_mtx))) - (set(idx0) & set(idx1)))

    return loc[idx_high, :], loc[idx_low, :]


def count_edges(adata, labels, rel_zero_indexing=False):
    """
    Counting edges.
    :param adata: AnnData
    :param labels: labels, type: np.array
    :param rel_zero_indexing: if False, then converts to 0-indexing, type: boolean
    :return: count matrix, type: np.array
    """
    edges = adata.uns["relation"]

    if not rel_zero_indexing:
        edges = edges - 1

    label_categories = sorted(list(pd.unique(labels)))
    cat_to_num_dict = dict(zip(label_categories, range(len(label_categories))))
    idx_label_dict = dict(zip(range(len(labels)), [cat_to_num_dict[i] for i in labels]))

    edge_count_matrix = np.zeros((len(label_categories), len(label_categories)), dtype=int)
    for u, v in edges:
        ul, vl = idx_label_dict[u], idx_label_dict[v]
        if ul == vl:
            edge_count_matrix[ul, vl] += 1
        else:
            if ul > vl:
                edge_count_matrix[ul, vl] += 1
            else:
                edge_count_matrix[vl, ul] += 1

    edge_df = pd.DataFrame(edge_count_matrix, columns=label_categories, index=label_categories)

    return edge_df
