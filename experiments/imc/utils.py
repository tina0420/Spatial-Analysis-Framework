"""
Useful functions used in other steps
@author: Tina Hsu
"""
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations


def update_labels(adata, cluster_key, labels):
    """
    Updating labels to categories.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param labels: labels, type: np.array with dtype: category
    :return:
    """
    adata.obs[cluster_key] = labels
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')


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



