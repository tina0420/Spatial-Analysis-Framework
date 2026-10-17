"""
@author: Tina Hsu
Functions for enrichment and sensitivity analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from itertools import combinations


def get_adjacency_enrichment_matrix(adata, cluster_key):
    """
    Get adjacency enrichment across all combinations of clusters.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in uns, type: str
    :return: adjacency enrichment matrix, pd.Dataframe
    """
    label_categories = sorted(list(pd.unique(adata.obs[cluster_key])))

    ae_mtx = np.array([(label_categories[a[0]], label_categories[a[1]],
                        adjacency_enrichment_score(adata, cluster_key, label_categories[a[0]], label_categories[a[1]]))
                       for a in combinations(range(len(label_categories)), 2)])

    ae_df = pd.DataFrame(ae_mtx, columns=["Cluster 1", "Cluster 2", "Enrichment Score"])

    return ae_df


def adjacency_enrichment_score(adata, cluster_key, type_i, type_j):
    """
    Calculate the adjacency enrichment score, defined by:
    Dividing the total number of edges between type i and j by
    2 * type i proportion * type j proportion * total # of edges,
    subtracted by 1.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in uns, type: str
    :param type_i: cell type 1, type: str/int
    :param type_j: cell type 2, type: str/int
    :return: adjacency enrichment score, type: float
    """
    # Transform relations by index to get pairs of cluster assignment
    neighbors = adata.uns["relation"]
    idx_label_map = dict(zip(range(len(adata.obs[cluster_key])), adata.obs[cluster_key]))
    mapped_list = [[idx_label_map[u], idx_label_map[v]] for u, v in neighbors]

    # Parameters for score
    total_pairs = mapped_list.count([type_i, type_j]) + mapped_list.count([type_j, type_i])
    type_i_prop = np.sum(adata.obs[cluster_key] == type_i) / len(adata)
    type_j_prop = np.sum(adata.obs[cluster_key] == type_j) / len(adata)
    total_edges = len(neighbors)

    return total_pairs / (2 * type_i_prop * type_j_prop * total_edges) - 1


def get_neg_binomial_regression_params(cell_count_arr):
    """
    Estimating negative binomial regression model parameters
    :param cell_count_arr: counts of cell of interest in each FOV, type: np.array
    :return: r, p shape parameters, type: float
    """
    mu = np.mean(cell_count_arr)
    var = np.mean(cell_count_arr ** 2) - mu ** 2
    r = mu ** 2 / (var - mu)
    p = mu / var

    return r, p


def plot_neg_binomial_regression(r, p, out_dir, figsize=(5, 5)):
    """
    Plotting the negative binomial distribution
    :param r: number of successes, type: float
    :param p: probability of a single success, type: float
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(nbinom.ppf(1e-3, r, p), nbinom.ppf(1 - 1e-3, r, p))

    ax.plot(x, nbinom.pmf(x, r, p), 'o')

    ax.set_xlabel('Number of cells of interest per FOV')
    ax.set_ylabel('Proportion of FOVs with X cells')

    plt.savefig(out_dir / f'nbinom_{round(r)}_{round(p, 2)}.png', dpi=500)
