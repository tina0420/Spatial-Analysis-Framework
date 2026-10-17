"""
@author: Tina Hsu
Functions for clustering.
"""
import pandas as pd
import scanpy as sc
from anndata import AnnData


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


def run_leiden(adata, columns, n_neighbors=15, n_pcs=None, name=None):
    """
    Leiden clustering with optional inputs.
    :param adata: AnnData
    :param columns: set of markers to cluster, dtype: list
    :param n_neighbors: parameter for setting sc.pp.neighbors, dtype: int
    :param n_pcs: parameter for setting sc.pp.neighbors, dtype: int
    :param name: additional label for the obs column name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in columns:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    leiden_adata = AnnData(adata.X[:, idxs], var=pd.DataFrame([], index=columns))
    sc.tl.pca(leiden_adata, svd_solver='arpack')
    sc.pp.neighbors(leiden_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.leiden(leiden_adata)

    if name is None:
        name = "leiden"

    adata.obs[name] = [int(i) for i in leiden_adata.obs["leiden"]]
    adata.obs[name] = adata.obs[name].astype('category')


def run_louvain(adata, columns, n_neighbors=15, n_pcs=None, name=None):
    """
    Louvain clustering with optional inputs.
    :param adata: AnnData
    :param columns: set of markers to cluster, dtype: list
    :param n_neighbors: parameter for setting sc.pp.neighbors, dtype: int
    :param n_pcs: parameter for setting sc.pp.neighbors, dtype: int
    :param name: additional label for the obs column name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in columns:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    louvain_adata = AnnData(adata.X[:, idxs], var=pd.DataFrame([], index=columns))
    sc.tl.pca(louvain_adata, svd_solver='arpack')
    sc.pp.neighbors(louvain_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.louvain(louvain_adata)

    if name is None:
        name = "louvain"

    adata.obs[name] = [int(i) for i in louvain_adata.obs["louvain"]]
    adata.obs[name] = adata.obs[name].astype('category')
