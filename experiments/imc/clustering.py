"""
Clustering methods for cell types/cell niches
@author: Tina Hsu
"""
import scanpy as sc
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def run_clustering(adata, cluster_key, marker_list, method, resolution=1.0):
    """
    Running clustering methods with certain columns
    :param adata: Anndata
    :param cluster_key: Name for the new clusters, type: str
    :param marker_list: list of the columns joining clustering, type: list[str]
    :param method: clustering method, type: str
    :param resolution: resolution used for clustering, type: float
    :return:
    """
    if method == 'leiden':
        temp_adata = adata[:, marker_list].copy()
        sc.tl.pca(temp_adata, n_comps=6, svd_solver='arpack')
        sc.pp.neighbors(temp_adata, n_neighbors=15, use_rep='X_pca')
        print("run leiden")
        sc.tl.leiden(temp_adata, resolution=resolution, key_added=cluster_key)

        adata.obs[cluster_key] = temp_adata.obs[cluster_key]
    else:
        print('Clustering method not implemented yet!')


def compare_clustering_resolutions(adata, resolutions, method):
    if method == 'calinski_harabasz':
        scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = calinski_harabasz_score(adata.obsm["X_pca"], adata.obs[col])
            scores[str(res)] = score
        adata.uns[method] = scores

    elif method == 'davies_bouldin':
        scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            col = f'layer_2_{res:.1f}'
            score = davies_bouldin_score(adata.obsm["X_pca"], adata.obs[col])
            scores[str(res)] = score
        adata.uns[method] = scores

    elif method == 'all':
        cal_scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = calinski_harabasz_score(adata.obsm["X_pca"], adata.obs[col])
            cal_scores[str(res)] = score
        adata.uns['calinski_harabasz'] = cal_scores

        db_sores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = davies_bouldin_score(adata.obsm["X_pca"], adata.obs[col])
            db_sores[str(res)] = score
        adata.uns['davies_bouldin'] = db_sores

    else:
        print("The method not implemented yet!"
              "Please select calinski_harabasz or davies_bouldin.")
