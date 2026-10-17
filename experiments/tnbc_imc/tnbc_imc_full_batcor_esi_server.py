"""
@author: Tina Hsu
Running ESI identification with full tnbc imc dataset after batch correction on the server, GMM and Leiden Clustering
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import run_leiden, update_labels
from src.visualize import (plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel, \
                           plot_zscore_cluster_mean_heatmap, plot_cluster_cell_count)
import scanpy as sc


def run_leiden_for_clusters(adata, marker_list, cluster_key, out_dir, plot_cluster_heatmap=True, name=None):
    """
    Running leiden clustering with specific markers
    :param adata: AnnData
    :param marker_list: markers used for clustering, type: list[str]
    :param cluster_key: indicate the column used in adata.obs, type:str
    :param out_dir: output directory, type:Path
    :param plot_cluster_heatmap: plot the heatmap with cluster result, type:bool
    :param name: name for the file, type: str
    :return:
    """
    sc.tl.pca(adata[:, marker_list], svd_solver='arpack')
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.leiden(adata)

    if plot_cluster_heatmap:
        plot_zscore_cluster_mean_heatmap(adata, marker_list, cluster_key,
                                         out_dir=out_dir, figsize=(7, 7), name=name)


def update_labels():
    pass


if __name__ == '__main__':
    # for full dataset
    EXPERIMENT = 'full_11_xp_batcor'
    EPITHELIAL_CLUSTER_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67']
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    in_dir = Path(f"data/{EXPERIMENT}")
    out_dir = Path(f"analysis/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata_batcor = read_h5ad(in_dir / f"{EXPERIMENT}_post.h5ad")

    adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
    tme_adata = read_h5ad(in_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    epi_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # need to run leiden clustering with epithelial and non-epithelial (tme) individually
    CLUSTER_KEY = 'leiden'
    # for non-epithelial
    print("running leiden clustering, for non-epithelial")
    run_leiden_for_clusters(tme_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY, out_dir,
                            plot_cluster_heatmap=True, name=f"{EXPERIMENT}_imm_str")
    tme_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    print("finish leiden clustering")

    # for epithelial
    print("running leiden clustering, for epithelial")
    run_leiden_for_clusters(epi_adata, EPITHELIAL_CLUSTER_MARKERS, CLUSTER_KEY, out_dir,
                            plot_cluster_heatmap=True, name=f"{EXPERIMENT}_epi")
    epi_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    print("finish leiden clustering")









