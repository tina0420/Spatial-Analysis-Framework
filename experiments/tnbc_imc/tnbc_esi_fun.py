"""
@author: Tina Hsu
Identifying ESI cell types in TNBC IMC experiments
"""
import os
from pathlib import Path
from anndata import read_h5ad
from tnbc_data_info import *
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels
from src.visualize import (plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel,
                           plot_zscore_cluster_mean_heatmap, plot_cluster_cell_count)
import scanpy as sc


def run_gmm_for_epithelial(adata, marker_list, cluster_key, out_dir, name=None):
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    # check which is epithelial and which is non-epithelial in the classification result
    if adata[np.where(adata.obs[CLUSTER_KEY] == 0)[0], marker_list].X.mean() > adata[
        np.where(adata.obs[CLUSTER_KEY] == 1)[0], marker_list].X.mean():
        # 0: epithelial, 1: non-epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'non-epithelial'})
    else:
        # 0: non-epithelial, 1: epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'non-epithelial', 1: 'epithelial'})

    adata.write(out_dir / f"{name}_merge.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0], :].write(out_dir / f"{name}_epithelial.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'non-epithelial')[0], :].write(out_dir / f"{name}_non-epithelial.h5ad")


def update_manual_labels(adata, old_cluster_key, new_cluster_key, new_labels, out_dir, name):
    """
    Update manual annotation to the immune, stromal and unknown cell types
    :param adata: Anndata
    :param old_cluster_key: indicate the column with the reference for mapping new labels, type: str
    :param new_cluster_key: indicate the column we need to update with the manual annotations, type: str
    :param new_labels: new labels for update, type: np.Array
    :param out_dir: output directory, type: Path
    :param name: extra name for the output file, type: str
    :return:
    """
    temp_adata = adata.copy()
    label_name_dict = dict(zip(range(len(new_labels)), new_labels))

    update_labels(temp_adata, new_cluster_key, [label_name_dict[int(i)] for i in adata.obs[old_cluster_key]])

    temp_adata.write(out_dir / f"{name}_non-epithelial.h5ad")


def update_all_labels(experiment, cluster_key, out_dir):
    """
    Update all the annotation together into the merged one
    :param experiment: experiment name, type: str
    :param cluster_key: indicate the column in obs, type: str
    :param out_dir: output directory, type: Path
    :return:
    """
    temp_adata = read_h5ad(out_dir / f"{experiment}_merge.h5ad")
    temp_epi = read_h5ad(out_dir / f"{experiment}_epithelial.h5ad")
    temp_tme = read_h5ad(out_dir / f"{experiment}_non-epithelial.h5ad")

    new_labels = np.zeros(temp_adata.shape[0], dtype=object)
    epi_labels = np.array(['epithelial'] * temp_epi.shape[0])
    tme_labels = temp_tme.obs[cluster_key].to_numpy()

    new_labels[np.where(temp_adata.obs[cluster_key] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(temp_adata.obs[cluster_key] != 'epithelial')[0]] = tme_labels

    update_labels(temp_adata, cluster_key, new_labels)

    temp_adata.write(out_dir / f"{experiment}_merge.h5ad")


if __name__ == '__main__':
    # parameter setting
    TYPE = 'tnbc_imc'
    EXPERIMENT = 'XP12446'

    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    EPITHELIAL_CLUSTER_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67']
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")

    # Step 1. Running 2-component GMM to identify epithelial cells
    CLUSTER_KEY = 'esi'
    run_gmm_for_epithelial(adata, EPITHELIAL_MARKERS, CLUSTER_KEY, in_dir, name=EXPERIMENT)

    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name=f"{EXPERIMENT }_find_epithelial")

    # plot UMAP to see if they separated well
    temp_adata = adata.copy()
    sc.pp.pca(temp_adata)
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{EXPERIMENT}_find_epithelial")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{EXPERIMENT}_find_epithelial")

    # Step 2. Running leiden clustering for epithelial and non-epithelial individually, epithelial cells for
    #         sub-clusters and non-epithelial cells for sub-clusters used to identify immune and stromal cells
    CLUSTER_KEY = 'leiden'

    # epithelial
    epi_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epithelial.h5ad")
    temp_adata = epi_adata[:, EPITHELIAL_CLUSTER_MARKERS].copy()
    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.leiden(temp_adata)

    plot_zscore_cluster_mean_heatmap(temp_adata, EPITHELIAL_CLUSTER_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_epithelial")

    # temp_adata.write(in_dir / f"{EXPERIMENT}_epithelial.h5ad")
    update_labels(epi_adata, cluster_key='epi_leiden', labels=temp_adata.obs[CLUSTER_KEY].to_numpy())
    epi_adata.write(in_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # non-epithelial
    tme_adata = read_h5ad(in_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    temp_adata = tme_adata[:, IMMUNE_CLUSTER_MARKERS].copy()
    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.leiden(temp_adata)

    plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_non-epithelial")

    update_labels(tme_adata, cluster_key=CLUSTER_KEY, labels=temp_adata.obs[CLUSTER_KEY].to_numpy())
    tme_adata.write(in_dir / f"{EXPERIMENT}_non-epithelial.h5ad")




