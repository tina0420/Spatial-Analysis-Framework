"""
@author: Tina Hsu
Testing for different parameters value setting in clustering.
"""
import os
from pathlib import Path
from anndata import read_h5ad, AnnData
import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import run_leiden, update_labels
from src.visualize import (plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel, \
                           plot_zscore_cluster_mean_heatmap, plot_cluster_cell_count, plot_spatial_cluster_distribution)
import scanpy as sc
from tnbc_esi import run_leiden_for_clusters


def run_leiden_clustering(adata, marker_list, cluster_key, resolution, out_dir, plot_zscore_heatmap=True, name=None):
    leiden_adata = AnnData(adata[:, marker_list].X, var=pd.DataFrame([], index=marker_list))
    print(leiden_adata)
    print("start running pca")
    sc.tl.pca(leiden_adata, svd_solver='arpack')
    sc.pp.neighbors(leiden_adata)
    sc.tl.leiden(leiden_adata, resolution=resolution)

    adata.obs[cluster_key] = [int(i) for i in leiden_adata.obs["leiden"]]
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')

    if plot_zscore_heatmap:
        plot_zscore_cluster_mean_heatmap(adata, marker_list, cluster_key,
                                         out_dir=out_dir, figsize=(7, 7), name=name)


def update_all_caell_types(adata, epi_adata, tme_adata, cluster_key, new_cluster_key):
    new_labels = np.zeros(adata.shape[0], dtype=object)
    epi_labels = np.array(['epithelial'] * epi_adata.shape[0])
    tme_labels = tme_adata.obs[cluster_key].to_numpy()
    new_labels[np.where(adata.obs['esi'] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(adata.obs['esi'] != 'epithelial')[0]] = tme_labels

    update_labels(adata, cluster_key, new_labels)


if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    EXPERIMENT = 'XP13086'

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")

    CLUSTER_KEY = 'esi'
    # test with one image, 'ROI10_ROI_011.ome.tiff'
    img = 'ROI10_ROI_011.ome.tiff'

    # check if setting the column category is necessary
    adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].astype('category')

    to_update = adata.obs[CLUSTER_KEY]
    update_labels(adata=adata, cluster_key=CLUSTER_KEY, labels=to_update)

    from src.visualize import COLOR_CODE

    COLOR_DICT = dict(zip(adata.obs[CLUSTER_KEY].cat.categories,
                          COLOR_CODE[:len(adata.obs[CLUSTER_KEY].cat.categories)]))

    plot_spatial_cluster_distribution(adata[np.where(adata.obs['image_id'] == img)[0], :], CLUSTER_KEY, out_dir,
                                      figsize=(5, 5), dotsize=1, color=COLOR_DICT, name="original")

    for pid in set(adata.obs['image_id']):
        print(pid)
        plot_spatial_cluster_distribution(adata[np.where(adata.obs['image_id'] == pid)[0], :], CLUSTER_KEY,
                                          out_dir, figsize=(5, 5), dotsize=1, color=COLOR_DICT, name=pid.split("_")[0])


    # test different resolution of leiden clustering (resolution=0.5)
    t_adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")
    tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    tme_idx = tme_adata.obs_names
    t_adata = adata.copy()
    t_adata = t_adata[tme_idx, :]

    CLUSTER_KEY_NEW = 'leiden_0.5'
    RESOLUTION = 0.5
    IMM_STR_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    out_dir = Path(f"output/{TYPE}/test/{EXPERIMENT}")
    # 1:26 starts
    run_leiden_clustering(t_adata, marker_list=IMM_STR_MARKERS, cluster_key=CLUSTER_KEY_NEW, resolution=RESOLUTION,
                          out_dir=out_dir, plot_zscore_heatmap=True, name=f"{EXPERIMENT}_imm_str_0.5")

    IMM_STR_LABELS = ['immune', 'immune', 'stromal', 'stromal', 'immune',
                      'immune', 'immune', 'stromal', 'undefined', 'immune',
                      'stromal', 'stromal', 'immune', 'undefined', 'immune']

    label_name_dict = dict(zip(range(len(IMM_STR_LABELS)), IMM_STR_LABELS))
    CLUSTER_KEY = 'esi_0.5'
    CLUSTER_KEY = 'leiden_0.5'
    NEW_CLUSTER_KEY = 'esi_0.5'
    update_labels(t_adata, NEW_CLUSTER_KEY, [label_name_dict[int(i)] for i in t_adata.obs[CLUSTER_KEY]])

    whole_adata = adata.copy()
    epi_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epithelial.h5ad")
    update_all_caell_types(whole_adata, epi_adata, t_adata, NEW_CLUSTER_KEY, NEW_CLUSTER_KEY)

    COLOR_DICT = dict(zip(whole_adata.obs[NEW_CLUSTER_KEY].cat.categories,
                          COLOR_CODE[:len(whole_adata.obs[NEW_CLUSTER_KEY].cat.categories)]))
    for pid in set(whole_adata.obs['image_id']):
        print(pid)
        file_name = "0.5_" + pid.split("_")[0]
        plot_spatial_cluster_distribution(whole_adata[np.where(whole_adata.obs['image_id'] == pid)[0], :],
                                          NEW_CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=1,
                                          color=COLOR_DICT, name=file_name)






