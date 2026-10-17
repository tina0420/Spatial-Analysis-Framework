"""
@author: Tina Hsu
Pipeline for identify ESI cell types and more sub-cell types in each group.
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


def run_gmm_for_epithelial(adata, marker_list, cluster_key, out_dir, name=None):
    """
    Running 2-component GMM to classify epithelial and non-epithelial cells
    :param adata: AnnData
    :param marker_list: markers used for GMM, type: list[str]
    :param cluster_key: indicate the column used in adata.obs, type: str
    :param out_dir: output directory, type: Path
    :param name: name for the output file, type: str
    :return:
    """
    print("run gmm")
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    print("finish updating labels")
    # check which is epithelial and which is non-epithelial in the classification result
    if adata[np.where(adata.obs[CLUSTER_KEY] == 0)[0], marker_list].X.mean() > adata[
        np.where(adata.obs[CLUSTER_KEY] == 1)[0], marker_list].X.mean():
        # 0: epithelial, 1: non-epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'non-epithelial'})
    else:
        # 0: non-epithelial, 1: epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'non-epithelial', 1: 'epithelial'})

    # store the result
    adata.write(out_dir / f"{name}_annotation.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0], :].write(out_dir / f"{name}_epithelial.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'non-epithelial')[0], :].write(out_dir / f"{name}_non-epithelial.h5ad")
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name=f"{name}_find_epithelial")
    print("start running pca")
    # running PCA to check the accuracy of classification (plot)
    temp_adata = adata.copy()
    sc.pp.pca(temp_adata)
    sc.pp.neighbors(temp_adata)
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{name}_find_epithelial")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{name}_find_epithelial")


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
    t_adata = adata[:, marker_list].copy()
    sc.tl.pca(t_adata, svd_solver='arpack')
    sc.pp.neighbors(t_adata, use_rep='X_pca')
    sc.tl.leiden(t_adata)

    update_labels(adata, cluster_key, t_adata['leiden'].to_numpy())

    if plot_cluster_heatmap:
        plot_zscore_cluster_mean_heatmap(adata, marker_list, cluster_key,
                                         out_dir=out_dir, figsize=(7, 7), name=name)


def update_esi_labels(xp, cluster_key, out_dir, new_cluster_key=None):
    """
    Update the cell type labels
    :param xp:
    :param cluster_key:
    :param out_dir:
    :param new_cluster_key:
    :return:
    """
    new_labels = []
    print("read files")
    adata_anno = read_h5ad(out_dir / f"{xp}_annotation.h5ad")
    adata_epi = read_h5ad(out_dir / f"{xp}_epithelial.h5ad")
    adata_tme = read_h5ad(out_dir / f"{xp}_non-epithelial.h5ad")

    print("merging")
    if new_cluster_key:
        print("get labels")
        epi_labels = adata_epi.obs[new_cluster_key].to_numpy()
        tme_labels = adata_tme.obs[new_cluster_key].to_numpy()
    else:
        print("get labels")
        epi_labels = adata_epi.obs[cluster_key].to_numpy()
        tme_labels = adata_tme.obs[cluster_key].to_numpy()

    print("check if the tme labels updated in tme_adata")
    new_labels = np.zeros(adata_anno.shape[0], dtype=object)
    new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'non-epithelial')[0]] = tme_labels

    print("check if the numbers of cells are same")
    print("shape of the anndata")
    print(adata_anno.shape[0])
    print("length of the labels")
    print(len(new_labels))

    adata_anno.obs[cluster_key] = new_labels
    adata_anno.write(out_dir / f"{xp}_annotation.h5ad")


if __name__ == '__main__':
    # parameter setting
    TYPE = 'tnbc_imc'
    EXPERIMENT = 'XP12446'
    EXPERIMENT_LIST = ["XP12446", "XP13085",
                       "XP12813", "XP13083",
                       "XP13087", "XP13084",
                       "XP12814", "XP13086",
                       "XP12274r1", "XP12274r23", "XP12274r34"]
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    EPITHELIAL_CLUSTER_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67']
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    IMM_STR_LABELS = {}
    # cluster 1, 10
    IMM_STR_LABELS['XP12446'] = ['stromal', 'stromal', 'immune', 'immune', 'stromal',
                                 'stromal', 'immune', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'immune', 'immune', 'stromal',
                                 'immune', 'immune', 'stromal']
    # cluster 9
    IMM_STR_LABELS['XP13085'] = ['immune', 'immune', 'immune', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'stromal', 'immune', 'immune']
    # cluster 4, 8, 9
    IMM_STR_LABELS['XP12813'] = ['stromal', 'immune', 'immune', 'stromal', 'stromal',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'stromal', 'immune', 'stromal',
                                 'stromal', 'immune', 'immune', 'immune', 'immune']

    IMM_STR_LABELS['XP13083'] = ['stromal', 'immune', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'stromal', 'stromal',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'immune', 'stromal']
    # cluster 3
    IMM_STR_LABELS['XP13087'] = ['immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'immune', 'immune',
                                 'stromal', 'stromal', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'stromal']
    # cluster 7
    IMM_STR_LABELS['XP13084'] = ['stromal', 'stromal', 'stromal', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'immune', 'stromal',
                                 'immune']
    # cluster 6, 14, 18
    IMM_STR_LABELS['XP12814'] = ['immune', 'stromal', 'stromal', 'immune', 'immune',
                                 'stromal', 'stromal', 'stromal', 'immune', 'immune',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'immune', 'stromal']
    # cluster 7
    IMM_STR_LABELS['XP13086'] = ['immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'immune', 'stromal',
                                 'immune', 'stromal', 'stromal', 'immune', 'immune',
                                 'stromal', 'immune']
    # cluster 2
    IMM_STR_LABELS['XP12274r1'] = ['stromal', 'stromal', 'stromal', 'immune', 'stromal',
                                   'stromal', 'stromal', 'immune', 'immune', 'immune',
                                   'immune', 'immune', 'immune', 'stromal', 'stromal',
                                   'immune', 'immune', 'stromal']
    # cluster 0, 4, 6
    IMM_STR_LABELS['XP12274r23'] = ['stromal', 'immune', 'stromal', 'stromal', 'stromal',
                                    'stromal', 'stromal', 'immune', 'immune', 'immune',
                                    'immune', 'stromal', 'stromal', 'stromal', 'stromal',
                                    'stromal', 'immune']
    # cluster 0
    IMM_STR_LABELS['XP12274r34'] = ['stromal', 'stromal', 'immune', 'stromal', 'stromal',
                                    'immune', 'stromal', 'immune', 'immune', 'immune',
                                    'immune', 'stromal', 'stromal', 'stromal', 'stromal',
                                    'immune']

    """
    Prototype-running starts from here.
    """
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")

    # Step 1. Running 2-component GMM to identify epithelial cells
    CLUSTER_KEY = 'esi'
    run_gmm_for_epithelial(adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=EXPERIMENT)

    # Step 2. Running leiden clustering for epithelial and non-epithelial individually, epithelial cells for
    #         sub-clusters and non-epithelial cells for sub-clusters and also identification of immune and stromal cells
    CLUSTER_KEY = 'leiden'

    # epithelial
    epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")
    run_leiden_for_clusters(epi_adata, EPITHELIAL_CLUSTER_MARKERS, CLUSTER_KEY, out_dir,
                            plot_cluster_heatmap=True, name=f"{EXPERIMENT}_find_epithelial")
    epi_adata.write(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # non-epithelial
    tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    run_leiden_for_clusters(tme_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY, out_dir,
                            plot_cluster_heatmap=True, name=f"{EXPERIMENT}_imm_str")
    tme_adata.write(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # for full dataset, update the immune and stromal clusters
    EXPERIMENT = 'full_11_xp_batcor'
    in_dir = Path(f"analysis/{TYPE}")
    out_dir = Path(f"analysis/{TYPE}")

    adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
    tme_adata = read_h5ad(in_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    epi_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epithelial.h5ad")

    # update immune and stromal clusters and update to obs.esi
    imm_str_labels = ['immune', 'immune', 'stromal', 'immune', 'stromal',
                      'immune', 'stromal', 'stromal', 'stromal', 'immune',
                      'immune', 'immune', 'immune', 'immune', 'stromal',
                      'immune', 'immune', 'stromal', 'stromal', 'immune',
                      'immune', 'stromal', 'immune']

    print("start mapping")
    label_name_dict = dict(zip(range(len(imm_str_labels)), imm_str_labels))
    CLUSTER_KEY = 'esi'
    NEW_CLUSTER_KEY = 'leiden'
    update_labels(tme_adata, CLUSTER_KEY, [label_name_dict[int(i)] for i in tme_adata.obs[NEW_CLUSTER_KEY]])
    tme_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")

    # merge the labels from epithelial, immune and stromal to annotation.h5ad
    update_esi_labels(EXPERIMENT, CLUSTER_KEY, out_dir)
    # finish, good! --> {'epithelial', 'immune', 'stromal'}

    # update the labels to EXPERIMENT_annotation.h5ad
    # check the update situation
    for EXPERIMENT in EXPERIMENT_LIST:
        print("______________________________")
        print(EXPERIMENT)

        in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
        out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

        CLUSTER_KEY = 'rsi'
        # update_esi_labels(EXPERIMENT, CLUSTER_KEY, out_dir)
        adata_anno = read_h5ad(out_dir / f"{EXPERIMENT}_annotation.h5ad")
        print(set(adata_anno.obs[CLUSTER_KEY]))



    # compare the results from individual experiment and full dataset
    full_labels = adata.obs.esi.to_numpy()

    new_labels_all = []
    CLUSTER_KEY = 'rsi'
    for EXPERIMENT in EXPERIMENT_LIST:
        print("_______________________")
        print(EXPERIMENT)

        in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
        out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

        adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
        labels = adata_anno.obs[CLUSTER_KEY].to_numpy()
        new_labels_all.append(labels)

    new_type_labels_all = np.concatenate(new_labels_all)
    diff_num_batcor = np.sum(new_type_labels_all != full_labels)

    # check with the original full dataset result
    EXPERIMENT = 'full_11_xp'
    CLUSTER_KEY = 'rsi'
    in_dir = Path(f"analysis/{TYPE}/original_data_results")
    full_adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
    original_full_labels = full_adata.obs[CLUSTER_KEY].to_numpy()

    diff_num_fulls = np.sum(full_labels != original_full_labels)

    # change the column name from rsi to esi
    CLUSTER_KEY = 'rsi'
    for EXPERIMENT in EXPERIMENT_LIST:
        print("_______________________")
        print(EXPERIMENT)

        in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
        out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

        adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
        print("before")
        print(adata_anno.obs.columns)
        adata_anno.obs.columns = ['image_id', 'area', 'esi', 'leiden']
        print("new columns:")
        print(adata_anno.obs.columns)
        adata_anno.write(out_dir / f"{EXPERIMENT}_annotation.h5ad")



