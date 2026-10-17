"""
@author: Tina Hsu
Different approaches for clustering and annotation of cell types in TNBC IMC.
"""
import os
from pathlib import Path
import numpy as np
from anndata import read_h5ad
# from tnbc_data_info import EXPERIMENT_MARKERS
from sklearn.mixture import BayesianGaussianMixture
from src.visualize import plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel, plot_zscore_cluster_mean_heatmap
import scanpy as sc
from src.annotate import run_leiden, update_labels
import scanpy as sc


def run_gmm(adata, marker_list):
    """
    Run 2-component GMM.
    :param adata: AnnData
    :param marker_list: markers used for GMM, type: list[str]
    :return: labels from GMM, type: list[int]
    """
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)

    return bgm_labels


if __name__ == '__main__':
    # experiment setting
    TYPE = "tnbc_imc"
    EXPERIMENT = "XP12813"
    # EXPERIMENT = "XP13085"
    # EXPERIMENT = "XP12446"
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the xp_filter.h5ad which includes only passed markers
    adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")

    passed_markers = EXPERIMENT_MARKERS[EXPERIMENT]

    """
    Compare all the cells and cells already separated into epithelium and TME cells
    """
    # separate all cells into epithelium and TME cells by 2-component GMM
    CLUSTER_KEY = 'gmm'
    MARKER_LIST = ['CK5', 'CK7', 'CK8-18', 'panCK']
    gmm_labels = run_gmm(adata, MARKER_LIST)
    update_labels(adata, cluster_key=CLUSTER_KEY, labels=gmm_labels)

    # identify which is 0 and which is 1 by mean expression of the markers
    if adata[np.where(adata.obs[CLUSTER_KEY] == 0)[0], MARKER_LIST].X.mean() > adata[
        np.where(adata.obs[CLUSTER_KEY] == 1)[0], MARKER_LIST].X.mean():
        adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].map({0: 'epithelial', 1: 'TME'})
        print("0, epithelial")
    else:
        adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].map({0: 'TME', 1: 'epithelial'})
        print("1, epithelial")

    # plot z-score heatmap to see if the classification accurately
    CLUSTER_KEY = 'gmm'
    plot_zscore_cell_expression_heatmap(adata, passed_markers, CLUSTER_KEY, out_dir, figsize=(7, 7), rotation=35,
                                        name="epi_tme")

    # store epithelium and TME anndata individually
    adata_epi = adata[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0], :]
    adata_epi.write(in_dir / f"{EXPERIMENT}_epithelial.h5ad")
    adata_tme = adata[np.where(adata.obs[CLUSTER_KEY] == 'TME')[0], :]
    adata_tme.write(in_dir / f"{EXPERIMENT}_tme.h5ad")

    # only for preparing and plotting the umap (t_adata, not change the original adata)
    t_adata = adata.copy()
    sc.pp.pca(t_adata)
    sc.pp.neighbors(t_adata)
    sc.tl.umap(t_adata)
    plot_umap(t_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name="epi_tme")
    plot_umap_panel(t_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name="epi_tme")
    run_leiden(t_adata, passed_markers, n_neighbors=150)
    CLUSTER_KEY = "leiden"
    plot_zscore_cluster_mean_heatmap(t_adata, passed_markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")
    # store the results
    t_adata.write(in_dir / f"{EXPERIMENT}_gmm.h5ad")

    """
    Focusing on TME cells.
    """
    """
    Method 1. Run 2-component GMM twice and classify into epithelial, immune and stromal.
    """
    CLUSTER_KEY = "eis"
    EPITHELIAL_LIST = ['CK5', 'CK7', 'CK8-18', 'panCK']
    IMMUNE_LIST = ['CD3e', 'CD20', 'CD45', 'CD68']
    if set(EPITHELIAL_LIST).issubset(adata_tme.var_names) is False or set(IMMUNE_LIST).issubset(
            adata_tme.var_names) is False:
        print("Epithelial or immune markers cannot be found in adata!")

    bgm_labels = run_gmm(adata_tme, EPITHELIAL_LIST)
    update_labels(adata_tme, cluster_key=CLUSTER_KEY, labels=bgm_labels)

    plot_zscore_cell_expression_heatmap(adata_tme, adata_tme.var_names, CLUSTER_KEY, out_dir, name='new_m1_epi')

    # check which is 0 and which is 1 to decide which one can be used for the second time GMM (not epithelial one)
    if adata_tme[np.where(adata_tme.obs[CLUSTER_KEY] == 0)[0], EPITHELIAL_LIST].X.mean() > adata_tme[
        np.where(adata_tme.obs[CLUSTER_KEY] == 1)[0], EPITHELIAL_LIST].X.mean():
        # 0: epithelial, 1: others
        FOLLOWING_IDX = 1
    else:
        # 0: others, 1: epithelial
        FOLLOWING_IDX = 0

    # second round of GMM to separate immune and stromal cells
    epi_rest_adata = adata_tme[np.where(adata_tme.obs[CLUSTER_KEY] == FOLLOWING_IDX)[0], :]

    # second round of epithelial markers as the performance is not good enough
    bgm_labels = run_gmm(epi_rest_adata, EPITHELIAL_LIST)
    update_labels(epi_rest_adata, cluster_key="sec_epi", labels=bgm_labels)

    bgm_labels = run_gmm(epi_rest_adata, IMMUNE_LIST)
    update_labels(epi_rest_adata, cluster_key=CLUSTER_KEY, labels=bgm_labels)

    plot_zscore_cell_expression_heatmap(epi_rest_adata, epi_rest_adata.var_names, CLUSTER_KEY, out_dir, name='new_m1_imm_str')

    plot_zscore_cell_expression_heatmap(epi_rest_adata, epi_rest_adata.var_names, "sec_epi", out_dir, name='new_m1_sec_epi')


    # update all the labels, including the epithelium one
    labels = np.array(adata_tme.obs[CLUSTER_KEY])
    new_labels = []
    j = 0
    for i in range(len(labels)):
        if labels[i] != 0:
            # epithelial
            new_labels.append('epithelial')
        else:
            # immune or stromal (check the second result) --> 0 is stromal and 1 is immune
            if bgm_labels[j] == 1:
                new_labels.append('immune')
                j += 1
            else:
                new_labels.append('stromal')
                j += 1

    print(j)
    adata_tme.obs[CLUSTER_KEY] = new_labels

    # visualize the final result of eis
    plot_zscore_cell_expression_heatmap(adata_tme, adata_tme.var_names, CLUSTER_KEY, out_dir, name='new_m1_esi_10')

    adata_tme.write(in_dir / f"{EXPERIMENT}_tme.h5ad")


    """
    Method 2.Leiden clustering by all the markers
    """



