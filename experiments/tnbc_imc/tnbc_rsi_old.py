"""
@author: Tina Hsu
RSI cell types identification for TNBC IMC.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from anndata import read_h5ad
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import run_leiden, update_labels
from src.visualize import plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel, \
    plot_zscore_cluster_mean_heatmap, plot_cluster_cell_count
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
    print("label length: ", str(len(bgm_labels)))
    print(type(bgm_labels))

    return np.array(bgm_labels)


def separate_epithelial_tme(adata, cluster_key, epithelial_markers):
    """
    Separating all cells into epithelium and TME cells.
    :param adata: anndata
    :param cluster_key: indicate the column used in obs, type: str
    :param epithelial_markers: epithelial marker list used in 2-component GMM, type: list[str]
    :return:
    """
    gmm_labels = run_gmm(adata, epithelial_markers)
    print("anndata shape: ", str(adata.obs.shape[0]))
    update_labels(adata, cluster_key=cluster_key, labels=gmm_labels)

    # identify which is 0 and which is 1 by mean expression of the markers
    if adata[np.where(adata.obs[cluster_key] == 0)[0], epithelial_markers].X.mean() > adata[
        np.where(adata.obs[cluster_key] == 1)[0], epithelial_markers].X.mean():
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'TME'})
        print("0 is epithelial and 1 is TME")
        print("epithelial cells: " + str(adata.obs[cluster_key].value_counts()['epithelial']))
        print("TME cells: " + str(adata.obs[cluster_key].value_counts()['TME']))
    else:
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'TME', 1: 'epithelial'})
        print("0 is TME and 1 is epithelial")
        print("TME cells: " + str(adata.obs[cluster_key].value_counts()['TME']))
        print("epithelial cells: " + str(adata.obs[cluster_key].value_counts()['epithelial']))


def identify_rsi(adata, cluster_key, epithelial_markers, immune_markers):
    if set(epithelial_markers).issubset(adata.var_names) is False or set(immune_markers).issubset(
            adata.var_names) is False:
        print("Epithelial or immune markers cannot be found in adata!")

    # first round GMM to identify epithelial cells
    bgm_labels = run_gmm(adata, epithelial_markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    # check which is 0 and which is 1 to decide which one can be used for the second time GMM (not epithelial one)
    if adata[np.where(adata.obs[CLUSTER_KEY] == 0)[0], epithelial_markers].X.mean() > adata[
        np.where(adata.obs[CLUSTER_KEY] == 1)[0], epithelial_markers].X.mean():
        # 0: epithelial, 1: others
        continue_cat = 1
    else:
        # 0: others, 1: epithelial
        continue_cat = 0

    # second round GMM to identify immune and stromal cells
    continue_adata = adata[np.where(adata.obs[CLUSTER_KEY] == continue_cat)[0], :]
    bgm_labels = run_gmm(continue_adata, immune_markers)

    # check which is 0 and which is 1 (for immune and stromal)
    if continue_adata[np.where(continue_adata.obs[CLUSTER_KEY] == 0)[0], immune_markers].X.mean() > continue_adata[
        np.where(continue_adata.obs[CLUSTER_KEY] == 1)[0], immune_markers].X.mean():
        # 0: immune, 1: stromal
        stromal_cat = 1
    else:
        # 0: stromal, 1: immune
        stromal_cat = 0

    # update all the labels, including the epithelium one
    labels = np.array(adata.obs[CLUSTER_KEY])
    new_labels = []
    j = 0
    for i in range(len(labels)):
        if labels[i] != continue_cat:
            # epithelial
            new_labels.append('epithelial')
        else:
            # immune or stromal
            # (can check the second result manually or mean expression) --> which is stromal and which is immune

            if bgm_labels[j] != stromal_cat:
                new_labels.append('immune')
                j += 1
            else:
                new_labels.append('stromal')
                j += 1

    adata.obs[CLUSTER_KEY] = new_labels


def plot_cell_type_proportion_pie(adata, cluster_key, out_dir, figsize=(7, 7), name=None):
    esi_counts = adata.obs[cluster_key].value_counts()

    plt.figure(figsize=figsize)
    plt.pie(esi_counts, labels=esi_counts.index, autopct='%1.1f%%', startangle=90)
    plt.legend(esi_counts.index, title="Cell Types")
    plt.title("Cell type proportions", weight='bold', fontsize=8)
    plt.axis('equal')
    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_type_proportion_pie.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_type_proportion_pie.png", dpi=500)
    plt.clf()


if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    EXPERIMENT = 'full_11_xp'

    # in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    # out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

    # for full data
    in_dir = Path(f"analysis/{TYPE}")
    out_dir = Path(f"analysis/{TYPE}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the xp_filter.h5ad which includes only passed markers
    adata = read_h5ad(in_dir / f"{EXPERIMENT}_pre.h5ad")

    print("1. separating epi and TME")
    # separate all the cells into epithelium and TME cells
    CLUSTER_KEY = 'gmm'
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    FILE_NAME = 'epi_tme'
    separate_epithelial_tme(adata, CLUSTER_KEY, EPITHELIAL_MARKERS)
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name=FILE_NAME)

    print("2. run and plot UMAP")
    # check the classification performance by plotting UMAP
    temp_adata = adata.copy()
    sc.pp.pca(temp_adata)
    sc.pp.neighbors(temp_adata)
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=FILE_NAME)
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=FILE_NAME)
    temp_adata.write(in_dir / f"{EXPERIMENT}_{FILE_NAME}_gmm.h5ad")

    # store the results individually and focus on TME part
    adata[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0], :].write(in_dir / f"{EXPERIMENT}_epithelial.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'TME')[0], :].write(in_dir / f"{EXPERIMENT}_tme.h5ad")

    adata = adata[np.where(adata.obs[CLUSTER_KEY] == 'TME')[0], :]

    print("3. running rsi identification")
    # identify epithelial, immune and stromal cells
    CLUSTER_KEY = 'rsi'
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    IMMUNE_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68']
    FILE_NAME = 'esi'
    identify_rsi(adata, CLUSTER_KEY, EPITHELIAL_MARKERS, IMMUNE_MARKERS)
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name=FILE_NAME)
    adata.write(in_dir / f"{EXPERIMENT}_analysis.5had")

    print("4. plot the pie chart")
    plot_cluster_cell_count(adata, CLUSTER_KEY, out_dir, figsize=(3.5, 2), name=FILE_NAME)
    # plot the proportions of esi
    plot_cell_type_proportion_pie(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), name=FILE_NAME)






