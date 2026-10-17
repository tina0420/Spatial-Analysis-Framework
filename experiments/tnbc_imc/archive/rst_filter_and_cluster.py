"""
@author: Tina Hsu
Double-checking and filtering out the markers fail or unclear, and run Leiden for clustering.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from src.annotate import run_leiden, update_labels
from src.visualize import (plot_cluster_mean_heatmap, plot_zscore_cluster_mean_heatmap,
                           plot_zscore_cell_expression_heatmap, plot_spatial_cluster_distribution, COLORS)
from src.visualize import plot_umap, plot_umap_panel, plot_cluster_cell_count
from scipy.stats import zscore
from scipy.sparse import issparse
from sklearn.mixture import BayesianGaussianMixture


def plot_zscore_cluster_mean_heatmap_here(adata, column_names, cluster_key, out_dir, figsize=(7, 7),
                                          v_max=None, v_min=None, name=None):
    """
    :param adata: AnnData
    :param column_names: set of column names to cluster, dtype: list(str)
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param v_max: max value shown in heatmap, type: float
    :param v_max: max value shown in heatmap, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    # create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
        exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories

    cluster_mean = np.array([np.mean(exp_mtx[np.where(labels == i)[0]], axis=0) for i in label_categories])
    cluster_df = pd.DataFrame(cluster_mean, columns=adata.var_names[idxs], index=label_categories)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if v_max and v_min:
        sns.heatmap(cluster_df, vmax=v_max, vmin=v_min, cmap="coolwarm", ax=ax)
    else:
        sns.heatmap(cluster_df, cmap="coolwarm", ax=ax)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Label", fontsize=8)
    plt.title(f"Z-scored cluster mean heatmap", weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_heatmap_z_cluster_mean.png", dpi=500)
    else:
        plt.savefig(out_dir / f"heatmap_z_cluster_mean.png", dpi=500)
    plt.clf()


# Similar code from 3_compare_cores_TMAs.py, line 57
def GMM_for_3_cell_types(adata, cluster_key, epithelial_list, immune_list, out_dir, name=None):
    if set(epithelial_list).issubset(adata.var_names) is False or set(immune_list).issubset(adata.var_names) is False:
        print("Epithelial or immune markers cannot be found in adata!")

    ck_sum = np.sum(adata[:, epithelial_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_sum)
    bgm_labels = bgm.predict(ck_sum)
    # update labels
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    # check first GMM performance
    plot_zscore_cell_expression_heatmap(adata, adata.var_names,
                                        cluster_key, out_dir,
                                        name=f'{name}_epithelial_separated')

    # start running the second time of GMM, separate immune and stromal cells
    imm_sum = np.sum(adata[np.where(adata.obs[cluster_key] == 0)[0], immune_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(imm_sum)
    bgm_labels = bgm.predict(imm_sum)

    # update all the labels
    labels = np.array(adata.obs[cluster_key])
    new_labels = []
    j = 0
    # immune:0, epithelial: 1, stromal: 2
    for i in range(len(labels)):
        if labels[i] == 0:
            if bgm_labels[j] == 0:
                new_labels.append('stromal')
                j += 1
            elif bgm_labels[j] == 1:
                new_labels.append('immune')
                j += 1
        elif labels[i] == 1:
            new_labels.append('epithelial')

    update_labels(adata, cluster_key=cluster_key, labels=new_labels)

    # check the final results
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, cluster_key, out_dir, rotation=90,
                                        name=f'{name}_final')

    # check cell counts for different cell types
    plot_cluster_cell_count(adata, cluster_key, out_dir, figsize=(3.5, 2), name=f"{name}_final")

    # check the proportion of each cell type (barplots, need to add)


if __name__ == '__main__':
    TYPE = "tnbc_imc"
    # EXPERIMENT = "XP12446"
    EXPERIMENT = "XP13085"

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")

    # XP12446
    # passed_markers = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
    #                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3',
    #                 'CD3e', 'CD20', 'CD45', 'CD68',
    #                 'CD31', 'SMA', 'Vimentin', 'vWF']

    # XP13085
    passed_markers = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                      'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                      'erbB2',
                      'CD3e', 'CD20', 'CD45', 'CD68',
                      'CD31', 'SMA', 'Vimentin', 'vWF']

    # filter the markers
    adata_available = adata[:, passed_markers]
    adata_available.write(in_dir / f"{EXPERIMENT}_filter.h5ad")

    """
    Need to separate the cells into epithelium and TME cells first!!!!!!!!!
    """
    # 2-component GMM
    # using CK5, CK7, CK8-18 and panCK in this experiment
    sub_markers = ['CK5', 'CK7', 'CK8-18', 'panCK']
    ck_markers = np.sum(adata_available[:, sub_markers].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_markers)
    bgm_labels = bgm.predict(ck_markers)
    CLUSTER_KEY = "GMM"
    update_labels(adata_available, cluster_key=CLUSTER_KEY, labels=bgm_labels)

    # replace the result with epithelial and TME
    adata_available.obs["GMM"] = adata_available.obs["GMM"].map({0: 'epithelial', 1: 'TME'})
    plot_umap(adata_available, "GMM", out_dir, figsize=(4.5, 4.5), dotsize=0.5, name="GMM")
    plot_umap_panel(adata_available, "GMM", out_dir, dotsize=0.1, name="GMM")

    # store the Anndata separately (epithelial and TME)
    adata_epi = adata_available[np.where(adata_available.obs['GMM'] == 'epithelial')[0], :]
    adata_tme = adata_available[np.where(adata_available.obs['GMM'] == 'TME')[0], :]

    """
    Method 1. Use Leiden clustering with all passed markers
    """
    # rerun clustering again
    run_leiden(adata_tme, passed_markers, n_neighbors=150)

    CLUSTER_KEY = "leiden"
    plot_cluster_mean_heatmap(adata_tme, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="TME_all")

    plot_zscore_cluster_mean_heatmap(adata_tme, passed_markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7),
                                     name="TME_all")

    # merge the clusters and annotate (into epithelial, immune , stromal and undefined) # 0, 1, 2, 3
    new_annotate = []

    # plot_zscore_cell_expression_heatmap(adata, passed_markers, "leiden", out_dir, figsize=(7, 7), rotation=35,
    #                                     name=None)

    # merge the clusters and annotate
    # XP12446
    # new_annotate = ["Epithelial", "Stromal", "Immune", "Epithelial", "Immune",
    #                 "Immune_undefined", "Stromal", "Stromal", "Stromal", "Epithelial",
    #                 "Epithelial_undefined", "Immune", "Epithelial", "Epithelial", "Epi_imm_undefined",
    #                 "Epithelial", "Epithelial_undefined", "Epithelial", "Epithelial_undefined", "Epithelial",
    #                 "Epithelial_undefined", "Epithelial", "Undefined"]

    # XP13085
    new_annotate = ["Immune", "Stromal", "Immune", "Epithelial_p53_enriched", "Epithelial",
                    "Stromal", "Epithelial_p53_enriched", "Epithelial", "Epithelial_p53_enriched", "Undefined",
                    "Epithelial_undefined", "Stromal", "Epithelial", "Immune", "Epi_stromal",
                    "Epithelial", "Epithelial_p53_enriched", "Immune", "Epi_stromal", "Epithelial",
                    "Immune"]

    clusters_mapped_dict = dict(zip(range(len(adata.obs["leiden"])), new_annotate))
    update_labels(adata=adata, cluster_key="cell_type",
                  labels=[clusters_mapped_dict[x] for x in adata.obs["leiden"]])
    # plot
    plot_zscore_cluster_mean_heatmap_here(adata, passed_markers, "cell_type", out_dir=out_dir,
                                          figsize=(7, 7), v_max=2, v_min=-1, name="3_types")

    # transfer the annotation into number (e.g. Immune: 1)
    new_annot_num = [3, 4, 3, 2, 1,
                     4, 2, 1, 2, 7,
                     6, 4, 1, 3, 5,
                     1, 2, 3, 5, 1,
                     3]

    new_annot_num = [2, 3, 2, 1, 1,
                     3, 1, 1, 1, 4,
                     1, 3, 1, 2, 4,
                     1, 1, 2, 4, 1,
                     2]

    clusters_mapped_dict = dict(zip(range(len(adata.obs["leiden"])), new_annot_num))
    update_labels(adata=adata, cluster_key="cell_type_num",
                  labels=[clusters_mapped_dict[x] for x in adata.obs["leiden"]])

    for pid in set(adata.obs.image_id):
        plot_spatial_cluster_distribution(adata[np.where(adata.obs.image_id == pid)[0], :],
                                          "cell_type_num", out_dir, figsize=(3, 3), dotsize=1,
                                          color=COLORS, name=pid.split("_")[-1].split(".")[0] + "_3_types")

    check_df = pd.DataFrame(adata.obsm['spatial'], columns=["x", "y"])
    check_df["3_cell_types"] = list(adata.obs['cell_type'])

    plt.figure(figsize=(7, 7))
    sns.scatterplot(check_df, x='x', y='y', hue='3_cell_types', sizes=2)
    plt.tight_layout()
    plt.show()

    """
    Method 2. Run 2-component GMM twice to identify epithelial, immune and stromal cells strictly 
    """
    CLUSTER_KEY = "eis"
    EPITHELIAL_LIST = ['CK5', 'CK7', 'CK8-18', 'panCK']
    IMMUNE_LIST = ['CD3e', 'CD20', 'CD45', 'CD68']
    t_adata = adata_available[np.where(adata_available.obs.GMM == 'TME')[0], :]
    GMM_for_3_cell_types(t_adata, CLUSTER_KEY, EPITHELIAL_LIST, IMMUNE_LIST, out_dir, name='B_esi')

    t_adata = adata_available[np.where(adata_available.obs.GMM == 'epithelial')[0], :]
    GMM_for_3_cell_types(t_adata, CLUSTER_KEY, EPITHELIAL_LIST, IMMUNE_LIST, out_dir, name='B_epi_esi')

    # adata_tme and adata_epi stored the wrong subset!!!! which is 0 and which is 1 need to check manually
    for pid in set(t_adata.obs.image_id):
        # plot_spatial_cluster_distribution(t_adata[np.where(t_adata.obs.image_id == pid)[0], :],
        #                                   "eis", out_dir, figsize=(4, 4), dotsize=1,
        #                                   color=COLORS, name=pid.split("_")[-1].split(".")[0] + "new")

        plot_spatial_cluster_distribution(t_adata[np.where(t_adata.obs.image_id == pid)[0], :],
                                          "eis", out_dir, figsize=(4, 4), dotsize=1,
                                          color=COLORS, name=pid + "new")

    for pid in set(adata.obs.image_id):
        plot_spatial_cluster_distribution(adata[np.where(adata.obs.image_id == pid)[0], :],
                                          "cell_type_num", out_dir, figsize=(3, 3), dotsize=1,
                                          color=COLORS, name=pid.split("_")[-1].split(".")[0] + "_3_types")

    labels = t_adata.obs['eis'].values
    label_categories = t_adata.obs['eis'].cat.categories
    is_digit_labels = all(str(e).isdigit() for e in labels)


