"""
@author: Tina Hsu
Running tnbc_tma_1_batch_effect.py on server
"""
# from experiments.tnbc_imc.tnbc_xp_setup import CORE_INFO, TMA_XP
import os
from pathlib import Path

import numpy as np
from anndata import read_h5ad
import scanpy as sc
from src.visualize import (plot_umap, plot_umap_panel, plot_zscore_cell_expression_heatmap,
                           plot_zscore_cluster_mean_heatmap, plot_spatial_cluster_distribution)
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels


# codes from nb_xenium_batch_effects.py
def compute_lisi(embedding, labels, perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI)
    :param embedding: NxD numpy array
    :param labels: array-like of categorical labels (e.g., batch or cell type)
    :param perplexity:
    :return:
    """
    n_neighbors = min(embedding.shape[0] - 1, perplexity * 3)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    lisi_scores = []

    for i, neighbors in enumerate(indices):
        neighbor_labels = [labels[j] for j in neighbors]
        label_counts = Counter(neighbor_labels)
        total = sum(label_counts.values())
        probs = np.array([count / total for count in label_counts.values()])
        simpson_index = np.sum(probs ** 2)
        inverse_simpson = 1.0 / simpson_index if simpson_index > 0 else 0
        lisi_scores.append(inverse_simpson)

    return np.array(lisi_scores)


def plot_marker_expression_distribution_violin(adata, cluster_key, out_dir, color_list,
                                               set_count=8, figsize=(8, 4), name=None):
    """
    Violin plot for more than one batches, currently only for 2 batches. If more, need to add more colors!
    :param adata: AnnData
    :param cluster_key: indicate the column from obs, type: str
    :param out_dir: output directory, type: Path
    :param color_list: color list for different batches (cluster_key), type: list
    :param set_count: number of the items showing in the x-axis for one figure, type: int
    :param figsize: figure size, type: tuple
    :param name: specific name for the file, type: str
    :return:
    """
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df['batch'] = adata.obs[cluster_key].values
    df_melted = df.melt(id_vars='batch', var_name='Marker', value_name='Expression')

    pairs = construct_range(len(adata.var_names), set_count)
    batch_colors = {
        batch: color for batch, color in zip(df['batch'].unique(), color_list)
    }

    for i in range(len(pairs)):
        print(i)
        subset = df_melted[df_melted.Marker.isin(adata.var_names[pairs[i][0]:pairs[i][1]])]
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.violinplot(data=subset, x="Marker", y="Expression", hue="batch", linewidth=1, ax=ax,
                       palette=batch_colors)

        plt.xlabel("Marker", fontsize=10)
        plt.ylabel("Expression", fontsize=10)
        plt.xticks(fontsize=10, rotation=35, ha="right")
        plt.yticks(fontsize=10)
        plt.grid(alpha=0.1)
        # plt.axhline(y=1, color='red', linestyle='--', linewidth=1)
        plt.ylim(-2, 6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)

        plt.clf()


def run_gmm_for_epithelial(adata, marker_list, cluster_key, out_dir, name=None):
    """
    Running 2-component Gaussian mixture model (GMM) to classify epithelial cells and non-epithelial cells
    :param adata: AnnData
    :param marker_list: subset of markers for GMM, type: list[str]
    :param cluster_key: indicate which column in obs to store the result, type: str
    :param out_dir: output directory, type: Path
    :param name: extra name for the output file, type: str
    :return:
    """
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    # check which is epithelial and which is non-epithelial in the classification result
    if adata[np.where(adata.obs[cluster_key] == 0)[0], marker_list].X.mean() > adata[
        np.where(adata.obs[cluster_key] == 1)[0], marker_list].X.mean():
        # 0: epithelial, 1: non-epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'non-epithelial'})
    else:
        # 0: non-epithelial, 1: epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'non-epithelial', 1: 'epithelial'})

    adata.write(out_dir / f"{name}_merge.h5ad")
    adata[np.where(adata.obs[cluster_key] == 'epithelial')[0], :].write(out_dir / f"{name}_epithelial.h5ad")
    adata[np.where(adata.obs[cluster_key] == 'non-epithelial')[0], :].write(out_dir / f"{name}_non-epithelial.h5ad")


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
    TYPE = 'tnbc_imc'
    # TMA = 'full_4_tma'
    TMA_LIST = ['BOB', '22-010A', '22-010C', '23-009']
    for TMA in TMA_LIST:
        print("_______________________________________")
        print(TMA)

        # create h5ad of each xp with the common markers only
        common_markers = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67',
                          'GATA3', 'CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

        in_dir = Path(f"analysis/{TYPE}/4_tma")
        out_dir = Path(f"analysis/{TYPE}/4_tma")
        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        adata = read_h5ad(in_dir / f"{TMA}.h5ad")

        # # run pca and plot UMAP
        # CLUSTER_KEY = 'batch'
        # sc.pp.pca(adata)
        # sc.pp.neighbors(adata, use_rep='X_pca')
        # sc.tl.umap(adata)
        # plot_umap(adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{TMA}")
        # plot_umap_panel(adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{TMA}")
        #
        # lisi_scores = compute_lisi(adata.obsm['X_pca'][:, :10], adata.obs['batch'].values)
        # adata.obs["lisi"] = lisi_scores

        adata.write(out_dir / f"{TMA}_merge.h5ad")

        # # plot the mean and variances of the marker expression (for marker expression across batches)
        # CLUSTER_KEY = 'batch'
        # TMA_4_colors = ['tab:blue', 'tab:pink', 'tab:orange', 'tab:green']
        # plot_marker_expression_distribution_violin(adata, CLUSTER_KEY, out_dir, color_list=TMA_4_colors, set_count=5,
        #                                            figsize=(8, 4), name=TMA)

        # run GMM to get epithelial and non-epithelial groups
        print("running gmm")
        EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
        CLUSTER_KEY = 'esi'

        temp_adata = adata.copy()
        run_gmm_for_epithelial(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=TMA)

        # check epithelial and non-epithelial zscore and UMAP
        plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                            rotation=35, name=f"{TMA}_find_epithelial")

        # run leiden for non-epithelial
        CLUSTER_KEY = 'leiden'
        IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

        adata = read_h5ad(in_dir / f"{TMA}_non-epithelial.h5ad")

        temp_adata = adata[:, IMMUNE_CLUSTER_MARKERS].copy()
        print("pca")
        sc.tl.pca(temp_adata, svd_solver='arpack')
        print("neighbors")
        sc.pp.neighbors(temp_adata, use_rep='X_pca')
        print("leiden")
        sc.tl.leiden(temp_adata)
        print("plotting")
        plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                         out_dir=out_dir, figsize=(7, 7), name=f"{TMA}_non-epithelial")
        temp_adata.write(out_dir / f"{TMA}_non-epithelial.h5ad")

    # manual annotations for immune, stromal and unknown
    imm_str_labels = ['unknown', 'immune', 'unknown', 'unknown', 'unknown',
                      'stromal', 'stromal', 'immune', 'immune', 'stromal',
                      'unknown', 'immune', 'immune', 'unknown', 'stromal',
                      'stromal', 'immune', 'unknown', 'immune', 'unknown',
                      'unknown', 'immune', 'immune', 'unknown', 'immune',
                      'stromal', 'stromal', 'unknown', 'unknown', 'unknown']

    # update esi labels based on leiden result
    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = imm_str_labels
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)

    # CHECK
    adata = read_h5ad(out_dir / f"{TMA}_merge.h5ad")

    # CHECK VISUALIZATION
    xp = '22-010C'
    temp_adata = adata.copy()
    temp_adata = temp_adata[np.where(temp_adata.obs.batch == xp)[0], :]
    # choose few images, 'ROI20_ROI_020.ome.tiff', 'ROI33_ROI_033.ome.tiff', 'ROI63_ROI_063.ome.tiff'
    ESI_COLOR_DICT = dict(
        zip(['epithelial', 'immune', 'stromal', 'unknown'], ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e']))
    img = 'ROI63_ROI_063.ome.tiff'
    plot_spatial_cluster_distribution(temp_adata[temp_adata.obs.image_id == img, :], 'esi', out_dir,
                                      figsize=(5, 5), dotsize=1, color=ESI_COLOR_DICT,
                                      name=f"{xp}_r2c10")
