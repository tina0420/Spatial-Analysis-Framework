"""
@author: Tina Hsu
Running cell type niches across cells
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, entropy
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from src.annotate import update_labels
from src.visualize import (plot_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap,
                           plot_spatial_cluster_panel, plot_spatial_cluster_distribution, COLOR_CODE)


def compute_entropy(x):
    return entropy(x.value_counts(normalize=True), base=2)


def compute_cell_cn_matrix(adata, radius, cluster_key):
    # Set up the matrix
    cell_cn_matrix = pd.DataFrame(0,
                                  index=adata.obs_names,
                                  columns=adata.obs[cluster_key].cat.categories)

    # Calculate pairwise distances
    dist_mtx = distance_matrix(adata.obsm["spatial"], adata.obsm["spatial"])

    # Calculate cell x cn matrix
    for i, cell_id in enumerate(adata.obs_names):
        neighbors = adata.obs.loc[dist_mtx[i] <= radius, cluster_key]
        comp = neighbors.value_counts()
        cell_cn_matrix.loc[cell_id, comp.index] = comp.values

    return cell_cn_matrix


if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")

    # Select for pre treatment
    FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]

    """
       In the literature:
       CN: cells within 200 pixels of the centroid of the center cell; each cell iteratively selected as the center.
       Cell x cell state composition matrix: column is a center cell; rows are counts of cell states within CN.
       Spearman used to assess the similarity of the CNs.
       k-means used on matrix to cluster center cells; similar clusters are merged.
    """

    CLUSTER_KEY = "unique_core_id"
    NEIGHBOR_THRESHOLD = 30
    cn_mtx_arr = []

    for pid in sorted(pd.unique(FULL_ADATA.obs[CLUSTER_KEY])):
        p_adata = FULL_ADATA[FULL_ADATA.obs[CLUSTER_KEY] == pid]

        print(f"Processing {pid}")

        # Compute the cell x cn matrix using found parameter
        cn_mtx = compute_cell_cn_matrix(p_adata, radius=NEIGHBOR_THRESHOLD, cluster_key="layer2")
        cn_mtx["core"] = pid
        cn_mtx_arr.append(cn_mtx)

    cn_mtx_arr = pd.concat(cn_mtx_arr)
    cn_mtx_arr = cn_mtx_arr.fillna(0)
    cn_mtx_arr.to_csv(out_dir / "cn_mtx_arr.csv", index=False)

    # Normalize data / proportion (divided by the sum of a row
    cn_mtx_arr = pd.read_csv(out_dir / "cn_mtx_arr.csv")
    cn_adata_raw = ad.AnnData(cn_mtx_arr.values[:, :-1])
    cn_adata_raw.var_names = list(cn_mtx_arr.columns[:-1])
    cn_adata_raw.X = cn_adata_raw.X / cn_adata_raw.X.sum(axis=1, keepdims=True)

    sns.histplot(cn_adata_raw.X[:, 0], kde=True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{XP}cn_multicase.png")

    # Write to file
    cn_adata_raw.X = cn_adata_raw.X.astype(float)
    cn_adata_raw.write(out_dir / "cn_adata_raw.h5ad")

    # Run clustering
    sc.pp.neighbors(cn_adata_raw, n_neighbors=30, use_rep='X', metric='cosine')

    sc.tl.leiden(cn_adata_raw, resolution=0.2, n_iterations=2)
    cn_adata_raw.obs['CN_cluster'] = cn_adata_raw.obs['leiden']
    clusters = cn_adata_raw.obs['CN_cluster']
    df_counts = pd.DataFrame(cn_adata_raw.X, columns=cn_adata_raw.var_names)
    df_counts['CN_cluster'] = cn_adata_raw.obs['CN_cluster'].values
    cluster_means = df_counts.groupby('CN_cluster').mean()

    plot_cell_expression_heatmap(cn_adata_raw, cluster_key="CN_cluster", out_dir=out_dir, name="CN_cluster")
    plot_zscore_cluster_mean_heatmap(adata, column_names, cluster_key, out_dir, vmax=None, vmin=None, figsize=(7, 7),
                                     name=None)

    cn_adata_raw.write(out_dir / "cn_adata_raw.h5ad")

    """
    Annotating CN clusters
    """
    cn_types = ['B_Mal_B', 'B_Macro', 'B_FDC', 'B_Immune', 'B_T-CM', 'B_TFH', 'B_Fibro', 'B_T-Reg',
                'non-B_Macro', 'non-B_Macro_Fibro', 'non-B_Fibro_T-CM_B', 'non-B_T-CM',
                'non-B_T-CM_TFH', 'non-B_T-Eff', 'non-B_Endo_Fibro', 'non-B_FDC_TFH_Cyc',
                'non-B_Non-mal-B', 'non-B_T-Eff_T-Reg_TFH', 'non-B_DC_Cyc_Non-mal-B',
                'non-B_Plasma', 'non-B_Fibro', 'non-B_DC_Cyc', 'non-B_pDC', 'non-B_undefined']

    cn_type_dict = dict(zip(range(len(cn_types)), cn_types))

    cn_type_color_dict = dict(zip(cn_types, COLOR_CODE))

    """
    Plot CN clusters
    """
    cn_tbl = pd.read_csv(out_dir / "cn_tbl.csv")

    cn_cluster_coarse = np.zeros((len(FULL_ADATA))).astype(int)
    cn_cluster = np.zeros((len(FULL_ADATA))).astype(int)
    cn_cluster_name = pd.Series(["0"] * len(FULL_ADATA))

    for c in sorted(pd.unique(FULL_ADATA.obs.core))[2:]:
        cn_cluster_coarse[FULL_ADATA.obs.core == c] = cn_tbl[cn_tbl["core"] == c]["round_1_cluster"].to_numpy()
        cn_cluster[FULL_ADATA.obs.core == c] = cn_tbl[cn_tbl["core"] == c]["round_2_cluster"].to_numpy()
        cn_cluster_name[np.where(FULL_ADATA.obs.core == c)[0]] = [cn_type_dict[i] for i in
                                                                  cn_tbl[cn_tbl["core"] == c]["round_2_cluster"].to_numpy()]

    update_labels(FULL_ADATA, "cn_cluster_coarse", cn_cluster_coarse)
    update_labels(FULL_ADATA, "cn_cluster", cn_cluster)
    update_labels(FULL_ADATA, "cn_cluster_name", cn_cluster_name.values)

    # FULL_ADATA.write(IN_DIR / "fl_tfl_cosmx_6000_new.h5ad")

    # for c in sorted(pd.unique(FULL_ADATA.obs.core))[2:]:
    #     c_adata = FULL_ADATA[FULL_ADATA.obs.core == c]
    #     core_id = c_adata.obs.study_id.iloc[0] + f"_{c}"
    #
    #     print(f"Processing {core_id}")
    #
    #     plot_spatial_cluster_distribution(c_adata, "cn_cluster_coarse", out_dir,
    #                                       figsize=(4.5, 4.5), dotsize=0.5, name=core_id + "_coarse_")
    #
    #     plot_spatial_cluster_distribution(c_adata, "cn_cluster_name", out_dir, color=cn_type_color_dict,
    #                                       figsize=(5, 4.5), dotsize=0.5, name=core_id)
    #
    #     plot_spatial_cluster_panel(c_adata, "cn_cluster_name", out_dir, color=cn_type_color_dict,
    #                                figsize=(11, 7), dotsize=0.5, name=core_id)


