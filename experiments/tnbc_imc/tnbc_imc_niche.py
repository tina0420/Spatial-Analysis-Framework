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
from src.annotate import update_labels
from src.visualize import (plot_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap,
                           plot_spatial_cluster_panel, plot_spatial_cluster_distribution, COLOR_CODE)
from experiments.tnbc_imc.tnbc_imc_setup_analysis import *


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
    out_dir = OUT_DIR / "niche"
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
       In the literature:
       CN: cells within 200 pixels of the centroid of the center cell; each cell iteratively selected as the center.
       Cell x cell state composition matrix: column is a center cell; rows are counts of cell states within CN.
       Spearman used to assess the similarity of the CNs.
       k-means used on matrix to cluster center cells; similar clusters are merged.
    """
    # CLUSTER_KEY = "unique_img_key"
    # NEIGHBOR_THRESHOLD = 30
    # cn_mtx_arr = []
    #
    # for pid in sorted(pd.unique(FULL_ADATA.obs[CLUSTER_KEY])):
    #     p_adata = FULL_ADATA[FULL_ADATA.obs[CLUSTER_KEY] == pid]
    #
    #     print(f"Processing {pid}")
    #
    #     # Compute the cell x cn matrix using found parameter
    #     cn_mtx = compute_cell_cn_matrix(p_adata, radius=NEIGHBOR_THRESHOLD, cluster_key="layer_2")
    #     cn_mtx["core"] = pid
    #     cn_mtx_arr.append(cn_mtx)
    #
    # cn_mtx_arr = pd.concat(cn_mtx_arr)
    # cn_mtx_arr = cn_mtx_arr.fillna(0)
    # cn_mtx_arr.to_csv(out_dir / "cn_mtx_arr.csv", index=False)


    """
    Cluster CNs
    """
    # # Normalize data / proportion (divided by the sum of a row
    # cn_mtx_arr = pd.read_csv(out_dir / "cn_mtx_arr.csv")
    # cn_adata_raw = ad.AnnData(cn_mtx_arr.values[:, :-1])
    # cn_adata_raw.var_names = list(cn_mtx_arr.columns[:-1])
    # cn_adata_raw.obs = pd.DataFrame(cn_mtx_arr.values[:, -1], columns=["core"])
    # cn_adata_raw.X = cn_adata_raw.X / cn_adata_raw.X.sum(axis=1, keepdims=True)
    #
    # # Write to file
    # cn_adata_raw.X = cn_adata_raw.X.astype(float)
    # cn_adata_raw.write(out_dir / "cn_adata_raw.h5ad")
    #
    # # Run clustering
    # sc.pp.neighbors(cn_adata_raw, n_neighbors=30, use_rep='X', metric='cosine')
    #
    # sc.tl.leiden(cn_adata_raw, resolution=0.2, n_iterations=2)
    # cn_adata_raw.obs['CN_cluster'] = cn_adata_raw.obs['leiden']
    # clusters = cn_adata_raw.obs['CN_cluster']
    # df_counts = pd.DataFrame(cn_adata_raw.X, columns=cn_adata_raw.var_names)
    # df_counts['CN_cluster'] = cn_adata_raw.obs['CN_cluster'].values
    # cluster_means = df_counts.groupby('CN_cluster').mean()
    #
    # # Visualize
    # cn_adata_raw.obs["temp"] = [i if i in ["0", "1", "2", "3", "4", "5", "6",
    #                                        "7", "8", "9", "10", "11", "12"] else '13' for i in cn_adata_raw.obs["leiden"]]
    # cn_adata_raw.obs["temp"] = cn_adata_raw.obs["temp"].astype("category")
    # plot_cell_expression_heatmap(cn_adata_raw, cluster_key="temp", out_dir=out_dir, name="temp")
    # plot_zscore_cluster_mean_heatmap(cn_adata_raw, column_names=cn_adata_raw.var_names, cluster_key="temp", out_dir=out_dir, name="temp")
    #
    # cn_adata_raw.write(out_dir / "cn_adata_raw.h5ad")


    """
    Classifying to major clusters
    """
    # cn_adata_raw = read_h5ad(out_dir / "cn_adata_raw.h5ad")
    #
    # # Force cluster B heavy clusters
    # training_cn_adata = cn_adata_raw[cn_adata_raw.obs["temp"].isin(["0", "1", "2", "3", "4", "5", "6",
    #                                        "7", "8", "9", "10", "11", "12"])]
    # testing_cn_adata = cn_adata_raw[cn_adata_raw.obs["temp"].isin(["13"])]
    #
    # # Train a random forest classifier
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # clf.fit(training_cn_adata.X, training_cn_adata.obs["temp"])
    # clf_labels = clf.predict(testing_cn_adata.X)
    #
    # # Replacing the temp labels with new labels
    # temp_labels = np.array(cn_adata_raw.obs["temp"]).astype(int)
    # temp_labels[cn_adata_raw.obs["temp"].isin(["13"])] = clf_labels
    # update_labels(cn_adata_raw, "temp", temp_labels)
    #
    # cn_adata_raw.write(out_dir / "cn_adata_raw.h5ad")


    """
    Validating CN clusters
    """
    # # Read file
    # cn_adata_raw.obs.to_csv(out_dir / "cn_tbl.csv", index=False)
    #
    # cn_tbl = pd.read_csv(out_dir / "cn_tbl.csv")
    #
    # core_by_cluster_tbl = cn_tbl.groupby(['core', 'temp']).size().unstack(fill_value=0)
    #
    # core_by_cluster_tbl.to_csv(out_dir / "core_by_cluster_tbl.csv")
    #
    # # How evenly distributed (diverse) are round_2_cluster within each core?
    # entropy_by_core = cn_tbl.groupby('core')['temp'].apply(compute_entropy)
    # entropy_by_core.to_csv(out_dir / "entropy_by_core.csv")
    #
    # # How diverse are cores within each round_2_cluster?
    # entropy_by_cluster = cn_tbl.groupby('temp')['core'].apply(compute_entropy)
    # entropy_by_cluster.to_csv(out_dir / "entropy_by_cluster.csv")

    """
    Annotating CN clusters
    """
    # cn_types = ['T_Mp_Aggregation',
    #             'T_Infiltration_Basal',
    #             'Epithelial_Predominant',
    #             'Endothelial_Predominant',
    #             'T_Infiltration_Luminal',
    #
    #             'Epithelial_Predominant',
    #             'T_Resident_Stromal',
    #             'T_Resident_Stromal',
    #             'Epithelial_Predominant',
    #             'Epithelial_Predominant',
    #
    #             'Epithelial_Predominant',
    #             'B_Patch',
    #             'Stromal_Predominant',]
    #
    # cn_type_dict = dict(zip(range(len(cn_types)), cn_types))
    #
    # cn_type_color_dict = dict(zip(cn_types, COLOR_CODE))
    #
    # cn_tbl = pd.read_csv(out_dir / "cn_tbl.csv")
    #
    # cn_cluster = np.zeros((len(FULL_ADATA))).astype(int)
    # cn_cluster_name = pd.Series(["0"] * len(FULL_ADATA))
    #
    # for c in pd.unique(FULL_ADATA.obs.unique_img_key):
    #     cn_cluster[FULL_ADATA.obs.unique_img_key == c] = cn_tbl[cn_tbl["core"] == c]["temp"].to_numpy()
    #     cn_cluster_name[np.where(FULL_ADATA.obs.unique_img_key == c)[0]] = [cn_type_dict[i] for i in
    #                                                               cn_tbl[cn_tbl["core"] == c]["temp"].to_numpy()]
    #
    # update_labels(FULL_ADATA, "cn_cluster", cn_cluster)
    # update_labels(FULL_ADATA, "cn_cluster_name", cn_cluster_name.values)
    #
    # FULL_ADATA.write(IN_DIR / f"TNBC_pretreatment.h5ad")


    """
    Plotting
    """
    cn_types = FULL_ADATA.obs.cn_cluster_name.cat.categories
    cn_type_color_dict = dict(zip(cn_types, COLOR_CODE))

    for c in sorted(pd.unique(FULL_ADATA.obs.unique_img_key)):
        c_adata = FULL_ADATA[FULL_ADATA.obs.unique_img_key == c]
        core_id = c_adata.obs.unique_img_key.iloc[0]

        print(f"Processing {core_id}")

        plot_spatial_cluster_distribution(c_adata, "cn_cluster_name", out_dir, color=cn_type_color_dict,
                                          figsize=(5, 4.5), dotsize=0.5, name=core_id + "_CN")

        plot_spatial_cluster_panel(c_adata, "cn_cluster_name", out_dir, color=cn_type_color_dict,
                                   figsize=(11, 2.4), dotsize=0.5, name=core_id + "_CN")


