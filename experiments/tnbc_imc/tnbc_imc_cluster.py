"""
@author: Tina Hsu
Running cell type niches across cells
"""
import os
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

if __name__ == "__main__":
    out_dir = OUT_DIR / "cluster"
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)


    """
    Round 1 Clustering
    """
    # # Cluster
    # CLUSTER_KEY = 'layer_1'
    # LAYER_1_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK',
    #                    'CD3e', 'CD20', 'CD45', 'CD68',
    #                    'CD31', 'SMA', 'Vimentin']
    #
    # LAYER_1_ADATA = FULL_ADATA[:, LAYER_1_MARKERS]
    # sc.pp.scale(LAYER_1_ADATA, max_value=6)
    # sc.tl.pca(LAYER_1_ADATA, n_comps=6, svd_solver='arpack')
    # sc.pp.neighbors(LAYER_1_ADATA, n_neighbors=15, use_rep='X_pca')
    # sc.tl.leiden(LAYER_1_ADATA, resolution=0.6, key_added=CLUSTER_KEY)
    #
    # # Plot heatmap
    # plot_zscore_cluster_mean_heatmap(LAYER_1_ADATA, LAYER_1_ADATA.var_names, CLUSTER_KEY,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"layer_1")
    #
    # esi = ["Immune", "Immune", "Epithelial", "Stromal", "Stromal",
    #        "Epithelial", "Stromal "Epithelial", "Epithelial", "Epithelial",
    #        "Epithelial", "Stromal", "Epithelial", "Epithelial", "Immune",
    #        "Immune", "Immune", "Epithelial", "Epithelial "Epithelial",
    #        "Epithelial", "U"]
    #
    # esi_map = dict(zip(range(len(esi)), esi))
    # LAYER_1_ADATA.obs.layer_1 = LAYER_1_ADATA.obs.layer_1.astype(int)
    # update_labels(LAYER_1_ADATA, "layer_1", [esi_map[i] for i in LAYER_1_ADATA.obs.layer_1])
    # FULL_ADATA.obs["layer_1"] = LAYER_1_ADATA.obs.layer_1
    # FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.layer_1 != "U"]
    #
    # # Plot heatmap
    # plot_zscore_cluster_mean_heatmap(FULL_ADATA, FULL_ADATA.var_names, CLUSTER_KEY,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"layer_1_final")
    #
    # # Save adata
    # FULL_ADATA.write(IN_DIR / f"TNBC_pretreatment.h5ad")


    """
    Round 2 Clustering
    """
    # # Cluster
    # CLUSTER_KEY = 'layer_2'
    #
    # LAYER_1_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK',
    #                    'CD3e', 'CD20', 'CD45', 'CD68',
    #                    'CD31', 'SMA', 'Vimentin']
    #
    # LAYER_2_TME_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68',
    #                        'CD31', 'SMA', 'Vimentin']
    #
    # LAYER_2_E_MARKERS = ['CK5', 'CK7', 'CK8-18']
    #
    # LAYER_2_ADATA = FULL_ADATA[:, LAYER_1_MARKERS]
    # sc.pp.scale(LAYER_2_ADATA, max_value=6)
    #
    # # Split to E/TME
    # LAYER_2_E_ADATA = LAYER_2_ADATA[:, LAYER_2_E_MARKERS]
    # LAYER_2_E_ADATA = LAYER_2_E_ADATA[LAYER_2_E_ADATA.obs.layer_1 == "E"]
    # LAYER_2_TME_ADATA = LAYER_2_ADATA[:, LAYER_2_TME_MARKERS]
    # LAYER_2_TME_ADATA = LAYER_2_TME_ADATA[LAYER_2_TME_ADATA.obs.layer_1 != "E"]
    #
    # sc.pp.neighbors(LAYER_2_E_ADATA, n_neighbors=15)
    # sc.tl.leiden(LAYER_2_E_ADATA, resolution=0.1, key_added=CLUSTER_KEY)
    #
    # sc.pp.neighbors(LAYER_2_TME_ADATA, n_neighbors=15)
    # sc.tl.leiden(LAYER_2_TME_ADATA, resolution=0.5, key_added=CLUSTER_KEY)
    #
    # # Plot heatmap
    # plot_zscore_cluster_mean_heatmap(LAYER_2_E_ADATA, LAYER_2_E_ADATA.var_names, CLUSTER_KEY,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"layer_2_E")
    # plot_zscore_cluster_mean_heatmap(LAYER_2_TME_ADATA, LAYER_2_TME_ADATA.var_names, CLUSTER_KEY,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"layer_2_TME")
    #
    # e_labels = ["CK+ Epithelial", "CK+ Epithelial", "CK8/18+ Luminal", "CK5+ Basal", "CK5+CK7+ Basal-luminal",
    #             "CK5+ Basal", "CK+ Epithelial", "CK+ Epithelial", "CK8/18+ Luminal", "CK7+ Luminal",
    #             "CK7+ Luminal", "CK+ Epithelial"]
    #
    # tme_labels = ["Endothelial", "T-cell", "Stromal", "SMA+ Stromal", "Stromal",
    #               "B-cell", "Macrophage", "SMA+ Stromal", "Macrophage", "Macrophage",
    #               "T-cell", "Vim+ Stromal", "SMA+ Vim+ Stromal", "Stromal"]
    #
    # e_map = dict(zip(range(len(e_labels)), e_labels))
    # tme_map = dict(zip(range(len(tme_labels)), tme_labels))
    #
    # LAYER_2_E_ADATA.obs.layer_2 = LAYER_2_E_ADATA.obs.layer_2.astype(int)
    # LAYER_2_TME_ADATA.obs.layer_2 = LAYER_2_TME_ADATA.obs.layer_2.astype(int)
    #
    # update_labels(LAYER_2_E_ADATA, "layer_2", [e_map[i] for i in LAYER_2_E_ADATA.obs.layer_2])
    # update_labels(LAYER_2_TME_ADATA, "layer_2", [tme_map[i] for i in LAYER_2_TME_ADATA.obs.layer_2])
    #
    # temp = np.zeros(len(FULL_ADATA)).astype(str)
    # temp[FULL_ADATA.obs.layer_1 == "E"] = LAYER_2_E_ADATA.obs.layer_2
    # temp[FULL_ADATA.obs.layer_1 != "E"] = LAYER_2_TME_ADATA.obs.layer_2
    # update_labels(FULL_ADATA, "layer_2", temp)
    #
    # # Change Vim+ Stromal to Stromal / Change SMA+ Vim+ Stromal to SMA+ Stromal / Basal-luminal to Basal
    # update_labels(FULL_ADATA, "layer_2",
    #               ["Stromal" if i == "Vim+ Stromal" else i for i in FULL_ADATA.obs.layer_2])
    #
    # update_labels(FULL_ADATA, "layer_2",
    #               ["SMA+ Stromal" if i == "SMA+ Vim+ Stromal" else i for i in FULL_ADATA.obs.layer_2])
    #
    # update_labels(FULL_ADATA, "layer_2",
    #               ["CK5+ Basal" if i == "CK5+CK7+ Basal-luminal" else i for i in FULL_ADATA.obs.layer_2])
    #
    # FULL_ADATA.obs["layer_2"] = FULL_ADATA.obs["layer_2"].cat.reorder_categories(
    #     ['CK5+ Basal', 'CK7+ Luminal', 'CK8/18+ Luminal', 'CK+ Epithelial',
    #      'T-cell', 'B-cell', 'Macrophage', 'Endothelial', 'SMA+ Stromal', 'Stromal'], ordered=True)
    #
    # # Plot heatmap
    # plot_zscore_cluster_mean_heatmap(FULL_ADATA, FULL_ADATA.var_names, CLUSTER_KEY,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"layer_2_final")
    #
    # # Save adata
    # FULL_ADATA.write(IN_DIR / f"TNBC_pretreatment.h5ad")


    """
    Plotting
    """
    for c in sorted(pd.unique(FULL_ADATA.obs.unique_img_key)):
        c_adata = FULL_ADATA[FULL_ADATA.obs.unique_img_key == c]
        core_id = c_adata.obs.unique_img_key.iloc[0]

        print(f"Processing {core_id}")

        plot_spatial_cluster_distribution(c_adata, "layer_1", out_dir,
                                          color=ESI_CELL_TYPE_COLOR_DICT,
                                          figsize=(3.5, 3.5), dotsize=0.5, name=core_id+"_L1")

        plot_spatial_cluster_distribution(c_adata, "layer_2", out_dir,
                                          color=CELL_TYPE_COLOR_DICT,
                                          figsize=(4, 3.5), dotsize=0.5, name=core_id+"_L2")

        plot_spatial_cluster_panel(c_adata, "layer_1", out_dir, color=ESI_CELL_TYPE_COLOR_DICT,
                                   figsize=(11, 5), dotsize=0.5, name=core_id+"_L1")

        plot_spatial_cluster_panel(c_adata, "layer_2", out_dir, color=CELL_TYPE_COLOR_DICT,
                                   figsize=(11, 5), dotsize=0.5, name=core_id+"_L2")
