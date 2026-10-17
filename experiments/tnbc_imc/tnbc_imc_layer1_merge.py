"""
@author: Tina Hsu
Merging the layer1 ESI annotations together
"""
import os
from pathlib import Path
import pandas as pd
from anndata import read_h5ad
import scanpy as sc
import numpy as np
from src.annotate import update_labels
from src.visualize import (plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap,
                           plot_umap, plot_umap_panel, plot_spatial_cluster_distribution)
from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT

if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific_5_tma'
    XP = '5_tma_new_7'

    in_dir = out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad
    adata = read_h5ad(in_dir / f"{XP}_finalize.h5ad")
    adata_tme = read_h5ad(in_dir / f"{XP}_tme.h5ad")

    # Create a processing h5ad
    adata_layer1 = adata.copy()

    CLUSTER_KEY = "layer1"
    NEW_CLUSTER_KEY = "esi"

    new_labels = np.zeros(adata_layer1.shape[0], dtype='object')
    new_labels[np.where(adata_layer1.obs[CLUSTER_KEY] == 'Epithelial')[0]] = np.array(
        ["Epithelial"] * len(np.where(adata_layer1.obs["layer1"] == 'Epithelial')[0]))
    new_labels[np.where(adata_layer1.obs[CLUSTER_KEY] == 'TME')[0]] = adata_tme.obs[NEW_CLUSTER_KEY].to_numpy()

    update_labels(adata_layer1, cluster_key=NEW_CLUSTER_KEY, labels=new_labels)

    # plot to check the undefined cluster
    # EXTRA WORK: adding a new column with the same format of image_key: TMA_ROI_number!!!!!!!!!!!!!!!!
    # new_labels = np.zeros(adata_layer1.shape[0], dtype='object')
    # new_labels[np.where(adata_layer1.obs["tma"] != '23-005')[0]] = adata_layer1[
    #                                                                np.where(adata_layer1.obs["tma"] != '23-005')[0],
    #                                                                :].obs.image_key.values.to_numpy()
    # mapping = {'23-005_R3C11': '23-005_ROI_002',
    #            '23-005_R3C13': '23-005_ROI_003',
    #            '23-005_R3C16': '23-005_ROI_004',
    #            '23-005_R4C6': '23-005_ROI_007',
    #            '23-005_R4C15': '23-005_ROI_005',
    #            '23-005_R5C3': '23-005_ROI_008',
    #            '23-005_R7C3': '23-005_ROI_011'}
    #
    # new_labels[np.where(adata_layer1.obs["tma"] == '23-005')[0]] = adata_layer1[
    #                                                                np.where(adata_layer1.obs["tma"] == '23-005')[0],
    #                                                                :].obs.image_key.map(mapping).to_numpy()
    # adata_layer1.obs["image_roi"] = new_labels

    CLUSTER_KEY = 'esi'
    UNIT_KEY = 'image_roi'
    ESI_CELL_TYPE_COLOR_DICT = {'Epithelial': '#b41f77',
                                'Immune': '#77b41f',
                                'Stromal': '#1f77b4'}
    for pid in pd.unique(adata_layer1.obs[UNIT_KEY]):
        p_adata = adata_layer1[np.where(adata_layer1.obs[UNIT_KEY] == pid)[0], :]
        if pid.split("_")[0] == "BOB":
            plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=4,
                                              color=ESI_CELL_TYPE_COLOR_DICT, name=pid)
        else:
            plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=2,
                                              color=ESI_CELL_TYPE_COLOR_DICT, name=pid)

    adata_layer1.write(in_dir / f"{XP}_finalize.h5ad")
    adata_layer1.write(in_dir / f"{XP}_esi_finalize.h5ad")

    # Plot UMAP to see how distinct are Epithelial, Immune and Stromal cells
    temp_adata = adata_layer1.copy()
    CLUSTER_KEY = 'esi'

    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.umap(temp_adata)

    plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=ESI_CELL_TYPE_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name=f"{XP}_esi")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=ESI_CELL_TYPE_COLOR_DICT,
                    dotsize=0.1, name=f"{XP}_esi")

    temp_adata.write(in_dir / f"{XP}_esi_finalize.h5ad")

    # Plot zscore expression of each cell to see if layer 1 annotated well (ESI)
    plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, color_dict=ESI_CELL_TYPE_COLOR_DICT, v_max=3, v_min=-3,
                                        name=f"{XP}_esi_3_-3")

    plot_zscore_cluster_mean_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, vmax=2, vmin=-2,
                                     figsize=(7, 7), name=f"{XP}_esi")


