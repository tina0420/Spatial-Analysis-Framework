"""
@author: Tina Hsu
Leiden Clustering for layer2 TME and annotate into immune and stromal cell subtypes
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.annotate import update_labels
from src.visualize import plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap, \
    plot_spatial_cluster_distribution, plot_umap, plot_umap_panel
import scanpy as sc
from experiments.tnbc_imc.tnbc_imc_setup_analysis import LAYER2_CELL_TYPE_COLOR_DICT

if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific_5_tma'
    XP = '5_tma_new_7'

    in_dir = out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad
    # adata = read_h5ad(in_dir / f"{XP}_finalize.h5ad")
    adata_tme = read_h5ad(in_dir / f"{XP}_tme.h5ad")

    # Create an adata for processing
    adata_layer2 = adata_tme.copy()

    # Annotate TME layer 2, resolution 0.8, not sure about cluster 15 (need to plot and then annotate)

    tme_layer2 = ['Blood Endothelial', 'T Cell', 'T Cell', 'Smooth Muscle Cell', 'Smooth Muscle Cell',
                  'Fibroblast', 'Blood Endothelial', 'Macrophage', 'Fibroblast', 'Macrophage',
                  'B Cell', 'Macrophage', 'T Cell', 'Vimentin+', 'B Cell',
                  'B Cell', 'Fibroblast', 'vWF+', 'B Cell']

    # Update layer 2 tme annotations
    CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'layer2'
    label_dict = dict(zip(range(len(set(adata_layer2.obs[CLUSTER_KEY]))), tme_layer2))
    update_labels(adata_layer2, cluster_key=NEW_CLUSTER_KEY,
                  labels=[label_dict[int(x)] for x in adata_layer2.obs[CLUSTER_KEY]])

    # Output the h5ad
    adata_layer2.write(out_dir / f"{XP}_tme.h5ad")
    adata_layer2.write(out_dir / f"{XP}_tme_layer2_finalize.h5ad")

    # # merge the layer2 tme annotation to finalize to plot and see what is cluster 15
    # adata_all = adata.copy()
    # import numpy as np
    #
    # CLUSTER_KEY = "layer1"
    # new_labels = np.zeros(adata_all.shape[0], dtype='object')
    # new_labels[np.where(adata_all.obs[CLUSTER_KEY] == 'Epithelial')[0]] = np.array(
    #     ["Epithelial"] * len(np.where(adata_all.obs["layer1"] == 'Epithelial')[0]))
    # new_labels[np.where(adata_all.obs[CLUSTER_KEY] == 'TME')[0]] = adata_layer2.obs["layer2"].to_numpy()
    #
    # update_labels(adata_all, cluster_key=NEW_CLUSTER_KEY, labels=new_labels)
    #
    # plot
    # CLUSTER_KEY = 'layer2'
    # UNIT_KEY = 'image_roi'
    #
    # out_dir = out_dir / "layer2_validation"
    # if os.path.exists(out_dir) == 0:
    #     os.makedirs(out_dir, exist_ok=True)
    #
    # for pid in pd.unique(adata_all.obs[UNIT_KEY]):
    #     p_adata = adata_all[np.where(adata_all.obs[UNIT_KEY] == pid)[0], :]
    #     if pid.split("_")[0] == "BOB":
    #         plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=4,
    #                                       color=LAYER2_CELL_TYPE_COLOR_DICT, name=pid)
    #     else:
    #         plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=2,
    #                                       color=LAYER2_CELL_TYPE_COLOR_DICT, name=pid)
    #
    # # Modify the annotation list (CLUSTER15_BT -> B Cell) and update

    # UMAP to see how distinct of TME subtypes
    temp_adata = adata_layer2.copy()
    CLUSTER_KEY = 'layer2'

    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.umap(temp_adata)

    plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name=f"{XP}_layer2_tme")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
                    dotsize=0.1, show_counts=False, name=f"{XP}_layer2_tme")

    temp_adata.write(out_dir / f"{XP}_tme_layer2_finalize.h5ad")

    # Plotting to see how well is the annotation
    plot_zscore_cluster_mean_heatmap(temp_adata, ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin',
                                                  'vWF'], CLUSTER_KEY, vmax=3, vmin=-3,
                                     out_dir=out_dir, figsize=(7, 7), name=f"{XP}_tme_layer2")

    plot_zscore_cell_expression_heatmap(temp_adata, ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin',
                                                     'vWF'], cluster_key=CLUSTER_KEY,
                                        out_dir=out_dir, figsize=(7, 7), rotation=35,
                                        color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
                                        name=f"{XP}_tme_layer2")
