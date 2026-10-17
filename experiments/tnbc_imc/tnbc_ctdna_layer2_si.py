"""
@author: Tina Hsu
Layer 2 immune and stromal cell type clustering and annotation for TNBC IMC BM+
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import scanpy as sc
from src.visualize import (plot_zscore_cluster_mean_heatmap, plot_zscore_cell_expression_heatmap,
                           plot_umap, plot_umap_panel)
from src.annotate import update_labels

if __name__ == '__mian__':
    TYPE = 'tnbc_imc'
    TMA = 'full_4_tma'

    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{TMA}.h5ad")
    adata_is = adata[np.where(adata.obs['layer1_classify'] != 'epithelial')[0], :].copy()
    IMM_STR_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']

    temp_adata = adata_is[:, IMM_STR_MARKERS].copy()

    # run leiden for epithelial cells, starting
    print("pca")
    sc.tl.pca(temp_adata, svd_solver='arpack')
    print("neighbors")
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    print("leiden")
    sc.tl.leiden(temp_adata)

    CLUSTER_KEY = 'leiden'
    plot_zscore_cluster_mean_heatmap(temp_adata, IMM_STR_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer2_is_2_-2")

    # manual annotate
    imm_str_labels = ['Blood endothelial', 'T cell', 'Undefined', 'Undefined stromal', 'T cell',
                      'B cell', 'Smooth muscle cell', 'Blood endothelial', 'Undefined', 'Undefined stromal',
                      'Macrophage', 'B cell', 'T cell', 'Vimentin', 'Macrophage',
                      'Undefined', 'Macrophage', 'B cell', 'B cell', 'Macrophage',
                      'Smooth muscle cell', 'Undefined stromal', 'Undefined immune', 'Macrophage', 'T cell',
                      'Smooth muscle cell', 'Undefined stromal']
    # Undefined: around 10-12 % are immune cells and the rest are stromal cells

    # peak

    temp_adata[np.where(temp_adata.obs.leiden == '26')[0], :].obs.layer1_classify.value_counts()

    # update layer 2 labels
    label_name_dict = dict(zip(range(len(imm_str_labels)), imm_str_labels))

    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'layer2_is'
    update_labels(temp_adata, NEW_CLUSTER_KEY, [label_name_dict[int(i)] for i in temp_adata.obs[OLD_CLUSTER_KEY]])

    temp_adata.write(out_dir / f"{TMA}_layer2_imm_str.h5ad")

    CLUSTER_KEY = 'layer2_is'
    plot_zscore_cluster_mean_heatmap(temp_adata, IMM_STR_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer2_is_annotate")

    # check

    plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name="layer2_is_annotate")

    CLUSTER_KEY = 'layer2_is'
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{TMA}_layer2_is")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{TMA}_layer2_is")

    temp_adata.write(out_dir / f"{TMA}_layer2_epi.h5ad")

    # check the distribution of cell types (temp_adata is layer 2 yet being stored), take XP13087 (22-010A) as example







