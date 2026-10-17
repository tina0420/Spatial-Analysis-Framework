"""
@author: Tina Hsu
Merging all TMAs together and running the pipeline in previous python files
"""
import os
from pathlib import Path
from tnbc_xp_setup import *
from anndata import read_h5ad
from tnbc_esi_fun import *


if __name__ == '__main__':
    # for full (combine all experiments together)
    in_dir = Path(f"data/{TYPE}")
    xp_1 = read_h5ad(in_dir / "XP12446/XP12446_filter.h5ad")
    xp_2 = read_h5ad(in_dir / "XP12813/XP12813_filter.h5ad")
    xp_3 = read_h5ad(in_dir / "XP12814/XP12814_filter.h5ad")
    xp_4 = read_h5ad(in_dir / "XP13087/XP13087_filter.h5ad")

    # intersection = set(list_1).intersection(list_2, list_3)
    common_markers = list(set(xp_1.var_names) & set(xp_2.var_names) & set(xp_3.var_names) & set(xp_4.var_names))
    common_markers = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3',
                      'CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    full_adata = xp_1[:, common_markers].concatenate(xp_2[:, common_markers], xp_3[:, common_markers],
                                                     xp_4[:, common_markers],
                                                     batch_key="batch")
    full_adata.obs['batch'] = full_adata.obs['batch'].map({'0': 'BOB', '1': '22-010A', '2': '23-009', '3': '22-010C'})

    out_dir = Path(f"analysis/{TYPE}_test")
    plot_marker_expression_distribution_violin(full_adata, 'batch', out_dir,
                                               set_count=4, figsize=(8, 4), name="Across_4_tma")

    import scanpy as sc

    sc.pp.normalize_total(full_adata)
    sc.pp.log1p(full_adata)

    CLUSTER_KEY = 'batch'
    sc.tl.pca(full_adata, svd_solver='arpack')
    sc.pp.neighbors(full_adata, use_rep='X_pca')
    sc.tl.umap(full_adata)
    plot_umap(full_adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)
    plot_umap_panel(full_adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)