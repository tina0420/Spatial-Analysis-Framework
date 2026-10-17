"""
@author: Tina Hsu
Layer 2 cell type clustering and annotation for TNBC IMC BM+
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import scanpy as sc
from src.visualize import plot_zscore_cluster_mean_heatmap
from src.annotate import update_labels
from src.visualize import plot_umap, plot_umap_panel

if __name__ == '__mian__':
    TYPE = 'tnbc_imc'
    TMA = 'full_4_tma'

    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{TMA}.h5ad")
    adata_e = adata[np.where(adata.obs['layer1_classify'] == 'epithelial')[0], :].copy()
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3']
    temp_adata = adata_e[:, EPITHELIAL_MARKERS].copy()

    # run leiden for epithelial cells, starting 11:23
    print("pca")
    sc.tl.pca(temp_adata, svd_solver='arpack')
    print("neighbors")
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    print("leiden")
    sc.tl.leiden(temp_adata)

    CLUSTER_KEY = 'leiden'
    plot_zscore_cluster_mean_heatmap(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer2_e")

    # briefly annotate the epithelial cels into luminal, myoepithelial abd undefined_epithelial
    e_labels = ['luminal', 'luminal', 'luminal', 'luminal', 'undefined_epithelial',
                'luminal', 'undefined_epithelial', 'luminal', 'myoepithelial', 'luminal',
                'myoepithelial', 'luminal', 'myoepithelial', 'luminal', 'undefined_epithelial',
                'myoepithelial', 'luminal', 'myoepithelial', 'luminal', 'myoepithelial',
                'luminal', 'luminal', 'luminal', 'myoepithelial', 'luminal',
                'myoepithelial', 'myoepithelial', 'luminal', 'myoepithelial', 'luminal',
                'undefined_epithelial', 'luminal', 'luminal', 'luminal', 'luminal',
                'luminal', 'luminal', 'luminal', 'luminal', 'undefined_epithelial']

    label_name_dict = dict(zip(range(len(e_labels)), e_labels))

    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'layer2_epi'
    update_labels(temp_adata, NEW_CLUSTER_KEY, [label_name_dict[int(i)] for i in temp_adata.obs[OLD_CLUSTER_KEY]])
    # Note: myoepithelial stands 25.87% (25.877487526796045)

    CLUSTER_KEY = 'layer2_epi'
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{TMA}_layer2_epi")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{TMA}_layer2_epi")

    temp_adata.write(out_dir / f"{TMA}_layer2_epi.h5ad")

    # check
    CLUSTER_KEY = 'layer2_epi'
    plot_zscore_cluster_mean_heatmap(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer2_epi_annotate")

    from src.visualize import plot_zscore_cell_expression_heatmap

    # check epithelial and non-epithelial zscore and UMAP
    plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name="layer2_epi_annotate")


    # other days: modified the first charaters of the cell types, luminal -> Luminal
    adata_epi.obs.layer2_epi = [ct_dict[i] for i in adata_epi.obs['layer2_epi']]



