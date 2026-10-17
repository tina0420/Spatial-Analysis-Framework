"""
@author: Tina Hsu
Leiden Clustering for layer1 TME and annotate into immune and stromal cell types
"""
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
from src.annotate import update_labels
from src.visualize import plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap, plot_umap, \
    plot_umap_panel
from experiments.tnbc_imc.tnbc_imc_setup_analysis import LAYER2_CELL_TYPE_COLOR_DICT
from experiments.tnbc_imc.tnbc_imc_layer1_tme_leiden import compare_clustering_resolutions

if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final_thesis'
    XP = 'TNBC_pretreatment'

    CLUSTER_KEY = 'leiden'
    EPITHELIAL_CLUSTER_MARKERS = ['CK5', 'CK7', 'CK8-18', 'SMA']

    in_dir = Path(f'data/{TYPE}/{PURPOSE}')
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad with epithelial cells only
    adata = read_h5ad(in_dir / f"{XP}_epi.h5ad")

    adata_epi = adata[:, EPITHELIAL_CLUSTER_MARKERS].copy()
    sc.tl.pca(adata_epi, svd_solver='arpack')
    sc.pp.neighbors(adata_epi, use_rep='X_pca')

    # Try different resolutions of leiden clustering
    resolutions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for res in resolutions:
        print(res)
        col = f'leiden_{res:.1f}'
        sc.tl.leiden(adata_epi, resolution=res, key_added=col)

    # Visualize zscore heatmap to see the cluster results from different resolutions
    for res in resolutions:
        CLUSTER_KEY = f'leiden_{res:.1f}'
        print(CLUSTER_KEY)
        plot_zscore_cluster_mean_heatmap(adata_epi, adata_epi.var_names, CLUSTER_KEY, vmax=3, vmin=-3,
                                         out_dir=out_dir, figsize=(7, 7), name=f"layer2_epi_{res}")

    # Compare the clustering performances across different resolutions
    compare_clustering_resolutions(adata=adata_epi,
                                   resolutions=resolutions,
                                   method='davies_bouldin',
                                   out_dir=out_dir,
                                   plot=True,
                                   figsize=(7, 5),
                                   name=f"{XP}_epi_layer2")

    # Annotation for the resolution of 0.6




    #
    # layer2_epi = ['Epithelial ' + x for x in layer2_epi]
    #
    # # Update layer 2 epithelial annotation to adata_epi
    # CLUSTER_KEY = 'leiden_0.7'
    # NEW_CLUSTER_KEY = 'layer2'
    # # !!! i added the update labels to the CLUSTER_KEY, which is the original one, wrong!!
    # # but for urgent case, just store it as layer2 and update the original epi h5ad!
    # labels_dict = dict(zip(range(len(set(adata_epi.obs[CLUSTER_KEY]))), layer2_epi))
    # labels_dict = dict(zip(layer2_epi, range(len(set(adata_epi.obs[CLUSTER_KEY])))))
    # update_labels(adata_epi, cluster_key=CLUSTER_KEY, labels=[labels_dict[x] for x in adata_epi.obs[CLUSTER_KEY]])
    # update_labels(adata_epi, cluster_key=CLUSTER_KEY, labels=[labels_dict[int(x)] for x in adata_epi.obs[CLUSTER_KEY]])
    #
    # update_labels(adata_epi, cluster_key=NEW_CLUSTER_KEY, labels=list(adata_epi.obs[CLUSTER_KEY]))
    # adata_epi.write(out_dir / f"{XP}_epi_layer2.h5ad")
    #
    # update_labels(adata, cluster_key=NEW_CLUSTER_KEY, labels=list(adata_epi.obs[CLUSTER_KEY]))
    # adata.write(in_dir / f"{XP}_epi.h5ad")
    #
    # # Visualize UMAP and zscore heatmap to see how well is the layer2 annotation of epithelial cells
    # temp_adata = adata.copy()
    # CLUSTER_KEY = 'layer2'
    #
    # sc.tl.pca(temp_adata, svd_solver='arpack')
    # sc.pp.neighbors(temp_adata, use_rep='X_pca')
    # sc.tl.umap(temp_adata)
    #
    # plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
    #           figsize=(4.5, 4.5), dotsize=0.5, name=f"{XP}_layer2_epi")
    # plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
    #                 dotsize=0.1, name=f"{XP}_layer2_epi")
    #
    # plot_zscore_cluster_mean_heatmap(temp_adata, ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67',
    #                                               'GATA3', 'SMA'], CLUSTER_KEY, vmax=3, vmin=-3,
    #                                  out_dir=out_dir, figsize=(7, 7), name=f"{XP}_epi_layer2")
    #
    # plot_zscore_cell_expression_heatmap(temp_adata, ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67',
    #                                                  'GATA3', 'SMA'], CLUSTER_KEY, out_dir, figsize=(7, 7), rotation=35,
    #                                     color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
    #                                     name=f"{XP}_epi_layer2")
