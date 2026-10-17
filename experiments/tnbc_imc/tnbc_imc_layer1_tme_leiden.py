"""
@author: Tina Hsu
Leiden Clustering for layer1 TME and annotate into immune and stromal cell types
"""
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
import numpy as np
from src.annotate import update_labels
from src.visualize import plot_zscore_cluster_mean_heatmap, plot_zscore_cell_expression_heatmap
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def compare_clustering_resolutions(adata, resolutions, method, out_dir=None, plot=True, figsize=(5, 7), name=None):
    if method == 'calinski_harabasz':
        scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = calinski_harabasz_score(adata.obsm["X_pca"], adata.obs[col])
            scores[str(res)] = score
        adata.uns[method] = scores

    elif method == 'davies_bouldin':
        scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = davies_bouldin_score(adata.obsm["X_pca"], adata.obs[col])
            scores[str(res)] = score
        adata.uns[method] = scores

    elif method == 'all':
        cal_scores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = calinski_harabasz_score(adata.obsm["X_pca"], adata.obs[col])
            cal_scores[str(res)] = score
        adata.uns['calinski_harabasz'] = cal_scores

        db_sores = {}
        for res in resolutions:
            col = f'leiden_{res:.1f}'
            score = davies_bouldin_score(adata.obsm["X_pca"], adata.obs[col])
            db_sores[str(res)] = score
        adata.uns['davies_bouldin'] = db_sores

    else:
        print("Please select each of (calinski_harabasz, davies_bouldin and all)")

    if plot:
        plt.figure(figsize=figsize)
        if method == 'all':
            # Normalize the scores
            nor_values_cal = (list(cal_scores.values()) - min(cal_scores.values())) / (
                    max(cal_scores.values()) - min(cal_scores.values()))
            plt.plot(nor_values_cal, marker='o', linestyle='-', color='b', label='Calinski score')

            nor_values_db = (list(db_sores.values()) - min(db_sores.values())) / (
                    max(db_sores.values()) - min(db_sores.values()))
            plt.plot(nor_values_db, marker='o', linestyle='-', color='r', label='Davies-Bouldin Index')
        else:
            plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='-', color='b', label=method)

        plt.xticks(ticks=np.arange(len(resolutions)), labels=resolutions, fontsize=10)
        plt.xlabel("Resolution", fontsize=12)
        plt.yticks(fontsize=10)

        plt.legend()
        plt.tight_layout()
        if name:
            plt.savefig(out_dir / f"{name}_clustering_performance.png", dpi=500)
        else:
            plt.savefig(out_dir / "clustering_performance.png", dpi=500)

        plt.clf()


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final_thesis'
    XP = 'TNBC_pretreatment'

    CLUSTER_KEY = 'leiden'
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']

    in_dir = Path(f'data/{TYPE}/{PURPOSE}')
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad with tme cells only
    adata = read_h5ad(in_dir / f"{XP}_tme.h5ad")

    print("running leiden")

    # Try different resolutions for leiden clustering
    adata_tme = adata[:, IMMUNE_CLUSTER_MARKERS].copy()
    sc.tl.pca(adata_tme, svd_solver='arpack')
    sc.pp.neighbors(adata_tme, use_rep='X_pca')

    # Try different resolutions of leiden clustering
    resolutions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for res in resolutions:
        print(res)
        col = f'leiden_{res:.1f}'
        sc.tl.leiden(adata_tme, resolution=res, key_added=col)

    # Visualize zscore heatmap to see the cluster results from different resolutions
    for res in resolutions:
        CLUSTER_KEY = f'leiden_{res:.1f}'
        print(CLUSTER_KEY)
        plot_zscore_cluster_mean_heatmap(adata_tme, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY, vmax=3, vmin=-3,
                                         out_dir=out_dir, figsize=(7, 7), name=f"layer2_tme_3_{res}")

    # Compare the clustering performances across different resolutions
    compare_clustering_resolutions(adata=adata_tme,
                                   resolutions=resolutions,
                                   # method='davies_bouldin',
                                   method='all',
                                   out_dir=out_dir,
                                   plot=True,
                                   figsize=(7, 5),
                                   name=f"{XP}_tme_layer2_all")

    # Annotation for clustering result from resolution 0.6 (lowest value in comparison)
    tme_layer1 = ["Immune", 'Immune', 'Immune', 'Immune', 'Stromal',
                  'Stromal', 'Stromal', 'Stromal', 'Stromal', 'Cluster9',
                  'Stromal', 'Immune', 'Stromal', 'Stromal', 'Immune',
                  'Immune', 'Immune']

    # Update the annotation to esi
    CLUSTER_KEY = 'leiden_0.6'
    NEW_CLUSTER_KEY = 'ESI'
    labels_dict = dict(zip(range(len(set(adata_tme.obs[CLUSTER_KEY]))), tme_layer1))
    temp_adata = adata_tme.copy()
    update_labels(temp_adata, cluster_key=NEW_CLUSTER_KEY,
                  labels=[labels_dict[int(x)] for x in temp_adata.obs[CLUSTER_KEY]])
    # Plot a cell expression heatmap to check what is Cluster9
    plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, NEW_CLUSTER_KEY, vmax=2, vmin=-2,
                                     out_dir=out_dir, figsize=(7, 7), name="tme_check_cluster")

    plot_zscore_cell_expression_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, NEW_CLUSTER_KEY, out_dir, figsize=(12, 12),
                                        rotation=35, color_dict={'Cluster9': '#b41f77',
                                                                 'Immune': '#77b41f',
                                                                 'Stromal': '#1f77b4'}, name=f"{XP}_layer1_default",
                                        v_max=2,
                                        v_min=-2)

    # Update the unsure cluster, Cluster9 here, and maker all the cell types updated
    tme_layer1 = ["Immune", 'Immune', 'Immune', 'Immune', 'Stromal',
                  'Stromal', 'Stromal', 'Stromal', 'Stromal', 'Immune',
                  'Stromal', 'Immune', 'Stromal', 'Stromal', 'Immune',
                  'Immune', 'Immune']

    CLUSTER_KEY = 'leiden_0.6'
    NEW_CLUSTER_KEY = 'ESI'
    labels_dict = dict(zip(range(len(set(adata_tme.obs[CLUSTER_KEY]))), tme_layer1))

    update_labels(adata_tme, cluster_key=NEW_CLUSTER_KEY,
                  labels=[labels_dict[int(x)] for x in adata_tme.obs[CLUSTER_KEY]])
    update_labels(adata, cluster_key=NEW_CLUSTER_KEY, labels=[labels_dict[int(x)] for x in adata_tme.obs[CLUSTER_KEY]])
    update_labels(adata, cluster_key="leiden", labels=list(adata_tme.obs[CLUSTER_KEY]))

    adata.write(out_dir / f"{XP}_tme.h5ad")
    adata_tme.write(out_dir / f"{XP}_esi_tme.h5ad")

    # Annotation for clustering result from resolution 0.6
    tme_layer2 = ["Macrophage", "T Cell", "T Cell", "B Cell", "Fibroblast",
                  "Blood Endothelial", "Smooth Muscle Cell", "Fibroblast", "Vimentin+", "",
                  "Blood Endothelial", "Macrophage", "Smooth Muscle Cell", "Smooth Muscle Cell", "Macrophage",
                  "Macrophage", "B Cell"]

    plot_zscore_cluster_mean_heatmap(adata, adata.var_names, "leiden", vmax=2, vmin=-2,
                                     out_dir=out_dir, figsize=(7, 7), name="tme_check_cluster_all_markers")
