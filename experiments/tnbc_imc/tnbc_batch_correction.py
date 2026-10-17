"""
@author: Tina Hsu
Batch effect checking for TNBC IMC, including filtering out markers which are fail or unclear.
"""
import os
from pathlib import Path
from tnbc_data_info import EXPERIMENT_LIST, EXPERIMENT_MARKERS
from anndata import read_h5ad
from functools import reduce
from src.annotate import run_leiden, update_labels
import pandas as pd
import numpy as np
import scanpy as sc
from src.visualize import plot_umap, plot_umap_panel, plot_cell_expression_violin
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import harmonypy as hm


def compute_lisi(embedding, labels, perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI)
    :param embedding: NxD numpy array
    :param labels: array-like of categorical labels (e.g., batch or cell type)
    :param perplexity:
    :return:
    """
    n_neighbors = min(embedding.shape[0] - 1, perplexity * 3)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    lisi_scores = []

    for i, neighbors in enumerate(indices):
        neighbor_labels = [labels[j] for j in neighbors]
        label_counts = Counter(neighbor_labels)
        total = sum(label_counts.values())
        probs = np.array([count / total for count in label_counts.values()])
        simpson_index = np.sum(probs ** 2)
        inverse_simpson = 1.0 / simpson_index if simpson_index > 0 else 0
        lisi_scores.append(inverse_simpson)

    return np.array(lisi_scores)


if __name__ == '__main__':
    # filtering out the markers failed or unclear in all TMAs
    TYPE = "tnbc_imc"
    # for EXPERIMENT in EXPERIMENT_LIST:
    #     print(EXPERIMENT)
    #
    #     if EXPERIMENT not in ('XP13328', 'XP13364', 'XP12274'):
    #         print("filter!!")
    #         in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    #         out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    #         if os.path.exists(out_dir) == 0:
    #             os.makedirs(out_dir, exist_ok=True)
    #
    #         adata = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")
    #         passed_markers = EXPERIMENT_MARKERS[EXPERIMENT]
    #         adata = adata[:, passed_markers]
    #         adata.write(in_dir / f"{EXPERIMENT}_filter.h5ad")
    #     else:
    #         print("not filter!!")
    #         continue

    # filtering out markers for all XP12274 experiments (3 in total)
    for xp in ("XP12274r1", "XP12274r23", "XP12274r34"):
        print(xp)
        in_dir = Path(f"data/{TYPE}/{xp}")

        adata = read_h5ad(in_dir / f"{xp}.h5ad")
        passed_markers = EXPERIMENT_MARKERS[EXPERIMENT]
        adata = adata[:, passed_markers]
        adata.write(in_dir / f"{xp}_filter.h5ad")

    """
    Real batch correction starts here!
    """
    # extract the common passed markers across all experiments
    common_markers = list(reduce(set.intersection, map(set, EXPERIMENT_MARKERS.values())))
    # 14, ['Ki67', 'CD68', 'CK8-18', 'SMA', 'CD3e', 'panCK', 'CK5', 'CK7', 'B-Catenin', 'H3K27me3', 'CD20', 'Vimentin',
    #      'CD45', 'CD31']
    # change ot to the order i want
    # change ot to the order i want
    common_markers = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'CD3e', 'CD20', 'CD45',
                      'CD68', 'CD31', 'SMA', 'Vimentin']

    # merging the slide expression data manually
    experiment_label = []
    all_df = pd.DataFrame()
    for EXPERIMENT in EXPERIMENT_LIST:
        print(EXPERIMENT)
        in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
        adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")

        experiment_label = experiment_label + [EXPERIMENT for _ in range(adata.X.shape[0])]
        exp_df = pd.DataFrame(adata[:, common_markers].X, columns=common_markers)

        adata = adata[:, common_markers]
        print(exp_df)
        all_df = pd.concat([all_df, exp_df], axis=0).reset_index(drop=True)

    print(all_df) # 2038297 cells
    out_dir = Path(f"analysis/{TYPE}/")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    all_df["slide"] = experiment_label
    all_df.to_csv(out_dir / "all_xp_exp.csv", index=False)

    # merging the slide expression data by scanpy in anndata format (codes from nb_xenium_batch_effects.py)
    in_dir = Path(f"data/{TYPE}")
    adata_list = [read_h5ad(in_dir / xp / f"{xp}_filter.h5ad")[:, common_markers] for xp in EXPERIMENT_LIST]

    # Normalizing adata by median total counts and log
    for i in range(len(adata_list)):
        sc.pp.normalize_total(adata_list[i])
        sc.pp.log1p(adata_list[i])

    # Concatenate data from all slides together
    full_adata = adata_list[0].concatenate(adata_list[1], adata_list[2], adata_list[3], adata_list[4],
                                           adata_list[5], adata_list[6], adata_list[7], adata_list[8], adata_list[9],
                                           adata_list[10], batch_key="batch")

    # Check for batch effects using PCA
    sc.tl.pca(full_adata, svd_solver='arpack')

    # Batch correcting
    ho = hm.run_harmony(full_adata.obsm['X_pca'], full_adata.obs, 'batch')
    full_adata.obsm['X_pca_harmony'] = ho.Z_corr.T

    # store the adata (original one and the one after batch correction)
    full_adata.write(in_dir / "xp_11_post.h5ad")

    # plotting the original UMAP
    CLUSTER_KEY = 'batch'
    adata = read_h5ad(out_dir / "full_11_xp.h5ad")
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    plot_umap(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)
    plot_umap_panel(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)
    adata.write(out_dir / "full_11_xp_pre.h5ad")

    # run the rsi function to identify cell types

    # plotting UMAP with the data after batch correction
    CLUSTER_KEY = 'batch'
    out_dir = Path(f"analysis/{TYPE}")
    adata = read_h5ad(out_dir / "full_11_xp_post.h5ad")
    # start running form 3:35 pm
    print("neighbors")
    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    print("sc.tl.umap()")
    sc.tl.umap(adata)
    print("start printing")
    plot_umap(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)
    plot_umap_panel(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)
    adata.write(out_dir / "xp_11_post.h5ad")


