"""
@author: Tina Hsu
Checking batch effect of the BM IMC dataset.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import read_neighbors
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import harmonypy as hm
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import plot_umap, plot_umap_panel


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
    # parameter setting
    # TYPE = "tnbc_imc"
    # experiment = "XP12813"

    TYPE = "tnbc_imc"
    EXPERIMENT = "XP13085"

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # adata13 = adata
    adata85 = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")
    # calculate the lisi score in raw data
    lisi_scores = compute_lisi(adata85.X[:, :], adata85.obs['image_id'].values) # use top 10: 3.05 & use all: 2.39
    adata85.obs["lisi"] = lisi_scores
    # adata = adata85

    adata = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")

    # running PCA and plot
    sc.tl.pca(adata, svd_solver='arpack')

    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(adata.X)
    # pca_result = pd.DataFrame(pca_result, columns=["pc1", "pc2"])
    # pca_result["image_id"] = list(adata.obs.image_id)
    # # change image name
    # pca_result["image_id"] = [x.split("_")[-1].split(".")[0] for x in adata.obs.image_id]
    # # plot
    # plt.figure(figsize=(7, 7))
    # sns.scatterplot(data=pca_result, x="pc1", y="pc2", hue="image_id", legend=False)
    # plt.title("PCA here", fontsize=10)
    # plt.tight_layout()
    # plt.savefig(out_dir / "pca_here.png", dpi=300)

    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    adata.write(in_dir / f"{EXPERIMENT}_pre.h5ad")

    # rename the name of ROI
    adata.obs.image_id = adata.obs.image_id.cat.rename_categories(lambda x: x.split("_")[-1].split(".")[0])

    # plot
    plot_umap(adata, cluster_key="image_id", out_dir=out_dir)
    plot_umap_panel(adata, cluster_key="image_id", out_dir=out_dir)

    # batch correction by harmony
    ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'image_id')  # 2:44 pm starts and 2:48 ends
    adata.obsm['X_pca_batch'] = ho.Z_corr.T

    # calcualte lisi scores
    lisi_scores = compute_lisi(adata.obsm['X_pca'][:, :10], adata.obs['image_id'].values)
    adata.obs["lisi"] = lisi_scores

    sc.pp.neighbors(adata, use_rep='X_pca_batch')
    sc.tl.umap(adata)
    adata.write(in_dir / f"{EXPERIMENT}_post.h5ad")

    # plot
    plot_umap(adata, cluster_key="image_id", out_dir=out_dir)
    plot_umap_panel(adata, cluster_key="image_id", out_dir=out_dir)

    # 4. Visualize batch effect
    sc.pl.umap(adata, color='batch')

    # check batch correction results
    lisi_scores = compute_lisi(adata.obsm['X_pca_batch'][:, :10], adata.obs['image_id'].values)
    adata.obs["lisi"] = lisi_scores







