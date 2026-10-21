"""
@author: Tina Hsu
Tutorial 1c: Annotating cells.
"""
from pathlib import Path
from anndata import read_h5ad
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from src.annotate import update_labels


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "example"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    out_dir = Path(f"example/output/{run_name}")

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(out_dir / f"{run_name}.h5ad")

    """
    We want to define the column that defines the clusters in obs.
    """
    CLUSTER_KEY = "cluster"

    """
    Next, we can annotate cells into cell types or cell states based on expression.
    
    For this dataset, we only expect four types of cells.
    Macrophages for CD68, B cells for CD20, T cells for CD3, and other cells for not expressing.
    
    Annotating cell types is usually done manually and requires expertise.
    For ease in this tutorial, we can classify them into four groups by 
    using a simple K-means and update the labels in obs.
    """
    for col in range(adata.shape[1]):
        adata.X[:, col] = zscore(adata.X[:, col])

    cluster_mean = np.array([np.mean(adata.X[np.where(adata.obs[CLUSTER_KEY] == i)[0]], axis=0) for i in
                             adata.obs[CLUSTER_KEY].cat.categories])[:, :3]

    cluster_labels = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(cluster_mean).labels_

    cluster_dict = dict(zip(range(len(cluster_labels)), cluster_labels))

    update_labels(adata, cluster_key="celltype", labels=[cluster_dict[i] for i in adata.obs[CLUSTER_KEY]])

    """
    We can now save the results by overwriting the h5ad file for visualizing.
    """
    adata.write(out_dir / f"{run_name}.h5ad")
