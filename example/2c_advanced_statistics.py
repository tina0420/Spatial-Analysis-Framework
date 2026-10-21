"""
@author: Tina Hsu
Tutorial 2c: Modeling and sensitivity analysis.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import read_neighbors
from src.enrichment import *


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "create"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    in_dir = Path(f"example/output/{run_name}")
    out_dir = Path(f"example/output/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(in_dir / "create.h5ad")

    """
    We want to define the column that defines the clusters in obs.
    """
    CLUSTER_KEY = "label"

    """
    For each image, we can subset each into an AnnData.

    In this example let us not use a for loop and just pick one ROI out, say, roi_0.
    """
    p_adata = adata[np.where(adata.obs["image_id"] == "roi_0")[0], :]

    """
    We can define neighboring cells per image by thresholding. The neighbors are saved in uns.
    """
    read_neighbors(adata, rel_path=out_dir / "rel.csv", threshold=0.5)

    """
    We can calculate adjacency enrichment scores for each cluster.
    """
    score_df = get_adjacency_enrichment_matrix(adata, cluster_key="label")

    """
    We can also plot a binomial regression model to model rare cell frequencies.
    """
    rare_cell_arr = np.array([2, 5, 8])
    r, p = get_neg_binomial_regression_params(rare_cell_arr)
    plot_neg_binomial_regression(r, p, out_dir)
