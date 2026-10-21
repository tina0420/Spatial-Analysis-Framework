"""
@author: Tina Hsu
Tutorial 2b: Performing spatial statistical analysis.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import read_neighbors
from src.spatial_stats import *


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
    Given an AnnData where clustering is performed, these are the following labels:
        - 0: tumor cell
        - 1: TME cell type 1
        - 2: TME cell type 2

    We can then find spatial clusters (not expression based, location based)
    to answer questions in each FOV, such as:

    1. Calculate the TME cell type distribution around each tumor cluster
    2. Calculate each TME cell type's average distance to each tumor cluster
    3. How different are the expressions between tumor clusters that have
       different distances from the TME
    4. Calculate a specific TME　cell proportion, and its average distance to tumor cells
    5. Validate spatial co-localization of two clusters
    """
    # Get spatial connected components for each cell type
    cc_0 = get_labeled_connected_components(adata, CLUSTER_KEY, label=0)
    cc_1 = get_labeled_connected_components(adata, CLUSTER_KEY, label=1)
    cc_2 = get_labeled_connected_components(adata, CLUSTER_KEY, label=2)

    # Get mean expression for components of interest
    exp_cc_2_0 = get_mean_expression(adata, cell_idxs=cc_2[0])
    exp_cc_2_1 = get_mean_expression(adata, cell_idxs=cc_2[1])

    # Get centroid cell indices for components of interest
    c_idx_cc_2_0 = get_centroid_cell_idx(adata, cell_idxs=cc_2[0])
    c_idx_cc_2_1 = get_centroid_cell_idx(adata, cell_idxs=cc_2[1])

    # Get all cell indices within radius around centroids
    range_idxs_cc_2_0 = get_peri_centroid_cell_idxs(adata, centroid_idx=c_idx_cc_2_0, radius=1)
    range_idxs_cc_2_1 = get_peri_centroid_cell_idxs(adata, centroid_idx=c_idx_cc_2_1, radius=1)

    # Get the distribution of cell types around centroids
    cell_distr_cc_2_0 = get_cell_distribution(adata, cell_idxs=range_idxs_cc_2_0, cluster_key=CLUSTER_KEY)
    cell_distr_cc_2_1 = get_cell_distribution(adata, cell_idxs=range_idxs_cc_2_1, cluster_key=CLUSTER_KEY)

    # Get mean distance of cell types to centroids
    d_cc_2_0 = get_mean_distance_from_centroid(adata, centroid_idx=c_idx_cc_2_0, cell_idxs=range_idxs_cc_2_0, cluster_key=CLUSTER_KEY)
    d_cc_2_1 = get_mean_distance_from_centroid(adata, centroid_idx=c_idx_cc_2_1, cell_idxs=range_idxs_cc_2_1, cluster_key=CLUSTER_KEY)

    # Get mean distance across all cells
    d_mean = get_global_mean_distance(adata)

    # Get interacting pairs of two selected cell types
    pairs = get_interacting_pairs(adata, cluster_key=CLUSTER_KEY, type_i=1, type_j=2)

    # Get interacting pairs of two selected cell types expressing specific markers
    filtered_pairs = get_filtered_interacting_pairs(adata, cluster_key=CLUSTER_KEY,
                                                    type_i=1, type_j=2,
                                                    col_idx_i=0, col_idx_j=1,
                                                    thres_i=-0.5, thres_j=-0.5)
