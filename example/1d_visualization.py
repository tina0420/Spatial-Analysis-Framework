"""
@author: Tina Hsu
Tutorial 1d: Visualizing clusters per image.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import read_neighbors
from src.visualize import *


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "example"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    in_dir = Path(f"example/output/{run_name}")
    out_dir = Path(f"example/output/{run_name}/visualize")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(in_dir / f"{run_name}.h5ad")

    """
    We want to define the column that defines the cluster index.
    """
    CLUSTER_KEY = "celltype"

    """
    As we have three images, we will iterate over all images to visualize.
    """
    for p in adata.obs["image_id"].cat.categories:
        """
        For each image, we can subset each into a AnnData
        """
        p_adata = adata[np.where(adata.obs["image_id"] == p)[0], :]

        """
        We can define neighboring cells per image by thresholding. The neighbors are saved in uns.
        """
        read_neighbors(adata, rel_path=out_dir / "rel.csv", threshold=10)

        """
        We can plot a series of plots to visualize our data.
        """
        # Plot a cell area distribution
        plot_cell_area(adata, out_dir, figsize=(3.5, 2), name=f"{p}")

        # Plot a cell mean intensity violin plot
        # You can also use plot_cell_expression_violin_split if there are too many markers to fit in one figure
        plot_cell_expression_violin(adata, out_dir, figsize=(7, 3), name=f"{p}")

        # Plot expression matrix for all cells
        plot_cell_expression_heatmap(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot a z-scored expression matrix for all cells
        plot_zscore_cell_expression_heatmap(adata, ['Cell: CD68 mean', 'Cell: CD20 mean', 'Cell: CD3 mean'],
                                            CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot a panel of normalized expression spatially per marker
        plot_spatial_expression_panel(adata, out_dir, name=f"{p}")

        # Plot spatial correlation with Spearman coefficient annotated
        plot_spatial_correlation(adata, out_dir, threshold=0, figsize=(15, 12), plot_coeff=False, fontsize=15,
                                 name=f"{p}")

        # Plot umap embedding
        plot_umap(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), dotsize=1, name=f"{p}")

        # Plot umap panel
        plot_umap_panel(adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{p}")

        # Plot cluster mean heatmap
        plot_cluster_mean_heatmap(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot z-scored cluster mean heatmap
        plot_zscore_cluster_mean_heatmap(adata, ['Cell: CD68 mean', 'Cell: CD20 mean', 'Cell: CD3 mean'],
                                         CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot cluster heatmaps
        plot_cluster_heatmap(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot z-scored cluster heatmaps
        plot_zscore_cluster_heatmap(adata, ['Cell: CD68 mean', 'Cell: CD20 mean', 'Cell: CD3 mean'],
                                    CLUSTER_KEY, out_dir, figsize=(7, 7), name=f"{p}")

        # Plot cell count by cluster
        plot_cluster_cell_count(adata, CLUSTER_KEY, out_dir, figsize=(3.5, 2), name=f"{p}")

        # Plot cluster map
        plot_spatial_cluster_distribution(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), dotsize=1, name=f"{p}")

        # Plot cluster panel
        plot_spatial_cluster_panel(adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{p}")

        # Plot edge count heatmap
        plot_edge_count(adata, CLUSTER_KEY, out_dir, figsize=(7, 7), rel_zero_indexing=True, annot=False, annot_size=8,
                        name=f"{p}")

        # Plot the neighbor graph
        plot_neighbor_graph(adata, out_dir, name=f"{p}")
