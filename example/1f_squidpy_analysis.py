"""
@author: Tina Hsu
Tutorial 1f: Squidpy statistical analysis.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.visualize import plot_cluster_cooccurrence, plot_cluster_interaction_matrix, plot_cluster_ripley_l
from src.squidpy_stats import *


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
    out_dir = Path(f"example/output/{run_name}/squidpy")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(in_dir / f"{run_name}.h5ad")

    """
    We want to define the column that defines the annotated cell types.
    """
    CLUSTER_KEY = "celltype"

    """
    We can run spatial statistics function prewritten by the squidpy package.
    This includes:
    [1] centrality scores, [2] co-occurence probabilities, [3] neighbourhood enrichment
    [4] Ripley's K [5] Moran's I
    """
    run_squidpy_statistics(adata, cluster_key=CLUSTER_KEY)

    """
    We can now save the AnnData to a h5ad file for visualizing later.
    """
    adata.write(out_dir / f"{run_name}_squidpy.h5ad")

    """
    To simulate that, let us read the saved file.
    """
    adata = read_h5ad(out_dir / f"{run_name}_squidpy.h5ad")

    """
    We can visualize and save our results in various ways.
    """
    # Plot cluster co-occurrence
    CELL_OF_INTEREST_KEY = 0
    plot_cluster_cooccurrence(adata, cluster_key=CLUSTER_KEY, label=CELL_OF_INTEREST_KEY, out_dir=out_dir)

    # Plot cluster interaction matrix from permutation test
    plot_cluster_interaction_matrix(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7))

    # Plot Ripley'L
    plot_cluster_ripley_l(adata, cluster_key=CLUSTER_KEY, out_dir=out_dir)

    # Get interaction entropy
    entropies = get_interaction_entropy(adata, CLUSTER_KEY)
    entropies.to_csv(out_dir / "entropies.csv", index=False)

    # Get spatially variable genes
    svg = get_spatially_variable_genes(adata, cluster_key=CLUSTER_KEY)
    svg.to_csv(out_dir / "svg.csv", index=False)
