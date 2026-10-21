"""
@author: Tina Hsu
Tutorial 2a: Creating a spatial dataset and performing clustering.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from anndata import AnnData
from src.create import simulate_spatial_ROI
from src.visualize import plot_spatial_cluster_distribution, plot_spatial_cluster_panel


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "create"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    out_dir = Path(f"example/output/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    We can simulate a region of interest with the following function.
    
    To generate a simulation, we can specify the amount of cells we want, the amount of genes as well as the amount 
    of clusters. We can also define interactiveness between cell types by specifying bias or uni, meaning cells likely
    to interact with like-types versus interact with different cell types. 

    Here, let us simulate a tissue with one large cancer cell type and four different TME cell types.
    """
    sdict_malignant = simulate_spatial_ROI(r_rois=3,
                                           i_cells=350,
                                           j_genes=250,
                                           k_clusters=1,
                                           pixel_size=0.5,
                                           mixture_type="bias",
                                           init_pos=(0, 0),
                                           seed=420)

    # Simulate 2 micro-environment clusters
    sdict_tme = simulate_spatial_ROI(r_rois=3,
                                     i_cells=150,
                                     j_genes=250,
                                     k_clusters=4,
                                     pixel_size=0.5,
                                     mixture_type="uni",
                                     init_pos=(1, 1),
                                     seed=420)

    # Adjust the labels to be one up to avoid merging wrong clusters
    sdict_tme["label"] = sdict_tme["label"] + 1
    
    """
    We can use the two dictionaries to create a common AnnData object.
    """
    adata = AnnData(np.concatenate([sdict_malignant["X"], sdict_tme["X"]]), var=pd.DataFrame([], index=range(250)))
    adata.obsm["spatial"] = np.concatenate([sdict_malignant["spatial"], sdict_tme["spatial"]])
    adata.obs["image_id"] = np.concatenate([sdict_malignant["image_id"], sdict_tme["image_id"]])
    adata.obs["image_id"] = adata.obs["image_id"].astype('category')
    adata.obs["label"] = np.concatenate([sdict_malignant["label"], sdict_tme["label"]])
    adata.obs["label"] = adata.obs["label"].astype('category')
    adata.obs["area"] = np.concatenate([sdict_malignant["area"], sdict_tme["area"]])

    """
    In order to see what the plots would look like, we can plot the graphs.
    """
    for p in adata.obs["image_id"].cat.categories:
        """
        For each image, we can subset each into an AnnData.
        """
        p_adata = adata[np.where(adata.obs["image_id"] == p)[0], :]

        """
        We can visualize the clusters.
        """
        # Plot cluster map
        plot_spatial_cluster_distribution(p_adata, "label", out_dir, figsize=(7, 7), dotsize=1, name=f"{p}")

        # Plot cluster panel
        plot_spatial_cluster_panel(p_adata, "label", out_dir, dotsize=0.1, name=f"{p}")

    """
    We can now save the AnnData to a h5ad file for clustering.
    """
    adata.write(out_dir / "create.h5ad")
