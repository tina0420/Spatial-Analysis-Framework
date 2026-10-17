"""
Visualization functions for TNBC IMC.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.visualize import plot_spatial_cluster_distribution, plot_cluster_cell_count, plot_spatial_cluster_panel
from experiments.tnbc_imc.tnbc_imc_setup_analysis import *

if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/summary")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    adata = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]

    """
    Cell type spatial distributions
    """
    UNIT_KEY = 'unique_core_id'
    for pid in set(adata.obs[UNIT_KEY]):
        print(pid)

        # layer1 (epithelial and TME) cell type distributions
        CLUSTER_KEY = 'layer1'
        plot_spatial_cluster_distribution(adata=adata,
                                          cluster_key=CLUSTER_KEY,
                                          out_dir=out_dir / "spatial_distribution",
                                          figsize=(5, 5),
                                          dotsize=2,
                                          color=E_TME_COLOR_DICT,
                                          name=f"{pid}_esi")
        # ESI cell type distributions
        CLUSTER_KEY = 'ESI'
        plot_spatial_cluster_distribution(adata=adata,
                                          cluster_key=CLUSTER_KEY,
                                          out_dir=out_dir / "spatial_distribution",
                                          figsize=(5, 5),
                                          dotsize=2,
                                          color=ESI_CELL_TYPE_COLOR_DICT,
                                          name=f"{pid}_layer1")
        # layer2 cell type distributions
        CLUSTER_KEY = 'layer2'
        plot_spatial_cluster_distribution(adata=adata,
                                          cluster_key=CLUSTER_KEY,
                                          out_dir=out_dir / "spatial_distribution",
                                          figsize=(5, 5),
                                          dotsize=2,
                                          color=LAYER2_CELL_TYPE_COLOR_DICT,
                                          name=f"{pid}_layer2")

        # layer2 cell type distribution panel, the original function includes cell count of each category
        # can use the modified plotting function without cell count numbers but magnify the texts
        # (function in tnbc_imc_h3k27me3.py)
        plot_spatial_cluster_panel(adata=adata,
                                   cluster_key=CLUSTER_KEY,
                                   out_dir=out_dir / "spatial_distribution",
                                   figsize=(15, 7),
                                   dotsize=0.8,
                                   color=LAYER2_CELL_TYPE_COLOR_DICT,
                                   name=pid)

        # cell type counts in each image
        CLUSTER_KEY = 'layer2'
        plot_cluster_cell_count(adata=adata,
                                cluster_key=CLUSTER_KEY,
                                out_dir=out_dir / "spatial_distribution",
                                figsize=(3.5, 2),
                                name=pid)


