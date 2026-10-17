"""
@author: Tina Hsu
Merging the layer 2 cell type annotations together
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import scanpy as sc
from src.annotate import update_labels
from src.visualize import plot_umap, plot_umap_panel


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific_5_tma'
    XP = '5_tma_new_7'

    in_dir = out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad
    adata = read_h5ad(in_dir / f"{XP}_finalize.h5ad")
    # adata_layer1 = read_h5ad(in_dir / f"{XP}_layer1_finalize.h5ad")
    adata_epi = read_h5ad(in_dir / f"{XP}_epi.h5ad")
    adata_tme = read_h5ad(in_dir / f"{XP}_tme.h5ad")

    # Create a processing h5ad
    adata_layer2 = adata.copy()

    # Update layer 2 cell types to the whole adata
    CLUSTER_KEY = 'layer1'
    NEW_CLUSTER_KEY = 'layer2'
    new_labels = np.zeros(adata_layer2.shape[0], dtype='object')
    new_labels[np.where(adata_layer2.obs['layer1'] == 'Epithelial')[0]] = adata_epi.obs[NEW_CLUSTER_KEY].to_numpy()
    new_labels[np.where(adata_layer2.obs['layer1'] != 'Epithelial')[0]] = adata_tme.obs[NEW_CLUSTER_KEY].to_numpy()

    # Update the new layer 1 labels to the adata
    update_labels(adata_layer2, cluster_key=NEW_CLUSTER_KEY, labels=new_labels)

    adata_layer2.write(out_dir / f"{XP}_layer2_finalize.h5ad")
    adata_layer2.write(out_dir / f"{XP}_finalize.h5ad")

    # plot cell distribution on layer 2
    from src.visualize import plot_spatial_cluster_distribution, plot_cluster_cell_count, plot_spatial_cluster_panel
    from experiments.tnbc_imc.tnbc_imc_setup_analysis import LAYER2_CELL_TYPE_COLOR_DICT

    CLUSTER_KEY = 'layer2'
    UNIT_KEY = "unique_core_id"
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/layer2")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # for img in np.unique(adata_layer2.obs[UNIT_KEY]):
    for img in np.unique(new_order_adata.obs[UNIT_KEY]):
        print(img)
        # p_adata = adata_layer2[np.where(adata_layer2.obs[UNIT_KEY] == img)[0], :]
        p_adata = new_order_adata[np.where(new_order_adata.obs[UNIT_KEY] == img)[0], :]
        pid = p_adata.obs.cid.values[0].split("_")[0]
        plot_spatial_cluster_distribution(adata=p_adata,
                                          cluster_key=CLUSTER_KEY,
                                          out_dir=out_dir,
                                          figsize=(7, 7),
                                          dotsize=3,
                                          color=LAYER2_CELL_TYPE_COLOR_DICT,
                                          name=f"{pid}_{img}_layer2"
                                          )

        plot_cluster_cell_count(adata=p_adata,
                                cluster_key=CLUSTER_KEY,
                                out_dir=out_dir,
                                figsize=(5.5, 4), name=f"{pid}_{img}_layer2")

        # Plot cluster panel
        plot_spatial_cluster_panel(adata=p_adata,
                                   cluster_key=CLUSTER_KEY,
                                   out_dir=out_dir,
                                   figsize=(12.7, 5.5),
                                   dotsize=0.3,
                                   color=LAYER2_CELL_TYPE_COLOR_DICT,
                                   name=f"{pid}_{img}_layer2")




