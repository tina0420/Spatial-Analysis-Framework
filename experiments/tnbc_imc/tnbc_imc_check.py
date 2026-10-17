import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
from src.visualize import plot_umap, plot_umap_panel, plot_spatial_cluster_distribution, plot_cluster_cell_count, \
    plot_spatial_cluster_panel
from experiments.tnbc_imc.tnbc_imc_setup_analysis import TMA_COLOR_DICT


if __name__ == '__main__':
    # Example for real data
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]

    """
    Compare batch effect across tmas
    """
    temp_adata = FULL_ADATA.copy()
    CLUSTER_KEY = 'tma_x'

    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.umap(temp_adata)

    plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=TMA_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name="tnbc_tma_batch")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=TMA_COLOR_DICT,
                    dotsize=0.1, name="tnbc_tma_batch")

    temp_adata.write(out_dir / "tma_batch_umap_record.h5ad")

    # plot esi and layer2 cell types umap
    CLUSTER_KEY = "relapse"
    RELAPSE_TF_COLOR_DICT = {'No recurrence': 'tab:blue',
                             'Recurrence': 'tab:pink'}
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=RELAPSE_TF_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name="tnbc_relapse")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=RELAPSE_TF_COLOR_DICT,
                    dotsize=0.1, name="tnbc_relapse")

