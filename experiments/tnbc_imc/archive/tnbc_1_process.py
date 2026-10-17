"""
@author: Tina Hsu
Processing TNBC IMC dataset.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from tnbc_data_info import EXPERIMENT_LIST, EXPERIMENT_MARKERS
from anndata import read_h5ad
from functools import reduce
import scanpy as sc
import harmonypy as hm

if __name__ == '__main__':
    """
    Filtering out the markers failed or unclear in each individual experiment by marker review.
    """
    TYPE = "tnbc_imc"
    for EXPERIMENT in EXPERIMENT_LIST:
        print(EXPERIMENT)

        in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
        out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        adata = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")
        passed_markers = EXPERIMENT_MARKERS[EXPERIMENT]
        adata = adata[:, passed_markers]
        adata.write(in_dir / f"{EXPERIMENT}_filter.h5ad")

    """
    Merging all experiments to one AnnData.
    """
    # extract the common passed markers across all experiments
    common_markers = list(reduce(set.intersection, map(set, EXPERIMENT_MARKERS.values())))
    # change the order we prefer manually
    common_markers = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'CD3e', 'CD20', 'CD45',
                      'CD68', 'CD31', 'SMA', 'Vimentin']

    # merging the slides
    in_dir = Path(f"data/{TYPE}")
    adata_list = [read_h5ad(in_dir / xp / f"{xp}_filter.h5ad")[:, common_markers] for xp in EXPERIMENT_LIST]

    # Normalizing adata by median total counts and log
    for i in range(len(adata_list)):
        sc.pp.normalize_total(adata_list[i])
        sc.pp.log1p(adata_list[i])

    # Concatenate data from all slides together
    full_adata = adata_list[0].concatenate(adata_list[1], adata_list[2], adata_list[3], adata_list[4],
                                           adata_list[5], adata_list[6], adata_list[7], adata_list[8], adata_list[9],
                                           adata_list[10], batch_key="batch")

    # store the full AnnData under directory for analysis
    CLUSTER_KEY = 'full_11_xp'
    out_dir = Path(f"analysis/{TYPE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    full_adata.write(out_dir / f"{CLUSTER_KEY}.h5ad")

    # running PCA to see the batches across all experiments
    sc.tl.pca(full_adata, svd_solver='arpack')
    full_adata.write(out_dir / f"{CLUSTER_KEY}_pre.h5ad")

    # running harmony to do batch correction
    ho = hm.run_harmony(full_adata.obsm['X_pca'], full_adata.obs, 'batch')
    full_adata.obsm['X_pca_harmony'] = ho.Z_corr.T
    full_adata.write(out_dir / f"{CLUSTER_KEY}_post.h5ad")



