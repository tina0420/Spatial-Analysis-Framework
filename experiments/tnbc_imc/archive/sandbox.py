"""
@author: Eric Lee
Example run file for preprocessing tnbc data
"""
import os
from pathlib import Path
import pandas as pd
import scanpy as sc
from src.preprocess import read_to_anndata_wrapper, normalize, read_neighbors
from src.extract import quantify_from_qupath
from src.annotate import run_leiden
from anndata import read_h5ad
from src.visualize import *

if __name__ == '__main__':
    # Which run
    run_name = "tnbc_imc"
    print(f"Processing {run_name}")

    # Set reading and writing paths
    path_dir = Path(f"data/{run_name}")
    out_dir = Path(f"figures/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    Step 1
    """
    # # Process measurements.csv from QuPath
    # exp_df, loc_df = process_qupath(path_dir)
    #
    # # Choose markers that you want
    # exp_df = pd.concat([exp_df.iloc[:, 0], exp_df.iloc[:, 4:-4]], axis=1)
    #
    # # Rename columns accordingly
    # exp_df.columns = ["Image"] + [i.split()[1] for i in exp_df.columns[1:]]
    #
    # # Output to input structure for Sakura
    # exp_df.to_csv(path_dir / "y.csv", index=False)
    # loc_df.to_csv(path_dir / "loc.csv", index=False)

    """
    Step 2
    """
    # # Read to y.csv and loc.csv to AnnData object
    # adata = read_to_anndata_wrapper(path_dir=path_dir)
    #
    # # Normalize expression data
    # normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    # normalize(adata, method="clip", clip_percentile=99.5)
    #
    # # Clustering using selected markers
    # subset_markers = ['CD20', 'CD31', 'CD3e', 'CD68', 'CK5',
    #                   'CK7', 'SMA', 'panCK']
    # neighbors_to_consider = 150
    # leiden_subset(adata, columns=subset_markers, n_neighbors=neighbors_to_consider, n_pcs=None)
    # CLUSTER_KEY = "leiden_subset"
    #
    # # Write to read later
    # adata.write(out_dir / f"{run_name}.h5ad")

    """
    Step 3
    """
    # Read saved file
    adata = read_h5ad(out_dir / f"{run_name}.h5ad")

    # Define which labels to use as basis
    CLUSTER_KEY = "leiden_subset"

    # Clustering using selected markers
    subset_markers = ['CD20',   #"B" 0
                      'CD31',   #"Endothelial" 1
                      'CD3e',   #"T" 2
                      'CD68',   #"Macrophage/Monocyte" 3
                      'CK5',    #"Basal" 4
                      'CK7',    #"Luminal" 5
                      'SMA',    #"Stromal" 6
                      'panCK']  #"General epithelial" 7
                                # Other 8

    # Plot z-scored cluster mean heatmaps
    plot_cluster_mean_heatmap(adata, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")

    plot_zscore_cluster_mean_heatmap(adata, subset_markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")

    from src.utils import get_coexpression_matrix
    all_names = adata.var.index
    m_dict = dict(zip(range(len(all_names)), all_names))
    r = get_coexpression_matrix(adata[np.where(adata.obs[CLUSTER_KEY] == 12)[0], :].X, range(len(all_names)))
    r = pd.DataFrame(r, columns=['m1', 'm2', 'r'])
    r.loc[:, "m1"] = [m_dict[i] for i in r.loc[:, "m1"]]
    r.loc[:, "m2"] = [m_dict[i] for i in r.loc[:, "m2"]]
    r = r.sort_values(by=["r"], ascending=False)

    """
    Step 4
    """
    # Plot by patient, mainly for spatial
    img_names = pd.read_csv(path_dir / "y.csv").iloc[:, 0]

    for name in pd.unique(img_names):
        # current id
        pid = name.split("_")[-2]

        # slice adata to patient level adata
        p_idxs = np.where(img_names == name)[0]
        p_adata = adata[p_idxs, :]

