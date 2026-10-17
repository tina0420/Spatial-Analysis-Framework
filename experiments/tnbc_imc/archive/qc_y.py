"""
@author: Tina Hsu
Testing the spatial and expression part for quality Control pipeline after the meeting with Y.
"""
import os
from pathlib import Path
from src.preprocess import read_to_anndata_wrapper, normalize
from src.extract import quantify_from_qupath
from src.annotate import run_leiden
from anndata import read_h5ad
from src.visualize import *
from experiments.tnbc_imc.archive.preprocessing import get_index_of_each_image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Which run
    run_name = "XP12446"
    print(f"Processing {run_name}")

    # Set reading and writing paths
    path_dir = Path(f"data/tnbc_imc/{run_name}")
    out_dir = Path(f"figures/tnbc_imc/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read and tidy the measurements.csv
    exp_df, loc_df = quantify_from_qupath(path_dir)

    columns = ['Image', 'Cell: CK5 mean',
               'Cell: CK19 mean', 'Cell: CK8-18 mean',
               'Cell: CD68 mean', 'Cell: vWF mean',
               'Cell: SMA mean', 'Cell: Vimentin mean',
               'Cell: erbB2 mean', 'Cell: CD3e mean',
               'Cell: Slug mean', 'Cell: EZH2 mean',
               'Cell: Prostaglandin mean', 'Cell: p53 mean',
               'Cell: H3K27me3 mean', 'Cell: PR-SP2 mean',
               'Cell: CD45 mean', 'Cell: GATA3 mean',
               'Cell: CD20 mean', 'Cell: B-Catenin mean',
               'Cell: CA-IX mean', 'Cell: ER-D8H8 mean',
               'Cell: Ki67 mean', 'Cell: EGFR mean',
               'Cell: ER-SP1 mean', 'Cell: CD31 mean',
               'Cell: CK7 mean', 'Cell: panCK mean',
               ]

    exp_df = exp_df[columns]
    exp_df.columns = [columns[0]] + [x.split(" ")[1] for x in columns[1:]]

    exp_df.to_csv(path_dir / "y.csv", index=False)
    loc_df.to_csv(path_dir / "loc.csv", index=False)

    adata = read_to_anndata_wrapper(path_dir=path_dir)

    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    image_index_dict = get_index_of_each_image(exp_df)

    subset_markers = ['CD20', 'CD31', 'CD3e', 'CD68', 'CK5', 'CK7', 'SMA', 'panCK']
    neighbors_to_consider = 150
    run_leiden(adata, columns=subset_markers, n_neighbors=neighbors_to_consider, n_pcs=None)
    CLUSTER_KEY = "leiden_subset"

    # Write to read later
    adata.write(out_dir / f"{run_name}.h5ad")

    #################### Start from here! ####################
    # Read the h5ad file
    adata = read_h5ad(out_dir / f"{run_name}.h5ad")

    # The first image
    image_1 = '2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff'

    idx = image_index_dict[image_1]  # 1782 cells

    image_1 = image_1.split("_")[-2]

    # MARKER = "CK5"
    # marker_loc = pd.concat([pd.DataFrame(adata.obsm["spatial"][idx, :], columns=["X", "Y"]),
    #                         pd.DataFrame(adata.X[idx, 0], columns=[MARKER])], axis=1)
    #
    # plt.figure(figsize=(8, 6))
    # marker_loc_high = marker_loc.iloc[np.where(marker_loc[MARKER] >= np.mean(marker_loc[MARKER])), :]
    # marker_loc_low = marker_loc.loc[np.where(marker_loc[MARKER] < np.mean(marker_loc[MARKER])), :]
    #
    # plt.scatter(x=marker_loc_high["X"], y=marker_loc_high["Y"],
    #             s=marker_loc_high[MARKER], cmap="r")
    # plt.scatter(x=marker_loc_low["X"], y=marker_loc_low["Y"],
    #             s=marker_loc_low[MARKER], cmap="b")
    #
    # plt.title(MARKER, fontsize=30)
    # plt.xlabel("X", fontsize=25)
    # plt.ylabel("Y", fontsize=25)
    # plt.tight_layout()
    # plt.colorbar()

    for i in range(len(adata.var_names)):
        marker = adata.var_names[i]
        marker_loc = pd.concat([pd.DataFrame(adata.obsm["spatial"][idx, :], columns=["X", "Y"]),
                                pd.DataFrame(adata.X[idx, i], columns=[marker])], axis=1)

        plt.figure(figsize=(8, 6))
        plt.scatter(x=marker_loc["X"], y=marker_loc["Y"], s=marker_loc[marker])
        plt.title(marker, fontsize=30)
        plt.xlabel("X", fontsize=25)
        plt.ylabel("Y", fontsize=25)
        plt.tight_layout()
        plt.colorbar()

        file_name = image_1 + "_" + marker + ".png"
        plt.savefig(out_dir / file_name, dpi=500)
        plt.clf()


