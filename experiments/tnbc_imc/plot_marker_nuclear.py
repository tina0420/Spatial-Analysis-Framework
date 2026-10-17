"""
@author: Tina Hsu
Tidy up all the data
"""
import os
from pathlib import Path
from src.extract import __extract_panel_from_mcd
from src.segment import run_mesmer
from src.preprocess import quantify_expression, normalize
from anndata import read_h5ad
from src.annotate import run_leiden
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import *
from src.segment import *


def plot_marker_nuclear(x, out_dir, name):
    """
    Plot the nuclear-and-one-marker-only image
    :param x: Pixel intensity of the image in 4-dimension, type: np.array
    :param out_dir: Directory to store the figure, type: str
    :param name: Name for the image, type: str
    :return:
    """
    rgb_images = create_rgb_image(x, channel_colors=['green', 'red'])

    plt.figure(figsize=(7, 7))

    plt.imshow(rgb_images[0])

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(out_dir / f"{name}.png", dpi=500)
    plt.clf()


if __name__ == '__main__':
    # path and file
    TYPE = "tnbc_imc"
    EXPERIMENTS = ["XP12446", "XP13085",
                   "XP12813", "XP13083",
                   "XP13087", "XP13084",
                   "XP12814", "XP13086",
                   "XP12274"]
    so_far_xp = ["XP13085"]  # mcd to tiff fails, error reading acquisition
    # so_far_xp = ["XP13087"]

    # already have ome-tiff files, and need segmentation
    for xp in EXPERIMENTS:
        if xp != "XP13085":
            print("===============================================================================")
            print("===============================================================================")
            print(f"Processing {xp}")

            # there's an error in the XP12814, ROI_46 (need to debug...)
            # but run the rest of experiments first (XP13086 and XP12274)

            in_dir = Path(f"data/{TYPE}/{xp}")
            out_dir = Path(f"output/{TYPE}/{xp}")
            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)


        """
        identify cell types in epithelial, immune, stromal cells
        -> to check the cell type distribution in each TMA
        """
        run_leiden(adata, columns=adata.var_names, n_neighbors=150, n_pcs=None, name="leiden_cluster")
        print()

        # adata.write(out_dir / f"{xp}.h5ad")

        adata = read_h5ad(out_dir / f"{xp}.h5ad")
        plot_zscore_cluster_mean_heatmap(adata, adata.var_names, "leiden_cluster", out_dir,
                                         figsize=(7, 7), name="leiden_cluster_all")
        subset_markers = ["CK5", "CK7", "panCK",
                          "CD3e", "CD20", "CD68", "CD45",
                          "SMA"]
        plot_cluster_mean_heatmap(adata, "leiden_cluster", out_dir=out_dir, figsize=(7, 7), name="all")

        plot_zscore_cluster_mean_heatmap(adata, subset_markers, "leiden_cluster", out_dir=out_dir, figsize=(7, 7),
                                         name="specific_markers")

        # # merge the clusters and identify them as epithelial, immune and stromal cell types
        # new_label = ["undefined", "epithelial", "epithelial", "immune", "stromal",  # 0-4
        #              "immune", "undefined", "stromal", "stromal", "epithelial",  # 5-9
        #              "epithelial", "epithelial", "immune", "undefined"]  #10-13
        #
        # clusters_mapped_dict = dict(zip(range(len(adata.obs["leiden_cluster"])), new_label))
        # update_labels(adata=adata, cluster_key="epi_clusters",
        #               labels=[clusters_mapped_dict[x] for x in adata.obs["leiden_cluster"]])

        # plot again to check if I merged the clusters correctly
        plot_zscore_cluster_mean_heatmap(adata, subset_markers, "epi_clusters",
                                         out_dir=out_dir, figsize=(7, 7), name="cell_types")

        """
        new way: using GMM twice to roughly separate the cells into epithelial, immune and stromal cells
        """


    # panel file (extract subcode from extract.py, extract_mcd_for_imc())
    file_list = list(in_dir.rglob("*.mcd"))
    for mcd_path in file_list:
        file_name = os.path.basename(mcd_path)
        print(f"Processing {file_name}")

        __extract_panel_from_mcd(mcd_path, in_dir)

    # segmentation
    DEEPCELL_TOKEN = "gLJAn8ZY.OmHKrKnth0mqobIljEi6o16StVeqhFVn"

    run_mesmer(nuclear_channel=[3], cytoplasm_channels=[3, 29],
               in_dir=in_dir, out_dir=out_dir,
               deepcell_token=DEEPCELL_TOKEN,
               mes_image_mpp=1, mes_interior_threshold=0.2, mes_maxima_threshold=0.075,
               plot_mask=True, plot_overlay=True, output_pred_df=False)

    # into AnnData
    adata = quantify_expression(out_dir, is_sparse=False)

    adata.write(out_dir / f"{xp}.h5ad")

    # generate and output the (marker & nuclear) images by cv2
    # get the roi from one experiment
    # testing with XP13084

    TYPE = "tnbc_imc"
    xp = "XP13084"

    in_dir = Path(f"data/{TYPE}/{xp}")
    out_dir = Path(f"marker/{TYPE}/{xp}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{xp}.h5ad")

    # marker info in panel.csv
    panel_df = pd.read_csv(in_dir / "panel.csv")
    panel = {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}

    for roi, image_path in enumerate(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")), start=1):
        file_name = os.path.basename(image_path)
        print(file_name.split("_")[0], file_name)

        # Read image
        img = skio.imread(image_path)

        for marker, channel_idx in panel.items():
            if channel_idx >= 3 and channel_idx <=29:
                print(marker, channel_idx)

                nuclei = img[31, :, :]
                cytoplasm = img[channel_idx, :, :]
                x = stack_for_mesmer(nuclei, cytoplasm)

                plot_marker_nuclear(x, out_dir, name=file_name.split("_")[0] + "_" + marker)


    # for checking the metadata
    TYPE = "tnbc_imc"

    in_dir = Path(f"data/{TYPE}")
    out_dir = Path(f"marker/{TYPE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)


    cohort_data = pd.read_csv(in_dir / "ctDNA_cohort.csv")
    metadata = pd.read_csv(in_dir / "for_IMC.csv")
    # relapse_all = metadata[np.where(type(metadata["First site of relapse if known"]) == str)[0], :]

    relapse_cid = []
    for i in range(metadata.shape[0]):
        if type(metadata["First site of relapse if known"][i]) == str:
            relapse_cid.append(metadata["CID"][i])

    cohort_name_all = set(cohort_data["Name"])
    count = 0
    for i in range(len(relapse_cid)):
        if relapse_cid[i] in cohort_name_all:
            count += 1







