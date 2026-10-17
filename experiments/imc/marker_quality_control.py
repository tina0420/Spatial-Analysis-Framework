"""
@author: Tina Hsu
3 ways of marker quality control for spatial datasets, which are marker_based, expression_based and marker-based QC
Here we start from h5ad file after normalization and column rename, which is the output from tnbc_imc_measurement_quantify.py
"""
from utils import get_coexpression_matrix
# import pandas as pd
# import numpy as np
from itertools import combinations
from src.preprocess import read_neighbors
from src.spatial_stats import get_hot_spots_graph
from visualization import plot_marker_correlations_heatmap
from pathlib import Path
from src.segment import *
import shutil


def run_qc(adata, out_dir, name=None, exp_match=True, exp_plot=True, loc_match=False, loc_plot=True,
           intersection_plot=True):
    """
    Running marker quality control pipeline.
    :param adata: Anndata, type: Anndata
    :param out_dir: output dictionary, type: Path
    :param name: name of the experiment, type: str
    :param exp_match: output the csv file with mapping between index and marker for expression_based correlation, type:bool
    :param exp_plot: plot a heatmap for expression_based marker correlation, type: bool
    :param loc_match: output the csv file with mapping between index and marker for location_based correlation, type:bool
    :param loc_plot: plot a heatmap for expression_based marker correlation, type: bool
    :param intersection_plot: plot a heatmap for intersection marker correlation, type: bool
    :return:
    """

    # Expression_based correlation
    get_exp_corr(adata, out_dir, name=name, match_info=exp_match, plot_corr=exp_plot)

    # Location_based correlation
    get_loc_corr(adata, out_dir, name=name, match_info=loc_match, plot_corr=loc_plot)

    # Integrating both expression_based and location_based correlation
    get_marker_correlation(out_dir, name, plot_corr=intersection_plot)


def get_exp_corr(adata, out_dir, name, match_info=True, plot_corr=True):
    """
    Getting expression correlation between markers.
    :param adata: AnnData
    :param out_dir: output directory for the correlation csv file, type: Path
    :param name: experiment name, type: str
    :param match_info: output index and marker matching file or not, type: bool
    :param plot_corr: plot the heatmap for marker correlations, type: bool
    :return:
    """
    if match_info:
        match = []
        for i in range(len(adata.var_names)):
            match.append([i, adata.var_names[i]])

        pd.DataFrame(match, columns=["index", "marker"]).to_csv(out_dir / "index_to_marker.csv", index=False)

    r = get_coexpression_matrix(adata.X, range(len(adata.var_names)))
    corr_df = pd.DataFrame(r, columns=["m1", "m2", "score"])
    corr_df.to_csv(out_dir / f"{name}_exp_corr.csv", index=False)

    if plot_corr:
        plot_marker_correlations_heatmap(out_dir, corr_df, figsize=(7, 7), name=f"{name}_exp")


def get_loc_corr(adata, out_dir, name, match_info=False, plot_corr=True):
    """
    Getting location correlation between markers.
    :param adata: AnnData
    :param out_dir: output directory for the correlation csv file, type:Path
    :param name: experiment name, type: str
    :param match_info: output index and marker matching file or not, type: bool
    :param plot_corr: plot the heatmap for marker correlations, type: bool
    :return:
    """
    # output the matching file between index and marker if needed
    if match_info:
        match = []
        for i in range(len(adata.var_names)):
            match.append([i, adata.var_names[i]])

        pd.DataFrame(match, columns=["index", "marker"]).to_csv(out_dir / "index_to_marker.csv", index=False)

    # Get each image id
    unique_id = pd.unique(adata.obs.image_id)

    # Patching: adding all columns of X to obs to run hot spots
    for i in range(len(adata.var_names)):
        adata.obs[adata.var_names[i]] = adata.X[:, i]

    scores = []
    # Iterative over unique IDs
    for i in range(len(unique_id)):
        # Get single image adata
        p_adata = adata[np.where(adata.obs.image_id == unique_id[i])[0], :]

        # Get neighbors to perform hotspot analysis
        read_neighbors(p_adata, threshold=30)

        # Get hotspot array for each marker column
        for j in range(len(p_adata.var_names)):
            p_adata.obs[f"{p_adata.var_names[j]}_hotspot"] = get_hot_spots_graph(p_adata, col_name=p_adata.var_names[j])

        # Accumulator
        common_hotspots = []

        # Find percentage of common hotspots between markers
        for a in combinations(range(len(p_adata.var_names)), 2):
            # If hotspot matches hotspot only
            col_1_hotspot_idx = np.where(p_adata.obs[f"{p_adata.var_names[a[0]]}_hotspot"] == 1)[0]
            col_2_hotspot_idx = np.where(p_adata.obs[f"{p_adata.var_names[a[1]]}_hotspot"] == 1)[0]
            intersect_hotspot_idx = set(col_1_hotspot_idx).intersection(col_2_hotspot_idx)

            col_1_pct = len(intersect_hotspot_idx) / len(col_1_hotspot_idx)
            col_2_pct = len(intersect_hotspot_idx) / len(col_2_hotspot_idx)

            common_hotspots.append([a[0], a[1], (col_1_pct + col_2_pct) / 2])

        scores.append(common_hotspots)

    # Get the marker index
    m1_m2 = np.array(scores[0])[:, :2]
    m3 = []

    # Get the mean percentages
    for k in range(len(scores)):
        m3.append(np.array([float(j) for j in np.array(scores[k])[:, 2]]))
    m3 = np.array(m3).mean(axis=0)

    corr_df = pd.concat([pd.DataFrame(m1_m2, columns=["m1", "m2"]), pd.DataFrame(m3, columns=["score"])], axis=1)
    corr_df.to_csv(out_dir / f"{name}_loc_corr.csv", index=False)

    # Plot a heatmap
    if plot_corr:
        plot_marker_correlations_heatmap(out_dir, corr_df, figsize=(7, 7), name=f"{name}_loc")


def get_marker_correlation(out_dir, name, plot_corr=True):
    """
    Getting the intersection of expression_based correlation and location_based correlation.
    :param out_dir: directory with the csv files, type: Path
    :param name: experiment name, type: str
    :param plot_corr: plot the heatmap for marker correlations, type: bool
    :return:
    """
    # Read the csv files from previous steps (qc_2_exp_based.py and qc_3_loc_based.py)
    exp_corr = pd.read_csv(out_dir / f"{name}_exp_corr.csv")
    loc_corr = pd.read_csv(out_dir / f"{name}_loc_corr.csv")

    if exp_corr.shape != loc_corr.shape:
        print("Different shapes of the correlation datas.")
        print("shape of exp_corr: " + str(exp_corr.shape))
        print("shape of loc_corr: " + str(loc_corr.shape))
    else:
        # Get the intersection between expression_based correlation and location_based correlation
        intersection_corr = []
        for i in range(exp_corr.shape[0]):
            m1 = exp_corr.iloc[i, 0]
            m2 = exp_corr.iloc[i, 1]
            avg = (exp_corr.iloc[i, 2] + loc_corr.iloc[i, 2]) / 2
            intersection_corr.append([m1, m2, avg])

        # Output the intersection correlation
        intersection_df = pd.DataFrame(intersection_corr, columns=["m1", "m2", "score"])
        intersection_df.to_csv(out_dir / f"{name}_intersection_corr.csv", index=False)

        if plot_corr:
            plot_marker_correlations_heatmap(out_dir, intersection_df, figsize=(7, 7), name=name)


def plot_marker_nuclear_per_image(marker_list, nuclei_name, in_dir, out_dir):
    """
    Plot the marker and nuclei images to check marker quality by each image
    :param marker_list: markers to plot, type: list[str]
    :param nuclei_name: marker name of nuclei, type: str
    :param in_dir: directory of the tiff file and panel information, type: Path
    :param out_dir: directory to store the output images, type: Path
    :return:
    """
    # Get all marker info from panel.csv
    panel_df = pd.read_csv(in_dir / "panel.csv")
    panel = {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}

    for roi, image_path in enumerate(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")), start=1):
        file_name = os.path.basename(image_path)
        print(file_name.split(".")[0], file_name)

        # Read image
        img = skio.imread(image_path)

        for marker in marker_list:

            temp_out_dir = out_dir / marker
            if os.path.exists(temp_out_dir) == 0:
                os.makedirs(temp_out_dir, exist_ok=True)

            channel_idx = panel[marker]

            nuclei = img[panel[nuclei_name], :, :]
            cytoplasm = img[channel_idx, :, :]
            x = stack_for_mesmer(nuclei, cytoplasm)

            plot_marker_nuclear(x, temp_out_dir, name=file_name.split(".")[0] + "_" + marker)


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

    plt.savefig(out_dir / f"{name}.png", dpi=300)
    plt.clf()


