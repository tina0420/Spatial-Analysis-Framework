"""
@author: Tina Hsu
Marker quality control for spatial data. (main, call functions)
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.utils import get_coexpression_matrix
import pandas as pd
import numpy as np
from itertools import combinations
from src.preprocess import read_neighbors
from src.spatial_stats import get_hot_spots_graph
import matplotlib.pyplot as plt
import seaborn as sns

"""
Need to check: users will start from which file, measurement after segmentation 
               or h5ad file after merging and normalization?
Here starting from h5ad file after normalization and column rename, which is the output from tnbc_imc_measurement_quantify.py
"""


def run_qc(disease_type, experiment, exp_match=True, exp_plot=True, loc_match=False, loc_plot=True, intersection_plot=True):
    """
    Running marker quality control pipeline.
    :param disease_type: disease type, type: str
    :param experiment: experiment name, type: str
    :param exp_match: output the csv file with mapping between index and marker for expression_based correlation, type:bool
    :param exp_plot: plot a heatmap for expression_based marker correlation, type: bool
    :param loc_match: output the csv file with mapping between index and marker for location_based correlation, type:bool
    :param loc_plot: plot a heatmap for expression_based marker correlation, type: bool
    :param intersection_plot: plot a heatmap for intersection marker correlation, type: bool
    :return:
    """
    in_dir = Path(f"data/{disease_type}/{experiment}")
    out_dir = Path(f"data/{disease_type}/{experiment}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{experiment}.h5ad")

    # expression_based correlation
    get_exp_corr(adata, out_dir, name=experiment, match_info=exp_match, plot_corr=exp_plot)

    # location_based correlation
    get_loc_corr(adata, out_dir, name=experiment, match_info=loc_match, plot_corr=loc_plot)

    # integration of expression_based and location_based correlation
    get_marker_correlation(out_dir, experiment, plot_corr=intersection_plot)


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

    # plot heatmap
    if plot_corr:
        plot_marker_correlations_heatmap(out_dir, corr_df, figsize=(7, 7), name=f"{name}_exp")


def get_marker_correlation(out_dir, name, plot_corr=True):
    """
    Getting the intersection of expression_based correlation and location_based correlation.
    :param out_dir: directory with the csv files, type: Path
    :param name: experiment name, type: str
    :param plot_corr: plot the heatmap for marker correlations, type: bool
    :return:
    """
    # Read the csv files from previous steps (exp-based and loc-based correlation metrix)
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


def plot_marker_correlations_heatmap(out_dir, corr_df, figsize, name=None):
    # Get the marker list from the index_to_marker.csv
    marker_list = list(pd.read_csv(out_dir / "index_to_marker.csv").marker)

    # Create a metrix for the heatmap with markers as columns and rows
    corr_mtx = np.ones((len(marker_list), len(marker_list)))

    corr_df = np.array(corr_df)
    # Fill in the correlation metrix
    for j in range(corr_df.shape[0]):
        u, v = corr_df[j][:2]

        corr_mtx[int(u)][int(v)] = corr_df[j][2]
        corr_mtx[int(v)][int(u)] = corr_df[j][2]

    # Plot a heatmap with the created correlation metrix
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(pd.DataFrame(corr_mtx, columns=marker_list, index=marker_list), cmap="coolwarm", ax=ax)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Marker", fontsize=8)
    plt.title("Marker correlations", weight='bold', fontsize=8)
    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{name}_marker_correlations_heatmap.png", dpi=500)
        plt.clf()
    else:
        plt.savefig(out_dir / "marker_correlations_heatmap.png", dpi=500)
        plt.clf()


