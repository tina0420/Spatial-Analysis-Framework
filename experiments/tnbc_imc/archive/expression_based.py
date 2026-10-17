"""
@author: Tina Hsu
Correlations between markers by protein expression (for quality control).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import read_h5ad
from scipy.stats import spearmanr
from src.utils import get_coexpression_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import *
from src.utils import get_coexpression_matrix

"""
where does the marker set come from?
- user selected marker set

helper fn for getting correlation of selected marker set(marker set, expression matrix):
- marker set (for marker i and marker j, where i does not equal to j, are they correlated?)
- expression matrix (are values in the columns of i and j correlated?)

use case:
+ rename the clusters
+ lets say you have 5 clusters, need to create a mapping dictionary
+ 0: epithelial
+ 1: immune
+ 2: stromal

# input cluster (from leiden): 0, 1, 2, 3, 4
# output cluster (manually by inspection): 1, 2, 0, 0, 0

mapped_dict = dict(zip(range(len(adata.obs["leiden_subset"])), [1, 2, 0, 0, 0]))

+ go back to the actual leiden assignment]
from src.annotate import update_labels

update_labels(adata, 
              cluster_key="new_labels", 
              labels=[mapped_dict[i] for i in adata.obs["leiden_subset"]])

+ end result: we have a new column in obs that is called new_labels with labels of epithelial, immune, and stromal cells

+ select indices for some cell type
selected_cell_type = 0     # epithelial
selected_cell_type_idxs = np.where(adata.obs["new_labels"] == selected_cell_type)[0]

+ compare correlation between all epithelial markers
helper fn (user defined marker set [CK5, CK7....], adata[selected_cell_type_idxs, :])

+ compare correlation between all immune markers
helper fn (user defined marker set [CD3, CD68....], adata[selected_cell_type_idxs, :])

+ compare correlation between all stromal markers
helper fn (user defined marker set [CD31, Vimentin....], adata[selected_cell_type_idxs, :])

helper fn(list of str, adata):
    selected_column_idxs = np.where....adata.var_name == ...
    pd.Dataframe(adata[:, selected_columns_idxs].X).corr..........


+ end result: df with columns of marker 1, marker 2, and correlation coefficient

FINAL FUNCTION:
final function(cluster key, adata, list of list of markers == [[CK5, CK7], 
                                                               [CD3, CD68],
                                                               [CD31, Vimentin]])
    for list of pd.unique(adata.obs[cluster key]):        # [0, 1 ,2], n=3
        selected_cell_type_idxs = np.where(adata.obs[cluster key] == selected_cell_type)[0]
        
        for each list of list of markers     # range(len(list of list of markers), n=3
            helper fn (list of markers, adata[selected_cell_type_idxs, :])

def plot_spearman(...)

"""

if __name__ == '__main__':
    """
    Parameter setting for the codes.
    """
    # Path to store the output files
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"

    CELL_TYPE = {0: "epithelial",
                 1: "immune",
                 2: "stromal",
                 3: "undefined"}

    LIST_MARKER_COMB_DICT = {"epithelial": ["CK5", "CK7", "CK8-18", "CK19", "panCK",
                                            "B-Catenin", "H3K27me3", "EZH2", "Ki67", "EGFR",
                                            "GATA3", "CA-IX", "Slug", "Prostaglandin", "p53",
                                            "ER-D8H8", "ER-SP1", "PR-SP2", "erbB2"],
                             "immune": ["CD3e", "CD20", "CD45", "CD68"],
                             "stromal": ["CD31", "SMA", "Vimentin", "vWF"]}

    CLUSTER_KEY = "3_types_labels"

    # Setting the paths for reading data and output figures
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")

    adata = read_h5ad(path_dir / f"{EXPERIMENT}_clustering.h5ad")
    # adata_all = read_h5ad(path_dir / "XP12446_before_batch_correction_clustering.h5ad")

    # Create a dictionary for markers to corresponded number (index)
    # all_markers = adata.var_names
    # marker_dict = dict(zip(all_markers, range(len(all_markers))))

    r = get_coexpression_matrix(adata.X, range(len(adata.var_names)))
    corr_df = pd.DataFrame(r, columns=["m1", "m2", "score"])
    corr_df.to_csv(path_dir / f"{EXPERIMENT}_exp_corr.csv", index=False)














    # """
    # Testing for a45
    # """
    # t_adata = adata[np.where(adata.obs.image_id == "2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a45_ac.ome.tiff")[0], :]



    # def calculate_spearman(exp_mtx, column_idx, marker_list):
    #     """
    #     Doing Spearman Correlation between the markers.
    #     :param exp_df: Expression data of the specific cell types and all markers, type: np.Array
    #     :param column_idx: Index of the column of markers, type: list[int]
    #     :param marker_list: List of the marker combination, type: list[str]
    #     :return: The correlation coefficient between the markers, type: np.Array[[str, str, float]]
    #     """
    #     exp_mtx = exp_mtx[:, column_idx]
    #     corr_list = []
    #     for m1, m2 in combinations(range(len(marker_list)), 2):
    #         corr_list.append([marker_list[m1], marker_list[m2], spearmanr(exp_mtx[:, [m1, m2]].T, axis=1)[0]])
    #
    #     return np.array(corr_list)
    #
    #
    # def plot_zscore_single_cell_type_single_marker_comb_corr_heatmap(exp_mtx, out_dir, cell_type,
    #                                                                  name, marker_idx, marker_comb_name):
    #     plt.figure(figsize=(12, 9))
    #     sns.heatmap(zscore(pd.DataFrame(exp_mtx[:, marker_idx], columns=marker_comb_name).corr(method='spearman')),
    #                 cmap=sns.diverging_palette(240, 10, n=9))
    #     plt.xticks(fontsize=8, rotation=35)
    #     plt.xlabel("Marker", fontsize=8)
    #     plt.yticks(fontsize=8)
    #     plt.ylabel("Marker", fontsize=8)
    #     plt.title(f"a45_cell type: {cell_type}, zscore_correlation_between_{name}_markers", weight='bold', fontsize=8)
    #     plt.savefig(out_dir / f"a45_{cell_type}_cell_z_expression_{name}_heatmap.png", dpi=500)
    #     plt.clf()
    #
    #
    # def all_markers_correlation(exp_mtx, marker_comb_dict, out_dir, cell_type):
    #     marker_idx = []
    #     marker_names = []
    #     for type, marker_comb in marker_comb_dict.items():
    #         marker_idx = marker_idx + [marker_dict[x] for x in marker_comb]
    #         marker_names = marker_names + marker_comb
    #         print(marker_idx)
    #         print(marker_names)
    #
    #     exp_mtx = exp_mtx[:, marker_idx]
    #
    #     plt.figure(figsize=(12, 9))
    #     # sns.heatmap(zscore(pd.DataFrame(exp_mtx, columns=marker_names).corr(method='spearman')),
    #     #             cmap=sns.diverging_palette(240, 10, n=9))
    #     sns.heatmap(pd.DataFrame(exp_mtx, columns=marker_names).corr(method='spearman'),
    #                 cmap=sns.diverging_palette(240, 10, n=9))
    #     plt.xticks(fontsize=8, rotation=35)
    #     plt.xlabel("Marker", fontsize=8)
    #     plt.yticks(fontsize=8)
    #     plt.ylabel("Marker", fontsize=8)
    #     plt.title(f"cell type: {cell_type}, correlation_between_all_markers", weight='bold', fontsize=10)
    #     # plt.show()
    #     plt.savefig(out_dir / f"{cell_type}_all_cells_correlation_heatmap.png", dpi=500)
    #     plt.clf()
    #
    # for cell_type_num, cell_type_name in CELL_TYPE.items():
    #     exp_mtx = np.array(
    #         pd.DataFrame(adata.X[np.where(adata.obs[CLUSTER_KEY] == cell_type_num)[0], :], columns=all_markers))
    #     cor_df = pd.DataFrame(columns=["m1", "m2", "r"])
    #
    #     all_markers_correlation(exp_mtx, LIST_MARKER_COMB_DICT, out_dir, cell_type_name)
    #
    #     for type, marker_comb in LIST_MARKER_COMB_DICT.items():
    #         marker_idx = [marker_dict[x] for x in marker_comb]
    #         r = calculate_spearman(np.array(exp_mtx), marker_idx, marker_comb)
    #         cor_df = pd.concat([cor_df, pd.DataFrame(r, columns=["m1", "m2", "r"])], axis=0)
    #
    #         #plot_zscore_single_cell_type_single_marker_comb_corr_heatmap(exp_mtx, out_dir,
    #                                                                      cell_type_name, type, marker_idx, marker_comb)
    #
    #     cor_df = cor_df.sort_values(by='r', ascending=False)
    #     #cor_df.to_csv(path_dir / f"{cell_type_name}_cell_type_corr.csv", index=False)
    #
    #
    #
    #
    #
    #
    #
    #
    # # helper function for doing spearman test
    # def calculate_spearman(exp_mtx, column_idx, marker_list):
    #     """
    #     Doing Spearman Correlation between the markers.
    #     :param exp_df: Expression data of the specific cell types and all markers, type: np.Array
    #     :param column_idx: Index of the column of markers, type: list[int]
    #     :param marker_list: List of the marker combination, type: list[str]
    #     :return: The correlation coefficient between the markers, type: np.Array[[str, str, float]]
    #     """
    #     exp_mtx = exp_mtx[:, column_idx]
    #     corr_list = []
    #     for m1, m2 in combinations(range(len(marker_list)), 2):
    #         corr_list.append([marker_list[m1], marker_list[m2], spearmanr(exp_mtx[:, [m1, m2]].T, axis=1)[0]])
    #
    #     return np.array(corr_list)
    #
    #
    # def plot_zscore_single_marker_combinations_heatmap(adata, cluster_key, out_dir, list_marker_comb, name):
    #     """
    #     PLot the correlation of markers for each marker combination which includes all cell types.
    #     :param adata: AnnData
    #     :param cluster_key: The cluster key (label/column name) used in obs, type: str
    #     :param out_dir: output path, type: Path
    #     :param list_marker_comb: All the marker combinations, type,: dict
    #     :param name: File name for the figure, type: str
    #     :return:
    #     """
    #     # Plot the marker coorelation for each marker combinations
    #     for type, marker_comb in list_marker_comb.items():
    #         plot_zscore_cell_expression_heatmap(adata, marker_comb, cluster_key, out_dir, figsize=(7, 7),
    #                                             name=f"{type}_marker_combination")
    #
    #
    # def plot_zscore_single_cell_type_single_marker_comb_corr_heatmap(exp_mtx, out_dir, cell_type,
    #                                                                  name, marker_idx, marker_comb_name):
    #     plt.figure(figsize=(12, 9))
    #     sns.heatmap(zscore(pd.DataFrame(exp_mtx[:, marker_idx], columns=marker_comb_name).corr(method='spearman')),
    #                 cmap=sns.diverging_palette(240, 10, n=9))
    #     plt.xticks(fontsize=8, rotation=35)
    #     plt.xlabel("Marker", fontsize=8)
    #     plt.yticks(fontsize=8)
    #     plt.ylabel("Marker", fontsize=8)
    #     plt.title(f"cell type: {cell_type}, zscore_correlation_between_{name}_markers", weight='bold', fontsize=8)
    #     plt.savefig(out_dir / f"{cell_type}_cell_z_expression_{name}_heatmap.png", dpi=500)
    #     plt.clf()
    #
    # plot_zscore_single_cell_type_single_marker_comb_corr_heatmap(exp_mtx=exp_mtx, marker_comb_type=name)
    #
    #
    # # def check_correlation(experiment_name, disease_data_type, cluster_key, cell_type, list_marker_comb):
    # #     # Setting the paths for reading data and output figures
    # #     path_dir = Path(f"data/{disease_data_type}/{experiment_name}")
    # #     out_dir = Path(f"figures/{disease_data_type}/{experiment_name}")
    # #
    # #     adata = read_h5ad(path_dir / f"{experiment_name}_clustering.h5ad")
    # #
    # #     # Create a dictionary for markers to corresponded number (index)
    # #     all_markers = adata.var_names
    # #     marker_dict = dict(zip(all_markers, range(len(all_markers))))
    # #
    # #     for cell_type_num, cell_type_name in cell_type.items():
    # #
    # #         exp_mtx = np.array(pd.DataFrame(adata.X[np.where(adata.obs[cluster_key] == cell_type_num)[0], :], columns=all_markers))
    # #         cor_df = pd.DataFrame(columns=["m1", "m2", "r"])
    # #
    # #         for type, marker_comb in list_marker_comb.items():
    # #             marker_idx = [marker_dict[x] for x in marker_comb]
    # #             r = calculate_spearman(np.array(exp_mtx), marker_idx, marker_comb)
    # #             cor_df = pd.concat([cor_df, pd.DataFrame(r, columns=["m1", "m2", "r"])], axis=0)
    # #
    # #             # plot_zscore_single_cell_type_single_marker_comb_corr_heatmap(exp_mtx, out_dir,
    # #             #                                                              cell_type_name, type, marker_idx, marker_comb)
    # #
    # #         cor_df = cor_df.sort_values(by='r', ascending=False)
    # #         cor_df.to_csv(path_dir / f"{cell_type_name}_cell_type_corr.csv", index=False)
    # #
    # #
    # # check_correlation(EXPERIMENT, TYPE, CLUSTER_KEY, CELL_TYPE, LIST_MARKER_COMB_DICT)
    #


