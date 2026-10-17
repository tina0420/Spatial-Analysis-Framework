"""
@author: Tina Hsu
Different combinations of the protein markers for the clustering (especially for differentiating cell types).
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
from anndata import read_h5ad
from src.annotate import run_leiden, update_labels
from src.visualize import plot_cluster_mean_heatmap, plot_zscore_cluster_mean_heatmap, plot_spatial_cluster_distribution
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Parameters for the codes
    # For the path to store the output files
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"

    # For doing leiden clustering
    CLUSTER_KEY = "leiden_subset"
    # MARKERS = ['CD20', 'CD31', 'CD3e', 'CD68', 'SMA', 'panCK']
    # MARKERS = list(adata.var_names)
    NEIGHBORS_TO_CONSIDER = 200


    NEW_CLUSTER_KEY = "3_types_labels"
    # For the cell types' identification
    # CELL_TYPE_LIST = [0, 3, 1, 3, 2,
    #                   0, 1, 0, 0, 0,
    #                   2, 2, 3, 3, 1,
    #                   3, 0, 3, 3, 2,
    #                   0, 0, 0, 1, 2,
    #                   1, 3, 0, 3, 1]

    CELL_TYPE_LIST = [3, 1, 2, 1, 3]

    # plotting modification from src (remove undefined cell type: cluster==3)
    COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
                  '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
                  '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909']

    COLORS = dict(zip(range(len(COLOR_CODE)), COLOR_CODE))

    print("Finish setting")
    # Setting the paths for reading data and output figures
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(path_dir / f"{EXPERIMENT}.h5ad")
    MARKERS = list(adata.var_names)

    # Clustering by Leiden
    run_leiden(adata, MARKERS, NEIGHBORS_TO_CONSIDER, n_pcs=None)

    plot_cluster_mean_heatmap(adata, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="All")

    plot_zscore_cluster_mean_heatmap(adata, MARKERS, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="All")

    # Update the cell types as new label
    clusters_mapped_dict = dict(zip(range(len(adata.obs[CLUSTER_KEY])), CELL_TYPE_LIST))

    update_labels(adata=adata, cluster_key=NEW_CLUSTER_KEY,
                  labels=[clusters_mapped_dict[x] for x in adata.obs[CLUSTER_KEY]])

    print("Plotting new labels")
    # plot the distribution of the new labels (cell types)
    def plot_spatial_cluster_distribution_here(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=3, name=None):
        """
        :param adata: AnnData
        :param cluster_key: indicate which cluster key to use in obs, type: str
        :param out_dir: output path, type: Path
        :param figsize: figure size, type: tuple
        :param dotsize: dot size, type: float
        :param name: additional label for the file name, type: str
        :return:
        """
        labels = adata.obs[cluster_key].values
        label_categories = sorted(list(pd.unique(labels)))
        is_digit_labels = all(str(e).isdigit() for e in labels)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for i in range(len(label_categories)-1):
            lbl_idxs = np.where(labels == label_categories[i])[0]

            if is_digit_labels:
                ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
                           -adata.obsm["spatial"][lbl_idxs][:, 1],
                           c=COLORS[int(label_categories[i])], s=dotsize, label=label_categories[i])
            else:
                ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
                           -adata.obsm["spatial"][lbl_idxs][:, 1],
                           c=COLORS[i], s=dotsize, label=label_categories[i])

        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])

        ax.axis('off')
        ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
        ax.set_title("Cluster distribution", weight="bold", fontsize=8)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, ncol=4, frameon=False, fontsize=6, markerscale=5)

        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_spatial_cluster_distribution.png", dpi=500)
        else:
            plt.savefig(out_dir / "spatial_cluster_distribution.png", dpi=500)
        plt.clf()

    # Plotting the clusters by single image
    for i in np.unique(adata.obs.image_id):
        plot_spatial_cluster_distribution_here(adata[np.where(adata.obs.image_id == i)[0], :], NEW_CLUSTER_KEY, out_dir,
                                               figsize=(7, 7), name=f"{i.split('_')[-2]}_{NEW_CLUSTER_KEY}")

    # Output the h5ad file
    adata.write(path_dir / f"{EXPERIMENT}_clustering.h5ad")

    """
    Function for clustering.
    """

#     def plot_spatial_cluster_distribution_here(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=1, name=None):
#         """
#         :param adata: AnnData
#         :param cluster_key: indicate which cluster key to use in obs, type: str
#         :param out_dir: output path, type: Path
#         :param figsize: figure size, type: tuple
#         :param dotsize: dot size, type: float
#         :param name: additional label for the file name, type: str
#         :return:
#         """
#         labels = adata.obs[cluster_key].values
#         label_categories = sorted(list(pd.unique(labels)))
#         is_digit_labels = all(str(e).isdigit() for e in labels)
#
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#         for i in range(len(label_categories)-1):
#             lbl_idxs = np.where(labels == label_categories[i])[0]
#
#             if is_digit_labels:
#                 ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
#                            -adata.obsm["spatial"][lbl_idxs][:, 1],
#                            c=COLORS[int(label_categories[i])], s=dotsize, label=label_categories[i])
#             else:
#                 ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
#                            -adata.obsm["spatial"][lbl_idxs][:, 1],
#                            c=COLORS[i], s=dotsize, label=label_categories[i])
#
#         # # Shrink current axis's height by 10% on the bottom
#         # box = ax.get_position()
#         # ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.9])
#
#         ax.axis('off')
#         ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
#         ax.set_title("Cluster distribution", weight="bold", fontsize=8)
#
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
#                   fancybox=True, ncol=4, frameon=False, fontsize=6, markerscale=5)
#
#         plt.tight_layout()
#
#         if name:
#             plt.savefig(out_dir / f"{name}_spatial_cluster_distribution.png", dpi=500)
#         else:
#             plt.savefig(out_dir / "spatial_cluster_distribution.png", dpi=500)
#         plt.clf()
#
#
#     def clustering(experiment_name, disease_data_type, cluster_key, markers, neighbors_to_consider, NEW_CLUSTER_KEY,cell_type_list):
#         # Setting the path for storing figures
#         path_dir = Path(f"data/{disease_data_type}/{experiment_name}")
#         out_dir = Path(f"figures/{disease_data_type}/{experiment_name}")
#
#         adata = read_h5ad(path_dir / f"{experiment_name}.h5ad")
#
#         leiden_subset(adata, markers, neighbors_to_consider, n_pcs=None)
#
#         plot_cluster_mean_heatmap(adata, cluster_key, out_dir=out_dir, figsize=(7, 7), name="All")
#
#         plot_zscore_cluster_mean_heatmap(adata, markers, cluster_key, out_dir=out_dir, figsize=(7, 7), name="All")
#
#         # Update the cell types as new label
#         clusters_mapped_dict = dict(zip(range(len(adata.obs[cluster_key])), cell_type_list))
#
#         update_labels(adata=adata, cluster_key=NEW_CLUSTER_KEY,
#                       labels=[clusters_mapped_dict[x] for x in adata.obs[cluster_key]])
#
#         # plot the distribution of the new labels (cell types)
#         plot_spatial_cluster_distribution_here(adata, NEW_CLUSTER_KEY, out_dir, figsize=(7, 7), name=NEW_CLUSTER_KEY)
#
#         # Output the h5ad file
#         adata.write(path_dir / f"{experiment_name}_clustering.h5ad")
#
#
#     clustering(EXPERIMENT, TYPE, CLUSTER_KEY, MARKERS, NEIGHBORS_TO_CONSIDER, NEW_CLUSTER_KEY, CELL_TYPE_LIST)
#
#
#
# # def identify_cell_types(adata, CLUSTER_KEY):
# #     # Identify the cell types into 0: epithelial, 1: immune, 2:stromal, 3: undefined
# #     # clusters_mapped_dict = dict(zip(range(len(adata.obs[CLUSTER_KEY])),
# #     #                                 [3, 0, 0, 3, 0,
# #     #                                  3, 2, 1, 2, 0,
# #     #                                  1, 3, 2, 0, 1,
# #     #                                  2, 0, 2, 0, 2,
# #     #                                  0, 3, 1, 0, 2,
# #     #                                  0, 0, 1, 1, 0,
# #     #                                  0, 0]))
# #
# #     clusters_mapped_dict = dict(zip(range(len(adata.obs[CLUSTER_KEY])),
# #                                     [0, 3, 1, 1, 2,
# #                                      0, 1, 0, 0, 2,
# #                                      2, 2, 1, 0, 1,
# #                                      3, 0, 1, 0, 2,
# #                                      2, 0, 0, 1, 2,
# #                                      1, 3, 0, 3, 1]))
# #
# #     # Update new labels of cell types in adata.obs
# #     update_labels(adata=adata, cluster_key="3_types_labels",
# #                   labels=[clusters_mapped_dict[x] for x in adata.obs[CLUSTER_KEY]])
# #
# #
