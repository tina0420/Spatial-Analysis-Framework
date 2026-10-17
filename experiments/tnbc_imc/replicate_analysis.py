"""
@author: Tina Hsu
Replicating the analysis from Raza Ali's paper in 2023/4.
"""
import os
from pathlib import Path
import scanpy as sc
from src.extract import extract_mcd_for_imc
from src.segment import run_mesmer
from src.preprocess import quantify_expression, normalize

from anndata import read_h5ad
from src.annotate import run_leiden
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels
from src.visualize import *

from src.preprocess import read_neighbors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Plan for the steps and functions.

-three data feed into the predictive model:
+cell density
helper fn (calculating the cell density for each cell type)
#input: image size (micro square meters)
#output: cell densities in float (number of the cell type / tissue area size)

helper fn (plot the odds ratio for the association of cell density and the clinical outcome, relapse)

+cell-cell interactions
helper fn (identify cell-cell interactions)

+proliferation
helper fn (calculate the proliferation proportion in each cell phenotype)

"""

if __name__ == "__main__":
    """
    Parameter setting for the codes.
    """
    # Path to store the output files
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"
    print(f"Processing {EXPERIMENT}")

    # Setting the paths for reading data and output figures
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    Step1: Preprocessing, mcd to tiff BY IMC Converter and segmentation by Mesmer
    """
    extract_mcd_for_imc(in_dir=in_dir, out_dir=in_dir)

    # segmentation
    DEEPCELL_TOKEN = "gLJAn8ZY.OmHKrKnth0mqobIljEi6o16StVeqhFVn"

    # run_mesmer(nuclear_channel=[31], cytoplasm_channels=[3, 29],
    #            in_dir=in_dir, out_dir=out_dir,
    #            deepcell_token=DEEPCELL_TOKEN,
    #            mes_image_mpp=1, mes_interior_threshold=0.1, mes_maxima_threshold=0.58)

    run_mesmer(nuclear_channel=[31], cytoplasm_channels=[3, 29],
               in_dir=in_dir, out_dir=out_dir,
               deepcell_token=DEEPCELL_TOKEN,
               mes_image_mpp=1, mes_interior_threshold=0.1, mes_maxima_threshold=0.58,
               plot_mask=True, plot_overlay=True, output_pred_df=False)

    adata = quantify_expression(out_dir)

    # Choose the column we want
    col_of_interest = ['Cell: CK5 mean', 'Cell: CK19 mean',
                       'Cell: CK8-18 mean', 'Cell: CD68 mean',
                       'Cell: vWF mean', 'Cell: SMA mean',
                       'Cell: Vimentin mean', 'Cell: CD3e mean',
                       'Cell: H3K27me3 mean', 'Cell: CD45 mean',
                       'Cell: GATA3 mean', 'Cell: CD20 mean',
                       'Cell: B-Catenin mean', 'Cell: Ki67 mean',
                       'Cell: CD31 mean', 'Cell: CK7 mean',
                       'Cell: panCK mean'
                       ]

    adata = adata[:, col_of_interest]

    # rename the columns
    adata.var_names = [x.split(" ")[1] for x in adata.var_names]

    # normalization
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    # clustering
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    # store the file in h5ad
    adata.write(out_dir / f"{EXPERIMENT}.h5ad")

    """
    Step2: Clustering, Leiden
    """
    # adata = read_h5ad(out_dir / f"{EXPERIMENT}.h5ad")

    run_leiden(adata, columns=adata.var_names, n_neighbors=150, n_pcs=None, name="leiden_cluster")

    adata.write(out_dir / f"{EXPERIMENT}.h5ad")

    """
    Step3: Annotation, epithelial and TME (2-component GMM)
    """
    # adata = read_h5ad(out_dir / f"{EXPERIMENT}.h5ad")

    # 2-component GMM
    ck_sum_here = np.sum(adata.X[:, [0, 2, 15, 16]], axis=1).reshape(-1, 1)  # CK 5, 8-18, 7, pan

    bgm_here = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_sum_here)

    bgm_labels = bgm_here.predict(ck_sum_here)
    update_labels(adata, cluster_key="gmm_4ck", labels=bgm_labels)

    plot_cell_expression_heatmap(adata, 'gmm_4ck',
                                 out_dir, name="GMM_4_CKs")

    plot_zscore_cell_expression_heatmap(adata, adata.var_names,
                                        'gmm_4ck', out_dir,
                                        name='GMM_4_CKs')

    # in gmm_4ck, 0: TME, 1:epithelial

    # Try different columns for GMM, CK5, CK7 and panCK (3_ck)

    ck_sum_there = np.sum(adata.X[:, [0, 15, 16]], axis=1).reshape(-1, 1)  # CK 5, 7, pan
    bgm_there = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_sum_there)
    bgm_labels = bgm_there.predict(ck_sum_there)
    update_labels(adata, cluster_key="bgm_3ck", labels=bgm_labels)

    # check the difference between two ways of GMM (3 CKs there and 4 CKs here)
    print(len(np.where(adata.obs.bgm_4ck != adata.obs.bgm_3ck)[0]))  # 6978

    # check which is epithelial and which is TME, by the heatmap
    plot_cell_expression_heatmap(adata, 'bgm_3ck',
                                 out_dir, name="3_CKs_new")

    plot_zscore_cell_expression_heatmap(adata, adata.var_names,
                                        'bgm_3ck', out_dir,
                                        name='3_CKs_new')

    """
    Step4: Annotation, identify the cell types in epithelial and TME cells separately (clustering and merge clusters)
    (Notice: Clustering without Ki67 based on the methods in the paper)
    """
    print("TME")
    adata_tme = adata[np.where(adata.obs.gmm_4ck == 0)[0], :]
    tme_markers = ["CD3e", "CD20", "CD45", "CD68", "CD31", "SMA", "Vimentin"]
    epi_markers = ["CK5", "CK7", "CK8-18", "panCK", "H3K27me3", "B-Catenin"]
    markers_order_plot = tme_markers + epi_markers + ["Ki67"]
    # clustering
    run_leiden(adata_tme, columns=tme_markers, n_neighbors=150, n_pcs=None, name="leiden_cluster")
    # heatmap plotting --> to merge clusters
    plot_zscore_cluster_mean_heatmap(adata_tme, markers_order_plot, "leiden_cluster",
                                     out_dir=out_dir, figsize=(7, 7), name="tme_all_markers")
    # new labels (28 clusters -> clusters)
    new_labels = ['B cell', 'Endothelial', 'T cell', 'Endothelial', 'T cell', 'Fibroblasts',  # 0-5
                  'Stromal', 'Stromal', 'Fibroblasts', 'Endothelial', 'T cell',  # 6-10
                  'Macrophage', 'T cell', 'Myofibroblasts', 'Stromal', 'Myofibroblasts',  # 11-15
                  'T cell', 'Stromal', 'T cell', 'Macrophage', 'Macrophage',  # 16-20
                  'Macrophage', 'Stromal', 'Macrophage', 'Fibroblasts', 'Myofibroblasts',  # 21-25
                  'Fibroblasts', 'Macrophage', 'Endothelial']  # 26-28

    # update the labels
    clusters_mapped_dict = dict(zip(range(len(adata_tme.obs["leiden_cluster"])), new_labels))
    update_labels(adata=adata_tme, cluster_key="tme_clusters",
                  labels=[clusters_mapped_dict[x] for x in adata_tme.obs["leiden_cluster"]])
    # plot again to check if I merged the clusters correctly
    plot_zscore_cluster_mean_heatmap(adata_tme, tme_markers, "tme_clusters",
                                     out_dir=out_dir, figsize=(7, 7), name="tme_cell_phenotypes")

    print("epithelial")
    adata_epi = adata[np.where(adata.obs.gmm_4ck == 1)[0], :]
    epi_markers = ["CK5", "CK7", "CK8-18", "panCK", "H3K27me3", "B-Catenin"]
    markers_order_plot = epi_markers + tme_markers + ["Ki67"]
    # clustering
    run_leiden(adata_epi, columns=epi_markers, n_neighbors=150, n_pcs=None, name="leiden_cluster")
    # heatmap plotting --> to merge clusters
    plot_zscore_cluster_mean_heatmap(adata_epi, markers_order_plot, "leiden_cluster",
                                     out_dir=out_dir, figsize=(7, 7), name="epi_all_markers")
    # new labels
    new_labels = ['CK7_enr_Luminal', 'CK7_enr_Luminal', 'Basal-like', 'Luminal', 'Luminal', 'CK7_enr_Luminal',  # 0-5
                  'CK7_enr_Luminal', 'Luminal', 'Luminal', 'Basal-like', 'Luminal',  # 6-10
                  'CK7_enr_Luminal', 'Luminal', 'Luminal', 'Basal-like', 'Luminal',  # 11-15
                  'Luminal', 'Basal-like', 'Basal-like', 'H3K27me3_enr_Luminal', 'H3K27me3_enr_Luminal',  # 16-20
                  'Luminal', 'Basal-like', 'Basal-like', 'CK7_enr_Luminal', 'Luminal',  # 21-25
                  'Luminal', 'B-Catenin_enr_Luminal', 'CK7_enr_Luminal', 'Luminal', 'Basal-like']  # 26-30

    # update the labels
    clusters_mapped_dict = dict(zip(range(len(adata_epi.obs["leiden_cluster"])), new_labels))
    update_labels(adata=adata_epi, cluster_key="epi_clusters",
                  labels=[clusters_mapped_dict[x] for x in adata_epi.obs["leiden_cluster"]])
    # plot again to check if I merged the clusters correctly
    plot_zscore_cluster_mean_heatmap(adata_epi, epi_markers, "epi_clusters",
                                     out_dir=out_dir, figsize=(7, 7), name="epi_cell_phenotypes_epi_markers")

    # tidy the clustering results of epithelial and tme and store them into the original adata (diff types -> na)
    tme_idx = np.where(adata.obs.gmm_4ck == 0)[0]
    epi_idx = np.where(adata.obs.gmm_4ck == 1)[0]
    tme_pointer = 0
    epi_pointer = 0
    cell_types = []
    for i in range(adata.X.shape[0]):
        if i in tme_idx:
            cell_types.append(adata_tme.obs.tme_clusters[tme_pointer])
            tme_pointer += 1
        else:
            cell_types.append(adata_epi.obs.epi_clusters[epi_pointer])
            epi_pointer += 1

    adata.obs["cell_type"] = cell_types
    adata.write(out_dir / f"{EXPERIMENT}.h5ad")

    # Go through all ROIs

    CLUSTER_KEY = "cell_type"
    for p_name in pd.unique(adata.obs.image_id):
        print(f"Processing {p_name}")

        p_adata = adata[np.where(adata.obs.image_id == p_name)]
        pid = p_adata.obs.image_id[0]

        # Plot cell count by cluster
        plot_cluster_cell_count(p_adata, CLUSTER_KEY, out_dir, name=pid)

        # Plot cluster map
        plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(7, 7), dotsize=2.5, name=pid)

        # Plot cluster panel
        plot_spatial_cluster_panel(p_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=pid)

    plot_cluster_cell_count(adata, "cell_type", out_dir, figsize=(4.5, 2))

    """
    Step5: Calculate the density and proportion.
    """
    # cell phenotype density
    # area of the tissue
    from scipy.spatial import ConvexHull

    # get the points/centroids of the cells, the first image for testing
    img = '2024-07-30 _row4-5_TNBC-TMA-AT19813_22-010A-TNBC-ctDNA_BMplus-FP_s0_a5.ome.tiff'
    img_idx = np.where(adata.obs.image_id == img)[0]
    img_idx_tme = adata.obs[(adata.obs.image_id == img) & (adata.obs.gmm_4ck == 0)].index.astype(int)
    img_idx_epi = adata.obs[(adata.obs.image_id == img) & (adata.obs.gmm_4ck == 1)].index.astype(int)

    hull = ConvexHull(adata.obsm["spatial"][img_idx])

    # spatial and cell type info
    loc_df = pd.DataFrame(adata.obsm["spatial"], columns=["x", "y"])
    loc_df.index = loc_df.index.astype(int)
    cell_type_df = pd.DataFrame(adata.obs.cell_type, columns=["cell_type"])
    cell_type_df.index = cell_type_df.index.astype(int)
    cell_loc_types_df = pd.concat([loc_df, cell_type_df], axis=1)

    # loc_df = pd.DataFrame(adata_tme.obsm["spatial"], columns=["x", "y"])
    # loc_df.index = loc_df.index.astype(int)
    # # cell_type_df = pd.DataFrame(adata_tme.obs.tme_clusters, columns=["cell_type"])
    # cell_type_df = adata_tme.obs.tme_clusters
    # cell_type_df.index = cell_type_df.index.astype(int)
    # cell_loc_types_df = pd.concat([loc_df, cell_type_df], axis=1)

    plt.figure(figsize=(12, 9))
    # sns.scatterplot(cell_loc_types_df, x="x", y="y", hue="cell_type")
    sns.scatterplot(cell_loc_types_df.iloc[img_idx, :], x="x", y="y", hue="cell_type")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # plt.plot(cell_loc_types_df.iloc[hull.vertices, 0], cell_loc_types_df.iloc[hull.vertices, 1], 'r--', lw=2)
    # plt.plot(cell_loc_types_df.iloc[hull.vertices[0], 0], cell_loc_types_df.iloc[hull.vertices[0], 1], 'ro')

    for simplex in hull.simplices:
        plt.plot(cell_loc_types_df.iloc[simplex, 0], cell_loc_types_df.iloc[simplex, 1], 'k-')

    plt.title(img, fontsize=15)
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y", fontsize=15)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

    color_map = {"B cell": '#77b41f',
                 "B-Catenin_enr_Luminal": '#b41f77',
                 "Basal-like": '#1f77b4',
                 'CK7_enr_Luminal': '#ff7f0e',
                 'Endothelial': '#ff9896',
                 'Fibroblasts': '#bd67bb',
                 'H3K27me3_enr_Luminal': '#8c564b',
                 'Luminal': '#fff80e',
                 'Macrophage': '#1fb45c',
                 'Myofibroblasts': '#b41f2d',
                 'Stromal': '#1f2db4',
                 'T cell': '#ff430e'
                 }

    img_area_estimated = []
    # plot the cell type results in all the images
    # for pid in np.unique(adata.obs.image_id):
    for pid in range(1):
        print(pid)
        img_idx = np.where(adata.obs.image_id == pid)[0]
        # hull = ConvexHull(adata.obsm["spatial"][img_idx])
        # img_area_estimated.append(hull.area)

        plt.figure(figsize=(12, 9))
        sns.scatterplot(cell_loc_types_df.iloc[img_idx, :], x="x", y="y", hue="cell_type", palette=color_map)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.title(img, fontsize=15)
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        plt.tight_layout()
        plt.axis('off')
        # plt.savefig(out_dir / f"{pid}_cell_types_distribution.png", dpi=500)
        # plt.clf()
        plt.show()

    # calculation for the density
    all_cell_types = np.unique(adata.obs.cell_type)
    density_dict = {}
    for i in all_cell_types:
        if i not in density_dict:
            density_dict[i] = []

    img_area_estimated = {}
    for pid in np.unique(adata.obs.image_id):
        print(pid)
        img_idx = np.where(adata.obs.image_id == pid)[0]
        hull = ConvexHull(adata.obsm["spatial"][img_idx])
        img_area_estimated[pid] = hull.area
        p_adata = adata[img_idx, :]
        for j in all_cell_types:
            numbers = len(np.where(p_adata.obs.cell_type == j)[0])
            density = numbers/hull.area
            density_dict[j].append(density)

    # plot densities of all the cell phenotypes
    density_df = pd.DataFrame(density_dict)
    plt.figure(figsize=(12, 9))
    plt.boxplot(density_df)

    plt.xticks(range(1, len(density_df.columns) + 1), density_df.columns, rotation=90, fontsize=15)
    plt.xlabel("cell type", fontsize=18)
    plt.yticks([0.0, 0.1, 0.3, 0.5, 0.7])
    plt.ylabel("density (%)", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_dir / f"{EXPERIMENT}_cell_type_density_distribution.png", dpi=500)
    plt.clf()

    # cell phenotype proportions across all the images



    """
    Step6: Identify the cell-cell interactions within or across cell phenotypes
    """
    read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=13)
    plot_neighbor_graph(p_adata, out_dir, name="13_")

    # count edges
    edges = p_adata.uns["relation"]
    labels = p_adata.obs.cell_type.values
    label_categories = sorted(list(pd.unique(labels)))
    cat_to_num_dict = dict(zip(label_categories, range(len(label_categories))))
    idx_label_dict = dict(zip(range(len(labels)), [cat_to_num_dict[i] for i in labels]))
    i=0
    cell_type_dict = dict(zip(all_cell_types, range(len(all_cell_types)))) # cell type to the number
    all_interactions = np.empty((0, 2), int)
    for pid in np.unique(adata.obs.image_id):
        print(pid)
        p_adata = adata[np.where(adata.obs.image_id == pid)[0], :]
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=10)
        # plot_neighbor_graph(p_adata, out_dir, name=f"{pid}")
        edges = p_adata.uns["relation"]
        labels = p_adata.obs.cell_type.values
        # label_categories = sorted(list(pd.unique(labels)))
        # cat_to_num_dict = dict(zip(label_categories, range(len(label_categories))))

        idx_label_dict = dict(zip(range(len(labels)), [cell_type_dict[i] for i in labels]))

        for u, v in edges:
            to_add = np.array([idx_label_dict[u], idx_label_dict[v]])
            all_interactions = np.vstack([all_interactions, to_add])

    # all the interactions are stored in all_interactions (122056, 2)
    edge_count_matrix = np.zeros((len(all_cell_types), len(all_cell_types)), dtype=int)
    for u, v in all_interactions:
        if u == v:
            edge_count_matrix[u, v] += 1
        else:
            if u > v:
                edge_count_matrix[u, v] += 1
            else:
                edge_count_matrix[v, u] += 1

    edge_df = pd.DataFrame(edge_count_matrix, columns=all_cell_types, index=all_cell_types)
    edge_df.to_csv(out_dir / "cell_cell_interaction_sum_in_cell_types.csv", index=False)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    mask = np.triu(np.ones_like(edge_df, dtype=bool))
    for i in range(len(edge_df)):
        mask[i, i] = False

    sns.heatmap(edge_df, annot=False,
                annot_kws={"size": 8}, cmap="RdPu", ax=ax, mask=mask, cbar=True)

    plt.xlabel("Cluster", fontsize=8)
    plt.ylabel("Cluster", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.title("Edge count", weight='bold', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"edge_count.png", dpi=500)
    plt.clf()


    """
    Step7: Calculating the proliferative fraction (Ki67)
    """
    # identify positive or negative of Ki67 for each cell
    """
    To determine the thresholds for marker positivity (based on the following description from the paper):
    We identified thresholds for assigning a cell as ‘positive’ for a given marker 
    by inspecting a random selection of at least 50 images for which cells passing a quantile threshold 
    (calculated using all data) were highlighted. 
    This procedure was repeated at differing quantile thresholds until the value that 
    most closely aligned with marker positivity was identified.
    """
    # use Ki67 for testing, index==13
    proliferation_dict = {}
    for i in all_cell_types:
        if i not in proliferation_dict:
            proliferation_dict[i] = []

    # calculate the mean of Ki67 in each cell type across all the images
    for pid in np.unique(adata.obs.image_id):
        print(pid)
        img_idx = np.where(adata.obs.image_id == pid)[0]
        p_adata = adata[img_idx, :]
        #exp_df = p_adata.X[img_idx, 13]  # Ki67
        for j in all_cell_types:
            print(j)
            if len(np.where(p_adata.obs.cell_type == j)[0]) == 0:
                mean_ki67 = 0
            else:
                mean_ki67 = np.mean(p_adata.X[np.where(p_adata.obs.cell_type == j)[0], 13].tolist())
            proliferation_dict[j].append(mean_ki67)

        # plot proliferation of all the cell phenotypes
        proliferation_df = pd.DataFrame(proliferation_dict)
        plt.figure(figsize=(12, 9))
        plt.boxplot(proliferation_df)

        plt.xticks(range(1, len(proliferation_df.columns) + 1), proliferation_df.columns, rotation=90, fontsize=15)
        plt.xlabel("cell type", fontsize=18)
        #plt.yticks([0.0, 0.1, 0.3, 0.5, 0.7])
        plt.ylabel("mean expression of Ki67", fontsize=18)
        plt.tight_layout()
        plt.savefig(out_dir / f"{EXPERIMENT}_cell_type_proliferation_mean.png", dpi=500)
        plt.clf()

