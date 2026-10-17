"""
@author: Tina Hsu
Analysis of TNBC IMC dataset.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.annotate import run_leiden
from src.visualize import plot_zscore_cluster_mean_heatmap, plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.stats import zscore
from scipy.sparse import issparse
from src.annotate import update_labels

COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
              '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
              '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909'
              ]

COLORS = dict(zip(range(len(COLOR_CODE)), COLOR_CODE))


def here_plot_zscore_cell_expression_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), rotation=35,
                                             name=None):
    """
    :param adata: AnnData
    :param column_names: set of column names to cluster, dtype: list(str)
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param rotation: rotation, type: int
    :param name: additional label for the file name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    # create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
        exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    print("labels:")
    print(labels)
    print("-------------------------------------------")
    df = pd.DataFrame(exp_mtx, columns=column_names).assign(label=labels)
    df = df.sort_values(by="label")

    print("df:")
    print(df)
    print("-------------------------------------------")
    if all(str(e).isdigit() for e in labels):
        row_colors = pd.DataFrame({'Cluster': [COLORS[int(i)] for i in df["label"]]})
    else:
        color_mapping = dict(zip(sorted(list(pd.unique(df["label"]))), range(len(pd.unique(df["label"])))))
        row_colors = pd.DataFrame({'Cluster': [COLORS[color_mapping[i]] for i in df["label"]]})

    print("color_mapping:")
    print(color_mapping)
    print("--------------------------------------------")
    print("row_colors:")
    print(row_colors)
    print("--------------------------------------------")
    df = df.drop(columns=["label"]).reset_index(drop=True)

    g = sns.clustermap(df, row_colors=row_colors, row_cluster=False, col_cluster=False, yticklabels=True,
                       figsize=figsize, cmap=sns.diverging_palette(240, 10, n=9), cbar_pos=None)
    ax = g.ax_heatmap
    ax.set_yticklabels([])

    ax.tick_params(right=False, length=1, width=0.5)
    g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    g.ax_row_colors.tick_params(length=1, width=0.5)

    plt.title("Z-scored expression heatmap", weight='bold', fontsize=8)

    # yticks = g.ax_heatmap.get_yticks()
    # print("yticks:")
    # print(yticks)
    # print("-----------------------------------")
    # print(g.ax_row_colors)
    # plt.xticks(fontsize=8, rotation=rotation, ha="right")
    #
    # ax.tick_params(right=False, length=1, width=0.5)
    # g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    # # g.ax_row_colors.set_yticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    # g.ax_row_colors.tick_params(length=1, width=0.5)
    #
    # plt.title("Z-scored expression heatmap", weight='bold', fontsize=8)

    plt.show()

    # if name:
    #     plt.savefig(out_dir / f"{name}_cell_z_expression_heatmap.png", dpi=500)
    # else:
    #     plt.savefig(out_dir / "cell_z_expression_heatmap.png", dpi=500)

    # plt.clf()


"""
Not testing the functions but checking the analysis with one xp, XP12813 (n=70)
"""
if __name__ == '__main__':
    TYPE = "tnbc_imc"
    EXPERIMENT = "XP12813"

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")

    # the list of the pass markers
    markers = ["CK5", "CK7", "CK8-18", "panCK", "B-Catenin",
               "H3K27me3", "Ki67", "GATA3", "CD3e", "CD20",
               "CD45", "CD68", "CD31", "SMA", "Vimentin",
               "vWF"]

    # running leiden clustering
    CLUSTER_KEY = "leiden_cluster"
    run_leiden(adata=adata, columns=markers, n_neighbors=15, name=CLUSTER_KEY)

    plot_zscore_cluster_mean_heatmap(adata=adata, column_names=markers, cluster_key=CLUSTER_KEY,
                                     out_dir=out_dir, name=f"{EXPERIMENT}_all")

    # running 2-component GMM to separate the cells into epithelial and TME by CK markers
    # using CK5, CK7, CK8-18 and panCK in this experiment
    ck_markers = np.sum(adata.X[:, [0, 1, 2, 3]], axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_markers)
    bgm_labels = bgm.predict(ck_markers)
    CLUSTER_KEY = "GMM"
    update_labels(adata, cluster_key=CLUSTER_KEY, labels=bgm_labels)

    # replace the result with epithelial and TME
    adata.obs["GMM"] = adata.obs["GMM"].map({0: 'epithelial', 1: 'TME'})

    # store the file in h5ad
    adata.write(out_dir / f"{EXPERIMENT}.h5ad")

    # check first GMM performance
    # plot_zscore_cell_expression_heatmap(adata=adata, column_names=markers, cluster_key=CLUSTER_KEY,
    #                                     out_dir=out_dir, name=f"{EXPERIMENT}_GMM")
    here_plot_zscore_cell_expression_heatmap(adata=adata, column_names=markers, cluster_key=CLUSTER_KEY,
                                             out_dir=out_dir, name=f"{EXPERIMENT}_GMM")

    # checking the classification result from GMM (epithelial and TME)
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    plot_umap(adata, "GMM", out_dir, figsize=(4.5, 4.5), dotsize=0.5, name="GMM")
    plot_umap_panel(adata, "GMM", out_dir, dotsize=0.1, name="GMM")

    # focusing on TME
    adata_tme = adata[np.where(adata.obs.GMM == "TME")[0], :]
    tme_markers = ["B-Catenin", "H3K27me3", "Ki67", "GATA3", "CD3e",
                   "CD20", "CD45", "CD68", "CD31", "SMA",
                   "Vimentin", "vWF"]
    CLUSTER_KEY = "TME"
    run_leiden(adata=adata_tme, columns=tme_markers, n_neighbors=15, name=CLUSTER_KEY)

    plot_zscore_cluster_mean_heatmap(adata=adata_tme, column_names=tme_markers, cluster_key=CLUSTER_KEY,
                                     out_dir=out_dir, name=f"{EXPERIMENT}_tme")

    # update_labels(adata_tme, cluster_key=CLUSTER_KEY, labels=bgm_labels)

    # checking one image (image_id == row4-5_ROI10_ROI_010.ome.tiff) by plotting
    loc_df = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])
    loc_df['type'] = list(adata.obs['GMM'])
    for img in set(adata.obs.image_id):
        roi = img.split("_")[1].split("I")[1]
        print(roi)
        plt.figure(figsize=(7, 7))
        img_df = loc_df.iloc[np.where(adata.obs.image_id == img)[0], :]
        hue_order = ['epithelial', 'TME']
        sns.scatterplot(img_df, x='x', y='y', hue='type', hue_order=hue_order)
        plt.title(f"ROI {roi}")
        plt.legend(bbox_to_anchor=(1, 1.04), loc="center right", ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{roi}_GMM.png", dpi=300)
        plt.clf()

    """
    Testing the heatmap function for adding label/cluster name on the left side of the color bar.
    """
    #
    # column_names = markers
    # cluster_key = CLUSTER_KEY
    # out_dir = out_dir
    # name = f"{EXPERIMENT}_GMM"
    #
    # # given columns, get column index
    # idxs = []
    # for i in column_names:
    #     for j in range(len(adata.var_names)):
    #         if i == adata.var_names[j]:
    #             idxs.append(j)
    #
    # # create zscore dataframe
    # if issparse(adata.X):
    #     exp_mtx = adata.X[:, idxs].toarray()
    # else:
    #     exp_mtx = adata.X[:, idxs]
    #
    # for col in range(exp_mtx.shape[1]):
    #     # check for divide by zero
    #     if exp_mtx[:, col].sum() == 0.0:
    #         continue
    #     exp_mtx[:, col] = zscore(exp_mtx[:, col])
    #
    # labels = adata.obs[cluster_key].values
    # print("labels:")
    # print(labels)
    # print("-------------------------------------------")
    # df = pd.DataFrame(exp_mtx, columns=column_names).assign(label=labels)
    # df = df.sort_values(by="label")
    #
    # print("df:")
    # print(df)
    # print("-------------------------------------------")
    # if all(str(e).isdigit() for e in labels):
    #     row_colors = pd.DataFrame({'Cluster': [COLORS[int(i)] for i in df["label"]]})
    # else:
    #     color_mapping = dict(zip(sorted(list(pd.unique(df["label"]))), range(len(pd.unique(df["label"])))))
    #     row_colors = pd.DataFrame({'Cluster': [COLORS[color_mapping[i]] for i in df["label"]]})
    #
    # df = df.drop(columns=["label"]).reset_index(drop=True)
    #
    # g = sns.clustermap(df, row_colors=row_colors, row_cluster=False, col_cluster=False, yticklabels=True,
    #                    figsize=(7,7), cmap=sns.diverging_palette(240, 10, n=9), cbar_pos=None)
    #
    # ax = g.ax_heatmap
    # #
    # # yticks = g.ax_heatmap.get_yticks()
    # # print("yticks:")
    # # print(yticks)
    # # print("-----------------------------------")
    # # print(g.ax_row_colors)
    # # plt.xticks(fontsize=8, rotation=rotation, ha="right")
    # ax.set_yticklabels([])
    # # ax.tick_params(right=False, length=1, width=0.5)
    # # g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    # # # g.ax_row_colors.set_yticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    # # g.ax_row_colors.tick_params(length=1, width=0.5)
    # #
    # # plt.title("Z-scored expression heatmap", weight='bold', fontsize=8)
    # #
    # # plt.show()
