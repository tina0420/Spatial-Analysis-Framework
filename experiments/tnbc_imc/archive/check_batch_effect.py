"""
@Author: Tina Hsu
Check batch effect and batch correction.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import read_h5ad
from src.annotate import run_leiden
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import plot_cluster_mean_heatmap, plot_zscore_cluster_mean_heatmap

if __name__ == "__main__":
    """
    Setting parameters for the codes.
    """
    # For the path to store the output files
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"

    # Setting the paths for reading data and output figures
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(path_dir / f"{EXPERIMENT}.h5ad")

    # For doing leiden clustering
    CLUSTER_KEY = "leiden_subset"
    MARKERS = list(adata.var_names)
    NEIGHBORS_TO_CONSIDER = 200


    """
    Start checking batch effect.
    """
    print("Start doing leiden clustering")
    # Clustering by Leiden
    run_leiden(adata, MARKERS, NEIGHBORS_TO_CONSIDER, n_pcs=None)
    plot_cluster_mean_heatmap(adata, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_all")

    MARKERS = ['CD20', 'CD31', 'CD3e', 'CD68', 'SMA', 'panCK']
    plot_zscore_cluster_mean_heatmap(adata, MARKERS, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name=F"{EXPERIMENT}_all")

    # Output the clustering result
    adata.write(path_dir / f"{EXPERIMENT}_before_batch_correction_clustering.h5ad")

    # Umap check batch effect
    def check_batch_effect_umap(adata, out_dir, name):
        umap_set = umap.UMAP(n_components=2, random_state=42).fit_transform(adata.X)
        umap_df = pd.DataFrame(umap_set)
        # short_image_name = pd.DataFrame([x.split("_")[-2] for x in adata.obs.image_id])
        umap_df = pd.concat([pd.DataFrame([x.split("_")[-2] for x in adata.obs.image_id]), umap_df], axis=1)
        umap_df.columns = ["Image", "X", "Y"]

        # Plot
        plt.figure(figsize=(12, 9))
        plt.title(f"Umap for batch effect_{name}", fontsize=30, pad=5)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        sns.scatterplot(data=umap_df, x="X", y="Y", hue="Image")
        plt.legend(bbox_to_anchor=(1.01, 1.05), loc="upper left", borderaxespad=0., ncol=2)
        # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0., ncol=4)
        plt.tight_layout()
        #plt.show()

        plt.savefig(out_dir / f"umap_batch_{name}", dpi=500)
        plt.clf()
        adata.obsm["X_umap_ori"] = np.array(umap_df.iloc[:, 1:])

    print("Start doing batch effect checking")
    check_batch_effect_umap(adata, out_dir, "before")


    """
    Check the cluster combination of each image. (boxplot)
    """
    # Data for each image
    image_dict = {}
    for i in range(adata.X.shape[0]):
        if adata.obs.image_id[i] in image_dict:
            continue
        else:
            image_dict[adata.obs.image_id[i]] = adata.obs.image_id[i].split("_")[-2]

    # Build the figure
    cols = 8
    fig, ax = plt.subplots(nrows=len(set(adata.obs.image_id)) // cols + 1, ncols=cols,
                           figsize = (12, 9), sharey=True, sharex=True)
    j = 0
    while (j < len(set(adata.obs.image_id))):

        j+=1


    # exp_df = pd.DataFrame(adata.X, columns=adata.var_names)
    for image_name, image_short in image_dict.items():
        # Calculating the percentages of the clusters in each image
        counts = adata.obs.iloc[np.where(adata.obs.image_id == image_name)[0], :]["leiden_subset"].value_counts()
        per = round(counts / counts.sum(), 3)
        per = per[per > 0]
        clusters = per.index.astype(str)


    #
    # cluster_pct = {}
    # for i in range(adata.X.shape[0]):
    #     if adata.obs.image_id[i] not in cluster_pct:
    #         cluster_pct[adata.obs.image_id[i]] =
    #     else:














    """
    Doing batch correction.
    """

    """
    Batch Correction 1: pyCombat
    pip install inmoose
    """
    from inmoose.pycombat import pycombat_norm, pycombat_seq







    """
    Batch Correction 2: mnnpy
    pip install mnnpy
    """
    import mnnpy





#
#
#
# def pca_plot_batch_effect(out_dir, file_name, adata_obj, image_name_df):
#     """
#     Run and plot PCA for checking batch effect
#     :param out_dir: Directory to store the output image, type: Path
#     :param file_name: Name of the file, type: Str
#     :param adata_obj: AnnData object for doing PCA, type: AnnData object
#     :param image_name_df: Original image names to concat, type: pd.DataFrame
#     :return: Output an image
#     """
#     # Prepare the data
#     exp_df = adata_obj.X
#     pca = PCA(n_components=2, random_state=42)
#     transform_df = pca.fit_transform(exp_df)
#     transform_df = pd.DataFrame(transform_df)
#     short_image_name = pd.DataFrame([x.split("_")[-2] for x in image_name_df])
#     pca_df = pd.concat([short_image_name, transform_df], axis=1)
#     pca_df.columns = ["Image", "PC1", "PC2"]
#
#     # Plot
#     plt.figure(figsize=(12, 9))
#     plt.title("PCA for batch effect", fontsize=30)
#     plt.xlabel("PC1", fontsize=20)
#     plt.ylabel("PC2", fontsize=20)
#     sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Image")
#     #ax.legend('upper left', ncol=10)
#     plt.tight_layout()
#
#     file_name = file_name + "_pca_batch_effect.png"
#     plt.savefig(out_dir / file_name, dpi=500)
#     plt.clf()
#
#     adata_obj.obsm["X_pca"] = np.array(pca_df.iloc[:, 1:])


pd.DataFrame(adata.obsm['X_umap_ori']).to_csv("XP12446_umap_embedding.csv", index=False)
