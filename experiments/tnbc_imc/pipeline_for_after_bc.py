"""
@author: Tina Hsu
Processing QC with data after batch correction
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
from anndata import read_h5ad
import scanpy as sc
from src.preprocess import normalize
import umap
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    """
    Setting parameters for the codes.
    """
    # For the path to store the output files
    EXPERIMENT = "XP12446_after_bc"
    TYPE = "tnbc_imc"

    # Setting the paths for reading data and output figures
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Get the expression data (.csv)
    data = pd.read_csv(path_dir / "XP12446_after_batch_correction.csv")
    adata = read_h5ad(path_dir / "XP12446_clustering.h5ad")

    """
    UMAP
    """
    def check_batch_effect_umap(adata, exp_data, out_dir, name):
        print("umap")
        umap_set = umap.UMAP(n_components=2, random_state=42).fit_transform(exp_data)
        umap_df = pd.DataFrame(umap_set)
        umap_df = pd.concat([pd.DataFrame([x.split("_")[-2] for x in adata.obs.image_id]), umap_df], axis=1)
        umap_df.columns = ["Image", "X", "Y"]

        # Plot
        print("plotting")
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
        pd.DataFrame(np.array(umap_df.iloc[:, 1:])).to_csv(path_dir / "XP12446_umap_after_bc_embedding.csv", index=False)
        return umap_df

    print("Start doing batch effect checking")
    umap_df = check_batch_effect_umap(adata, data, out_dir, "after")

    """
    Expression based
    """
    from src.utils import get_coexpression_matrix
    r = get_coexpression_matrix(np.array(data), range(len(adata.var_names)))
    corr_df = pd.DataFrame(r, columns=["m1", "m2", "score"])

    corr_df.to_csv(path_dir / f"{EXPERIMENT}_exp_corr.csv", index=False)

    """
    Location based
    """
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

        """
        # Plotting neighbors
        from src.visualize import plot_neighbor_graph
        plot_neighbor_graph(p_adata, out_dir, figsize=(7, 7), dotsize=1, linewidth=0.2, name=unique_id[i])
        """

        # Get hotspot array for each marker column
        for j in range(len(p_adata.var_names)):
            p_adata.obs[f"{p_adata.var_names[j]}_hotspot"] = \
                get_hot_spots_graph(p_adata, col_name=p_adata.var_names[j])

        # p_adata.write(path_dir / f"{EXPERIMENT}_hotspots.h5ad")


        # Accumulator
        common_hotspots = []
        marker_combination_score = []
        # Find percentage of common hotspots between markers
        for a in combinations(range(len(p_adata.var_names)), 2):
            """
               cd3 cd68
            0   0   1
            1   1   1
            2   1   0
            3   0   0
            4   1   0
            """
            # If hotspot matches hotspot and non-hotspot matches non-hotspot
            common_count = sum(1 for u, v in zip(p_adata.obs[f"{p_adata.var_names[a[0]]}_hotspot"],
                                                 p_adata.obs[f"{p_adata.var_names[a[1]]}_hotspot"]) if u == v)
            common_pct = common_count / len(p_adata.obs)

            # If hotspot matches hotspot only
            col_1_hotspot_idx = np.where(p_adata.obs[f"{p_adata.var_names[a[0]]}_hotspot"] == 1)[0]
            col_2_hotspot_idx = np.where(p_adata.obs[f"{p_adata.var_names[a[1]]}_hotspot"] == 1)[0]
            intersect_hotspot_idx = set(col_1_hotspot_idx).intersection(col_2_hotspot_idx)

            col_1_pct = len(intersect_hotspot_idx) / len(col_1_hotspot_idx)
            col_2_pct = len(intersect_hotspot_idx) / len(col_2_hotspot_idx)

            common_hotspots.append([p_adata.var_names[a[0]], p_adata.var_names[a[1]],
                                    (col_1_pct + col_2_pct) / 2])

        scores.append(common_hotspots)







