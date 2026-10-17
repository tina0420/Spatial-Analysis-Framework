"""
@author: Tina Hsu
Spatially resolved info for quality control.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import read_h5ad
from itertools import combinations
from src.preprocess import read_neighbors
from src.spatial_stats import get_hot_spots_graph

if __name__ == '__main__':
    # Which run
    RUN = "tnbc_imc"
    EXPERIMENT = "XP12446"

    # Set paths
    path_dir = Path(f"data/{RUN}/{EXPERIMENT}")
    out_dir = Path(f"figures/{RUN}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read AnnData
    adata = read_h5ad(path_dir / f"{EXPERIMENT}_clustering.h5ad")

    # Get unique ID
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

        #p_adata.write(path_dir / f"{EXPERIMENT}_hotspots.h5ad")

    #     """
    #     # Plotting obs
    #     obs_out_dir = out_dir / f"{unique_id[i].split('_')[-2]}"
    #     if os.path.exists(obs_out_dir) == 0:
    #         os.makedirs(obs_out_dir, exist_ok=True)
    #
    #     from src.visualize import plot_observation_distribution
    #     plot_observation_distribution(p_adata,
    #                                   obs_key=f"{p_adata.var_names[j]}_hotspot",
    #                                   out_dir=obs_out_dir,
    #                                   dotsize=7,
    #                                   name=f"{p_adata.var_names[j]}_hotspot")
    #     """

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
                                    (col_1_pct + col_2_pct)/2])

        scores.append(common_hotspots)

    m1_m2 = np.array(scores[0])[:, :2]
    m3 = []
    for i in range(len(scores)):
        m3.append(np.array([float(j) for j in np.array(scores[i])[:, 2]]))
    m3 = np.array(m3).mean(axis=0)
    # np.concatenate([m1_m2, m3], axis=0)
    pd.concat([pd.DataFrame(m1_m2, columns=["m1", "m2"]), pd.DataFrame(m3, columns=["score"])], axis=1).to_csv(path_dir / f"{EXPERIMENT}_loc_corr.csv", index=False)

        # Output df, from here we can set rules to decide the spatial correlation between markers
        # hotspot_df = pd.DataFrame(common_hotspots, columns=["m1", "m2", "common_pct", "m1_pct", "m2_pct"])
        # hotspot_df = hotspot_df.sort_values(by=["common_pct"], ascending=False)
        #
        # hotspot_df.to_csv(path_dir / f"{unique_id[i].split('_')[-2]}_hotspot.csv", index=False)
