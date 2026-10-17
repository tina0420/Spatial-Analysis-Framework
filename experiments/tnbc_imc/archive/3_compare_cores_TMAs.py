"""
@author: Tina Hsu
Compare the cores in each TMA
"""
import os
from pathlib import Path
import numpy as np
from anndata import read_h5ad
from src.visualize import plot_cell_expression_violin_split
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels
from src.visualize import *

if __name__ == '__main__':
    # parameter setting
    TYPE = "tnbc_imc"
    EXPERIMENTS = ["XP12446", "XP13085",
                   "XP12813", "XP13083",
                   "XP13087", "XP13084",
                   "XP12814", "XP13086",
                   "XP12274"]

    for xp in EXPERIMENTS:
        if xp == "XP13087" or xp == "XP12446":
            continue
        else:
            # not merge perfect well, so
            if xp == "XP12813" or xp == "XP12814":
                # check the path
                in_dir = Path(f"data/{TYPE}/{xp}")
                out_dir = Path(f"output/{TYPE}/{xp}")
                if os.path.exists(out_dir) == 0:
                    os.makedirs(out_dir, exist_ok=True)

                adata = read_h5ad(in_dir / f"{xp}_merge_exp.h5ad")
            else:
                # check the path
                in_dir = Path(f"data/{TYPE}/{xp}")
                out_dir = Path(f"output/{TYPE}/{xp}")
                if os.path.exists(out_dir) == 0:
                    os.makedirs(out_dir, exist_ok=True)

                adata = read_h5ad(in_dir / f"{xp}.h5ad")

            print(".....")
            """
            Check marker expression distribution of all the ROIs in a TMA
            -> to check the marker expression across cores in a TMA
            """
            # all markers in one figure
            plot_cell_expression_violin_split(adata, out_dir, set_count=len(adata.var_names) + 1, rotation=90,
                                              figsize=(12, 5), name=f"{xp}")

            """
            Identify cells into epithelial, immune and stromal cells by doing GMM twice
            """
            ck_sum = np.sum(adata.X[:, [0, 1, 2, 4]], axis=1).reshape(-1, 1)  # CK 5, 7, 8-18, pan

            bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(ck_sum)

            bgm_labels = bgm.predict(ck_sum)
            update_labels(adata, cluster_key="cell_types", labels=bgm_labels)

            # check first GMM performance
            plot_zscore_cell_expression_heatmap(adata, adata.var_names,
                                                'cell_types', out_dir,
                                                name='epithelial_separated_cell_types')

            imm_sum = np.sum(adata[adata.obs.cell_types == 0].X[:, [19, 20, 21, 22]], axis=1).reshape(-1, 1)  # CD3e, CD20, CD45, CD68
            bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(imm_sum)
            bgm_labels = bgm.predict(imm_sum)

            # update labels
            labels = np.array(adata.obs.cell_types)
            new_labels = []
            j = 0
            # immune:0, epithelial: 1, stromal: 2
            for i in range(len(labels)):
                if labels[i] == 0:
                    if bgm_labels[j] == 0:
                        new_labels.append('stromal')
                        j += 1
                    elif bgm_labels[j] == 1:
                        new_labels.append('immune')
                        j += 1
                elif labels[i] == 1:
                    new_labels.append('epithelial')

            update_labels(adata, cluster_key="cell_types", labels=new_labels)
            plot_zscore_cell_expression_heatmap(adata, adata.var_names, 'cell_types', out_dir, rotation=90,
                                                name=f'{xp}_3_cell_types')
            # for proportion
            plot_cluster_cell_count(adata, "cell_types", out_dir, figsize=(3.5, 2), name=f"{xp}_3_cell_types")

