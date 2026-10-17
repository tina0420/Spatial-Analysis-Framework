"""
@author: Tina Hsu
Compare the cores across TMAs
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from anndata import read_h5ad
from src.visualize import plot_cell_expression_violin_split
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plr
import seaborn as sns


if __name__ == '__main__':
    # parameter setting
    TYPE = "tnbc_imc"
    EXPERIMENTS = ["XP12446", "XP13085",
                   "XP12813", "XP13083",
                   "XP13087", "XP13084",
                   "XP12814", "XP13086",
                   "XP12274"]

    # EXPERIMENTS = ["XP12274", "XP12446", "XP12813", "XP12814",
    #                "XP13083", "XP13084", "XP13085", "XP13086",
    #                "XP13087"]
    # EXPERIMENTS = ["XP13085"]
    exp_all = []
    # calculate the mean of marker expression in each TMA and store them in the dictionary
    out_dir = Path(f"output/{TYPE}")
    for xp in EXPERIMENTS:
        print(xp)
        in_dir = Path(f"data/{TYPE}/{xp}")

        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        adata = read_h5ad(in_dir / f"{xp}.h5ad")
        # marker_mean = np.mean(adata.X, axis=0)
        # exp_mean_dict[xp] = marker_mean
        marker_exp = adata.X
        exp_all = np.append(exp_all, marker_exp)


    exp_mean_df = pd.DataFrame.from_dict(exp_mean_dict).T
    exp_mean_df.columns = adata.var_names

    # box plot for the
    df = pd.DataFrame(data=np.random.random(size=(4, 4)), columns=['A', 'B', 'C', 'D'])

    plt.figure(figsize=(12, 5))
    sns.boxplot(x="variable", y="value", data=pd.melt(exp_mean_df))
    plt.xticks(rotation=90)
    plt.xlabel("marker")
    plt.ylabel("intensity")
    plt.tight_layout()
    plt.savefig(out_dir / "marker_expression_across_tma.png")
    plt.show()

