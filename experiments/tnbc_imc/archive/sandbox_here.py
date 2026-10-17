import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    TYPE = "tnbc_imc"
    EXPERIMENT = "XP12446"

    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")

    exp_corr = pd.read_csv(path_dir / "XP12446_exp_corr.csv")
    loc_corr = pd.read_csv(path_dir / "XP12446_loc_corr.csv")

    intersection_corr = []
    for i in range(exp_corr.shape[0]):
        m1 = exp_corr.iloc[i, 0]
        m2 = exp_corr.iloc[i, 1]
        #avg = (exp_corr.iloc[i, 2] + loc_corr.iloc[i, 2]) / 2
        # intersection_corr.append([m1, m2, avg])
        intersection_corr.append([m1, m2, loc_corr.iloc[i, 2]])

    # pd.DataFrame(intersection_corr, columns=["m1", "m2", "score"]).to_csv(
    #     path_dir / f"{EXPERIMENT}_intersection_corr.csv")

    marker_list = ["CK5", "CK7", "CK8-18", "CK19", "panCK",
                   "B-Catenin", "H3K27me3", "EZH2", "Ki67", "EGFR",
                   "GATA3", "CA-IX", "Slug", "Prostaglandin", "p53",
                   "ER-D8H8", "ER-SP1", "PR-SP2", "erbB2",
                   "CD3e", "CD20", "CD45", "CD68",
                   "CD31", "SMA", "Vimentin", "vWF"]

    intersection_mtx = np.ones((27, 27))
    for i in range(len(intersection_corr)):
        u, v = intersection_corr[i][:2]

        intersection_mtx[int(u)][int(v)] = intersection_corr[i][2]
        intersection_mtx[int(v)][int(u)] = intersection_corr[i][2]

    intersection_df = pd.DataFrame(intersection_mtx, columns=['CK5', 'CK19', 'CK8-18', 'CD68', 'vWF', 'SMA',
                                                               'Vimentin', 'erbB2', 'CD3e', 'Slug', 'EZH2',
                                                               'Prostaglandin', 'p53', 'H3K27me3', 'PR-SP2',
                                                               'CD45', 'GATA3', 'CD20', 'B-Catenin', 'CA-IX', 'ER-D8H8',
                                                               'Ki67', 'EGFR', 'ER-SP1', 'CD31', 'CK7', 'panCK'],
                                    index=['CK5', 'CK19', 'CK8-18', 'CD68', 'vWF', 'SMA',
                                                               'Vimentin', 'erbB2', 'CD3e', 'Slug', 'EZH2',
                                                               'Prostaglandin', 'p53', 'H3K27me3', 'PR-SP2',
                                                               'CD45', 'GATA3', 'CD20', 'B-Catenin', 'CA-IX', 'ER-D8H8',
                                                               'Ki67', 'EGFR', 'ER-SP1', 'CD31', 'CK7', 'panCK'])

    intersection_df = intersection_df[marker_list]
    intersection_df = intersection_df.T
    intersection_df = intersection_df[marker_list]

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    sns.heatmap(intersection_df, cmap="coolwarm", ax=ax)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Marker", fontsize=8)
    plt.title("Correlation between markers", weight='bold', fontsize=8)

    plt.tight_layout()

    plt.savefig(out_dir / f"{EXPERIMENT}_heatmap_marker_correlations_loc_corr.png", dpi=500)
    plt.clf()

