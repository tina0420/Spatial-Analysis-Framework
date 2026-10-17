import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    """
    Parameter setting for the codes.
    """
    TYPE = "tnbc_imc"
    EXPERIMENT = "XP12446"

    SUB_TYPE = { 0: "epithelial",
                 1: "immune",
                 2: "stromal",
                 3: "undefined",
                 4: "all"}

    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")

    for i in SUB_TYPE.items():
        cell_type = i[1]
        exp_corr = pd.read_csv(path_dir / f"{EXPERIMENT}_{cell_type}_exp_corr.csv")
        loc_corr = pd.read_csv(path_dir / f"{EXPERIMENT}_{cell_type}_loc_corr.csv")




    exp_corr = pd.read_csv(path_dir / f"{EXPERIMENT}_exp_corr.csv")
    loc_corr = pd.read_csv(path_dir / f"{EXPERIMENT}_loc_corr.csv")

    combination = []
    if exp_corr.shape != loc_corr:
        print("Different shapes of the correlation datas.")
        print(f"{exp_corr}: " + str(exp_corr.shape))
        print(f"{loc_corr}: " + str(loc_corr.shape))
    else:
        for i in range(exp_corr.shape[0]):
            m1 = exp_corr.iloc[i, 0]
            m2 = exp_corr.iloc[i, 1]
            avg = (exp_corr.iloc[i, 2] + loc_corr.iloc[i, 2]) / 2
            combination.append([m1, m2, avg])





