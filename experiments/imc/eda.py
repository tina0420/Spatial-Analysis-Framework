"""
Exploratory Data Analysis for spatial-omics data on expression and proportion comparison.
@author: Tina Hsu
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import pandas as pd
from src.annotate import update_labels
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, PCR_TF_COLOR_DICT


def compare_exp_distribution():
    pass


def compute_cell_type_proportion(adata, cluster_key, unit_key, cell_type, population_type, observation_key):
    proportions = []
    for pid in np.unique(adata.obs[unit_key]):
        # For all cells
        if population_type == 'all':
            p_obs = adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[cluster_key]
            if cell_type in p_obs.value_counts():
                prop = p_obs.value_counts()[cell_type] / len(p_obs)
            else:
                prop = 0
            proportions.append([pid,
                                adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[observation_key].values[0],
                                prop])

        # For specific cell types only, e.g. Epithelial, Immune, Stromal
        else:
            p_obs = adata[np.where((adata.obs[unit_key] == pid) &
                                   (adata.obs['esi'] == population_type))[0], :].obs[cluster_key]
            if cell_type in p_obs.value_counts():
                prop = p_obs.value_counts()[cell_type] / len(p_obs)
            else:
                prop = 0

            # List with patient, observation, cell proportion
            proportions.append([pid,
                                adata[np.where(adata.obs[unit_key] == pid)[0], :].obs[observation_key].values[0],
                                prop])

    # Store the calculation result in adata.uns
    proportions_df = pd.DataFrame(proportions, columns=[unit_key, observation_key, f"{cell_type}_prop"])
    adata.uns[f"{cell_type}_prop"] = proportions_df

    return proportions_df




