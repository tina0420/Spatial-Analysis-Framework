import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from experiments.tnbc_imc.tnbc_imc_cliff_delta import (calculate_cliff_delta,
                                                       bootstrap_for_cliffs_delta_ci, plot_effect_size)
from src.visualize import COLOR_CODE
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu


if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    TMA = 'full_4_tma'

    # Data directory
    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in recurrence info
    patient_df = pd.read_csv(in_dir / "full_4_tma_cohort_patient_only_reset_index.csv")[
        ["TMA", "CID", 'Core img_1', 'Core img_2', 'relapse']]

    # Read in cohort only h5ad
    adata = read_h5ad(in_dir / "full_4_tma_cohort_patient_only.h5ad")

    # Compute cell type matrix
    ct_prop = []
    CLUSTER_KEY = 'layer2_classify'

    cell_ids = []
    cell_counts = []

    for i in range(patient_df.shape[0]):
        # Use first image, if NaN then use the second image
        img = patient_df.iloc[i, :]["Core img_1"]
        if pd.isna(img):
            img = patient_df.iloc[i, :]["Core img_2"]

        patient_obs = adata[(adata.obs.batch == patient_df.iloc[i, :]["TMA"]) &
                            (adata.obs['image_id'] == img)].obs[CLUSTER_KEY]

        cell_ids.append(patient_df.iloc[i, :]["TMA"] + "_" + img)
        cell_counts.append(patient_obs.value_counts())

    cell_counts = pd.concat(cell_counts, axis=1)
    cell_counts = cell_counts.fillna(0)
    cell_counts.columns = cell_ids

    # Transpose the df, patient by cell type, and rearrange order
    cell_counts = cell_counts.T
    cell_counts = cell_counts[['Luminal', 'Myoepithelial', 'Undefined epithelial', 'T cell', 'B cell', 'Macrophage',
                               'Undefined immune', 'Blood endothelial', 'Smooth muscle cell', 'Vimentin',
                               'Undefined stromal', 'Undefined']]

    # Get subset of cell counts, for epithelial
    e_arr = cell_counts.iloc[:, :3].to_numpy()
    e_arr = np.where(e_arr.sum(axis=1, keepdims=True) == 0, 0, e_arr / e_arr.sum(axis=1, keepdims=True))

    # Get subset of cell counts, for immune
    i_arr = cell_counts.iloc[:, 3:7].to_numpy()
    i_arr = np.where(i_arr.sum(axis=1, keepdims=True) == 0, 0, i_arr / i_arr.sum(axis=1, keepdims=True))

    # Get subset of cell counts, for stromal
    s_arr = cell_counts.iloc[:, 7:].to_numpy()
    s_arr = np.where(s_arr.sum(axis=1, keepdims=True) == 0, 0, s_arr / s_arr.sum(axis=1, keepdims=True))

    # Merge the three arrays back to the dataframe
    cell_proportions = pd.DataFrame(np.hstack([e_arr, i_arr, s_arr]), columns=cell_counts.columns,
                                    index=cell_counts.index)

    # Add relapse information of patients into the cell_proportions dataframe
    cell_proportions["relapse"] = patient_df.relapse.values

    # Calculate cliff's delta with the cell proportion data (there are some rows with 0 but including them here)
    CLUSTER_KEY = 'relapse'
    outcome = ['relapse', 'non-relapse']
    CI = 95

    delta_results = []

    for cell_type in cell_proportions.columns[:-1]:
        group_1 = cell_proportions.iloc[np.where(cell_proportions[CLUSTER_KEY] == outcome[0])[0], :][cell_type].values
        group_2 = cell_proportions.iloc[np.where(cell_proportions[CLUSTER_KEY] == outcome[1])[0], :][cell_type].values
        delta, (lower_ci, higher_ci) = bootstrap_for_cliffs_delta_ci(group1_list=group_1,
                                                                     group2_list=group_2,
                                                                     n_boot=1000, ci_num=CI, ran_seed=420)
        stat, pval = mannwhitneyu(group_1, group_2, alternative='two-sided')

        delta_results.append([cell_type, delta, lower_ci, higher_ci, pval])

    delta_results_df = pd.DataFrame(delta_results, columns=['cell_type', 'cliff', 'lower_ci', 'upper_ci', 'pvalue'])
    delta_results_df["adj_pvalue"] = multipletests(delta_results_df['pvalue'], method='fdr_bh')[1]

    # Plot effect size for the cliff's delta result
    COLOR_DICT = dict(zip(delta_results_df.cell_type.values, COLOR_CODE[:len(delta_results_df.cell_type.values)]))
    plot_effect_size(delta_results_df, "cliff", COLOR_DICT, out_dir, figsize = (5, 5), name="cell_type_prop")



