"""
@author: Tina Hsu
Spatial functions.
"""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from analysis_setup import *


def run_shannon_entropy(adata, unit_key, cluster_key):
    counts = adata.obs.pivot_table(
        index=unit_key,
        columns=cluster_key,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)

    # Shannon diversity
    score_shannon = []
    for i in range(prop_df.shape[0]):
        score = __compute_spatial_entropy(prop_df.iloc[i, :])
        score_shannon.append(score)

    return score_shannon


def __compute_spatial_entropy(p_list):
    """
    Compute spatial entropy for each cell
    :param p_list: probabilities of all cell types among the neighbors of each cell
    :return: entropy value
    """
    entropy = (-1) * sum([x * np.log(x) if x > 0 else 0 for x in p_list])
    return entropy


def compare_effect_size(adata, cluster_key, unit_key, observation_key, observation_order, cell_type_list, cell_type_len):
    """
    Run Cliff Delta for effect size comparison.
    :param adata: Anndata object.
    :param cluster_key: the column in
    :param unit_key:
    :param observation_key:
    :param observation_order:
    :param cell_type_list:
    :param cell_type_len:
    :return:
    """
    ct_mtx = __calculate_cell_type_proportions(adata=adata,
                                               cluster_key=cluster_key,
                                               unit_key=unit_key,
                                               observation_key=observation_key,
                                               cell_types=cell_type_list,
                                               cell_type_len=cell_type_len)

    delta_df = __run_cliffs_delta(calculate_df=ct_mtx,
                                  cluster_key=cluster_key,
                                  observation_key=observation_key,
                                  observation_order=observation_order,
                                  stats_type="cliff_delta",
                                  n_boot=1000,
                                  ci_num=95,
                                  ran_seed=57)

    return ct_mtx, delta_df


def __calculate_cell_type_proportions(adata, cluster_key, unit_key, observation_key, cell_types, cell_type_len):
    """
    Calculate cell type proportions for cliff delta calculation.
    :param adata: anndata
    :param cluster_key: indicate the column used in obs, type: str
    :param unit_key: indicate the column representing the unit in obs, type: str
    :param observation_key: indicate the column representing observation, type:str
    :param cell_types: list of cell types, type: list[str]
    :param cell_type_len: list includes the length of bigger categories of cell type, type: list[int]
    :return: cell type proportions matrix, type:pd.DataFrame
    """
    # Store the proportions
    prop = []

    cell_ids = []
    observations = []
    cell_counts = []

    for pid in set(adata.obs[unit_key]):
        # Get the cluster_key information (cell type) of the patient
        patient_obs = adata[adata.obs[unit_key] == pid].obs[[cluster_key, observation_key]]

        # Compute cell counts
        cell_ids.append(pid)
        observations.append(patient_obs[observation_key].values[0])
        cell_counts.append(patient_obs[cluster_key].value_counts())

    cell_counts = pd.concat(cell_counts, axis=1)
    cell_counts = cell_counts.fillna(0)
    cell_counts.columns = cell_ids

    # Transpose the df, patient by cell type, and rearrange order
    cell_counts = cell_counts.T
    cell_counts = cell_counts[cell_types]

    # Get subset of cell counts for cell types one by one
    ct_prop = {}
    start_idx = 0
    for i in range(len(cell_type_len)):
        arr = cell_counts.iloc[:, start_idx:(start_idx + cell_type_len[i])].to_numpy()
        arr = np.where(arr.sum(axis=1, keepdims=True) == 0, 0, arr / arr.sum(axis=1, keepdims=True))
        ct_prop[i] = arr

        start_idx = cell_type_len[i]

    # Merge the three arrays back to the dataframe
    cell_proportions = pd.DataFrame(np.hstack([ct_prop[i] for i in range(len(cell_type_len))]),
                                    columns=cell_counts.columns, index=cell_counts.index)

    # Add observation information of patients into the cell_proportions dataframe
    cell_proportions[observation_key] = observations

    return cell_proportions


def __run_cliffs_delta(calculate_df, cluster_key, observation_key, observation_order, stats_type, n_boot=1000,
                       ci_num=95,
                       ran_seed=57):
    """
    Running Cliff's Delta.
    :param calculate_df: data prepared for cliff delta (e.g. cell type proportions), type: pd.DataFrame
    :param cluster_key: indicate the column used in obs, type: str
    :param observation_key: indicate the column representing observation (outcome), type:str
    :param n_boot: number for bootstrapping, type: int
    :param ci_num: number of confident interval, type: float
    :param ran_seed: randon seed number used in bootstrap, type:int
    :return: matrix with cliff delta result, type:pd.DataFrame
    """
    delta_results = []
    observation_1 = observation_order[0]
    observation_2 = observation_order[1]

    if stats_type == "cliff_delta":
        for cell_type in calculate_df.columns[:-1]:
            # Separate two subgroups based on observations
            group_1 = calculate_df.iloc[np.where(calculate_df[observation_key] == observation_1)[0], :][
                cell_type].values
            group_2 = calculate_df.iloc[np.where(calculate_df[observation_key] == observation_2)[0], :][
                cell_type].values

            # Run bootstrap to get confidence interval
            np.random.seed(ran_seed)
            deltas = []
            for _ in range(n_boot):
                sample_1 = np.random.choice(group_1, size=len(group_1), replace=True)
                sample_2 = np.random.choice(group_2, size=len(group_2), replace=True)
                deltas.append(__calculate_cliff_delta(sample_1, sample_2))

            lower_ci = np.percentile(deltas, (100 - ci_num) / 2)
            upper_ci = np.percentile(deltas, 100 - (100 - ci_num) / 2)

            stat, pval = mannwhitneyu(group_1, group_2, alternative='two-sided')

            delta_results.append([cell_type, np.mean(deltas), lower_ci, upper_ci, pval])

        delta_results_df = pd.DataFrame(delta_results,
                                        columns=[cluster_key, 'cliff_delta', 'lower_ci', 'upper_ci', 'pvalue'])
        delta_results_df["adj_pvalue"] = multipletests(delta_results_df['pvalue'], method='fdr_bh')[1]
    elif stats_type == "odds_ratio":
        for cell_type in calculate_df.columns[:-1]:
            # Separate two subgroups based on observations
            group_1 = calculate_df.iloc[np.where(calculate_df[observation_key] == observation_1)[0], :][
                cell_type].values
            group_2 = calculate_df.iloc[np.where(calculate_df[observation_key] == observation_2)[0], :][
                cell_type].values

            # Run bootstrap to get confidence interval
            np.random.seed(ran_seed)
            deltas = []
            for _ in range(n_boot):
                sample_1 = np.random.choice(group_1, size=len(group_1), replace=True)
                sample_2 = np.random.choice(group_2, size=len(group_2), replace=True)
                deltas.append(__calculate_odds_ratio(sample_1, sample_2))

            lower_ci = np.percentile(deltas, (100 - ci_num) / 2)
            upper_ci = np.percentile(deltas, 100 - (100 - ci_num) / 2)

            stat, pval = mannwhitneyu(group_1, group_2, alternative='two-sided')

            delta_results.append([cell_type, np.mean(deltas), lower_ci, upper_ci, pval])

        delta_results_df = pd.DataFrame(delta_results,
                                        columns=[cluster_key, 'cliff_delta', 'lower_ci', 'upper_ci', 'pvalue'])
        delta_results_df["adj_pvalue"] = multipletests(delta_results_df['pvalue'], method='fdr_bh')[1]

    return delta_results_df


def __calculate_cliff_delta(group1_list, group2_list):
    """
    Calculate cliff delta
    :param group1_list: list of the group 1, type: list
    :param group2_list: list of the group 2, type: list
    :return: delta value, type: float
    """
    more = less = 0
    for x in group1_list:
        for y in group2_list:
            if x > y:
                more += 1
            elif x < y:
                less += 1

    delta = (more - less) / (len(group1_list) * len(group2_list))

    return delta


def __calculate_odds_ratio(group1_list, group2_list):
    """
    Calculate odds ratio
    :param group1_list: list of the group 1, type: list
    :param group2_list: list of the group 2, type: list
    :return: ratio value, type: float
    """
    a, c = group1_list
    b, d = group2_list

    if 0 in [a, b, c, d]:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    return (a * d) / (b * c)


def quantify_tumour_infiltrate_lymphocyte():
    pass


def quantify_immune_impact():
    pass
