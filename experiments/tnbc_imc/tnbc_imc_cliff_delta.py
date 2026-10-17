"""
@author: Tina Hsu
Cliff's Delta and effect size plotting for TNBC IMC, currently for cell type proportions only
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from experiments.tnbc_imc.tnbc_imc_setup_analysis import LAYER2_CELL_TYPE_COLOR_DICT, ESI_CELL_TYPE_COLOR_DICT
from experiments.tnbc_imc.tnbc_imc_setup_analysis_e import *


# Calculate the target matrix for cliff delta (e.g. cell type proportions)
def calculate_cell_type_proportions(adata, cluster_key, unit_key, observation_key, cell_types, cell_type_len):
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


def run_cliffs_delta(calculate_df, cluster_key, observation_key, observation_order, stats_type, n_boot=1000, ci_num=95,
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


# calculate cliff's delta
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


def plot_effect_size(delta_results_df, cluster_key, calculate_key, observation_order, observation_counts, observation_dict,
                     out_dir, color_dict, figsize=(7, 7), name=None):
    """
    Plot effect size.
    :param delta_results_df: cliff delta result, type: pd.DataFrame
    :param cluster_key: indicate the column used in obs (e.g. cell type), type: str
    :param calculate_key: indicate the column representing calculation result to plot in obs (e.g. cliff_delta), type: str
    :param observation_order: indicate the column representing observation (outcome), type:str
    :param out_dir: output directory, type: Path
    :param color_dict: color dictionary for each category in cluster_key, type: dict
    :param figsize: figure size, type: tuple
    :param name: extra name for output file, type: str
    :return:
    """
    # Reorder the dataframe by cliff's delta values
    cal_df = delta_results_df.sort_values(by=calculate_key, ascending=False, inplace=False).reset_index(drop=True,
                                                                                                        inplace=False)
    plt.subplots(figsize=figsize)

    # Compute the value of confidence interval
    ci = [cal_df.iloc[::-1][calculate_key].values - cal_df.iloc[::-1]['lower_ci'].values,
          cal_df.iloc[::-1]['upper_ci'].values - cal_df.iloc[::-1][calculate_key].values]

    for i in range(cal_df.shape[0]):
        # Give the significant results (p_value<0.05 lightblue, otherwise gray
        if cal_df['adj_pvalue'].values[i] < 0.05:
            plt.errorbar(x=cal_df.iloc[::-1][calculate_key][i],
                         y=cal_df.iloc[::-1].index.values[i],
                         xerr=[[ci[0][i]], [ci[1][i]]],
                         color='black',
                         capsize=5,
                         linestyle='none',
                         elinewidth=1,
                         marker="o",
                         markersize=15,
                         mfc='lightblue',
                         mec='lightblue')
        else:
            plt.errorbar(x=cal_df.iloc[::-1][calculate_key][i],
                         y=cal_df.index.values[i],
                         xerr=[[ci[0][i]], [ci[1][i]]],
                         color='black',
                         capsize=5,
                         linestyle='none',
                         elinewidth=1,
                         marker="o",
                         markersize=15,
                         mfc='darkgray',
                         mec='darkgray')

    plt.axvline(x=0, linewidth=1.2, linestyle='--', color='gray')

    # Get the max and min values from ci
    x_left = int(min(cal_df.iloc[::-1]['lower_ci'].values) * 10) / 10.0
    x_right = int(max(cal_df.iloc[::-1]['upper_ci'].values) * 10) / 10.0

    if x_right == 1.0:
        x_right = 1.0 + (1.0 - x_left)

    # Set y-ticks and y-tick labels
    plt.xticks([x_left, 0, x_right], fontsize=13)
    plt.yticks(cal_df.iloc[::-1].index.values, cal_df.iloc[::-1][cluster_key], fontsize=13)
    plt.xlabel("Cliff's delta effect size", fontsize=14)

    # Remove the upper and right spines of the plot
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get the coordinates for better setting the texts
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Create a rectangle representing the color strip for each category
    color_strip_width = 0.05
    for i, category in enumerate(cal_df[cluster_key]):
        rect = patches.Rectangle((x_min, i - 0.4), color_strip_width, 0.98, linewidth=0,
                                 edgecolor='none', facecolor=color_dict[category])
        ax.add_patch(rect)

    # Add the adj-pvalue on the right side of figure
    ax.text(x_max, y_max + 0.2, "Adj. pval", va='center', ha='left', fontsize=16)
    for idx, pvalue in enumerate(cal_df['adj_pvalue']):
        if pvalue < 0.05:
            exp = int(f"{pvalue:e}".split('e')[1])
            mantissa = pvalue / (10 ** exp)
            formatted_text = f"${mantissa:.1f} \\times 10^{{{exp}}}$"
            ax.text(x_max, idx, formatted_text, va='center', ha='left', fontsize=15)
        else:
            ax.text(x_max, idx, f"{pvalue:.2f}", color='grey', va='center', ha='left', fontsize=15)

    # Add legend showing the circles
    legend_elements = [Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                              marker="o", markeredgecolor='black', markerfacecolor='darkgray', label='n.s.'),

                       Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                              marker="o", markeredgecolor='black', markerfacecolor='lightblue', label='p < 0.05')]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.5, -0.15))

    # Add arrows on the upper region to show
    ax.annotate('', xy=(x_min, (y_max)), xytext=(0, (y_max)),
                arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='blue'))
    # ax.text((0 + x_min) / 2, y_max + 0.45, observation_order[1], va='center', ha='center', fontsize=15)
    ax.text((0 + x_min) / 2, y_max + 0.66, f"{observation_dict[observation_order[1]]}\n(n={observation_counts[observation_order[1]]})",
            va='center', ha='center', fontsize=15)

    ax.annotate('', xy=(x_max, (y_max)), xytext=(0, (y_max)),
                arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='red'))
    # ax.text((0 + x_max) / 2, y_max + 0.45, observation_order[0], va='center', ha='center', fontsize=15)
    ax.text((0 + x_max) / 2, y_max + 0.66, f"{observation_dict[observation_order[0]]}\n(n={observation_counts[observation_order[0]]})",
            va='center', ha='center', fontsize=15)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_errorbar.png", dpi=500)
    else:
        plt.savefig(out_dir / "errorbar.png", dpi=500)

    plt.clf()


if __name__ == '__main__':
    # Example for real data
    """
    Starting with Anndata with expression in adata.X, image information, cell type annotations from different layers and 
    patient clinical outcome in adata.obs, and patient information stored in adata.uns['cohort_patient'].
    """
    # Set disease type and which experiment to process
    # TYPE = 'tnbc_imc'
    # PURPOSE = 'final'
    #
    # in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    # out_dir = Path(f"data/{TYPE}/{PURPOSE}/pCR")
    # if os.path.exists(out_dir) == 0:
    #     os.makedirs(out_dir, exist_ok=True)

    # FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # # Select for pre treatment
    # FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]
    #
    # FULL_ADATA = FULL_ADATA[FULL_ADATA.obs.chemo_type == "NAT"]

    """
    Structure of adata looks like:
    AnnData object with n_obs × n_vars
    obs: 'image_id', 'area', 'tma', 'image_key', 'layer1', 'layer2', 'patient_key', 'outcome'
    uns: 'ctdna_cohort'
    obsm: 'spatial'
    """
    # Calculate cell type matrix (img by cell types)
    # Set the columns storing cell type, patient id and outcome
    CLUSTER_KEY = 'layer_2'
    UNIT_KEY = 'CID'
    OBSERVATION_KEY = 'Recurrence status'

    # Cell type list need to write down in an order with the same ESI cell type together
    CELL_TYPE_LIST = ['CK5+ Basal',  # Epithelial
                      'CK7+ Luminal',
                      'CK8/18+ Luminal',
                      'CK+ Epithelial',
                      'T-cell',  # Immune
                      'B-cell',
                      'Macrophage',
                      'Endothelial',  # Stromal
                      'SMA+ Stromal',
                      'Stromal'
                      ]
    #
    # # CELL_TYPE_LIST = ['B',
    # #                   'CK5_Basal',
    # #                   'CK5_CK7_Mp',
    # #                   'CK7_Lum',
    # #                   'CK8-Lum_Myo',
    # #                   'CK8_Lum',
    # #                   'Fibro',
    # #                   'H3K27me3_Epi',
    # #                   'T_B',
    # #                   'T_Mp_Endo']
    #
    ESI_CELL_TYPE_LENGTH = [4, 6]

    # Set the order of observation (the right one then the left one on the figure)
    OBSERVATION_KEY_ORDER = [1, 0]

    # IMG_KEY is for the dataset when there's more than one image from the patient (to select one image)
    ct_mtx = calculate_cell_type_proportions(adata=FULL_ADATA,
                                             cluster_key=CLUSTER_KEY,
                                             unit_key=UNIT_KEY,
                                             observation_key=OBSERVATION_KEY,
                                             cell_types=CELL_TYPE_LIST,
                                             cell_type_len=ESI_CELL_TYPE_LENGTH)

    # Run cliff delta, can change the numbers for specific confidence interval
    delta_df = run_cliffs_delta(calculate_df=ct_mtx,
                                cluster_key=CLUSTER_KEY,
                                observation_key=OBSERVATION_KEY,
                                observation_order=OBSERVATION_KEY_ORDER,
                                stats_type="cliff_delta",
                                n_boot=1000,
                                ci_num=95,
                                ran_seed=57)

    # count pid based on observation
    df = FULL_ADATA.obs[[OBSERVATION_KEY, "CID"]]

    # Plot effect size the see the result of cliff's delta
    plot_effect_size(delta_results_df=delta_df,
                     cluster_key=CLUSTER_KEY,
                     calculate_key='cliff_delta',
                     observation_order=OBSERVATION_KEY_ORDER,
                     observation_counts=pd.Series(
                         dict(zip(df.CID.values, df[OBSERVATION_KEY].values)).values()).value_counts(),
                     observation_dict={0: 'No recurrence', 1: 'Recurrence'},
                     color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
                     out_dir=OUT_DIR,
                     figsize=(8, 8),
                     name="Recurrence status_cliff_cell_type_cid")
