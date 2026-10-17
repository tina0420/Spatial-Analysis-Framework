"""
@author: Tina Hsu
Quick test for effect size.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import statsmodels.api as sm
# from sklearn.linear_model import LogisticRegression
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


def calculate_odds_ratio(adata, cluster_key, values, condition, ci_number):
    """
    Calculating odds ratio
    :param adata: Anndata object
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param values: values to calculate odds ratio, type: np.array
    :param condition: indicate which condition key to use in obs, type: str
    :param ci_number: percentage number of confidence interval, type: int
    :return: calculation result, type: pd.DataFrame
    """
    cal_result = pd.DataFrame(columns=["Odds Ratio", "CI_low", "CI_high", "pval"])
    for type in np.unique(adata.obs[cluster_key]):
        # independent variables
        x_train = values[np.where(adata.obs[cluster_key] == type)[0]]
        if len(x_train) != len(np.where(adata.obs[cluster_key] == type)[0]):
            print("Values should in the same length to adata.obs.index!")
        y_train = adata.obs.iloc[np.where(adata.obs.cell_type == type)[0], :].loc[:, condition].values

        # run logistic regression model
        lr_model = sm.Logit(y_train, x_train).fit()
        params = lr_model.params
        conf = lr_model.conf_int(alpha=(1 - ci_number * 0.01))
        odds_ratios = np.exp(params)
        conf_intervals = np.exp(conf)
        pvalue = lr_model.pvalues
        cal_result.loc[type] = [odds_ratios[0], conf_intervals[0][0], conf_intervals[0][1], pvalue[0]]

    cal_result = cal_result.sort_values(by=['Odds Ratio'], ascending=False)

    return cal_result


# def calculate_cliffs_delta(adata, cluster_key, situation_key, ci_number):
#     """
#     Calculating cliff's delta effect size.
#     :param adata: Anndata
#     :param cluster_key: indicate which cluster key to use in obs, type: str
#     :param situation_key: indicate which situation key to use in obs, type: strs
#     :param ci_number: percentage number of confidence interval, type: int
#     :return: cliff's delta effect size result, type: pd.DataFrame
#     """
#     cliff_list = []
#     for type in pd.unique(adata.obs[cluster_key]):
#         print("_____________________________________")
#         print(type)
#         group_a = adata.obs[adata.obs[situation_key] == 0].values
#         group_b = adata.obs[adata.obs[situation_key] == 1].values
#
#     cal_result = pd.DataFrame(cliff_list, columns=[])
#
#     return cal_result


def plot_effect_size(calculate_df, method, ci_number, condition_name, figsize=(7, 7), name=None):
    """
    Effect size plotting
    :param calculate_df: calculated result form the method, type:pd.DataFrame
    :param method: method that did for the calculation, type: str
    :param ci_number: percentage number of confidence interval, type: int
    :param condition_name: names of the conditions (with the order of 0 to 1), type:list
    :param figsize: figure size, type: tuple
    :param name: name of the png, type: str
    :return:
    """
    plt.subplots(figsize=(7, 7))

    # define significant threshold
    p_threshold = round(1-ci_number/100, 2)
    for idx, row in calculate_df.iterrows():
        mfc = 'lightblue' if row['pval'] < p_threshold else 'darkgray'
        mec = 'black' if row['pval'] < p_threshold else 'grey'
        ecolor = 'black' if row['pval'] < p_threshold else 'grey'

        plt.errorbar(row[method], idx,
                     xerr=[[row[method] - row['CI_low']], [row['CI_high'] - row[method]]],
                     marker='o',
                     ecolor=ecolor,
                     capsize=5,
                     linestyle='none',
                     elinewidth=1,
                     markersize=15,
                     mfc=mfc,
                     mec=mec
                     )

    plt.axvline(x=1, linewidth=1.2, linestyle='--', color='lightgrey')

    # xticks setting
    x_left = int(min(calculate_df.iloc[::-1]['CI_low'].values) * 10) / 10.0
    x_right = int(max(calculate_df.iloc[::-1]['CI_high'].values) * 10) / 10.0

    if x_right == 1.0:
        x_right = 1.0 + (1.0 - x_left)

    plt.xticks([x_left, 1, x_right], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(method, fontsize=15)

    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # remove the upper and right spines of the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # create a rectangle representing the color strip for each category
    color_strip_width = (1.0-x_min)/10
    for i, category in enumerate(calculate_df.index):
        rect = patches.Rectangle((x_min, i - 0.4), color_strip_width, 0.98, linewidth=0,
                                 edgecolor='none', facecolor=COLORS[i])
        ax.add_patch(rect)

    # add p-value on the right side
    ax.text(x_max, y_max + 0.2, "Adj. pval", va='center', ha='left', fontsize=18)
    for idx, pvalue in enumerate(calculate_df['pval']):
        if pvalue < p_threshold:
            exp = int(f"{pvalue:e}".split('e')[1])
            mantissa = pvalue / (10 ** exp)
            formatted_text = f"${mantissa:.1f} \\times 10^{{{exp}}}$"
            ax.text(x_max, idx, formatted_text, va='center', ha='left', fontsize=15)
        else:
            ax.text(x_max, idx, f"{pvalue:.2f}", color='grey', va='center', ha='left', fontsize=15)

    # show the different conditions
    ax.annotate('', xy=(x_min, (y_max)), xytext=(1, (y_max)),
                arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='blue'))
    ax.text((1 + x_min) / 2, y_max + 0.45, condition_name[0], va='center', ha='center', fontsize=18)

    ax.annotate('', xy=(x_max, (y_max)), xytext=(1, (y_max)),
                arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='red'))
    ax.text((1 + x_max) / 2, y_max + 0.45, condition_name[1], va='center', ha='center', fontsize=18)

    # legend for the forest plot
    legend_elements = [Line2D([0], [0], linestyle="-", color='gray', gapcolor='gray',
                              marker="o", markeredgecolor='gray', markerfacecolor='darkgray', label='n.s.'),

                       Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                              marker="o", markeredgecolor='black', markerfacecolor='lightblue',
                              label='p < '+str(p_threshold))]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.5, -0.15), frameon=False)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_effect_size.png", dpi=500)
    else:
        plt.savefig(out_dir / "effect_size.png", dpi=500)
    plt.clf()


if __name__ == '__main__':
    # path setting
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"
    print(f"Processing {EXPERIMENT}")

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(out_dir / f"{EXPERIMENT}.h5ad")

    # parameters setting
    CLUSTER_KEY = "cell_type"
    VALUES = np.array(adata.obs["test_val"])
    SITUATION = "outcome"
    CI_NUMBER = 95
    METHOD = "Odds Ratio"
    CONDITION_NAME = ["Duct", "Lobule"]
    NAME = "odds_ratio_plot_testing"

    # can get from src.visualize
    COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
                  '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
                  '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
                  '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
                  '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
                  '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909'
                  ]
    COLORS = dict(zip(range(len(COLOR_CODE)), COLOR_CODE))

    # example run
    plot_effect_size(calculate_odds_ratio(adata,
                                          CLUSTER_KEY,
                                          np.array(adata.obs["test_val"]),
                                          SITUATION,
                                          CI_NUMBER),
                     METHOD,
                     CI_NUMBER,
                     CONDITION_NAME,
                     figsize=(7, 7), name=NAME)

