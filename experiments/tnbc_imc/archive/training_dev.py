"""
@author: Tina Hsu
Functions developed for Sakura
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

"""
# to-do:
(calculation)
1. Cliff's Delta effect size (whole)

There's a problem that which python package or function is better to run cliff's delta effect size
One is cliff-delta, the other is cliff.delta


2. Odds Ratio (new parameter for setting the number of CI)

(visualization)
1. new parameters of (1) [the names of situations] 
                     (2) name of the figure
2. set the fixed scales of the x-axis


# structure:
1. 2 helper fn for calculating the values by OR, Cliff's: odds_ratio(), cliffs_delta()
2. 1 visualization fn to plot the result: plot_effect_size()

"""


def odds_ratio(adata, cluster_key, annot_key, control_key):
    """
    Calculating odds ratio
    :param adata: Anndata object
    :param cluster_key: Indicate which cluster key to use in obs, type: str
    :param annot_key: Annotation of the situations, type: str
    :param control_key: Column names for the independent variables, type: str
    :return: Odds ratio info, type: pd.DataFrame
    """
    cal_results = pd.DataFrame()

    # calculating odds ratio through each cell type
    for type in np.unique(adata.obs[cluster_key]):
        # independent variables
        x_train = adata.obs.iloc[np.where(adata.obs.cell_type == type)[0], :].loc[:, control_key]
        # column for the situations
        y_train = adata.obs.iloc[np.where(adata.obs.cell_type == type)[0], :].loc[:, annot_key]
        logi_model = sm.Logit(y_train, x_train).fit()
        params = logi_model.params
        conf = logi_model.conf_int()
        conf['odds ratio'] = params
        conf = pd.DataFrame(np.exp(conf))
        conf['pvalues'] = logi_model.pvalues
        conf.index = [type]
        cal_results = pd.concat([cal_results, conf.iloc[0, :]], axis=1)

    cal_results = cal_results.T
    cal_results.columns = ['2.5%', '97.5%', 'Odds Ratio', 'pval']
    cal_results = cal_results.sort_values(by=['Odds Ratio'])

    return cal_results


def effect_size_plot(adata, method, cluster_key, annot_key, control_key):
    if method == "odds_ratio":
        or_results_df = odds_ratio(adata, cluster_key, annot_key, control_key)

    plt.subplots(figsize=(7, 7))
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for idx in range(or_results_df.shape[0]):
        if or_results_df['pval'][idx] < 0.05:
            pass
        else:
            print(idx)
            plt.errorbar(x=or_results_df['Odds Ratio'][idx],
                         y=or_results_df.index.values[idx],
                         xerr=[or_results_df['Odds Ratio'][idx] - or_results_df['2.5%'].values[idx],
                               or_results_df['97.5%'].values[idx] - or_results_df['Odds Ratio'][idx]],
                         color='darkgray',
                         capsize=5,
                         linestyle='none',
                         elinewidth=1,
                         marker="o",
                         markersize=15,
                         mfc='gray',
                         mec='gray')

            ax.text(x_max, idx, f"{or_results_df['pval']:.2f}", color='grey', va='center', ha='left', fontsize=15)

            ci = [or_results_df.iloc[::-1]['Odds Ratio'] - or_results_df.iloc[::-1]['2.5%'].values,
                  or_results_df.iloc[::-1]['97.5%'].values - or_results_df.iloc[::-1]['Odds Ratio']]

            plt.errorbar(x=or_results_df.iloc[::-1]['Odds Ratio'],
                         y=or_results_df.iloc[::-1].index.values,
                         xerr=ci,
                         color='black',
                         capsize=5,
                         linestyle='none',
                         elinewidth=1,
                         marker="o",
                         markersize=15,
                         mfc='lightblue',
                         mec='lightblue')

            plt.axvline(x=1, linewidth=1.2, linestyle='--', color='gray')

            plt.tight_layout()
            plt.savefig(out_dir / "errorbar.png", dpi=500)

            x_left = int(min(or_results_df.iloc[::-1]['2.5%'].values) * 10) / 10.0
            x_right = int(max(or_results_df.iloc[::-1]['97.5%'].values) * 10) / 10.0

            if x_right == 1.0:
                x_right = 1.0 + (1.0 - x_left)

            plt.xticks([x_left, 1, x_right], fontsize=15)

            plt.yticks(fontsize=15)
            plt.xlabel("Odds Ratio", fontsize=15)

            # remove the upper and right spines of the plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Create a rectangle representing the color strip for each category

            color_strip_width = 0.025
            for i, category in enumerate(or_results_df.index):
                rect = patches.Rectangle((x_min, i - 0.4), color_strip_width, 0.98, linewidth=0,
                                         edgecolor='none', facecolor=COLORS[i])
            ax.add_patch(rect)

            # add the p-value on the right side of figure

            ax.text(x_max, y_max + 0.2, "Adj. pval", va='center', ha='left', fontsize=18)
            for idx, pvalue in enumerate(or_results_df['pval']):
                if pvalue < 0.05:
            exp = int(f"{pvalue:e}".split('e')[1])
            mantissa = pvalue / (10 ** exp)
            # formatted_text = f"{mantissa:.1f} × 10^{exp}"
            formatted_text = f"${mantissa:.1f} \\times 10^{{{exp}}}$"
            ax.text(x_max, idx, formatted_text, va='center', ha='left', fontsize=15)
            else:
            ax.text(x_max, idx, f"{pvalue:.2f}", color='grey', va='center', ha='left', fontsize=15)

            # show the sides of different conditions, ex: RD or Relapse, Duct or Lobule
            ax.annotate('', xy=(x_min, (y_max)), xytext=(1, (y_max)),
                        arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='blue'))
            ax.text((1 + x_min) / 2, y_max + 0.45, "Duct", va='center', ha='center', fontsize=18)

            ax.annotate('', xy=(x_max, (y_max)), xytext=(1, (y_max)),
                        arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='red'))
            ax.text((1 + x_max) / 2, y_max + 0.45, "Lobule", va='center', ha='center', fontsize=18)

            # other information shown on the figure

            ax.text(x_min - 0.3, y_max + 0.6, "Paired", va='center', ha='center', fontsize=18)
            ax.text(x_min - 0.3, y_max + 0.2, f"(n = {str(adata.X.shape[0])})", va='center', ha='center', fontsize=18)

            legend_elements = [Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                                      marker="o", markeredgecolor='black', markerfacecolor='darkgray', label='n.s.'),

                               Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                                      marker="o", markeredgecolor='black', markerfacecolor='lightblue',
                                      label='p < 0.05')]
            ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.5, -0.15), frameon=False)

            plt.tight_layout()
            # plt.show()
            plt.savefig(out_dir / "effect_size_plot.png", dpi=500)



def cliffs_delta_cal(adata, cluster_key, outcome, target_val):
    cliffs = []
    for cluster, sub_df in adata.obs.groupby(cluster_key):
        situation_0 = sub_df[sub_df[outcome] == 0][target_val]
        situation_1 = sub_df[sub_df[outcome] == 1][target_val]
        delta, _ = cliffs_delta(situation_0, situation_1)
        _, mannwhitney_p_value = mannwhitneyu(situation_0, situation_1, alternative='two-sided')
        print(cluster)
        print(delta)
        print(mannwhitney_p_value)
        print("---------------------------------------------------------------")
        cliffs.append([cluster, delta, mannwhitney_p_value])

    cliff_df = pd.DataFrame(cliffs, columns=["cell_type", "delta", "pval"]).sort_values(by="delta").reset_index(drop=True)

    return cliff_df

df = cliff.delta()


if __name__ == "__main__":
    """
    Testing start.(with .h5ad) 
    """
    # path and file
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"
    print(f"Processing {EXPERIMENT}")

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # adata = read_h5ad(out_dir / f"{EXPERIMENT}.h5ad")
    adata = read_h5ad(out_dir / f"{EXPERIMENT}_subset_fortest.h5ad")

    METHODS = "odds_ratio"
    # column names of cell types, target variable, and categories of two situations
    # cell types
    CLUSTER_KEY = "cell_type"
    # two situations
    ANNOTATION = "outcome"
    # target variable
    OBSERVATION = "test_val"

    COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
                  '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
                  '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
                  '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
                  '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
                  '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909'
                  ]

    calculation_result = odds_ratio(adata=adata, cluster_key=CLUSTER_KEY, annot_key=ANNOTATION,
                                    control_key=OBSERVATION)
    effect_size_plot(results_df=calculation_result,
                     method=METHODS,
                     cluster_key=CLUSTER_KEY,
                     annot_key=ANNOTATION,
                     control_key=OBSERVATION
                     )

    or_results_df = odds_ratio(adata, 'cell_type', 'outcome', 'test_val')

    """
    Testing end.
    """

    """
    Function development start.
    """

    """
    Parameter setting for the codes.
    """
    # Path to store the output files
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"
    print(f"Processing {EXPERIMENT}")

    # Setting the paths for reading data and output figures
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(out_dir / f"{EXPERIMENT}_subset_fortest.h5ad")

    # for test
    # to_add = list(range(1503))
    # adata.obs["test_val"] = to_add
    to_add = [random.randint(0, 100) for _ in range(1503)]
    adata.obs["test_val"] = to_add
    to_add = [random.choice([0, 1]) for _ in range(1503)]
    adata.obs["outcome"] = to_add

    """
    Functions for the calculation
    """


def cliffs_delta_calculations(adata, cluster_key, annot_key, control_key):

    return adata


"""
Functions for the calculation
"""

COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
              '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909'
              ]

COLORS = dict(zip(range(len(COLOR_CODE)), COLOR_CODE))
# colors_map = dict(zip([or_results_df.index], COLOR_CODE[:or_results_df.index]))
colors_map = {}
for i in range(or_results_df.shape[0]):
    colors_map[or_results_df.index[i]] = COLOR_CODE[i]


def plot_effect_size(adata, cluster_key, annot_key, control_key, method, fig_size=(12, 9), name=None):
    print(method)
    if method == "odds_ratio":
        cal_df = odds_ratio(adata, cluster_key, annot_key, control_key)

    elif method == "cliffs_delta":
        cal_df = cliffs_delta(adata, cluster_key, annot_key, control_key)
    else:
        print("Please use the method in [odds_ratio, cliffs_delta].")

    # plotting
    print("Plotting")
    plt.figure(figsize=fig_size)
    ci = [cal_df.iloc[::-1]['Odds Ratio'] - cal_df.iloc[::-1]['2.5%'].values,
          cal_df.iloc[::-1]['97.5%'].values - cal_df.iloc[::-1]['Odds Ratio']]

    plt.errorbar(x=cal_df.iloc[::-1]['Odds Ratio'],
                 y=cal_df.iloc[::-1].index.values,
                 xerr=ci,
                 color='black',
                 capsize=3,
                 linestyle='none',
                 elinewidth=1,
                 marker="o",
                 markersize=5,
                 mfc='red',
                 mec='red')

    # line on x=1
    # plt.axvline(x=1, linewidth=0.8, linestyle='--', color='black')
    plt.axvline(x=1, linewidth=0.8, linestyle='-', color='gray')

    plt.xlabel(method)
    plt.ylabel(cluster_key)
    plt.tight_layout()

    plt.savefig(out_dir / f"{name}_method.png", dpi=500)
    plt.clf()


# for plotting test with the data: cal_results
plt.subplots(figsize=(7, 7))

ci = [or_results_df.iloc[::-1]['Odds Ratio'] - or_results_df.iloc[::-1]['2.5%'].values,
      or_results_df.iloc[::-1]['97.5%'].values - or_results_df.iloc[::-1]['Odds Ratio']]

plt.errorbar(x=or_results_df.iloc[::-1]['Odds Ratio'],
             y=or_results_df.iloc[::-1].index.values,
             xerr=ci,
             color='black',
             capsize=5,
             linestyle='none',
             elinewidth=1,
             marker="o",
             markersize=15,
             mfc='lightblue',
             mec='lightblue')

plt.axvline(x=1, linewidth=1.2, linestyle='--', color='gray')

x_left = int(min(or_results_df.iloc[::-1]['2.5%'].values) * 10) / 10.0
x_right = int(max(or_results_df.iloc[::-1]['97.5%'].values) * 10) / 10.0

if x_right == 1.0:
    x_right = 1.0 + (1.0 - x_left)

plt.xticks([x_left, 1, x_right], fontsize=15)

plt.yticks(fontsize=15)
plt.xlabel("Odds Ratio", fontsize=15)

# remove the upper and right spines of the plot
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# get the coordinates for better setting the texts
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Create a rectangle representing the color strip for each category

color_strip_width = 0.025
for i, category in enumerate(or_results_df.index):
    rect = patches.Rectangle((x_min, i - 0.4), color_strip_width, 0.98, linewidth=0,
                             edgecolor='none', facecolor=COLORS[i])
    ax.add_patch(rect)

# add the p-value on the right side of figure

ax.text(x_max, y_max + 0.2, "Adj. pval", va='center', ha='left', fontsize=18)
for idx, pvalue in enumerate(or_results_df['pval']):
    if pvalue < 0.05:
        exp = int(f"{pvalue:e}".split('e')[1])
        mantissa = pvalue / (10 ** exp)
        # formatted_text = f"{mantissa:.1f} × 10^{exp}"
        formatted_text = f"${mantissa:.1f} \\times 10^{{{exp}}}$"
        ax.text(x_max, idx, formatted_text, va='center', ha='left', fontsize=15)
    else:
        ax.text(x_max, idx, f"{pvalue:.2f}", color='grey', va='center', ha='left', fontsize=15)

# show the sides of different conditions, ex: RD or Relapse, Duct or Lobule
ax.annotate('', xy=(x_min, (y_max)), xytext=(1, (y_max)),
            arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='blue'))
ax.text((1 + x_min) / 2, y_max + 0.45, "Duct", va='center', ha='center', fontsize=18)

ax.annotate('', xy=(x_max, (y_max)), xytext=(1, (y_max)),
            arrowprops=dict(arrowstyle='->', lw=2, mutation_scale=20, color='red'))
ax.text((1 + x_max) / 2, y_max + 0.45, "Lobule", va='center', ha='center', fontsize=18)

# other information shown on the figure

ax.text(x_min - 0.3, y_max + 0.6, "Paired", va='center', ha='center', fontsize=18)
ax.text(x_min - 0.3, y_max + 0.2, f"(n = {str(adata.X.shape[0])})", va='center', ha='center', fontsize=18)

from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# ci_line = Line2D([0, 1], [0, 0], color='black', lw=2)
# point_estimate_circle = Circle((0.5, 0), radius=0.05, color='lightblue')
# combined_handle = [ci_line, point_estimate_circle]
#
# # Add the combined handle to the plot
# ax.add_artist(ci_line)
# ax.add_artist(point_estimate_circle)
#
# # Custom legend with the combined sign "|---o----|"
# ax.legend(combined_handle, ['95% Confidence Interval (CI)', 'Point Estimate (OR)'], loc='lower left')

legend_elements = [Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                          marker="o", markeredgecolor='black', markerfacecolor='darkgray', label='n.s.'),

                   Line2D([0], [0], linestyle="-", color='black', gapcolor='black',
                          marker="o", markeredgecolor='black', markerfacecolor='lightblue', label='p < 0.05')]
ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.5, -0.15))

plt.tight_layout()
# plt.show()
plt.savefig(out_dir / "visualization_test.png.png", dpi=500)

# ci_line = Line2D([0, 0], [0, 0], marker='_', color='black', markersize=15, label='95% Confidence Interval (CI)')
# ax.legend(handles=[ci_line], loc='lower left')

# legend_line = Line2D([0, 1], [0, 0], color='black', lw=2)  # Horizontal line for CI
# legend_circle = Circle((0.5, 0), radius=0.05, color='black')  # Circle for OR in the legend
#
# # Add the legend to the plot
# ax.legend([legend_line, legend_circle], ['95% Confidence Interval (CI)', 'Point Estimate (OR)'], loc='lower left',
#           bbox_to_anchor=(-1, -0.04))

# line_with_marker = Line2D([0, 1], [0, 0], color='black', lw=2, marker='o', markersize=10, markerfacecolor='black')
#
# # Add the line with marker to the plot
# ax.add_line(line_with_marker)
#
# # Add a legend for the combined symbol "|---o----|"
# ax.legend([line_with_marker], ['95% Confidence Interval (CI) with Point Estimate (OR)'], loc='lower left')


# ax.text(-0.5, 1.1, "Paired", va='center', ha='center', fontsize=18, transform=ax.transAxes)
# ax.text(-0.5, 1.05, "(n = 345)", va='center', ha='center', fontsize=18, transform=ax.transAxes)

# lower-left for the description of the points (p)

# Add the legend to describe the significance
# significant_patch = plt.Line2D([0], [0], marker='o',
#                                color='w', markerfacecolor="lightblue", markersize=15, label='p < 0.05')
# non_significant_patch = plt.Line2D([0], [0], marker='o',
#                                    color='w', markerfacecolor="darkblue", markersize=15, label='n.s.')
#
# # Add the legend to the bottom-left corner of the plot
# ax.legend(handles=[significant_patch, non_significant_patch], loc='upper left', bbox_to_anchor=(-1, -0.04), ncol=1)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

# Dummy effect size data
x = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1]
y = range(len(x))
ax.errorbar(x, y, xerr=0.05, fmt='o')

# Plot horizontal axis for Cliff's delta
ax.axvline(0, color='gray', linestyle='--')
ax.set_xlabel("Cliff's delta effect size")
ax.set_yticks(y)
ax.set_yticklabels([f'Feature {i+1}' for i in y])

# ------ Add arrow labels above the plot ------

# Sample sizes
n1, n2 = 402, 120

# Define x range for the two groups
x_left = -0.6
x_right = 0.6
y_arrow = 1.05  # vertical position above the axis (in axes coords)

# Use axis transform to stay relative to axes
ax.annotate('', xy=(0.5, y_arrow), xytext=(0.05, y_arrow),
            textcoords='axes fraction', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='<|-', linewidth=1.5))

ax.annotate('', xy=(0.95, y_arrow), xytext=(0.5, y_arrow),
            textcoords='axes fraction', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>', linewidth=1.5))

# Add text labels: "Age < 50" and "Age ≥ 50"
ax.text(0.05, y_arrow + 0.03, "Age < 50", ha='center', transform=ax.transAxes)
ax.text(0.5, y_arrow + 0.03, "Age ≥ 50", ha='center', transform=ax.transAxes)

# Add "n" and sample sizes
ax.text(0.05, y_arrow - 0.05, "n", ha='center', transform=ax.transAxes)
ax.text(0.05, y_arrow - 0.10, str(n1), ha='center', transform=ax.transAxes)

ax.text(0.5, y_arrow - 0.05, "n", ha='center', transform=ax.transAxes)
ax.text(0.5, y_arrow - 0.10, str(n2), ha='center', transform=ax.transAxes)

plt.tight_layout()
plt.show()


