"""
@Author: Tina Hsu
Investigate the correlations between different markers.
"""
import pandas as pd
import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


#exp_df = pd.DataFrame(adata.X, columns=adata.var_names)


def correlation_for_b_cells():
    """
    Run all the correlation listed to check quality control.
    :return: Output figures of correlation
    """
    # First Correlation: ER, PR and HER2
    correlation_bw_three_markers(exp_df=exp_df, marker_1="ER-SP1", marker_2="PR-SP2", marker_3="erbB2")

    # The second correlation: CK5, CK8-18, CK19 and panCK
    correlation_bw_three_markers(exp_df=exp_df, marker_1="CK5", marker_2="CK8-18", marker_3="CK19")

    # The third correlation:
    correlation_bw_two_markers(exp_df=exp_df, marker_1="CD20", marker_2="CD45")


def correlation_bw_two_markers(exp_df, marker_1, marker_2):
    """
    Run and plot the correlation of two markers by Spearman Correlation.
    :param exp_df: Expression data of all the markers, type: pd.DataFrame
    :param marker_1: The first marker, type: str
    :param marker_2: The second marker, type: str
    :return: A scatter plot of the correlation of two markers.
    """
    expression_to_compare = np.array(exp_df[[marker_1, marker_2]].T)
    rho = calculate_spearman_cor(expression_to_compare)
    plot_spearman_correlation(expression_to_compare, [marker_1, marker_2], rho, "scatter")


def correlation_bw_three_markers(exp_df, marker_1, marker_2, marker_3):
    """
    Run and plot the correlation of three markers by Spearman Correlation.
    :param exp_df: Expression data of all the markers, type: pd.DataFrame
    :param marker_1: The first marker, type: str
    :param marker_2: The second marker, type: str
    :param marker_3: The third marker, type: str
    :return: A heatmap plot of the correlation of three markers.
    """
    expression_to_compare = np.array(exp_df[[marker_1, marker_2, marker_3]].T)
    rho = calculate_spearman_cor(expression_to_compare)
    plot_spearman_correlation(rho, [marker_1, marker_2, marker_3], rho, "heatmap")


def calculate_spearman_cor(exp_arr):
    """
    Run Spearman Correlation.
    :param exp_arr: Expression data of the markers, type: np.Array
    :return: Correlation coefficient from Spearman Correlation
    """
    rho, p_value = stats.spearmanr(exp_arr, axis=1)

    return rho


def plot_spearman_correlation(data_to_plot, markers, corr_val, figure_type):
    """
    Plot the correlation of two markers by Spearman Correlation.
    :param data_to_plot: Data used for plotting, type: pd.DataFrame or np.Array
    :param markers: Markers for the correlation test, type: list
    :param corr_val: Correlation coefficient of Spearman test, type: float
    :param figure_type: A kind of plotting for showing the correlation, type: str
    :return:
    """
    plt.figure(figsize=(12, 9))
    if figure_type == "heatmap":
        sns.heatmap(pd.DataFrame(data_to_plot, columns=markers, index=markers),
                    cmap="YlGnBu", annot=True, annot_kws={'size': 25})
        plt.title("Correlation: " + str(markers), fontsize=35)
        file_name = "_".join(markers) + "_heatmap.png"

    elif figure_type == "scatter":
        plt.scatter(x=data_to_plot[0], y=data_to_plot[1])
        plt.title("Correlation Coefficient: " + str(corr_val) + ", " + str(markers), fontsize=35)
        file_name = "_".join(markers) + "_scatter.png"

    plt.xticks(fontsize=25)
    plt.xlabel("Markers", fontsize=30)
    plt.yticks(fontsize=25)
    plt.ylabel("Markers", fontsize=30)
    plt.tight_layout()
    plt.savefig(out_dir / file_name, dpi=500)
    plt.clf()


#
#
# run_name = "XP12446"
# out_dir = Path(f"figures/tnbc_imc/{run_name}")
#
#
# plt.figure(figsize=(20, 15))
# sns.heatmap(image_1_exp.corr(method="spearman"), cmap="YlGnBu", annot=True, vmax=1, vmin=-1)
# plt.title("Correlation across all markers", fontsize=30)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.tight_layout()
# # plt.savefig(out_dir / "Correlation_heatmap_all_markers_all_images.png", dpi=500)
# # plt.clf()
# plt.show()

#testtest = adata.X