"""
@Author: Tina Hsu
Investigate the intensity of each marker.
"""
import pandas as pd
import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_all_markers_expression(out_dir, exp_df):
    """
    Visualization of the expression of all markers.
    :param out_dir: The directory to store the figure, type: Path
    :param exp_df: Expression data from all the markers, type: pd.DataFrame
    :return: Boxplot figure
    """
    # Plot
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=exp_df)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.xlabel("Marker", fontsize=25)
    plt.ylabel("Intensity", fontsize=25)
    plt.title("Expression of all the markers", fontsize=30)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(out_dir / "expression_all_markers.png", dpi=500)
    plt.clf()


def correlation_for_immune_cells():
    pass


def correlation_for_epithelial_cells():
    pass

def correlation_for_t_cells():
    pass


