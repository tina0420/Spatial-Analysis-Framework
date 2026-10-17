"""
@author: Tina Hsu
Plotting and visualization for TNBC IMC analysis
"""
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from src.visualize import plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap, plot_umap, \
    plot_umap_panel
from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT, LAYER2_CELL_TYPE_COLOR_DICT


def plot_marker_expression_distribution_violin(adata, cluster_key, out_dir, color_dict,
                                               set_count=8, figsize=(8, 4), name=None):
    """
    Violin plot for more than one batches, currently only for 2 batches. If more, need to add more colors!
    :param adata: AnnData
    :param cluster_key: indicate the column from obs, type: str
    :param out_dir: output directory, type: Path
    :param color_dict: color dictionary for different batches (cluster_key), type: dict
    :param set_count: number of the items showing in the x-axis for one figure, type: int
    :param figsize: figure size, type: tuple
    :param name: specific name for the file, type: str
    :return:
    """
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df['batch'] = adata.obs[cluster_key].values
    df_melted = df.melt(id_vars='batch', var_name='Marker', value_name='Expression')

    pairs = construct_range(len(adata.var_names), set_count)

    # batch_colors = {
    #     batch: color for batch, color in zip(df['batch'].unique(), color_list)
    # }

    for i in range(len(pairs)):
        print(i)
        subset = df_melted[df_melted.Marker.isin(adata.var_names[pairs[i][0]:pairs[i][1]])]
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.violinplot(data=subset, x="Marker", y="Expression", hue="batch", linewidth=1, ax=ax,
                       palette=color_dict)

        plt.xlabel("Marker", fontsize=10)
        plt.ylabel("Expression", fontsize=10)
        plt.xticks(fontsize=10, rotation=35, ha="right")
        plt.yticks(fontsize=10)
        plt.grid(alpha=0.1)
        # plt.axhline(y=1, color='red', linestyle='--', linewidth=1)
        plt.ylim(-2, 6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)

        plt.clf()


def plot_marker_expression_distribution_violin_for_one(adata, out_dir, set_count=8, figsize=(8, 4), name=None):
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    pairs = construct_range(len(adata.var_names), set_count)

    for i in range(len(pairs)):
        print(i)
        subset = df.iloc[:, pairs[i][0]:pairs[i][1]]
        df_melted = subset.melt(var_name='Marker', value_name='Expression')

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.violinplot(data=df_melted, x="Marker", y="Expression", linewidth=1, ax=ax)

        plt.xlabel("Marker", fontsize=10)
        plt.ylabel("Expression", fontsize=10)
        plt.xticks(fontsize=10, rotation=35, ha="right")
        plt.yticks(fontsize=10)
        plt.grid(alpha=0.1)
        # plt.axhline(y=1, color='red', linestyle='--', linewidth=1)
        plt.ylim(-2, 6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"marker_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)

        plt.clf()


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific_8_tma'
    XP = '8_tma_cohort'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}")

    adata = read_h5ad(in_dir / f"{XP}_analysis_pretreat.h5ad")
    patient_info = pd.read_csv(Path("analysis/tnbc_imc/summary") / "all_8_tma_to_analysis_patient_info.csv")
    mapping_dict = dict(zip(patient_info.cid.values, patient_info.recurrence.values))
    pd.Series([mapping_dict[cid] for cid in set(adata.obs.cid)]).value_counts()

    # check why there's less one patient from testing (pre_4) to the merging
    pre_4_pid = ['CID0000314',
                 'CID0000051',
                 'CID0001307',
                 'CID0001212',
                 'CID0003269',
                 'CID0000789',
                 'CID0000044',
                 'CID0000333',
                 'CID0006435',
                 'CID0000135',
                 'CID0005629',
                 'CID0003721',
                 'CID0005751',
                 'CID0000479',
                 'CID0001007',
                 'CID0000305',
                 'CID0004207',
                 'CID0003046',
                 'CID0001646',
                 'CID0004921',
                 'CID0000300',
                 'CID0004121']

    merge_pre_4 = list(set(adata[np.where(~adata.obs.tma.isin(["Brian", "23-005", "22-010B", "23-010"]))[0], :].obs.cid))

    # For distribution of cell expression in different markers (across batches)
    plot_marker_expression_distribution_violin()

    # For spatially distribution of all cell types (
    for pid in set(adata.obs.image_key):
        print(pid)
        tma = pid.split("_")[0]

        # Set the output directory of the image based on which tma it is from
        out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/tma")
        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        # layer 2

        # test for one img
        img = '22-010A_ROI_033'
        p_adata = adata[np.where(adata.obs.image_key == img)[0], :]

        from src.visualize import plot_spatial_cluster_distribution

        CLUSTER_KEY = 'layer2'
        plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5.5, 5.5),
                                          dotsize=2, color=LAYER2_CELL_TYPE_COLOR_DICT, name=f"{img}_layer2")

    plot_marker_expression_distribution_violin_for_one(adata, out_dir, set_count=9, figsize=(8, 4), name=XP)
