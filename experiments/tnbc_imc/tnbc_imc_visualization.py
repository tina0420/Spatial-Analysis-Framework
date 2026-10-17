"""
Functions for visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import zscore

COLOR_CODE = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
              '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909',
              '#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e',
              '#1fb45c', '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00',
              '#c7e8ae', '#e8aec7', '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909'
              ]

COLORS = dict(zip(range(len(COLOR_CODE)), COLOR_CODE))


def plot_zscore_cell_expression_heatmap(adata, column_names, cluster_key, out_dir, color_dict=COLORS, figsize=(7, 7),
                                        rotation=35, name=None, v_max=None, v_min=None):
    """
    :param adata: AnnData
    :param column_names: set of column names to cluster, dtype: list(str)
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param rotation: rotation, type: int
    :param name: additional label for the file name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    df = pd.DataFrame(exp_mtx, columns=column_names).assign(label=labels)
    df = df.sort_values(by="label", ascending=False)

    if all(str(e).isdigit() for e in labels):
        row_colors = pd.DataFrame({'Cluster': [COLORS[int(i)] for i in df["label"]]})
    elif type(color_dict) is not dict:
        color_mapping = dict(zip(sorted(list(pd.unique(df["label"]))), range(len(pd.unique(df["label"])))))
        row_colors = pd.DataFrame({'Cluster': [COLORS[color_mapping[i]] for i in df["label"]]})
    else:
        row_colors = pd.DataFrame({'Cluster': [color_dict[i] for i in df["label"]]})

    df = df.drop(columns=["label"]).reset_index(drop=True)

    g = sns.clustermap(df, row_colors=row_colors, row_cluster=False, col_cluster=False,
                       figsize=figsize, cmap=sns.diverging_palette(240, 10, n=9), cbar_kws={"label": "Z-score"},
                       vmax=v_max, vmin=v_min)

    ax = g.ax_heatmap
    plt.xticks(fontsize=8, rotation=rotation, ha="right")
    ax.set_yticklabels([])
    ax.tick_params(right=False, length=1, width=0.5)
    g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    g.ax_row_colors.tick_params(length=1, width=0.5)

    g.ax_cbar.tick_params(labelsize=8)
    g.ax_cbar.set_title("Z-score", fontsize=10, pad=10)

    plt.title("Z-scored expression heatmap", weight='bold', fontsize=8)

    if name:
        plt.savefig(out_dir / f"{name}_cell_z_expression_heatmap.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_z_expression_heatmap.png", dpi=500)
    plt.clf()
