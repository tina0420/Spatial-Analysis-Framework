"""
@author: Tina Hsu
Functions for plotting.
Only 25 colors are used here as more than this range can be uninterpretable. 3 repeats are used temporarily.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from itertools import combinations
from scipy.stats import zscore
from scipy.sparse import issparse

# from src.squidpy_stats import get_squidpy_data

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


def plot_cell_area(adata, out_dir, figsize=(3.5, 2), name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    cell_area = pd.DataFrame({"cell_area": adata.obs["area"]})

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.histplot(data=cell_area, x="cell_area", binwidth=3)

    plt.xlabel("Cell Area", fontsize=8)
    plt.ylabel("Cell Count", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.grid(alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_area.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_area.png", dpi=500)
    plt.clf()


def plot_cell_expression_violin(adata, out_dir, figsize=(7, 3), name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    if issparse(adata.X):
        exp_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    else:
        exp_df = pd.DataFrame(adata.X, columns=adata.var_names)

    exp_melt = pd.melt(exp_df, var_name="Marker", value_name="Intensity")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.violinplot(data=exp_melt, x="Marker", y="Intensity", linewidth=1, color="white", ax=ax)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Intensity", fontsize=8)
    plt.xticks(fontsize=8, rotation=35, ha="right")
    plt.yticks(fontsize=8)

    plt.grid(alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylim(0, exp_melt["Intensity"].max() + exp_melt["Intensity"].std())

    plt.title("Marker intensity", weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_expression_violin.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_expression_violin.png", dpi=500)
    plt.clf()


def plot_cell_expression_violin_split(adata, out_dir, set_count=8, figsize=(7, 3), rotation=0, name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param set_count: count of markers per figure, type: int
    :param figsize: figure size, type: tuple
    :param rotation: rotation, type: int
    :param name: additional label for the file name, type: str
    :return:
    """
    if issparse(adata.X):
        exp_df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    else:
        exp_df = pd.DataFrame(adata.X, columns=adata.var_names)

    exp_melt = pd.melt(exp_df, var_name="Marker", value_name="Intensity")
    pairs = construct_range(len(adata.var_names), set_count)

    for i in range(len(pairs)):
        subset = exp_melt[exp_melt.Marker.isin(adata.var_names[pairs[i][0]:pairs[i][1]])]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.stripplot(data=subset, x="Marker", y="Intensity", jitter=True, size=1, ax=ax, alpha=0.08)
        sns.violinplot(data=subset, x="Marker", y="Intensity", linewidth=1, color="white", ax=ax)

        plt.xticks(fontsize=8, rotation=rotation)
        plt.yticks(fontsize=8)

        plt.grid(alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.ylim(0, exp_melt["Intensity"].max() + exp_melt["Intensity"].std())

        plt.xlabel("Marker", fontsize=8)
        plt.ylabel("Intensity", fontsize=8)
        plt.title(f"Marker intensities split {i + 1}", weight='bold', fontsize=8)

        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_cell_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"cell_expression_violin_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
        plt.clf()


def plot_cell_expression_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), rotation=35, name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param rotation: rotation, type: int
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values

    if issparse(adata.X):
        df = pd.DataFrame(adata.X.toarray(), columns=adata.var.index).assign(label=labels)
    else:
        df = pd.DataFrame(adata.X, columns=adata.var.index).assign(label=labels)

    df = df.sort_values(by="label")

    if all(str(e).isdigit() for e in labels):
        row_colors = pd.DataFrame({'Cluster': [COLORS[int(i)] for i in df["label"]]})
    else:
        color_mapping = dict(zip(sorted(list(pd.unique(df["label"]))), range(len(pd.unique(df["label"])))))
        row_colors = pd.DataFrame({'Cluster': [COLORS[color_mapping[i]] for i in df["label"]]})

    df = df.drop(columns=["label"]).reset_index(drop=True)

    g = sns.clustermap(df, row_colors=row_colors, row_cluster=False, col_cluster=False,
                       figsize=figsize, cmap=sns.diverging_palette(240, 10, n=9), cbar_pos=None)

    ax = g.ax_heatmap
    plt.xticks(fontsize=8, rotation=rotation, ha="right")
    ax.set_yticklabels([])
    ax.tick_params(right=False, length=1, width=0.5)
    g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    g.ax_row_colors.tick_params(length=1, width=0.5)

    plt.title("Expression heatmap", weight='bold', fontsize=8)

    if name:
        plt.savefig(out_dir / f"{name}_cell_expression_heatmap.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_expression_heatmap.png", dpi=500)
    plt.clf()


def plot_zscore_cell_expression_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), rotation=35,
                                        color_dict=COLORS, name=None, v_max=None, v_min=None):
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

    # create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
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
                       figsize=figsize, cmap=sns.diverging_palette(240, 10, n=9), cbar_pos=None,
                       vmax=v_max, vmin=v_min)

    ax = g.ax_heatmap
    plt.xticks(fontsize=8, rotation=rotation, ha="right")
    ax.set_yticklabels([])
    ax.tick_params(right=False, length=1, width=0.5)
    g.ax_row_colors.set_xticklabels(["Cluster"], fontsize=8, rotation=rotation, ha="right")
    g.ax_row_colors.tick_params(length=1, width=0.5)

    plt.title("Z-scored expression heatmap", weight='bold', fontsize=8)

    if name:
        plt.savefig(out_dir / f"{name}_cell_z_expression_heatmap.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_z_expression_heatmap.png", dpi=500)
    plt.clf()


def plot_spatial_expression_panel(adata, out_dir, name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param name: additional label for the file name, type: str
    :return:
    """
    rows = len(construct_range(len(adata.var_names) - 1, 5))

    fig, axes = plt.subplots(rows, 5, figsize=(7, rows * 1.4))

    for i, ax in enumerate(fig.axes):
        if i > len(adata.var_names) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        g = ax.scatter(adata.obsm["spatial"][:, 0], -adata.obsm["spatial"][:, 1], label=None,
                       c=adata.X[:, i].toarray() if issparse(adata.X) else adata.X[:, i], cmap='rocket_r', s=1)

        ax.set_facecolor('white')
        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(adata.var_names[i], weight="bold", fontsize=8)
        cbar = fig.colorbar(g, ax=ax)
        cbar.ax.tick_params(length=1, width=0.5, labelsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_spatial_expression_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "spatial_expression_panel.png", dpi=500)
    plt.clf()


def plot_spatial_correlation(adata, out_dir, threshold=0, figsize=(15, 12), plot_coeff=False, fontsize=15, name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param threshold: threshold for Pearson correlation to determine which markers to plot, type: float
    :param figsize: figure size, type: tuple
    :param plot_coeff: whether to include Pearson correlation coefficient, type: bool
    :param fontsize: font size, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    if issparse(adata.X):
        high_r_cols, low_r_cols = threshold_column_correlation(adata.X.toarray(), threshold=threshold)
    else:
        high_r_cols, low_r_cols = threshold_column_correlation(adata.X, threshold=threshold)

    if len(high_r_cols) == 0:
        print(f"Threshold of {threshold} is too high, no columns are above threshold.")
        return

    for exp in ["high", "low"]:
        if exp == "high":
            if issparse(adata.X):
                subset_exp = adata.X[:, high_r_cols].toarray()
            else:
                subset_exp = adata.X[:, high_r_cols]

        else:
            if len(low_r_cols) == 0:
                continue
            if issparse(adata.X):
                subset_exp = adata.X[:, low_r_cols].toarray()
            else:
                subset_exp = adata.X[:, low_r_cols]

        high_r_col = adata.var_names[high_r_cols]

        fig, ax = plt.subplots(len(high_r_col), len(high_r_col), figsize=figsize)

        # Plot intensity histogram
        for a in range(0, len(high_r_col)):
            ax[a, a].hist(subset_exp[:, a])

        for a in combinations(range(0, len(high_r_col)), 2):
            # Plot intensity correlation
            ax[a[0], a[1]].scatter(subset_exp[:, a[0]], subset_exp[:, a[1]], s=0.1)

            # Annotate Spearman correlation
            if plot_coeff:
                if issparse(adata.X):
                    r_mtx = get_coexpression_matrix(adata.X.toarray(), high_r_cols)
                else:
                    r_mtx = get_coexpression_matrix(adata.X, high_r_cols)
                r_mtx = pd.DataFrame(r_mtx)
                sub = r_mtx[r_mtx[0] == a[0]]
                plt.text(.05, .8, f'r={round(sub[sub[1] == a[1]][2].values[0], 2)}',
                         transform=ax[a[0], a[1]].transAxes, fontsize=fontsize)

            # Plot spatial co-expression
            subset_loc0, subset_loc1 = spatial_expression_threshold_by_mean(
                subset_exp, adata.obsm["spatial"], np.mean(subset_exp, axis=0), a)

            ax[a[1], a[0]].scatter(subset_loc0[:, 0], -subset_loc0[:, 1], s=0.1, c="blue")
            ax[a[1], a[0]].scatter(subset_loc1[:, 0], -subset_loc1[:, 1], s=0.1, c="white")

            # Fix labels on the boundary
            if a[0] == 0:
                ax[a[1], a[0]].set_ylabel(f"{high_r_col[a[1]]}", fontsize=fontsize)

            if a[1] == len(high_r_col) - 1:
                ax[a[1], a[0]].set_xlabel(f"{high_r_col[a[0]]}", fontsize=fontsize)

        # Fix labels on the boundary
        ax[0, 0].set_ylabel(f"{high_r_col[0]}", fontsize=fontsize)
        ax[len(high_r_col) - 1, len(high_r_col) - 1].set_xlabel(f"{high_r_col[len(high_r_col) - 1]}", fontsize=fontsize)

        if plot_coeff:
            if name:
                plt.savefig(out_dir / f"{name}_{exp}_threshold_{threshold}_coeff.png", dpi=500)
            else:
                plt.savefig(out_dir / f"{exp}_threshold_{threshold}_coeff.png", dpi=500)

        else:
            plt.tight_layout()
            if name:
                plt.savefig(out_dir / f"{name}_{exp}_threshold_{threshold}.png", dpi=500)
            else:
                plt.savefig(out_dir / f"{exp}_threshold_{threshold}.png", dpi=500)

        plt.clf()


def plot_umap(adata, cluster_key, out_dir, color_dict=COLORS, figsize=(7, 7), dotsize=1, name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param color_dict: color dictionary for plotting, type: dict
    :param figsize: figure size, type: tuple
    :param dotsize: dot size, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    is_digit_labels = all(str(e).isdigit() for e in labels)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(len(label_categories)):
        lbl_idxs = np.where(labels == label_categories[i])[0]

        if color_dict:
            ax.scatter(adata.obsm["X_umap"][:, 0][lbl_idxs], adata.obsm["X_umap"][:, 1][lbl_idxs],
                       c=color_dict[label_categories[i]], s=dotsize, label=label_categories[i])
        else:
            if is_digit_labels:
                ax.scatter(adata.obsm["X_umap"][:, 0][lbl_idxs], adata.obsm["X_umap"][:, 1][lbl_idxs],
                           c=COLORS[int(label_categories[i])], s=dotsize, label=label_categories[i])
            else:
                ax.scatter(adata.obsm["X_umap"][:, 0][lbl_idxs], adata.obsm["X_umap"][:, 1][lbl_idxs],
                           c=COLORS[i], s=dotsize, label=label_categories[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.xlabel("UMAP1", fontsize=8)
    plt.ylabel("UMAP2", fontsize=8)

    plt.legend(prop={'size': 8}, loc='upper right', bbox_to_anchor=(1.12, 1.1))

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_umap.png", dpi=500)
    else:
        plt.savefig(out_dir / "umap.png", dpi=500)
    plt.clf()


def plot_umap_panel(adata, cluster_key, out_dir, color_dict=COLORS, dotsize=0.1, show_counts=False, name=None, dpi=300):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param color_dict: color dictionary for plotting, type: dict
    :param dotsize: dot size, type: float
    :param show_counts: show the counts of each label or not, type: bool
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    is_digit_labels = all(str(e).isdigit() for e in labels)
    rows = len(construct_range(len(pd.unique(labels)) - 1, 5))
    # fig, axes = plt.subplots(rows, 5, figsize=(7, rows * 1.4))
    fig, axes = plt.subplots(rows, 4, figsize=(10, rows * 1.7))
    for i, ax in enumerate(fig.axes):
        if i > len(pd.unique(labels)) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        lbl_idxs = np.where(labels == label_categories[i])[0]
        non_lbl_idxs = np.where(labels != label_categories[i])[0]
        subset = adata.obsm["X_umap"][lbl_idxs]
        inv_subset = adata.obsm["X_umap"][non_lbl_idxs]
        # Plot the background
        ax.scatter(inv_subset[:, 0], inv_subset[:, 1], s=dotsize, c="#e0e0e0")
        if color_dict:
            ax.scatter(subset[:, 0], subset[:, 1], s=dotsize, c=color_dict[label_categories[i]])
        else:
            if is_digit_labels:
                ax.scatter(subset[:, 0], subset[:, 1], s=dotsize, c=COLORS[int(label_categories[i])])
            else:
                ax.scatter(subset[:, 0], subset[:, 1], s=dotsize, c=COLORS[i])
        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_counts:
            ax.set_title(
                label_categories[i] + " (n=" + str(adata.obs[cluster_key].value_counts()[label_categories[i]]) + ")",
                weight="bold", fontsize=10)
        else:
            ax.set_title(
                label_categories[i], weight="bold", fontsize=10)
    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{name}_umap_panel.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "umap_panel.png", dpi=dpi)
    plt.clf()


def plot_cluster_mean_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories

    if issparse(adata.X):
        cluster_mean = np.array(
            [np.mean(adata.X[np.where(labels == i)[0]].toarray(), axis=0) for i in label_categories])
    else:
        cluster_mean = np.array(
            [np.mean(adata.X[np.where(labels == i)[0]], axis=0) for i in label_categories])

    cluster_df = pd.DataFrame(cluster_mean, columns=adata.var_names, index=label_categories)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(cluster_df, cmap="coolwarm", ax=ax)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Label", fontsize=8)
    plt.title(f"Cluster mean heatmap", weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_heatmap_cluster_mean.png", dpi=500)
    else:
        plt.savefig(out_dir / f"heatmap_cluster_mean.png", dpi=500)
    plt.clf()


def plot_zscore_cluster_mean_heatmap(adata, column_names, cluster_key, out_dir, vmax=None, vmin=None, figsize=(7, 7),
                                     name=None):
    """
    :param adata: AnnData
    :param column_names: set of column names to cluster, dtype: list(str)
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    # create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
        exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories

    cluster_mean = np.array([np.mean(exp_mtx[np.where(labels == i)[0]], axis=0) for i in label_categories])
    cluster_df = pd.DataFrame(cluster_mean, columns=adata.var_names[idxs], index=label_categories)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmax and vmin:
        print("vmax and vmin setting manually")
        sns.heatmap(cluster_df, cmap="coolwarm", ax=ax, vmax=vmax, vmin=vmin, square=False)
    else:
        sns.heatmap(cluster_df, cmap="coolwarm", ax=ax, vmax=2, vmin=-2, square=False)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Label", fontsize=8)
    plt.title(f"Z-scored cluster mean heatmap", weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_heatmap_z_cluster_mean.png", dpi=500)
    else:
        plt.savefig(out_dir / f"heatmap_z_cluster_mean.png", dpi=500)
    plt.clf()


def plot_cluster_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = sorted(list(pd.unique(labels)))

    if issparse(adata.X):
        vmax = adata.X.toarray().max()
        vmin = adata.X.toarray().min()
    else:
        vmax = adata.X.max()
        vmin = adata.X.min()

    for i in label_categories:
        if issparse(adata.X):
            cluster_subset = adata.X[np.where(labels == i)[0]].toarray()
        else:
            cluster_subset = adata.X[np.where(labels == i)[0]]

        cluster_subset = pd.DataFrame(cluster_subset, columns=adata.var_names)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.heatmap(cluster_subset, cmap="coolwarm", ax=ax, vmax=vmax, vmin=vmin)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=6)

        plt.xlabel("Marker", fontsize=8)
        plt.ylabel("Cells", fontsize=8)
        plt.title(f"Cluster {i} heatmap", weight='bold', fontsize=8)

        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_heatmap_{i}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"heatmap_{i}.png", dpi=500)
        plt.clf()


def plot_zscore_cluster_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None):
    """
    :param adata: AnnData
    :param column_names: set of column names to cluster, dtype: list(str)
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    # create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
        exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    vmax = exp_mtx.max()
    vmin = exp_mtx.min()

    for i in label_categories:
        cluster_subset = exp_mtx[np.where(labels == i)[0]]
        cluster_subset = pd.DataFrame(cluster_subset, columns=adata.var_names[idxs])

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.heatmap(cluster_subset, cmap="coolwarm", ax=ax, vmax=vmax, vmin=vmin)

        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        cax = ax.figure.axes[-1]
        cax.tick_params(labelsize=6)

        plt.xlabel("Marker", fontsize=8)
        plt.ylabel("Cells", fontsize=8)
        plt.title(f"Cluster {i} heatmap", weight='bold', fontsize=8)

        plt.tight_layout()

        if name:
            plt.savefig(out_dir / f"{name}_heatmap_z_{i}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"heatmap_z_{i}.png", dpi=500)
        plt.clf()


def plot_cluster_cell_count(adata, cluster_key, out_dir, figsize=(3.5, 2), name=None, dpi=100):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param name: additional label for the file name, type: str
    :return:
    """
    label_df = pd.DataFrame(adata.obs[cluster_key].values, columns=["label"])
    cluster_cell_count = label_df["label"].value_counts().reset_index()
    cluster_cell_count.columns = ["Cluster", "Count"]
    cluster_cell_count = cluster_cell_count.sort_values(by="Cluster")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.barplot(data=cluster_cell_count, x="Cluster", y="Count", color="#1f77b4")

    plt.xticks(fontsize=8, rotation=35, ha="right")
    plt.yticks([0, 50000, 100000], fontsize=8)
    plt.yticks(fontsize=8)

    plt.grid(alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("Cluster", fontsize=8)
    plt.ylabel("Cell Count", fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cluster_cell_count.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "cluster_cell_count.png", dpi=dpi)
    plt.clf()


def plot_spatial_cluster_distribution(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=1,
                                      color=COLORS, name=None, dpi=150):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param dotsize: dot size, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    is_digit_labels = all(str(e).isdigit() for e in labels)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i in range(len(label_categories)):
        lbl_idxs = np.where(labels == label_categories[i])[0]

        if is_digit_labels:
            ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
                       -adata.obsm["spatial"][lbl_idxs][:, 1],
                       c=COLORS[int(label_categories[i])], s=dotsize, label=label_categories[i])
        else:
            ax.scatter(adata.obsm["spatial"][lbl_idxs][:, 0],
                       -adata.obsm["spatial"][lbl_idxs][:, 1],
                       c=color[label_categories[i]], s=dotsize, label=label_categories[i])

    ax.axis('off')
    ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
    # ax.set_title(f"{name}", weight="bold", fontsize=12)
    if adata.obs["Recurrence status"].values[0] == 0:
        observation = "NR"
    else:
        observation = "R"

    ax.set_title(f"{observation}-Patient {adata.obs.image_id.values[0]}", weight="bold", fontsize=12)

    # # Reorder the items in the legend, e.g. Yes, No, T-Cell and Others
    # handles, labels = ax.get_legend_handles_labels()
    # # Original order: No, Others, T-Cell and Yes -> 0, 1, 2, 3
    # order = [3, 0, 2, 1]
    #
    # ax.legend([handles[idx] for idx in order],
    #           [labels[idx] for idx in order],
    #           loc='upper center',
    #           bbox_to_anchor=(0.5, -0.02),
    #           fancybox=True,
    #           ncol=2,
    #           frameon=False,
    #           fontsize=11,
    #           markerscale=5)

    # fontsize 7 for cell niche, fontsize 8 for layer2
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, ncol=3, frameon=False, fontsize=5, markerscale=5)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_spatial_cluster_distribution.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "spatial_cluster_distribution.png", dpi=dpi)

    plt.clf()


def plot_spatial_cluster_panel(adata, cluster_key, out_dir, ncol=7, figsize=(7, 7), dotsize=0.1, color=COLORS, name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param dotsize: dot size, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories
    is_digit_labels = all(str(e).isdigit() for e in labels)
    rows = len(construct_range(len(pd.unique(labels)) - 1, 7))

    fig, axes = plt.subplots(rows, ncol, figsize=figsize)
    for i, ax in enumerate(fig.axes):
        if i > len(pd.unique(labels)) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        lbl_idxs = np.where(labels == label_categories[i])[0]
        non_lbl_idxs = np.where(labels != label_categories[i])[0]

        subset = adata.obsm["spatial"][lbl_idxs]
        inv_subset = adata.obsm["spatial"][non_lbl_idxs]

        # Plot the background
        ax.scatter(inv_subset[:, 0], -inv_subset[:, 1], s=dotsize, c="#e0e0e0")

        if is_digit_labels:
            ax.scatter(subset[:, 0], -subset[:, 1], s=dotsize, c=COLORS[int(label_categories[i])])
        else:
            ax.scatter(subset[:, 0], -subset[:, 1], s=dotsize, c=color[label_categories[i]])

        ax.grid(False)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"{label_categories[i]} (n={len(lbl_idxs)})", weight="bold", fontsize=7)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_spatial_cluster_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "spatial_cluster_panel.png", dpi=500)
    plt.clf()


def plot_edge_count(adata, cluster_key, out_dir, figsize=(7, 7), rel_zero_indexing=True, annot=False, annot_size=8,
                    name=None):
    """
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param rel_zero_indexing: if False, then converts to 0-indexing, type: bool
    :param annot: whether to annotate, type: bool
    :param annot_size: size of annotation, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    labels = adata.obs[cluster_key].values
    edge_count_df = count_edges(adata, labels, rel_zero_indexing=rel_zero_indexing)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    mask = np.triu(np.ones_like(edge_count_df, dtype=bool))
    for i in range(len(edge_count_df)):
        mask[i, i] = False

    sns.heatmap(edge_count_df, annot=annot,
                annot_kws={"size": annot_size}, cmap="RdPu", ax=ax, mask=mask, cbar=False)

    plt.xlabel("Cluster", fontsize=8)
    plt.ylabel("Cluster", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.title("Edge count", weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_edge_count.png", dpi=500)
    else:
        plt.savefig(out_dir / f"edge_count.png", dpi=500)
    plt.clf()


# def plot_cluster_cooccurrence(adata, cluster_key, label, out_dir, name=None):
#     """
#     :param adata: AnnData
#     :param cluster_key: indicate which cluster key to use in obs, type: str
#     :param label: indicate which cluster to use, type: str
#     :param out_dir: output path, type: Path
#     :param name: additional label for the file name, type: str
#     :return:
#     """
#     co_occur = get_squidpy_data(adata=adata, cluster_key=cluster_key)["co_occur"]
#     co_occur_interval = get_squidpy_data(adata=adata, cluster_key=cluster_key)["co_occur_interval"]
#
#     label_categories = list(adata.obs[cluster_key].value_counts(sort=False).index)
#     rows = len(construct_range(len(label_categories) - 1, 5))
#
#     type_idx = np.where(adata.obs[cluster_key].value_counts(sort=False).index == label)[0][0]
#
#     fig, axes = plt.subplots(rows, 5, figsize=(7, rows * 1.4))
#     for i, ax in enumerate(fig.axes):
#         if i > len(label_categories) - 1:
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['bottom'].set_visible(False)
#             ax.spines['left'].set_visible(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             continue
#
#         ax.plot(co_occur_interval[1:], co_occur[type_idx][i], c="red")
#
#         ax.set_ylim(0, co_occur[type_idx].max())
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         ax.set_title(label_categories[i], weight="bold", fontsize=8)
#
#     plt.tight_layout()
#
#     if name:
#         plt.savefig(out_dir / f"{name}_co_occur_{label}.png", dpi=500)
#     else:
#         plt.savefig(out_dir / f"co_occur_{label}.png", dpi=500)
#     plt.clf()
#
#
# def plot_cluster_interaction_matrix(adata, cluster_key, out_dir, figsize=(7, 7), name=None):
#     """
#     :param adata: AnnData
#     :param cluster_key: indicate which cluster key to use in obs, type: str
#     :param out_dir: output path, type: Path
#     :param figsize: figure size, type: tuple
#     :param name: additional label for the file name, type: str
#     :return:
#     """
#     nhood_zscore = get_squidpy_data(adata=adata, cluster_key=cluster_key)["nhood_zscore"]
#
#     label_categories = list(adata.obs[cluster_key].value_counts(sort=False).index)
#
#     interaction_matrix = pd.DataFrame(nhood_zscore, columns=label_categories, index=label_categories)
#
#     fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#     sns.heatmap(interaction_matrix, cmap="coolwarm", annot=True, annot_kws={"size": 8}, ax=ax)
#
#     plt.xlabel("Cluster", fontsize=8)
#     plt.ylabel("Cluster", fontsize=8)
#     plt.xticks(fontsize=8)
#     plt.yticks(fontsize=8)
#
#     plt.tight_layout()
#
#     if name:
#         plt.savefig(out_dir / f"{name}_interaction_matrix.png", dpi=500)
#     else:
#         plt.savefig(out_dir / "interaction_matrix.png", dpi=500)
#     plt.clf()
#
#
# def plot_cluster_ripley_l(adata, cluster_key, out_dir, name=None):
#     """
#     :param adata: AnnData
#     :param cluster_key: indicate which cluster key to use in obs, type: str
#     :param out_dir: output path, type: Path
#     :param name: additional label for the file name, type: str
#     :return:
#     """
#     ripley_l = get_squidpy_data(adata=adata, cluster_key=cluster_key)["ripley_L"]
#     label_categories = list(adata.obs[cluster_key].value_counts(sort=False).index)
#     rows = len(construct_range(len(label_categories) - 1, 5))
#
#     fig, axes = plt.subplots(rows, 5, figsize=(7, rows * 1.4))
#     for i, ax in enumerate(fig.axes):
#         if i > len(label_categories) - 1:
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['bottom'].set_visible(False)
#             ax.spines['left'].set_visible(False)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             continue
#
#         subset = ripley_l[ripley_l[cluster_key] == label_categories[i]]
#
#         ax.plot(subset["bins"], subset["stats"], c="red")
#
#         ax.set_ylim(0, ripley_l.stats.max())
#
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         ax.set_title(label_categories[i], weight="bold", fontsize=8)
#
#     plt.tight_layout()
#
#     if name:
#         plt.savefig(out_dir / f"{name}_ripley_L.png", dpi=500)
#     else:
#         plt.savefig(out_dir / f"ripley_L.png", dpi=500)
#     plt.clf()


def plot_neighbor_graph(adata, out_dir, figsize=(7, 7), dotsize=1, linewidth=0.2, name=None):
    """
    :param adata: AnnData
    :param out_dir: output path, type: Path
    :param figsize: figure size, type: tuple
    :param dotsize: dot size, type: float
    :param linewidth: line width, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot edges
    for u, v in adata.uns["relation"]:
        x1, y1 = adata.obsm["spatial"][u]
        x2, y2 = adata.obsm["spatial"][v]
        ax.plot([x1, x2], [-y1, -y2], '-', zorder=1, c="black", linewidth=linewidth)

    # Plot nodes
    ax.scatter(adata.obsm["spatial"][:, 0], -adata.obsm["spatial"][:, 1], s=dotsize, zorder=2)

    ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
    ax.set_title("Neighbor Graph", weight="bold", fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_neighbors.png", dpi=500)
    else:
        plt.savefig(out_dir / "neighbors.png", dpi=500)

    plt.clf()


def plot_observation_distribution(adata, obs_key, out_dir, figsize=(7, 7), cell_idxs=None,
                                  color_non_cell_idxs=False, cmap="coolwarm", dotsize=1, name=None):
    """
    Plotting transformation score.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :param obs_key: indicate which observation key, type: str
    :param out_dir: output path, type: Path
    :param color_non_cell_idxs: color gray for non cell indices of interest, type: bool
    :param cmap: color map, type: str
    :param dotsize: dot size, type: float
    :param name: additional label for the file name, type: str
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    loc = adata.obsm["spatial"]

    if color_non_cell_idxs:
        ax.scatter(loc[:, 0], -loc[:, 1], s=dotsize, c="#e0e0e0")

    if cell_idxs is not None:
        selected_loc = loc[cell_idxs, :]
        sc = ax.scatter(selected_loc[:, 0], -selected_loc[:, 1], s=dotsize, c=adata.obs[obs_key].values[cell_idxs],
                        cmap=cmap)
    else:
        sc = ax.scatter(loc[:, 0], -loc[:, 1], s=dotsize, c=adata.obs[obs_key].values, cmap=cmap)

    ax.axis('off')
    ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
    ax.set_title(f"{name}", weight="bold", fontsize=10)

    plt.colorbar(sc, orientation='horizontal', fraction=0.02, pad=0.01)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_{obs_key}_distribution.png", dpi=500)
    else:
        plt.savefig(out_dir / f"{obs_key}_distribution.png", dpi=500)
    plt.clf()
