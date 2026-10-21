"""
Visualization functions
@author: Tina Hsu
"""
import matplotlib.pyplot as plt
import seaborn as sns
from segmentation import stack_for_mesmer
from src.segment import *
from src.utils import *
import shutil
from scipy.stats import zscore
from scipy.sparse import issparse
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import roc_curve, auc

# Color code panel if
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


def plot_marker_correlations_heatmap(out_dir, corr_df, figsize=(7, 7), name=None):
    """
    Heatmap for correlation metric
    :param out_dir: output directory, type: Path
    :param corr_df: correlation metric, type: pd.DataFrame
    :param figsize: figure size, type: tuple
    :param name: name of the figure, type: str
    :return:
    """
    # Get the marker list from the index_to_marker.csv
    marker_list = list(pd.read_csv(out_dir / "index_to_marker.csv").marker)

    # Create a metrix for the heatmap with markers as columns and rows
    corr_mtx = np.ones((len(marker_list), len(marker_list)))

    corr_df = np.array(corr_df)
    # fill in the correlation metrix
    for j in range(corr_df.shape[0]):
        u, v = corr_df[j][:2]

        corr_mtx[int(u)][int(v)] = corr_df[j][2]
        corr_mtx[int(v)][int(u)] = corr_df[j][2]

    # Plot a heatmap with the created correlation metrix
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(pd.DataFrame(corr_mtx, columns=marker_list, index=marker_list), cmap="coolwarm", ax=ax)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=6)

    plt.xlabel("Marker", fontsize=8)
    plt.ylabel("Marker", fontsize=8)
    plt.title("Marker correlations", weight='bold', fontsize=8)
    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{name}_marker_correlations_heatmap.png", dpi=500)
    else:
        plt.savefig(out_dir / "marker_correlations_heatmap.png", dpi=500)

    plt.close('all')


def plot_marker_nuclei_staining(marker_list, nuclei_name, in_dir, out_dir):
    """
    Plot the marker and nuclei images to check marker quality by each image
    :param marker_list: markers to plot, type: list[str]
    :param nuclei_name: marker name of nuclei, type: str
    :param in_dir: directory of the tiff file and panel information, type: Path
    :param out_dir: directory to store the output images, type: Path
    :return:
    """
    # Get all marker info from panel.csv
    panel_df = pd.read_csv(in_dir / "panel.csv")
    panel = {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}

    for roi, image_path in enumerate(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")), start=1):
        file_name = os.path.basename(image_path)
        print(file_name.split("_")[0], file_name)

        # Read image
        img = skio.imread(image_path)

        for marker in marker_list:
            # Set the dictionary to store the staining images for corresponding markers
            temp_out_dir = out_dir / marker
            if os.path.exists(temp_out_dir) == 0:
                os.makedirs(temp_out_dir, exist_ok=True)

            channel_idx = panel[marker]
            nuclei = img[panel[nuclei_name], :, :]
            cytoplasm = img[channel_idx, :, :]
            x = stack_for_mesmer(nuclei, cytoplasm)

            plot_marker_nuclear(x, temp_out_dir, name=file_name.split("_")[0] + "_" + marker)


def plot_marker_nuclear(x, out_dir, name):
    """
    Plot the nuclear-and-one-marker-only image
    :param x: Pixel intensity of the image in 4-dimension, type: np.array
    :param out_dir: Directory to store the figure, type: str
    :param name: Name for the image, type: str
    :return:
    """
    rgb_images = create_rgb_image(x, channel_colors=['green', 'red'])

    plt.figure(figsize=(7, 7))

    plt.imshow(rgb_images[0])

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(out_dir / f"{name}.png", dpi=300)
    plt.clf()


def plot_zscore_cluster_mean_heatmap(adata, column_names, cluster_key, out_dir, vmax=None, vmin=None, figsize=(7, 7),
                                     name=None):
    # given columns, get column index
    idxs = []
    for i in column_names:
        for j in range(len(adata.var_names)):
            if i == adata.var_names[j]:
                idxs.append(j)

    # Create zscore dataframe
    if issparse(adata.X):
        exp_mtx = adata.X[:, idxs].toarray()
    else:
        exp_mtx = adata.X[:, idxs]

    for col in range(exp_mtx.shape[1]):
        # Check for divide by zero
        if exp_mtx[:, col].sum() == 0.0:
            continue
        exp_mtx[:, col] = zscore(exp_mtx[:, col])

    labels = adata.obs[cluster_key].values
    label_categories = adata.obs[cluster_key].cat.categories

    cluster_mean = np.array([np.mean(exp_mtx[np.where(labels == i)[0]], axis=0) for i in label_categories])
    cluster_df = pd.DataFrame(cluster_mean, columns=adata.var_names[idxs], index=label_categories)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vmax and vmin:
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
        plt.savefig(out_dir / f"{name}_heatmap_z_cluster_mean.png", dpi=150)
    else:
        plt.savefig(out_dir / f"heatmap_z_cluster_mean.png", dpi=150)

    plt.clf()


def plot_compare_clustering_line(adata, method, out_dir, figsize=(5, 7), name=None):

    res_scores = adata.uns[method]

    plt.figure(figsize=figsize)
    plt.plot(list(res_scores.keys()), list(res_scores.values()), marker='o', linestyle='-', color='b', label=method)

    plt.xticks(ticks=np.arange(len(res_scores)), labels=list(res_scores.keys()), fontsize=10)
    plt.xlabel("Resolution", fontsize=12)
    plt.yticks(fontsize=10)

    plt.legend()
    plt.tight_layout()
    if name:
        plt.savefig(out_dir / f"{name}_clustering_performance.png", dpi=500)
    else:
        plt.savefig(out_dir / "clustering_performance.png", dpi=500)

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


def plot_cluster_cell_count(adata, cluster_key, out_dir, figsize=(3.5, 2), name=None, dpi=150):
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
    # plt.yticks([0, 50000, 100000], fontsize=8)
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


def plot_cell_area_count(adata, out_dir, figsize=(3.5, 2), name=None, dpi=150):
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
        plt.savefig(out_dir / f"{name}_cell_area.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "cell_area.png", dpi=dpi)

    plt.clf()


"""
# Plotting neighbors
plot_neighbor_graph(p_adata, out_dir, figsize=(7, 7), dotsize=1, linewidth=0.2, name=unique_id[i])
"""


# Plot Cliff Delta (effect size)
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


def plot_marker_intensity_across_clusters(adata, marker_list, cluster_key, out_dir, cluster_order=None, dpi=150):
    """
    Plot marker intensity across clusters in boxplot.
    :param adata: Anndata
    :param marker_list: list of markers, type: list[str]
    :param cluster_key: column in adata.obs representing the cluster, type: str
    :param out_dir: output directory, type: Path
    :param cluster_order: order of the clusters to show on the figure, type: list[str]
    :param dpi: dpi setting, type: int
    :return:
    """
    for marker in marker_list:
        idx = np.where(adata.var_names == marker)[0][0]
        exp = adata.X[:, idx]

        # Make a df of the marker to plot (cell_type, expression)
        df = pd.DataFrame({"cell_type": list(adata.obs[cluster_key]),
                           "expression": list(exp)})

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        sns.boxplot(data=df, x="cell_type", y="expression", order=cluster_order, color="lightblue", ax=ax)

        sns.despine(ax=ax, top=True, right=True)
        plt.xlabel("Cell type", fontsize=20)
        plt.xticks(fontsize=15, rotation=90)
        plt.ylabel(f"{marker} intensity", fontsize=20)

        plt.tight_layout()
        plt.savefig(out_dir / f"{marker}_exp_cell_type.png", dpi=dpi)

        plt.clf()


def plot_cell_type_prop_per_img_bar(adata, unit_key, cluster_key, color_dict, out_dir, cluster_key_order=None,
                                    figsize=(50, 10), name=None, dpi=150):
    """
    Plot cluster proportion in bars.
    :param adata: Anndata
    :param unit_key: the column in obs representing the basic unit, type: str
    :param cluster_key: the column in obs representing the cluster, type: str
    :param color_dict: colour dictionary for the clusters, type: dict
    :param out_dir: output directory, type: Path
    :param cluster_key_order: the order of the clusters, type: list[str]
    :param figsize: figure size, type: tuple
    :param name: name of the output file, type: str
    :param dpi: dpi setting for the output image, type: int
    :return:
    """
    # Compute the proportion of each cell type
    counts = adata.obs.pivot_table(
        index=unit_key,
        columns=cluster_key,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)
    # print(prop_df)

    # Run hierachical clustering
    Z = linkage(prop_df, method="complete", metric="euclidean")
    # reorder the images based on clustering result
    new_idx_order = leaves_list(Z)

    if cluster_key_order:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], cluster_key_order]
    else:
        new_order_df = prop_df.loc[prop_df.index[new_idx_order], :]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot stacked bars
    bottom = np.zeros(new_order_df.shape[0])
    for cell_type in new_order_df.columns:
        bars = ax.bar(new_order_df.index, new_order_df[cell_type],
                      bottom=bottom, color=color_dict[cell_type], label=cell_type)
        bottom += new_order_df[cell_type].values

    # Some setting for the figure
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks(range(new_order_df.shape[0]))
    ax.set_xticklabels(list(new_order_df.index), rotation=90, fontsize=30, va='top')
    ax.set_xlim(-0.25, new_order_df.shape[0] - 0.75)
    ax.tick_params(axis='x', pad=2)
    plt.subplots_adjust(bottom=0.25)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='y', labelsize=35)
    ax.set_ylabel('Proportion', fontsize=35)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Cell Types", fontsize=30, title_fontsize=35,
              bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False,
              )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_proportion_bar.png", dpi=dpi)
    else:
        plt.savefig(out_dir / "proportion_bar.png", dpi=dpi)

    plt.clf()


### TILs
def plot_tils_comparison(infiltrating_df, observation_key, cell_types, tils_types, threshold,
                         color_dict, out_dir, figtype='box', show_palue=False, figsize=(15, 6), name=None):
    # Compute p-value of Mann-whitney U test for each category between two observation groups
    pvalues = []
    group1_df = infiltrating_df[infiltrating_df[observation_key] == list(set(infiltrating_df[observation_key]))[0]]
    group2_df = infiltrating_df[infiltrating_df[observation_key] == list(set(infiltrating_df[observation_key]))[1]]

    for col in [f"{ct}_{tils}" for ct in cell_types for tils in tils_types]:
        # If there's no this cell type in the image, then continue
        if (len(group1_df[col].values) == 0) | (len(group2_df[col].values) == 0):
            continue

        stat, p_value = mannwhitneyu(group1_df[col].values, group2_df[col].values, alternative='two-sided')

        pvalues.append([col, p_value])

    pvalues_df = pd.DataFrame(pvalues, columns=['type', 'pvalue'])
    pvalues_df.to_csv(out_dir / f"pval_{threshold}.csv", index=False)

    # Plotting
    df_melt = infiltrating_df.iloc[:, 1:].melt(id_vars=observation_key, var_name='variable', value_name='value')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if figtype == 'box':
        sns.boxplot(x="variable", y="value", hue=observation_key, data=df_melt, palette=color_dict, ax=ax)
    elif figtype == 'violin':
        sns.violinplot(x="variable", y="value", hue=observation_key, data=df_melt, palette=color_dict, ax=ax)

    # Add the p-values we calculated above to the plot
    if show_palue:
        for i, type in enumerate([f"{ct}_{tils}" for ct in cell_types for tils in tils_types]):
            # Position of the lines
            x1, x2 = i - 0.2, i + 0.2
            y_max = max(list(group1_df[type].values) + list(group2_df[type].values)) + 0.05

            hight = 0.015

            # Draw bracket (|-| shape)
            ax.plot([x1, x1, x2, x2], [y_max, y_max + hight, y_max + hight, y_max], lw=1.5, c='black')

            # Add p-value text
            if pvalues_df['pvalue'][i] < 0.01:
                ax.text((x1 + x2) / 2, y_max + hight + 0.01, f"p<0.01",
                        ha='center', va='bottom', fontsize=8)
            else:
                ax.text((x1 + x2) / 2, y_max + hight + 0.01, f"p = {pvalues_df['pvalue'][i]:.2f}",
                        ha='center', va='bottom', fontsize=8)

    plt.xlabel('', fontsize=8)
    plt.ylabel('Proportion', fontsize=10)
    plt.xticks(fontsize=10, rotation=35)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-0.1, 1.2)

    # Add patient counts to the legend block
    handles, labels = ax.get_legend_handles_labels()

    # Create new labels with count numbers
    new_labels = [f"{label} (n={infiltrating_df[observation_key].value_counts()[label]})" for label in labels]

    # Update legend
    ax.legend(
        handles,
        new_labels,
        title=name.split("_")[0],
        loc='upper left',
        prop={'size': 8}
    )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_tils.png", dpi=500)
    else:
        plt.savefig(out_dir / f"infiltration_{figtype}.png", dpi=500)

    plt.clf()






### Features correlations and model
def plot_feature_correlation(corr_matrix, out_dir, figsize=(13, 12), dpi=150):
    """
    Plots a lower-triangle heatmap for feature correlation.
    :param corr_matrix: correlation matrix, type: pd.DataFrame
    :param out_dir: output directory, type: Path
    :param figsize: figure size, type: tuple
    :param dpi: dpi setting, type: int
    :return:
    """

    # Hide the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": 0.5, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
                          "label": "Spearman Correlation Coefficient"},
                annot=True,
                fmt=".2f",
                annot_kws={"size": 16},
                ax=ax)

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(f'Feature Correlation Matrix', fontsize=25, pad=15)

    plt.tight_layout()
    plt.savefig(out_dir / "feature_correlation.png", dpi=dpi)

    plt.clf()


def plot_roc_curve(y, y_pred_probs, out_dir, figsize=(6, 5), name=None, dpi=150):
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # AUROC
    fpr, tpr, _ = roc_curve(y, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='#d62728', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Random chance line
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    if "/" in name:
        temp_name = name.replace("/", "_")
        plt.savefig(out_dir / f"{temp_name}_ROC.png", dpi=dpi)
    else:
        plt.savefig(out_dir / f"{name}_ROC.png", dpi=dpi)

    plt.clf()


# odds ratio plot
def plot_odds_ratio(inference_df, model_type, out_dir, figsize=(9, 4), name=None, dpi=150):
    """
    PLot odds ratio.
    :param inference_df: odds ratio matrix, type: pd.DataFrame
    :param model_type: type of model, univariate or multivariate, type: str
    :param out_dir: output directory, type: Path
    :param figsize: figure size, type: tuple
    :param name: name of the figure, type: str
    :param dpi: dpi setting, type: int
    :return:
    """
    # Set global plot style for publication-quality figures
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    plot_df = inference_df.sort_values(by='Odds Ratio', ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(plot_df))
    ors = plot_df['Odds Ratio']
    lowers = plot_df['95% CI Lower']
    uppers = plot_df['95% CI Upper']
    pvals = plot_df['p-val']
    feature_names = plot_df['Features']

    # Define colors based on significance
    colors = []
    for o, p in zip(ors, pvals):
        if p >= 0.05:
            colors.append('#808080')  # Non-significant (background noise)
        elif o > 1:
            colors.append('#d62728')  # Significant Risk Factor (Red)
        else:
            colors.append('#1f77b4')  # Significant Protective Factor (Blue)

    fig, ax = plt.subplots(figsize=figsize)

    # 95% CI
    xerr_lower = ors - lowers
    xerr_upper = uppers - ors

    ax.errorbar(ors, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', ecolor="gray", capsize=6, elinewidth=3.5, zorder=2)

    # Odds Ratios
    ax.scatter(ors, y_pos, color=colors, s=110, zorder=2)

    ax.axvline(x=1.0, color='black', linestyle='--', lw=1.5, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=14, va='center')
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (Log Scale)', fontsize=13)
    if model_type == "univariate":
        ax.set_title('Univariate Predictor of Recurrence', weight="bold", fontsize=13)
    elif model_type == "multivariate":
        ax.set_title('Multivariate Predictor of Recurrence', weight="bold", fontsize=13)

    for i in range(len(y_pos)):
        p_str = "p < 0.01" if pvals[i] < 0.01 else f"p = {pvals[i]:.2f}"
        anno_text = f"{p_str}"
        ax.text(1.02, y_pos[i], anno_text, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=13, color=colors[i])

    plt.xticks(fontsize=10)
    plt.tight_layout()
    if "/" in name:
        temp_name = name.replace("/", "_")
        plt.savefig(out_dir / f"{temp_name}_OR.png", dpi=dpi)
    else:
        plt.savefig(out_dir / f"{name}_OR.png", dpi=dpi)

    plt.clf()

