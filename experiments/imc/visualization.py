"""
Visualization functions
@author: Tina Hsu
"""
import matplotlib.pyplot as plt
import seaborn as sns
from segmentation import stack_for_mesmer
from src.segment import *
import shutil
from scipy.stats import zscore
from scipy.sparse import issparse

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



