"""
@author: Tina Hsu
Compare the experiments from the same TNBC TMA, merging, UMAP, LISI and marker intensities
"""
from experiments.tnbc_imc.tnbc_imc_setup_xp import CORE_INFO, TMA_XP
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
from src.visualize import (plot_umap, plot_umap_panel, plot_zscore_cell_expression_heatmap,
                           plot_zscore_cluster_mean_heatmap)
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *


# codes from nb_xenium_batch_effects.py
def compute_lisi(embedding, labels, perplexity=30):
    """
    Compute Local Inverse Simpson's Index (LISI)
    :param embedding: NxD numpy array
    :param labels: array-like of categorical labels (e.g., batch or cell type)
    :param perplexity:
    :return:
    """
    n_neighbors = min(embedding.shape[0] - 1, perplexity * 3)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    lisi_scores = []

    for i, neighbors in enumerate(indices):
        neighbor_labels = [labels[j] for j in neighbors]
        label_counts = Counter(neighbor_labels)
        total = sum(label_counts.values())
        probs = np.array([count / total for count in label_counts.values()])
        simpson_index = np.sum(probs ** 2)
        inverse_simpson = 1.0 / simpson_index if simpson_index > 0 else 0
        lisi_scores.append(inverse_simpson)

    return np.array(lisi_scores)


if __name__ == '__main__':
    """
    Parameters for the python python:
    TYPE -> disease type
    TMA -> TMA name in the dictionary
    EPITHELIAL_MARKERS -> subset of epithelial markers for GMM
    IMMUNE_CLUSTER_MARKERS -> subset of immune and stromal markers for Leiden Clustering
    """
    # parameter setting
    TYPE = 'tnbc_imc'
    TMA = 'BOB'
    EXPERIMENT_1 = TMA_XP[TMA][0]
    EXPERIMENT_2 = TMA_XP[TMA][1]

    print(EXPERIMENT_1)
    print(EXPERIMENT_2)

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT_1}")
    adata_1 = read_h5ad(in_dir / f"{EXPERIMENT_1}_filter.h5ad")
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT_2}")
    adata_2 = read_h5ad(in_dir / f"{EXPERIMENT_2}_filter.h5ad")

    out_dir = Path(f"analysis/{TYPE}/{TMA}_test")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the common cores on both slides (core location)
    matched_cores = [k for k in CORE_INFO[EXPERIMENT_1] if 'roi' in CORE_INFO[EXPERIMENT_1][k]
                     and 'roi' in CORE_INFO[EXPERIMENT_2][k]]

    # get the roi number list in each experiment
    roi_1 = [CORE_INFO[EXPERIMENT_1][k]['roi'] for k in matched_cores]
    roi_2 = [CORE_INFO[EXPERIMENT_2][k]['roi'] for k in matched_cores]

    # to extract the part with the common cores in the adata separately
    CLUSTER_KEY = 'image_id'
    idx_1 = []
    idx_2 = []
    for i in range(len(roi_1)):
        mask = adata_1.obs[CLUSTER_KEY].str.contains(roi_1[i])
        matching_rows = list(adata_1.obs.index[mask])
        idx_1 = idx_1 + matching_rows

        mask = adata_2.obs[CLUSTER_KEY].str.contains(roi_2[i])
        matching_rows = list(adata_2.obs.index[mask])
        idx_2 = idx_2 + matching_rows

    # get the common markers and merge the adata together
    common_markers = list(adata_1.var_names.intersection(adata_2.var_names))

    adata_1 = adata_1[idx_1, common_markers]
    adata_2 = adata_2[idx_2, common_markers]

    # output the individual experiment first
    adata_1.write(out_dir / f"{EXPERIMENT_1}_comm.h5ad")
    adata_2.write(out_dir / f"{EXPERIMENT_2}_comm.h5ad")

    # Concatenate data from all slides together
    adata = adata_1.concatenate(adata_2, batch_key="batch")
    adata.obs['batch'] = adata.obs['batch'].map({'0': EXPERIMENT_1, '1': EXPERIMENT_2})

    adata.write(out_dir / f"{TMA}_merge.h5ad")

    print("UMAP for batch")
    # plot UMAP and calculate the LISI score (it should be close to 2)
    CLUSTER_KEY = 'batch'
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    plot_umap(adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{TMA}")
    plot_umap_panel(adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{TMA}")

    # calculate LISI score
    lisi_scores = compute_lisi(adata.obsm['X_pca'][:, :10], adata.obs['batch'].values)
    adata.obs["lisi"] = lisi_scores

    adata.write(out_dir / f"{TMA}_merge.h5ad")

    # plot the mean and variances of the marker expression (for marker expression across batches)
    CLUSTER_KEY = 'batch'
    TMA_4_colors = ['tab:blue', 'tab:pink', 'tab:orange', 'tab:green']
    plot_marker_expression_distribution_violin(adata, CLUSTER_KEY, out_dir, color_list=TMA_4_colors, set_count=8,
                                               figsize=(8, 4), name=TMA)

    print("running for each individual xp")
    # running the same process for each individual xp
    for xp in TMA_XP[TMA]:
        print("_________________________________")
        print(xp)
        print("running gmm")
        adata = read_h5ad(out_dir / f"{xp}_comm.h5ad")
        EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
        CLUSTER_KEY = 'esi'

        temp_adata = adata.copy()
        run_gmm_for_epithelial(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=xp)

        # check epithelial and non-epithelial zscore and UMAP
        plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                            rotation=35, name=f"{xp}_find_epithelial")

        print("running leiden")
        # non-epithelial (tme)
        CLUSTER_KEY = 'leiden'
        IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']
        adata_tme = read_h5ad(out_dir / f"{xp}_non-epithelial.h5ad")
        temp_adata = adata_tme[:, IMMUNE_CLUSTER_MARKERS].copy()
        sc.tl.pca(temp_adata, svd_solver='arpack')
        sc.pp.neighbors(temp_adata, use_rep='X_pca')
        sc.tl.leiden(temp_adata)

        plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                         out_dir=out_dir, figsize=(7, 7), name=f"{xp}_non-epithelial")

        temp_adata.write(out_dir / f"{xp}_non-epithelial.h5ad")

# this function can be used in the boxplot with mean+sd from different source overlay (thesis figure)
# def plot_marker_expression_box_batch_split(adata, cluster_key, out_dir, set_count=9, figsize=(10, 4), name=None):
#     df = pd.DataFrame(adata.X, columns=adata.var_names)
#     df['batch'] = adata.obs[cluster_key].values
#
#     df_melted = df.melt(id_vars='batch', var_name='Marker', value_name='Expression')
#
#     # split into several figs
#     pairs = construct_range(len(adata.var_names), set_count)
#     for i in range(len(pairs)):
#         print(i)
#         subset = df_melted[df_melted.Marker.isin(adata.var_names[pairs[i][0]:pairs[i][1]])]
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#
#         sns.boxplot(data=subset, x="Marker", y="Expression", color="lightgray", width=0.5)
#
#         # Compute batch-specific mean ± SD for overlay
#         group_stats = subset.groupby(['Marker', 'batch'])['Expression'].agg(['mean', 'std']).reset_index()
#         batch_colors = {
#             batch: color for batch, color in zip(df['batch'].unique(), ['tab:blue', 'tab:pink'])
#         }
#
#         for j, marker in enumerate(adata.var_names[pairs[i][0]:pairs[i][1]]):
#             for k, batch in enumerate(df['batch'].unique()):
#                 row = group_stats[(group_stats['Marker'] == marker) & (group_stats['batch'] == batch)]
#                 if not row.empty:
#                     mean = row['mean'].values[0]
#                     std = row['std'].values[0]
#                     color = batch_colors[batch]
#
#                     # Position the dot (mean)
#                     ax.plot(j, mean, 'o', color=color, label=batch if j == 0 else "", markersize=6)
#
#                     # Draw the line for mean ± std
#                     ax.vlines(j, mean - std, mean + std, colors=color, linewidth=1.5)
#
#         plt.xlabel("Marker", fontsize=8)
#         plt.ylabel("Expression", fontsize=8)
#         plt.xticks(fontsize=8, rotation=35, ha="right")
#         plt.yticks(fontsize=8)
#
#         plt.grid(alpha=0.1)
#         plt.ylim(0, subset["Expression"].max() + subset["Expression"].std())
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#
#         plt.title("Marker Expression", weight='bold', fontsize=8)
#         plt.legend(title="Experiment", loc='upper right')
#         plt.tight_layout()
#
#         if name:
#             plt.savefig(out_dir / f"{TMA}_marker_expression_box_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
#         else:
#             plt.savefig(out_dir / f"marker_expression_box_{pairs[i][0]}_{pairs[i][1] - 1}.png", dpi=500)
#         plt.clf()
