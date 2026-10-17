"""
@author: Tina Hsu
Spatial entropy.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import read_neighbors
from src.annotate import update_labels


def compute_spatial_entropy(p_list):
    """
    Compute spatial entropy for each cell
    :param p_list: probabilities of all cell types among the neighbors of each cell
    :return: entropy value
    """
    entropy = (-1) * sum([x * np.log(x) if x > 0 else 0 for x in p_list])

    return entropy


def plot_spatial_entropy_distribution(adata, idx_list, entropy_score, out_dir, figsize=(7, 7), dotsize=1, name=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sc = ax.scatter(adata.obsm["spatial"][idx_list, 0],
                    -adata.obsm["spatial"][idx_list, 1],
                    c=entropy_score, s=dotsize, cmap="coolwarm")

    ax.axis('off')
    ax.set(xlabel="X Axis", ylabel="Y Axis", xticklabels=[], yticklabels=[])
    ax.set_title(f"{name}", weight="bold", fontsize=10)

    cbar = plt.colorbar(sc, orientation='horizontal', fraction=0.02, pad=0.01)
    # cbar.set_label('Value')

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_spatial_entropy_distribution.png", dpi=500)
    else:
        plt.savefig(out_dir / "spatial_entropy_distribution.png", dpi=500)

    plt.clf()


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/entropy")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    FULL_ADATA = read_h5ad(in_dir / "TNBC_adata.h5ad")
    # Select for pre treatment
    adata = FULL_ADATA[FULL_ADATA.obs.collection_type == "pre_treatment"]
    adata = adata[adata.obs.chemo_type == "NAT"]

    # h3k27me3 only
    idx = np.concatenate((np.where(adata.obs.layer2 == "Epithelial H3K27me3+")[0],
                          np.where(adata.obs.layer1 == "TME")[0]))

    adata = adata[idx, :].copy()

    # # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal in there's no fj
    CLUSTER_KEY = "ESI"
    NEW_CLUSTER_KEY = "layer2"
    new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0]] = ['Epithelial'] * len(
        np.where(adata.obs[CLUSTER_KEY] == 'Epithelial')[0])
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0]] = ['Stromal'] * len(
        np.where(adata.obs[CLUSTER_KEY] == 'Stromal')[0])
    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]] = adata.obs[NEW_CLUSTER_KEY].values[
        np.where(adata.obs[CLUSTER_KEY] == 'Immune')[0]]

    CLUSTER_KEY = "immune_ct"
    update_labels(adata, CLUSTER_KEY, new_labels)
    # adata.write(out_dir / f"{XP}_immune_abundance.h5ad")

    CLUSTER_KEY = 'immune_ct'
    UNIT_KEY = 'unique_core_id'
    OBSERVATION_KEY = 'pCR'
    THRESHOLD = 10
    CELL_TYPES = ['B Cell', 'T Cell', 'Macrophage']
    TARGET_CELL = 'Epithelial'

    all_records = {}
    for pid in pd.unique(adata.obs[UNIT_KEY]):
        print(pid)

        p_adata = adata[adata.obs[UNIT_KEY] == pid, :]
        cid = p_adata.obs.cid.values[0]

        # get the neighbors
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=THRESHOLD)

        # Get the map of each row to the cell type
        idx_label_map = dict(zip(range(len(p_adata.obs[CLUSTER_KEY])), p_adata.obs[CLUSTER_KEY]))

        neighbors = p_adata.uns['relation']
        idx_to_show = np.where(p_adata.obs[CLUSTER_KEY] == TARGET_CELL)[0]
        entropy_record = []
        for i in idx_to_show:
            pairs = neighbors[(neighbors[:, 0] == i) | (neighbors[:, 1] == i)]
            # need to optimize!
            neighbor_list = []
            for j in range(pairs.shape[0]):
                if pairs[j, 0] == i and idx_label_map[pairs[j, 1]] != TARGET_CELL:
                    # neighbor is the second one
                    neighbor_list.append(idx_label_map[pairs[j, 1]])
                elif pairs[j, 1] == i and idx_label_map[pairs[j, 0]] != TARGET_CELL:
                    neighbor_list.append(idx_label_map[pairs[j, 0]])
                else:
                    # print(f"the neighbors: {idx_label_map[pairs[j, 0]]} and {idx_label_map[pairs[j, 1]]}")
                    continue

            # neighbor_list = [idx_label_map[x] for x in pairs[:, 1]]

            # Compute the proportions of the specific cell types in list
            ct_counts = dict(pd.Series(neighbor_list).value_counts())

            total_counts = sum([ct_counts.get(ct, 0) for ct in CELL_TYPES])

            if total_counts == 0:
                entropy = 0
            else:
                entropy = compute_spatial_entropy([ct_counts.get(ct, 0) / total_counts for ct in CELL_TYPES])

            entropy_record.append(entropy)

        all_records[pid] = entropy_record

        # Plots in different folder based on observation
        # if p_adata.obs[OBSERVATION_KEY].values[0] == 'Recurrence':
        #     out_dir = Path(in_dir / f"entropy/{TARGET_CELL}/Recurrence")
        #     if os.path.exists(out_dir) == 0:
        #         os.makedirs(out_dir, exist_ok=True)
        #     # Plot the distribution of the entropy
        #     plot_spatial_entropy_distribution(p_adata, idx_to_show, all_records[pid], out_dir, figsize=(5, 5),
        #                                       dotsize=2, name=f'{cid}_{pid}_BTM')
        #
        # elif p_adata.obs[OBSERVATION_KEY].values[0] == 'No recurrence':
        #     out_dir = Path(in_dir / f"entropy/{TARGET_CELL}/No recurrence")
        #     if os.path.exists(out_dir) == 0:
        #         os.makedirs(out_dir, exist_ok=True)
        #     # Plot the distribution of the entropy
        #     plot_spatial_entropy_distribution(p_adata, idx_to_show, all_records[pid], out_dir, figsize=(5, 5),
        #                                       dotsize=2, name=f'{cid.split("_")[0]}_{pid}_BTM')
        # else:
        #     print(f"observation is not found here: {p_adata.obs[OBSERVATION_KEY].values[0]}")

    df = pd.DataFrame(columns=["observation", "scores"])
    # Create a table and plotting for comparing entropy score of recurrence and non-recurrence patients
    for pid in pd.unique(adata.obs[UNIT_KEY]):
        print(pid)

        p_adata = adata[adata.obs[UNIT_KEY] == pid, :]
        observation = p_adata.obs[OBSERVATION_KEY].values[0]
        temp_df = pd.DataFrame({"observation": [observation] * len(all_records[pid]),
                                "scores": all_records[pid]})

        df = pd.concat([df, temp_df], axis=0)

    df = df[df.scores > 0]
    from scipy.stats import mannwhitneyu

    stat, p_value = mannwhitneyu(df[df["observation"] == "Yes"].scores.values,
                                 df[df["observation"] == "No"].scores.values, alternative='two-sided')

    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    sns.boxplot(data=df, x="observation", y="scores", palette=["tab:blue", "tab:pink"])
    if p_value < 0.01:
        ax.text(1.05, 1, 'p<0.01', ha='left', va='top', fontsize=8)
    else:
        ax.text(1.05, 1, f'p = {p_value:.3f}', ha='left', va='top', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title('Epithelial-Immune Spatial Entropy', weight='bold', fontsize=10)
    plt.xlabel('', fontsize=8)
    plt.ylabel('Scores', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "pCR_entropy_distribution_no_zero_H3K27me3.png", dpi=500)
    plt.show()
    plt.clf()



