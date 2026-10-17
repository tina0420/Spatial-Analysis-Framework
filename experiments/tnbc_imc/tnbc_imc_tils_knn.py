"""
@author: Tina Hsu
Calculating TILs (tumour infiltration lymphocytes) in each image of TNBC IMC dataset
"""
import os
from pathlib import Path
from anndata import read_h5ad
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import read_neighbors
from src.spatial_stats import get_interacting_pairs
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from src.annotate import update_labels
from src.spatial_stats import get_interacting_neighborhood_cell_idxs
from src.visualize import plot_neighbor_graph
from sklearn.neighbors import NearestNeighbors


def plot_cell_distribution_boxplot(adata, cluster_key, cell_type, patient_key, observation_key,
                                   out_dir, color_list, figsize=(5, 4), name=None):
    # Compute cell type proportions
    proportions = []
    for pid in np.unique(adata.obs[patient_key]):
        # For all cells
        # p_obs = adata[np.where(adata.obs[patient_key] == pid)[0], :].obs[cluster_key]
        # if cell_type in p_obs.value_counts():
        #     prop = p_obs.value_counts()[cell_type] / len(p_obs)
        # else:
        #     prop = 0
        # proportions.append([pid,
        #                     adata[np.where(adata.obs[patient_key] == pid)[0], :].obs[observation_key].values[0],
        #                     prop])

        # For immune cells only
        p_obs = adata[np.where((adata.obs[patient_key] == pid) & (adata.obs['layer1'] == 'Immune'))[0], :].obs[
            cluster_key]
        if cell_type in p_obs.value_counts():
            prop = p_obs.value_counts()[cell_type] / len(p_obs)
        else:
            prop = 0
        proportions.append([pid,
                            adata[np.where(adata.obs[patient_key] == pid)[0], :].obs[observation_key].values[0],
                            prop])

    proportions_df = pd.DataFrame(proportions, columns=[patient_key, observation_key, f"{cell_type}_prop"])

    adata.uns[f"{cell_type}_prop"] = proportions_df

    # Boxplot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.boxplot(x=observation_key, y=f"{cell_type}_prop", data=proportions_df,
                palette=dict(zip(list(set(proportions_df[observation_key])), color_list)), ax=ax)

    # Calculate t-tset and add the p-value to the figure
    t_stat, p_value = ttest_ind(
        proportions_df[proportions_df[observation_key] == list(set(proportions_df[observation_key]))[0]][
            f"{cell_type}_prop"].values,
        proportions_df[proportions_df[observation_key] == list(set(proportions_df[observation_key]))[1]][
            f"{cell_type}_prop"].values,
        equal_var=False)

    ax.text(1.05, 1, f'p = {p_value:.3f}', ha='left', va='top', fontsize=10)

    plt.xlabel('Patient', fontsize=8)
    plt.ylabel('Proportion', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-0.1, 1)

    plt.title(f'{cell_type} proportions', weight='bold', fontsize=8)

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_distribution.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_type_distribution.png", dpi=500)
    plt.clf()


# def plot_immune_infiltration_boxplot(adata, infiltration, observation_key, cell_types, out_dir, figsize, name):


# Plot


if __name__ == '__main__':
    # Method 1: Use neighborhood directly to get the interaction (>=3 -> infiltration)

    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    XP = '22-23_3_tma'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/tils")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in adata for tils analysis
    adata = read_h5ad(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")

    # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
    new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    new_labels[np.where(adata.obs.layer1 == 'Epithelial')[0]] = ['Epithelial'] * len(
        np.where(adata.obs.layer1 == 'Epithelial')[0])
    new_labels[np.where(adata.obs.layer1 == 'Stromal')[0]] = ['Stromal'] * len(
        np.where(adata.obs.layer1 == 'Stromal')[0])
    new_labels[np.where(adata.obs.layer1 == 'Immune')[0]] = adata.obs.layer2.values[
        np.where(adata.obs.layer1 == 'Immune')[0]]

    update_labels(adata, 'ct_tils', new_labels)

    # # Compute immune cell proportion in each patient/image
    # CLUSTER_KEY = 'layer2'
    # PATIENT_KEY = 'patient_key'
    # OBSERVATION_KEY = 'relapse'
    # COLOR_LIST = ['tab:pink', 'tab:blue']
    # CELL_TYPE = 'B Cell'
    # # denominator is the total number of the cells
    # plot_cell_distribution_boxplot(adata=adata,
    #                                cluster_key=CLUSTER_KEY,
    #                                cell_type=CELL_TYPE,
    #                                patient_key=PATIENT_KEY,
    #                                observation_key=OBSERVATION_KEY,
    #                                out_dir=out_dir,
    #                                color_list=COLOR_LIST,
    #                                figsize=(3, 4),
    #                                name=f"{XP}_{CELL_TYPE}_imm")

    # Infiltration
    THRESHOLD = 20
    PATIENT_KEY = 'patient_key'
    result = {}
    til_type = ['Infiltrating', 'Bystanding', 'Ignored']
    cell_types = ['Immune', 'B Cell', 'T Cell', 'Macrophage']

    for pid in np.unique(adata.obs[PATIENT_KEY]):
        print(pid)
        result[pid] = {}
        p_adata = adata[np.where(adata.obs[PATIENT_KEY] == pid)[0], :]

        # # Get the total number of immune cells (layer1)
        # CLUSTER_KEY = 'layer1'
        # total_immune = p_adata.obs[CLUSTER_KEY].value_counts()['Immune']
        # result[pid]['total_immune'] = total_immune

        knn = NearestNeighbors(n_neighbors=THRESHOLD + 1)
        knn.fit(p_adata.obsm['spatial'])

        distances, indices = knn.kneighbors(p_adata.obsm['spatial'])
        p_adata.uns['relation'] = np.array([[row[0], x] for row in indices for x in row[1:]])
        p_adata.uns['threshold'] = THRESHOLD

        # Construct a neighborhood network for one image
        # read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=THRESHOLD)

        for cell_type in cell_types:
            if cell_type == 'Immune':
                result[pid][cell_type] = {}

                # Count edges from Epithelial and Immune
                CLUSTER_KEY = 'layer1'
                interactions = get_interacting_pairs(p_adata, cluster_key=CLUSTER_KEY,
                                                     type_i='Epithelial', type_j=cell_type)

                # Compute the number of neighbor does each immune cell has
                infiltrating = []
                bystanding = []
                ignored = []
                for idx in np.where(p_adata.obs[CLUSTER_KEY] == cell_type)[0]:
                    # When the cell has 3 or more tumour neighbors
                    if np.sum(interactions.flatten() == idx) >= 3:
                        infiltrating.append(idx)
                    # When the cell has one or two tumour neighbors
                    elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                        bystanding.append(idx)
                    # When the cell has no tumour neighbors
                    else:
                        ignored.append(idx)

                result[pid][cell_type]['infiltrating'] = infiltrating
                result[pid][cell_type]['bystanding'] = bystanding
                result[pid][cell_type]['ignored'] = ignored

            else:
                result[pid][cell_type] = {}

                # Count edges from Epithelial and Immune
                CLUSTER_KEY = 'ct_tils'
                interactions = get_interacting_pairs(p_adata, cluster_key=CLUSTER_KEY,
                                                     type_i='Epithelial', type_j=cell_type)

                # Compute the number of neighbor does each immune cell have
                infiltrating = []
                bystanding = []
                ignored = []
                for idx in np.where(p_adata.obs[CLUSTER_KEY] == cell_type)[0]:
                    # When the cell has 3 or more tumour neighbors
                    if np.sum(interactions.flatten() == idx) >= 3:
                        infiltrating.append(idx)
                    # When the cell has one or two tumour neighbors
                    elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                        bystanding.append(idx)
                    # When the cell has no tumour neighbors
                    else:
                        ignored.append(idx)

                result[pid][cell_type]['infiltrating'] = infiltrating
                result[pid][cell_type]['bystanding'] = bystanding
                result[pid][cell_type]['ignored'] = ignored

    # Compute the percentage of infiltrating cells in each patient and store in a dataframe
    infiltrating_calculation = []
    for pid, info in result.items():
        observation = adata.obs[adata.obs.patient_key == pid]['relapse'].values[0]
        info_to_add = [pid, observation]
        for cell_type in cell_types:
            total_count = len(info[cell_type]['infiltrating']) + len(info[cell_type]['bystanding']) + len(
                info[cell_type]['ignored'])
            if total_count == 0:
                info_to_add.append(0)
                info_to_add.append(0)
                info_to_add.append(0)
            else:
                info_to_add.append(len(info[cell_type]['infiltrating']) / total_count)
                info_to_add.append(len(info[cell_type]['bystanding']) / total_count)
                info_to_add.append(len(info[cell_type]['ignored']) / total_count)

        infiltrating_calculation.append(info_to_add)

    infiltrating_df = pd.DataFrame(infiltrating_calculation, columns=['patient_key', 'relapse',
                                                                      'Immune_infiltrating', 'Immune_bystanding',
                                                                      'Immune_ignored',
                                                                      'B Cell_infiltrating', 'B Cell_bystanding',
                                                                      'B Cell_ignored',
                                                                      'T Cell_infiltrating', 'T Cell_bystanding',
                                                                      'T Cell_ignored',
                                                                      'Macrophage_infiltrating',
                                                                      'Macrophage_bystanding',
                                                                      'Macrophage_ignored'])

    # Plot
    # Melt the data to long format
    df_melted = infiltrating_df[['relapse', 'Immune_infiltrating', 'Immune_bystanding',
                                 'Immune_ignored',
                                 'B Cell_infiltrating', 'B Cell_bystanding',
                                 'B Cell_ignored',
                                 'T Cell_infiltrating', 'T Cell_bystanding',
                                 'T Cell_ignored',
                                 'Macrophage_infiltrating',
                                 'Macrophage_bystanding',
                                 'Macrophage_ignored']].melt(id_vars='relapse', var_name='variable', value_name='value')

    # Calculate t-test in each category
    pvalues = []
    for col in ['Immune_infiltrating', 'Immune_bystanding',
                'Immune_ignored',
                'B Cell_infiltrating', 'B Cell_bystanding',
                'B Cell_ignored',
                'T Cell_infiltrating', 'T Cell_bystanding',
                'T Cell_ignored',
                'Macrophage_infiltrating',
                'Macrophage_bystanding',
                'Macrophage_ignored']:

        non_relapse_df = infiltrating_df[infiltrating_df['relapse'] == 'non-relapse']
        relapse_df = infiltrating_df[infiltrating_df['relapse'] == 'relapse']
        if (len(non_relapse_df[col].values) == 0) | (len(relapse_df[col].values) == 0):
            continue
        t_stat, p_value = ttest_ind(non_relapse_df[col].values,
                                    relapse_df[col].values, equal_var=False)

        pvalues.append([col, p_value])

    pvalues_df = pd.DataFrame(pvalues, columns=['type', 'pvalue'])
    pvalues_df.to_csv(out_dir / f"pval_{THRESHOLD}_knn.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    color_dict = {'non-relapse': 'tab:blue',
                  'relapse': 'tab:pink'}
    sns.violinplot(x="variable", y="value", hue="relapse", data=df_melted,
                   palette=color_dict, ax=ax)

    plt.title('Infiltration proportions', weight='bold', fontsize=8)

    plt.xlabel('Patient', fontsize=8)
    plt.ylabel('Proportion', fontsize=8)
    plt.xticks(fontsize=8, rotation=35)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-0.1, 1)

    plt.tight_layout()
    plt.savefig(out_dir / f"{XP}_infiltration_{THRESHOLD}_knn.png", dpi=500)
    plt.clf()

    # Boxplot for infiltration
    # plot_immune_infiltration_boxplot(adata, result, cell_types, til_type, out_dir, figsize, name)

    # #Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
    # new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    # new_labels[np.where(adata.obs.layer1 == 'Epithelial')[0]] = ['Epithelial']*len(np.where(adata.obs.layer1 == 'Epithelial')[0])
    # new_labels[np.where(adata.obs.layer1 == 'Stromal')[0]] = ['Stromal']*len(np.where(adata.obs.layer1 == 'Stromal')[0])
    # new_labels[np.where(adata.obs.layer1 == 'Immune')[0]] = adata.obs.layer2.values[np.where(adata.obs.layer1 == 'Immune')[0]]
    #
    # update_labels(adata, 'ct_tils', new_labels)

    # Modify the labels in layer2: convert 'SMA+T Cell' to 'T Cell'
    # adata = read_h5ad(in_dir / f"{XP}_one_img_per_pid.h5ad")
    # adata.obs.loc[adata.obs['layer2'] == 'SMA+T Cell', 'layer2'] = 'T Cell'
    # adata.write(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")

    # # Check the radius happened most in each image
    # radius_list = []
    # for pid in np.unique(adata.obs.patient_key):
    #     p_adata = adata[np.where(adata.obs.patient_key == pid)[0], :]
    #     radius = np.sqrt(p_adata.obs.area / np.pi)
    #     radius_list.append(math.ceil(pd.Series(radius).value_counts().index[0]) * 2)
    #
    # # Compute threshold=8, 9, 10, 11, 12 and plot to see the performance
    # thresholds = [15]
    # for pid in np.unique(adata.obs.patient_key):
    #     print(pid)
    #     p_adata = adata[np.where(adata.obs.patient_key == pid)[0], :]
    #
    #     out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/tils/{pid}")
    #     if os.path.exists(out_dir) == 0:
    #         os.makedirs(out_dir, exist_ok=True)
    #
    #     # Compute neighbors with different thresholds
    #     for threshold in thresholds:
    #         read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=threshold)
    #         plot_neighbor_graph(p_adata, out_dir, name=f"{pid}_{threshold}")

    # Method 2: Use convex to get the estimated area of tumour and count how many immune cells in it
