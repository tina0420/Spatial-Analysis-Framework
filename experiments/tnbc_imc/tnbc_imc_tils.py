"""
@author: Tina Hsu
Calculating TILs (tumour infiltration lymphocytes) in each image of TNBC IMC dataset
"""
import os
from pathlib import Path
from anndata import read_h5ad
import matplotlib.pyplot as plt
import seaborn as sns
import math
from src.preprocess import read_neighbors
from src.spatial_stats import get_interacting_pairs
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from src.annotate import update_labels
from src.spatial_stats import get_interacting_neighborhood_cell_idxs
from src.visualize import plot_neighbor_graph, plot_spatial_cluster_distribution
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, ESI_TILS_COLOR_DICT, CT_TILS_COLOR_DICT, \
    PCR_TF_COLOR_DICT


def compute_tils_categories(adata, unit_key, observation_key, cluster_key, cell_types, target_cell,
                            tils_types, threshold):
    result = {}
    infiltrating_labels = {}

    for pid in np.unique(adata.obs[unit_key]):
        print(pid)
        result[pid] = {}
        infiltrating_labels[pid] = {}

        p_adata = adata[np.where(adata.obs[unit_key] == pid)[0], :]

        # Read the neighbors whose distances are in the threshold
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=threshold)

        for cell_type in cell_types:
            print(cell_type)
            if cell_type == 'Immune':
                result[pid][cell_type] = {}

                # Count edges from Epithelial and Immune (for esi)
                interactions = get_interacting_pairs(p_adata, cluster_key='ESI',
                                                     type_i=target_cell, type_j=cell_type)

                # Compute the number of neighbor does each immune cell has
                infiltrating = []
                bystanding = []
                ignored = []
                tils_labels = p_adata.obs['ESI'].to_list()
                for idx in np.where(p_adata.obs['ESI'] == cell_type)[0]:
                    # When the cell has 3 or more tumour neighbors
                    if np.sum(interactions.flatten() == idx) >= 3:
                        infiltrating.append(idx)
                        tils_labels[idx] = 'Infiltrating'
                    # When the cell has one or two tumour neighbors
                    elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                        bystanding.append(idx)
                        tils_labels[idx] = 'Bystanding'
                    # When the cell has no tumour neighbors
                    else:
                        ignored.append(idx)
                        tils_labels[idx] = 'Ignored'

                result[pid][cell_type][tils_types[0]] = infiltrating
                result[pid][cell_type][tils_types[1]] = bystanding
                result[pid][cell_type][tils_types[2]] = ignored

                infiltrating_labels[pid][cell_type] = tils_labels

            else:
                result[pid][cell_type] = {}

                # Count edges between the cell type and the target cell type (e.g. T Cell and Epithelial)
                interactions = get_interacting_pairs(p_adata, cluster_key=cluster_key,
                                                     type_i=target_cell, type_j=cell_type)

                # Compute the number of neighbor does each immune cell have
                infiltrating = []
                bystanding = []
                ignored = []
                tils_labels = p_adata.obs['ESI'].to_list()
                for idx in np.where(p_adata.obs[cluster_key] == cell_type)[0]:
                    # When the cell has 3 or more tumour neighbors
                    if np.sum(interactions.flatten() == idx) >= 3:
                        infiltrating.append(idx)
                        tils_labels[idx] = 'Infiltrating'
                    # When the cell has one or two tumour neighbors
                    elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                        bystanding.append(idx)
                        tils_labels[idx] = 'Bystanding'
                    # When the cell has no tumour neighbors
                    else:
                        ignored.append(idx)
                        tils_labels[idx] = 'Ignored'

                result[pid][cell_type][tils_types[0]] = infiltrating
                result[pid][cell_type][tils_types[1]] = bystanding
                result[pid][cell_type][tils_types[2]] = ignored

                infiltrating_labels[pid][cell_type] = tils_labels

    # Compute the percentages of each tils categories for different cell types
    print("compute the percentage")
    infiltrating_calculation = []

    for pid, info in result.items():
        observation = adata.obs[adata.obs[unit_key] == pid][observation_key].values[0]
        info_to_add = [pid, observation]
        for cell_type in cell_types:
            total_count = (len(info[cell_type][tils_types[0]]) + len(info[cell_type][tils_types[1]]) +
                           len(info[cell_type][tils_types[2]]))
            if total_count == 0:
                info_to_add.append(0)
                info_to_add.append(0)
                info_to_add.append(0)
            else:
                info_to_add.append(len(info[cell_type][tils_types[0]]) / total_count)
                info_to_add.append(len(info[cell_type][tils_types[1]]) / total_count)
                info_to_add.append(len(info[cell_type][tils_types[2]]) / total_count)

        infiltrating_calculation.append(info_to_add)

    # Create a new dataframe for the infiltration result
    infiltrating_df = pd.DataFrame(infiltrating_calculation,
                                   columns=[unit_key, observation_key] +
                                           [f"{ct}_{tils}" for ct in cell_types for tils in tils_types])

    return result, infiltrating_df, infiltrating_labels


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


if __name__ == '__main__':
    # Method 1: Use neighborhood directly to get the interaction (>=3 -> infiltration)

    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final_thesis'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/summary")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / "TNBC_adata_pretreatment.h5ad")
    # Select for pre treatment
    adata_nat = adata[adata.obs["chemo_type"] == 'NAT']

    if "imm_tils" not in adata_analysis.obs.columns:
        # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial and Stromal
        CLUSTER_KEY = "ESI"
        new_labels = np.zeros(adata_analysis.obs.shape[0], dtype=object)
        new_labels[np.where(adata_analysis.obs[CLUSTER_KEY] == 'Epithelial')[0]] = ['Epithelial'] * len(
            np.where(adata_analysis.obs[CLUSTER_KEY] == 'Epithelial')[0])
        new_labels[np.where(adata_analysis.obs[CLUSTER_KEY] == 'Stromal')[0]] = ['Stromal'] * len(
            np.where(adata_analysis.obs[CLUSTER_KEY] == 'Stromal')[0])
        new_labels[np.where(adata_analysis.obs[CLUSTER_KEY] == 'Immune')[0]] = adata_analysis.obs.layer2.values[
            np.where(adata_analysis.obs[CLUSTER_KEY] == 'Immune')[0]]

        update_labels(adata_analysis, 'imm_tils', new_labels)

    # Infiltration
    # threshold: cell-cell distance under the threshold will be considered as neighbors
    # Check the radius happened most in each image

    """
    Extract epithelial cells labeled as H3K27me3+, all immune cells and all stromal cells
    """
    idx = np.concatenate((np.where(adata_analysis.obs.layer2 == "Epithelial CK5+CK7+")[0],
                          np.where(adata_analysis.obs.layer1 == "TME")[0]))

    adata_analysis_h3k27m3e = adata_analysis[idx, :].copy()
    adata_analysis_basoluminal = adata_analysis[idx, :].copy()
    CHEMOTYPE_COLOR_DICT = {"ADJ": "tab:blue", "NAT": "tab:pink"}

    CLUSTER_KEY = "unique_core_id"
    radius_list = []
    for pid in pd.unique(adata_analysis_basoluminal.obs[CLUSTER_KEY]):
        p_adata = adata_analysis_basoluminal[np.where(adata_analysis_basoluminal.obs[CLUSTER_KEY] == pid)[0], :]
        radius = np.sqrt(p_adata.obs.area / np.pi)
        radius_list.append(math.ceil(pd.Series(radius).value_counts().index[0]) * 2)

    THRESHOLD = np.median(radius_list)  # 10
    UNIT_KEY = 'unique_core_id'
    OBSERVATION_KEY = 'relapse'
    CLUSTER_KEY = 'imm_tils'
    TARGET_CELL = 'Epithelial'
    CELL_TYPES = ['Immune', 'B Cell', 'T Cell', 'Macrophage']
    TILS_TYPES = ['Infiltrating', 'Bystanding', 'Ignored']

    result_dict, infiltration_df, infiltrating_labels_to_add = compute_tils_categories(adata=adata_analysis_basoluminal,
                                                                                       unit_key=UNIT_KEY,
                                                                                       observation_key=OBSERVATION_KEY,
                                                                                       cluster_key=CLUSTER_KEY,
                                                                                       cell_types=CELL_TYPES,
                                                                                       target_cell=TARGET_CELL,
                                                                                       tils_types=TILS_TYPES,
                                                                                       threshold=THRESHOLD)
    PCR_TF_COLOR_DICT = {'No': 'tab:blue',
                         'Yes': 'tab:pink'}

    plot_tils_comparison(infiltrating_df=infiltration_df,
                         observation_key=OBSERVATION_KEY,
                         cell_types=CELL_TYPES,
                         tils_types=TILS_TYPES,
                         threshold=THRESHOLD,
                         color_dict=RELAPSE_TF_COLOR_DICT,
                         out_dir=out_dir,
                         figtype='box',
                         show_palue=True,
                         figsize=(15, 6),
                         name="Recurrence_roi_basoluminal")

    # Average the infiltration from the same patients and plot again
    # Add a new patient column to infiltration_df (link the image_roi to cid)
    mapping_dict = dict(zip(adata_analysis_basoluminal.obs[UNIT_KEY].values, adata_analysis_basoluminal.obs.cid.values))
    infiltration_df["cid"] = [mapping_dict[x] for x in infiltration_df[UNIT_KEY].values]

    infiltration_df_by_pid = infiltration_df.groupby("cid", as_index=False)[['Immune_Infiltrating', 'Immune_Bystanding',
                                                                             'Immune_Ignored', 'B Cell_Infiltrating',
                                                                             'B Cell_Bystanding',
                                                                             'B Cell_Ignored', 'T Cell_Infiltrating',
                                                                             'T Cell_Bystanding',
                                                                             'T Cell_Ignored',
                                                                             'Macrophage_Infiltrating',
                                                                             'Macrophage_Bystanding',
                                                                             'Macrophage_Ignored']].mean()

    mapping_dict = dict(
        zip(adata_analysis_basoluminal.obs.cid.values, adata_analysis_basoluminal.obs[OBSERVATION_KEY].values))
    infiltration_df_by_pid[OBSERVATION_KEY] = [mapping_dict[x] for x in infiltration_df_by_pid.cid.values]

    plot_tils_comparison(infiltrating_df=infiltration_df_by_pid,
                         observation_key=OBSERVATION_KEY,
                         cell_types=CELL_TYPES,
                         tils_types=TILS_TYPES,
                         threshold=THRESHOLD,
                         color_dict=RELAPSE_TF_COLOR_DICT,
                         out_dir=out_dir,
                         figtype='box',
                         show_palue=True,
                         figsize=(15, 6),
                         name="Recurrence_pid_basoluminal")

    """
    Group the patients based on the infiltration result for each immune cell type, one by one separately
    NO NEED PROPORTION, COUNT ONLY
    (Not for comparing the recurrence and pathCR)
    """
    CELL_TYPE = "Macrophage"
    UNIT_KEY = "unique_core_id"
    THRESHOLD = 10
    CLUSTER_KEY = 'imm_tils'
    TARGET_CELL = 'Epithelial'
    # CELL_TYPES = ['Immune', 'B Cell', 'T Cell', 'Macrophage']
    TILS_TYPES = ['Infiltrating', 'Bystanding', 'Ignored']

    result = {}
    infiltrating_labels = {}
    count_cells = []
    for pid in np.unique(adata.obs[UNIT_KEY]):
        print(pid)
        result[pid] = {}
        infiltrating_labels[pid] = {}

        p_adata = adata[np.where(adata.obs[UNIT_KEY] == pid)[0], :]

        # Read the neighbors whose distances are in the threshold
        read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=THRESHOLD)

        if CELL_TYPE == 'Immune':
            result[pid][CELL_TYPE] = {}

            # Count edges from Epithelial and Immune (for esi)
            interactions = get_interacting_pairs(p_adata, cluster_key='ESI',
                                                 type_i=TARGET_CELL, type_j=CELL_TYPE)

            # Compute the number of neighbor does each immune cell has
            infiltrating = []
            bystanding = []
            ignored = []
            tils_labels = p_adata.obs['ESI'].to_list()
            for idx in np.where(p_adata.obs['ESI'] == CELL_TYPE)[0]:
                # When the cell has 3 or more tumour neighbors
                if np.sum(interactions.flatten() == idx) >= 3:
                    infiltrating.append(idx)
                    tils_labels[idx] = 'Infiltrating'
                # When the cell has one or two tumour neighbors
                elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                    bystanding.append(idx)
                    tils_labels[idx] = 'Bystanding'
                # When the cell has no tumour neighbors
                else:
                    ignored.append(idx)
                    tils_labels[idx] = 'Ignored'

            result[pid][CELL_TYPE][TILS_TYPES[0]] = infiltrating
            result[pid][CELL_TYPE][TILS_TYPES[1]] = bystanding
            result[pid][CELL_TYPE][TILS_TYPES[2]] = ignored

            infiltrating_labels[pid][CELL_TYPE] = tils_labels

            # Count the cells in each group, image_key, patient_key, #infiltrating, #bystanding, #ignored
            count_cells.append([pid, p_adata.obs.cid.values[0], len(result[pid][CELL_TYPE][TILS_TYPES[0]]),
                                len(result[pid][CELL_TYPE][TILS_TYPES[1]]), len(result[pid][CELL_TYPE][TILS_TYPES[2]])])

        else:
            result[pid][CELL_TYPE] = {}

            # Count edges between the cell type and the target cell type (e.g. T Cell and Epithelial)
            interactions = get_interacting_pairs(p_adata, cluster_key=CLUSTER_KEY,
                                                 type_i=TARGET_CELL, type_j=CELL_TYPE)

            # Compute the number of neighbor does each immune cell have
            infiltrating = []
            bystanding = []
            ignored = []
            tils_labels = p_adata.obs['ESI'].to_list()
            for idx in np.where(p_adata.obs[CLUSTER_KEY] == CELL_TYPE)[0]:
                # When the cell has 3 or more tumour neighbors
                if np.sum(interactions.flatten() == idx) >= 3:
                    infiltrating.append(idx)
                    tils_labels[idx] = 'Infiltrating'
                # When the cell has one or two tumour neighbors
                elif (np.sum(interactions.flatten() == idx) == 2) | (np.sum(interactions.flatten() == idx) == 1):
                    bystanding.append(idx)
                    tils_labels[idx] = 'Bystanding'
                # When the cell has no tumour neighbors
                else:
                    ignored.append(idx)
                    tils_labels[idx] = 'Ignored'

            result[pid][CELL_TYPE][TILS_TYPES[0]] = infiltrating
            result[pid][CELL_TYPE][TILS_TYPES[1]] = bystanding
            result[pid][CELL_TYPE][TILS_TYPES[2]] = ignored

            infiltrating_labels[pid][CELL_TYPE] = tils_labels
            # Count the cells in each group, image_key, patient_key, #infiltrating, #bystanding, #ignored
            count_cells.append([pid, p_adata.obs.cid.values[0], len(result[pid][CELL_TYPE][TILS_TYPES[0]]),
                                len(result[pid][CELL_TYPE][TILS_TYPES[1]]), len(result[pid][CELL_TYPE][TILS_TYPES[2]])])

    # Summary the count_cells list to df
    tils_cell_counts_df = pd.DataFrame(count_cells,
                                       columns=[UNIT_KEY, "cid", TILS_TYPES[0], TILS_TYPES[1], TILS_TYPES[2]])
    tils_cell_counts_df['Ratio'] = np.where(tils_cell_counts_df[TILS_TYPES[0]] != 0,
                                            tils_cell_counts_df[TILS_TYPES[1]] / tils_cell_counts_df[TILS_TYPES[0]], 0)

    # Give the groups of each image (then need to check if the group assigned are the same form one patient)
    groups = []
    for i in range(tils_cell_counts_df.shape[0]):

        if tils_cell_counts_df["Ratio"].values[i] < 10 and tils_cell_counts_df[TILS_TYPES[0]].values[i] > 20:
            groups.append("Inflamed")
        elif tils_cell_counts_df["Ratio"].values[i] >= 8:
            groups.append("Excluded")
        elif tils_cell_counts_df[TILS_TYPES[0]].values[i] + tils_cell_counts_df[TILS_TYPES[1]].values[i] <= 20:
            groups.append("Cold")
        else:

            groups.append("Pending")
            # groups.append("Cold")

    tils_cell_counts_df["Groups"] = groups

    # Check if a patient has more than one groups from the corresponding images
    for pid in set(tils_cell_counts_df["cid"]):
        if len(set(tils_cell_counts_df[tils_cell_counts_df["cid"] == pid]["Groups"])) > 1:
            print(pid)

    # Output the csv file
    # # I forgot to remove the images with no epithelial cells
    # keep_img = []
    # UNIT_KEY = "unique_core_id"
    # CLUSTER_KEY = 'layer1'
    # for img in pd.unique(adata.obs[UNIT_KEY]):
    #     p_adata = adata[adata.obs[UNIT_KEY] == img]
    #     if p_adata.obs[CLUSTER_KEY].value_counts()["Epithelial"] / p_adata.shape[0] > 0.1:
    #         keep_img.append(img)  # keep 83 images, there were 90 initially
    # output_csv_tils_hot_cold = tils_cell_counts_df[tils_cell_counts_df.unique_core_id.isin(keep_img)]
    # output_csv_tils_hot_cold.to_csv(out_dir / "Immune_hot_cold.csv", index=False)
    tils_cell_counts_df.to_csv(out_dir / "T Cell_hot_cold.csv", index=False)

    # Read the tils Immune file
    df = pd.read_csv(out_dir / "T Cell_hot_cold.csv")

    # Modify the groups of patients if one has more than more result
    get_idx = []
    for pid in set(df.cid):
        sub_df = df[df.cid == pid]
        if len(set(sub_df.Groups)) > 1:
            if "Inflamed" in set(sub_df["Groups"]):
                get_idx.append(sub_df[sub_df.Groups == "Inflamed"].index[0])
            elif "Excluded" in set(sub_df["Groups"]):
                get_idx.append(sub_df[sub_df.Groups == "Excluded"].index[0])
        else:
            idx = sub_df.index[0]
            get_idx.append(idx)

    df_recu = df.iloc[get_idx, :].copy()

    # For neoadj patients
    neo_pid = list(set(adata_analysis[np.where(adata_analysis.obs.chemo_type == "NAT")[0], :].obs.cid))
    df_nat = df_recu[df_recu.cid.isin(neo_pid)].copy()

    # Merge outcome into it
    outcome_mapping = {}
    for pid in set(adata_analysis.obs.cid):
        p_adata = adata_analysis[adata_analysis.obs["cid"] == pid]
        outcome_mapping[pid] = p_adata.obs.pCR.values[0]

    df_nat["PathCR"] = [outcome_mapping[pid] for pid in df_nat.cid.values]

    counts = pd.crosstab(df_nat["PathCR"], df_nat["Groups"])

    # 2. Convert counts to proportions (normalize by the total per store)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    # 3. Create the stacked bar plot
    ax = proportions.plot(kind='bar',
                          stacked=True,
                          figsize=(8, 6),
                          color=['#66b3ff', '#99ff99', '#ff9999'])  # Custom colors for apple, banana, mango

    # Add labels and formatting
    plt.title('Proportion of Fruits per Store')
    plt.xlabel('Store')
    plt.ylabel('Proportion (0.0 to 1.0)')
    plt.xticks(rotation=0)  # Keep store names horizontal
    plt.legend(title='Fruits', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Use Chi-Square test to compare these two groups
    from scipy.stats import chi2_contingency

    list_a = list(proportions.iloc[0, :])
    list_a = [0.233333, 0.7666660000000001]
    list_b = list(proportions.iloc[1, :])
    list_b = [0.333333, 0.666667]

    n_a = 100
    n_b = 100

    # 3. Convert proportions to counts
    counts_a = [p * n_a for p in list_a]
    counts_b = [p * n_b for p in list_b]

    # 4. Create the contingency table
    table = [counts_a, counts_b]

    # 5. Run the Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(table)

    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    """
    Visualization of the cell distribution per core 
    """
    # Add the tils category of each cell for plotting and plot the distribution of tils
    pids = []
    for i in adata.obs[UNIT_KEY].values:
        if i in pids:
            continue
        else:
            pids.append(i)

    for ct in CELL_TYPES:
        new_labels = []
        for pid in pids:
            print(pid)
            new_labels = new_labels + infiltrating_labels_to_add[pid][ct]

        update_labels(adata, cluster_key=f"{ct}_infiltrates", labels=new_labels)

    # Plot the tils distribution for each patient and each cell type
    cell_type = 'Immune'
    in_dir = out_dir
    for pid in np.unique(adata.obs[UNIT_KEY]):
        print(pid)
        p_adata = adata[np.where(adata.obs[UNIT_KEY] == pid)[0], :]
        idx = p_adata.obs.cid.values[0] + "_" + pid
        CLUSTER_KEY = f"{cell_type}_infiltrates"

        if p_adata.obs[OBSERVATION_KEY].values[0] == 'No':
            out_dir = Path(in_dir / f"{cell_type}/No")
            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)

        elif p_adata.obs[OBSERVATION_KEY].values[0] == 'Yes':
            out_dir = Path(in_dir / f"{cell_type}/Yes")
            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)

        else:
            print(f"{pid}, with wrong outcome!")

        if list(set(p_adata.obs.tma))[0] == 'BOB':
            plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=3,
                                              color=CT_TILS_COLOR_DICT, name=f"{idx}_{cell_type}")
        else:
            plot_spatial_cluster_distribution(p_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=2,
                                              color=CT_TILS_COLOR_DICT, name=f"{idx}_{cell_type}")

    # adata.write(out_dir / f"imm_tils_record_all_epi.h5ad")
    #

    #
    # # Add a new column in adata.obs with T Cell, B Cell, Macrophage, Epithelial, Stromal and Endothelial
    #
    # # new_labels = np.zeros(adata.obs.shape[0], dtype=object)
    # # new_labels[np.where(adata.obs.layer1 == 'Epithelial')[0]] = ['Epithelial']*len(np.where(adata.obs.layer1 == 'Epithelial')[0])
    # # new_labels[np.where(adata.obs.layer1 == 'Stromal')[0]] = adata.obs.layer2.values[np.where(adata.obs.layer1 == 'Stromal')[0]]
    # # # new_labels[np.where(adata.obs.layer1 == 'Immune')[0]] = adata.obs.layer2.values[np.where(adata.obs.layer1 == 'Immune')[0]]
    # # new_labels[np.where(adata.obs.layer1 == 'Immune')[0]] = ['Immune']*len(np.where(adata.obs.layer1 == 'Immune')[0])
    # #
    # # update_labels(adata, 'immune_endo', new_labels)
    #
    # # # Compute threshold=8, 9, 10, 11, 12 and plot to see the performance
    # # thresholds = [15]
    # # for pid in np.unique(adata.obs.patient_key):
    # #     print(pid)
    # #     p_adata = adata[np.where(adata.obs.patient_key == pid)[0], :]
    # #
    # #     out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/tils/{pid}")
    # #     if os.path.exists(out_dir) == 0:
    # #         os.makedirs(out_dir, exist_ok=True)
    # #
    # #     # Compute neighbors with different thresholds
    # #     for threshold in thresholds:
    # #         read_neighbors(p_adata, rel_path=out_dir / "rel.csv", threshold=threshold)
    # #         plot_neighbor_graph(p_adata, out_dir, name=f"{pid}_{threshold}")
    #
    # # Method 2: Use convex to get the estimated area of tumour and count how many immune cells in it
