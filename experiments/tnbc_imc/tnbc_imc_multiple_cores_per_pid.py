"""
@author: Tina Hsu
Comparing the images/cores from the same patients
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT, LAYER2_CELL_TYPE_COLOR_DICT
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import gridspec


def compute_img_correlation(adata, patient_key, image_key, cluster_key):
    """
    Computing the correlation/similarity of the images by cell type proportions
    :param adata: Anndata
    :param patient_key: indicate the primary key of image used in adata.obs , type: str
    :param image_key:indicate the primary key of patient in adata.obs, type: str
    :param cluster_key: indicate the column used in adata.obs, type: str
    :return:
    """
    # Get the cid-img mapping dictionary
    pid_img = {}
    for pid in np.unique(adata.obs[patient_key]):
        pid_num = pid.split("_")[0]
        pid_img[pid_num] = list(set(adata.obs[image_key][adata.obs[patient_key].str.startswith(pid_num)]))

    # computation of cell type proportions and the correlation between the images
    all_pid_prop = {}
    for pid, img in pid_img.items():
        pid_record = {}
        # If the patient only has one image then no need to compare
        if len(img) == 1:
            continue

        img_prop = []
        for roi in sorted(img):
            # Compute the proportion of each cell type in the cluster_key
            img_prop.append([roi] +
                            list(adata[adata.obs[image_key] == roi].obs[cluster_key].value_counts().sort_index() /
                                 adata[adata.obs[image_key] == roi].shape[0])
                            )

            # Get the index name (in the order)
        idx_name = list(adata[adata.obs[image_key] == roi].obs[cluster_key].value_counts().sort_index().index)

        pid_record['prop'] = pd.DataFrame(img_prop, columns=['roi'] + idx_name).set_index('roi')

        # Calculate the correlation coefficient by Spearman Correlation
        corr, pval = spearmanr(pid_record['prop'], axis=1)

        # The format of corr becomes an array when there are more than two variables
        if isinstance(corr, np.ndarray):
            corr_df = pd.DataFrame(corr,
                                   index=sorted(img),
                                   columns=sorted(img))
            pid_record["corr"] = corr_df

        else:
            pid_record["corr"] = corr

        all_pid_prop[pid] = pid_record

    return all_pid_prop


def plot_ct_prop_pid_panel_split(all_pid_prop, patient_key, image_key, set_count,
                                 out_dir, color_dict, name=None):
    # Get the pairs for split pages
    pairs = construct_range(len(all_pid_prop.keys()), set_count)

    # Plotting, bar plot for proportions in each patient
    for pair in range(len(pairs)):
        print(pairs[pair])
        pid_list = list(all_pid_prop.keys())[pairs[pair][0]: pairs[pair][1]]

        fig, axes = plt.subplots(2, 5, figsize=(12, 2 * 1.4))
        for i, ax in enumerate(fig.axes):
            # Make the blocks empty if we don't need them
            if i > len(pid_list) - 1:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            else:
                # Get the info of the patient
                pid = pid_list[i]

                # Basic numbers for x-axis
                bottom = [0] * all_pid_prop[pid]['prop'].shape[0]

                # Plot and label each component (e.g. each sub-bar for ESI)
                for ct in all_pid_prop[pid]['prop'].columns:
                    values = list(all_pid_prop[pid]['prop'][ct])
                    bars = ax.bar(range(all_pid_prop[pid]['prop'].shape[0]),
                                  values, bottom=bottom, label=ct, color=color_dict[ct])

                    # Add proportions to each bar section
                    for idx, (bar, value) in enumerate(zip(bars, values)):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bottom[idx] + value / 2,
                            f"{value:.2f}",
                            ha='center', va='center', color='black', fontsize=6
                        )

                    # Update bottom for stacking
                    bottom = [j + k for j, k in zip(bottom, values)]

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # show the corr on the title (temporary)
                if isinstance(all_pid_prop[pid]["corr"], pd.DataFrame):
                    ax.set_title(f'{pid}', weight='bold', fontsize=7)
                else:
                    ax.set_title(f'{pid}, coef={all_pid_prop[pid]["corr"]}', weight='bold', fontsize=7)

                ax.set_xlabel('', fontsize=6)
                ax.set_ylabel('Proportion', fontsize=6)

                ax.set_xticks(range(all_pid_prop[pid]['prop'].shape[0]))

                xtick_labels = [label for label in all_pid_prop[pid]['prop'].index]

                # Get the count of each group (cell numbers)
                counts = adata[np.where((adata.obs[patient_key].str.startswith(pid)))[0], :].obs[
                    image_key].value_counts()

                # Build new labels in the same order as plotted
                new_labels = [f"{cat}\n(cells={counts[cat]})" for cat in xtick_labels]

                # Set the new labels
                ax.set_xticklabels(new_labels)
                ax.tick_params(axis='x', labelsize=6)

                ax.set_ylim(-0.1, 1.1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.tick_params(axis='y', labelsize=6)

                # only remain the y-label and y-ticks at the figure 'L'
                if i != 0 or i != 5:
                    ax.set_yticks([])
                    ax.set_ylabel("")
                    continue

        # plt.legend(prop={'size': 6}, loc='upper left', bbox_to_anchor=(1.05, 1.9))

        plt.tight_layout()
        if name:
            plt.savefig(out_dir / f"{name}_pid_panel_{pair}.png", dpi=500)
        else:
            plt.savefig(out_dir / f"pid_panel_{pair}.png", dpi=500)

        plt.clf()


def plot_cell_type_prop_bar(adata, cluster_key, unit_key, out_dir, color_dict,
                            figsize=(55, 25), cell_type_list=None,
                            show_dendrogram=False, name=None):
    # Compute the cell type proportions and stored them into a dataframe
    if cell_type_list is None:
        cell_type_list = list(set(adata.obs[cluster_key]))

    ct_prop_each_unit = []
    for i in np.unique(adata.obs[unit_key]):
        i_obs = adata[adata.obs[unit_key] == i, :].obs[cluster_key]
        counts = dict(i_obs.value_counts())

        ct_prop_each_unit.append([counts.get(ct, 0) / sum(counts.values()) for ct in cell_type_list])

    ct_prop_df = pd.DataFrame(ct_prop_each_unit, columns=cell_type_list,
                              index=list(np.unique(adata.obs[unit_key])))

    # Store the calculation result into adata.uns
    adata.uns[f"{unit_key}_all_ct_prop"] = ct_prop_df

    # Running hierarchical clustering
    hc = linkage(ct_prop_df, method='ward')

    # Bar plot for visualization, including stacked bar chart and dendrogram
    print("start plotting")

    if show_dendrogram:
        fig = plt.subplots(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.05)
        # Create subplots for bar chart and dendrogram individually
        ax_0 = plt.subplot(gs[0])
        ax_1 = plt.subplot(gs[1])

        # Add dendrogram above the bar plot
        # Get dendrogram result
        dendrograms = dendrogram(hc, labels=ct_prop_df.index, ax=ax_0)

        # Remove x-axis of dendrogram
        ax_0.set_xticks([])
        ax_0.set_yticks([])
        ax_0.spines['right'].set_visible(False)
        ax_0.spines['top'].set_visible(False)
        ax_0.spines['left'].set_visible(False)
        ax_0.spines['bottom'].set_visible(False)

        # Reorder the bars based on the dendrogram result
        new_order = [ct_prop_df.index[i] for i in dendrograms['leaves']]

        ct_prop_df = ct_prop_df.loc[new_order, :]

        # Identification of stacking which starts from 0 (y=0)
        bottom = np.zeros(ct_prop_df.shape[0])

        # Plot through each bar and then stack the bar
        for cell_type in cell_type_list:
            values = ct_prop_df[cell_type]

            bars = ax_1.bar(range(ct_prop_df.shape[0]), values, bottom=bottom, label=cell_type,
                            color=color_dict[cell_type],
                            edgecolor='black', width=1.0)

            # Update bottom for stacking
            bottom = [i + j for i, j in zip(bottom, values)]

        # Remove the spines
        ax_1.spines['top'].set_visible(False)
        ax_1.spines['bottom'].set_visible(False)
        ax_1.spines['right'].set_visible(False)
        ax_1.spines['left'].set_visible(False)

        ax_1.set_xticks(range(ct_prop_df.shape[0]))
        ax_1.set_xticklabels(['' for _ in range(ct_prop_df.shape[0])])
        ax_1.set_ylim(-0.1, 1.1)
        ax_1.set_yticks([0, 0.5, 1])
        ax_1.tick_params(axis='y', labelsize=10)
        ax_1.set_ylabel('Proportion', fontsize=10)

    else:
        fix, ax = plt.subplots(1, 1, figsize=figsize)

        # Get dendrogram result
        dendrograms = dendrogram(hc, labels=ct_prop_df.index, no_plot=True)

        # Reorder the bars based on the dendrogram result
        new_order = [ct_prop_df.index[i] for i in dendrograms['leaves']]

        ct_prop_df = ct_prop_df.loc[new_order, :]

        # Identification of stacking which starts from 0 (y=0)
        bottom = np.zeros(ct_prop_df.shape[0])

        # Plot through each bar and then stack the bar
        for cell_type in cell_type_list:
            values = ct_prop_df[cell_type]

            bars = ax.bar(range(ct_prop_df.shape[0]), values, bottom=bottom, label=cell_type,
                            color=color_dict[cell_type],
                            edgecolor='black', width=1.0)

            # Update bottom for stacking
            bottom = [i + j for i, j in zip(bottom, values)]

        # Remove the spines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_xticks(range(ct_prop_df.shape[0]))
        ax.set_xticklabels(list(ct_prop_df.index), rotation=90, ha='right')
        ax.tick_params(axis='x', labelsize=20)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylabel('Proportion', fontsize=20)

    plt.legend(fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_multiple_ct_prop_bar.png", dpi=500)
    else:
        plt.savefig(out_dir / "esi_multiple_ct_prop_bar.png", dpi=500)

    plt.clf()


if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific'
    XP = 'cohort_tnbc'
    UNIT_KEY = 'image_key'
    IMAGE_KEY = 'image_key'
    PATIENT_KEY = 'patient_key'

    in_dir = out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    # out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/multiple_cores")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad file with all cohort patients in 3 TMAs, 99 images
    adata = read_h5ad(in_dir / f"{XP}_all_TNBC_imgs.h5ad")

    # Check the numbers of patients and images
    print("images: ", str(len(set(adata.obs[IMAGE_KEY]))))
    print("patients: ", str(len(set([x.split("_")[0] for x in set(adata.obs[PATIENT_KEY])]))))

    # Compare cell type distribution in the images from the same patients

    # start to compare the distributions of cell types
    # For quickly finish the task here but need to be integrated and modified
    # with the functions in tnbc_imc_cell_type_abundance.py
    CLUSTER_KEY = 'layer1'

    all_pid_prop = compute_img_correlation(adata=adata,
                                           patient_key=PATIENT_KEY,
                                           image_key=IMAGE_KEY,
                                           cluster_key=CLUSTER_KEY)

    # Bar subplots for showing the ct proportions of all patients one by one
    plot_ct_prop_pid_panel_split(all_pid_prop=all_pid_prop,
                                 patient_key=PATIENT_KEY,
                                 image_key=IMAGE_KEY,
                                 set_count=10,
                                 out_dir=out_dir,
                                 color_dict=ESI_CELL_TYPE_COLOR_DICT,
                                 name=f"{XP}_all")

    # get the subset of the adata by removing the images with no or less than 10% of epithelial cells
    len(set(adata.obs.image_key))  # 99
    image_list = []
    for img in pd.unique(adata.obs[IMAGE_KEY]):
        img_adata = adata[np.where(adata.obs.image_key == img)[0], :].copy()
        # Check the epithelial proportion in layer 1
        if img_adata.obs['layer1'].value_counts()['Epithelial'] / img_adata.shape[0] >= 0.1:
            image_list.append(img)

    """
    REMOVE the images with no or less than 10% epithelial cells 
    before this step (checking image correlations from the same patient)
    """
    image_to_remove = []
    for img in pd.unique(adata.obs[IMAGE_KEY]):
        img_adata = adata[np.where(adata.obs[IMAGE_KEY] == img)[0], :].copy()
        # Check the epithelial proportion in layer 1
        if img_adata.obs['layer1'].value_counts()['Epithelial'] / img_adata.shape[0] < 0.1:
            image_to_remove.append(img)

    filter_img_list = [x for x in set(adata.obs.image_key) if x not in image_to_remove]

    # temp to see if any images missed
    temp_adata = adata[adata.obs.image_key.isin(filter_img_list)]
    # plot to see how different comparing to the previous one

    all_pid_prop = compute_img_correlation(adata=temp_adata,
                                           patient_key=PATIENT_KEY,
                                           image_key=IMAGE_KEY,
                                           cluster_key=CLUSTER_KEY)

    plot_ct_prop_pid_panel_split(all_pid_prop=all_pid_prop,
                                 patient_key=PATIENT_KEY,
                                 image_key=IMAGE_KEY,
                                 set_count=10,
                                 out_dir=out_dir,
                                 color_dict=ESI_CELL_TYPE_COLOR_DICT,
                                 name=f"{XP}_rm-no_epi")

    # Plot the cell type bar charts in the order by dendrogram
    CELL_TYPE_LIST = ['Stromal', 'Immune', 'Epithelial']
    CELL_TYPE_LIST = ['TME Cluster 9', 'Vimentin+', 'Fibroblast', 'Smooth Muscle Cell', 'Blood Endothelial',
                      'Macrophage', 'T Cell', 'B Cell',
                      'Myoepithelial H3K27me3+', 'Myoepithelial', 'Ki67+', 'Epithelial H3K27me3+',
                      'Epithelial CK_low', 'Epithelial CK8-18+', 'Epithelial CK7+', 'Epithelial CK5+CK8-18+',
                      'Epithelial CK5+CK7+', 'Epithelial CK5+']

    CLUSTER_KEY = 'layer1'
    plot_cell_type_prop_bar(adata=adata,
                            cluster_key=CLUSTER_KEY,
                            unit_key=UNIT_KEY,
                            out_dir=out_dir,
                            color_dict=ESI_CELL_TYPE_COLOR_DICT,
                            figsize=(55, 30),
                            cell_type_list=CELL_TYPE_LIST,
                            show_dendrogram=False,
                            name=f"{XP}_layer1_with_img_names")

