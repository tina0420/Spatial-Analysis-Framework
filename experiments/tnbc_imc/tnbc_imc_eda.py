"""
TNBC IMC Exploratory Data Analysis
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, PCR_TF_COLOR_DICT, CHEMOTYPE_COLOR_DICT
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels


def compare_intensity_panel(adata, marker, cluster_key, cell_type_list, observation_key, out_dir, color_dict,
                            observation_order=None, name=None):
    """
    Compare marker intensity of each category (e.g. cell type) across different observations
    :param adata: Anndata
    :param marker: indicate which marker intensity to compare, type: str
    :param cluster_key: indicate the column used in obs to show comparison, type: str
    :param observation_key: indicate the column in obs with observation, type: str
    :param out_dir: output directory, type: Path
    :param color_dict: colour setting for observations, type: dict
    :param observation_order: the order of observations showing in the figure, type: list[str]
    :param name: name for the output file, type: str
    :return:
    """
    # Get the index of the marker in X
    idx = np.where(adata.var_names == marker)[0][0]

    # Get two groups in the observations
    group1 = adata[(adata.obs[observation_key] == observation_order[0])]
    group2 = adata[(adata.obs[observation_key] == observation_order[1])]

    # Plot
    rows = int(len(cell_type_list) / 5) + 1

    fig, axes = plt.subplots(rows, 5, figsize=(12, rows * 1.4))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(cell_type_list) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = cell_type_list[i]
        print(ct)
        sub_group1 = list(group1.X[np.where(group1.obs[cluster_key] == ct)[0], idx])
        sub_group2 = list(group2.X[np.where(group2.obs[cluster_key] == ct)[0], idx])

        sns.violinplot(data=[sub_group1, sub_group2],
                       palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        # Add lines from the median values
        ax.plot([0, 1], [np.median(sub_group1), np.median(sub_group2)], color='red', linestyle='--', linewidth=1,
                zorder=1)

        ax.set_yticks([0, np.round(max(max(sub_group1), max(sub_group2)), 1)])

        # Calculate p-value
        stat, pval = mannwhitneyu(sub_group1, sub_group2, alternative='two-sided')

        # Add p-value to the figure
        if pval < 0.01:
            ax.text(0.5, 0.8, "p < 0.01", ha='center', va='center', fontsize=7, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.8, f'p = {pval:.3f}', ha='center', va='center', fontsize=7, transform=ax.transAxes)

        # Add x-ticks and legend
        ax.set_xticks([0, 1])
        ax.set_xticklabels(observation_order, size=8)

        ax.grid(False)

        ax.set_title(
            cell_type_list[i], weight="bold", fontsize=8)

    # Add the marker name to the y-label of the figure
    fig.supylabel(f"{marker} intensity", fontsize=10, fontweight='bold')

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]], label=observation_order[0])
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]], label=observation_order[1])

    fig.legend(title=name,
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=1,
               fontsize=8,
               title_fontsize=8
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_{marker}_intensity_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "ct_marker_intensity_panel.png", dpi=500)

    plt.clf()

# Useful code for extracting info from adata.X and adata.obs
# df = pd.DataFrame(adata.X[:, idx], columns=[target_marker]).assign(**{cluster_key: adata.obs[cluster_key].values,


def compare_cell_type_proportion(adata, unit_key, target_ct_group_dict, cluster_key, observation_key, observation_order,
                                 out_dir, color_dict, name=None, marker_focus_key=None):
    props = []
    for unit in pd.unique(adata.obs[unit_key]):
        # To focus on a specific marker PN column in obs
        if marker_focus_key:
            p_adata = adata[adata.obs[unit_key] == unit].obs[
                ["unique_core_id", "cid", cluster_key, marker_focus_key, observation_key]]

            for target_ct in list(target_ct_group_dict.keys()):
                print(target_ct)
                if target_ct not in p_adata[cluster_key].values:
                    prop = 0
                else:
                    # Need to modify based on the representative of groups 0 and 1
                    sub_df = p_adata[p_adata[cluster_key] == target_ct]
                    prop = sub_df[marker_focus_key].value_counts()[0] / sub_df[cluster_key].value_counts()[target_ct]
                    # prop = p_adata[marker_focus_key].value_counts()[0] / p_adata[cluster_key].value_counts()[
                    #     target_ct_group_dict[target_ct]]

                props.append([unit, target_ct, p_adata[observation_key].values[0], prop])

        else:
            p_adata = adata[adata.obs[unit_key] == unit].obs[
                ["unique_core_id", "cid", "layer1", cluster_key, observation_key]]

            for target_ct in list(target_ct_group_dict.keys()):
                if target_ct not in p_adata[cluster_key].values:
                    prop = 0
                else:
                    prop = p_adata[cluster_key].value_counts()[target_ct] / p_adata["layer1"].value_counts()[
                        target_ct_group_dict[target_ct]]

            props.append([unit, target_ct, p_adata[observation_key].values[0], prop])

    props_df = pd.DataFrame(props, columns=[unit_key, "cell type", observation_key, "prop"])

    # Split groups based on the observation, calculate p-value and plot
    rows = int(len(target_ct_group_dict.keys()) / 5) + 1

    if observation_order:
        # Get two groups in the observations
        group1 = props_df[props_df[observation_key] == observation_order[0]]
        group2 = props_df[props_df[observation_key] == observation_order[1]]
    else:
        group1 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[0]]
        group2 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[1]]

    fig, axes = plt.subplots(rows, 5, figsize=(10, rows * 1.4))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(target_ct_group_dict.keys()) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = list(target_ct_group_dict.keys())[i]

        sns.boxplot(data=[list(group1[group1["cell type"] == ct]["prop"].values),
                          list(group2[group2["cell type"] == ct]["prop"].values)],
                    palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        stat, pval = mannwhitneyu(list(group1[group1["cell type"] == ct]["prop"].values),
                                  list(group2[group2["cell type"] == ct]["prop"].values),
                                  alternative='two-sided')

        # Add p-value to the figure
        if pval < 0.01:
            ax.text(0.5, 0.9, "p<0.01", ha='center', va='top', fontsize=7)
            # ax.text(1.4, 0.9, "p<0.01", ha='center', va='top', fontsize=7)
        else:
            ax.text(0.5, 0.9, f'p = {pval:.2f}', ha='center', va='top', fontsize=7)
        print(ct)
        print(f'p = {pval:.2f}')
        print("____________________________")

        # Add x-ticks and legend
        ax.set_xticks([0, 1])
        ax.set_xticklabels(observation_order, size=8)

        ax.grid(False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([])

        ax.set_title(ct, weight="bold", fontsize=9)

    # Add the marker name to the y-label of the figure
    fig.supylabel(f"{marker_focus_key}+ fraction", fontsize=10, fontweight='bold')

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]],
                            label=observation_order[0] + " (n=" + str(len(set(group1["cid"]))) + ")")
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]],
                            label=observation_order[1] + " (n=" + str(len(set(group2["cid"]))) + ")")

    fig.legend(title=name.split("_")[0],
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=1,
               fontsize=8,
               title_fontsize=8
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}+_cell_type_proportions_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_type_proportions_panel.png", dpi=500)

    plt.clf()


def compare_cell_niche_proportion(adata, unit_key, target_ct_group_dict, cluster_key, observation_key,
                                  observation_order,
                                  out_dir, color_dict, name=None):
    props = []
    for unit in pd.unique(adata.obs[unit_key]):
        p_adata = adata[adata.obs[unit_key] == unit].obs[
            ["unique_core_id", "cid", "layer1", cluster_key, observation_key]]

        for target_ct in list(target_ct_group_dict.keys()):
            if target_ct not in p_adata[cluster_key].values:
                prop = 0
            else:
                prop = p_adata[cluster_key].value_counts()[target_ct] / p_adata.shape[0]

            props.append([unit, target_ct, p_adata[observation_key].values[0], prop])

    props_df = pd.DataFrame(props, columns=[unit_key, "cell type", observation_key, "prop"])

    # Split groups based on the observation, calculate p-value and plot
    rows = int(len(target_ct_group_dict.keys()) / 5) + 1

    if observation_order:
        # Get two groups in the observations
        group1 = props_df[props_df[observation_key] == observation_order[0]]
        group2 = props_df[props_df[observation_key] == observation_order[1]]
    else:
        group1 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[0]]
        group2 = props_df[props_df[observation_key] == list(set(adata.obs[cluster_key]))[1]]

    fig, axes = plt.subplots(rows, 5, figsize=(10, rows * 1.4))
    for i, ax in enumerate(fig.axes):
        print(i)
        if i > len(target_ct_group_dict.keys()) - 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Plot the violin subplot
        ct = list(target_ct_group_dict.keys())[i]

        sns.boxplot(data=[list(group1[group1["cell type"] == ct]["prop"].values),
                          list(group2[group2["cell type"] == ct]["prop"].values)],
                    palette=[color_dict[observation_order[0]], color_dict[observation_order[1]]], ax=ax)

        stat, pval = mannwhitneyu(list(group1[group1["cell type"] == ct]["prop"].values),
                                  list(group2[group2["cell type"] == ct]["prop"].values),
                                  alternative='two-sided')

        # Add p-value to the figure
        if pval < 0.01:
            ax.text(0.5, 0.9, "p<0.01", ha='center', va='top', fontsize=7)
            # ax.text(1.4, 0.9, "p<0.01", ha='center', va='top', fontsize=7)
        else:
            ax.text(0.5, 0.9, f'p = {pval:.2f}', ha='center', va='top', fontsize=7)
        print(ct)
        print(f'p = {pval:.2f}')
        print("____________________________")

        # Add x-ticks and legend
        ax.set_xticks([0, 1])
        ax.set_xticklabels(observation_order, size=8)

        ax.grid(False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([])

        ax.set_title(ct, weight="bold", fontsize=9)

    # Add legend
    patch1 = mpatches.Patch(color=color_dict[observation_order[0]],
                            label=observation_order[0] + " (n=" + str(len(set(group1["cid"]))) + ")")
    patch2 = mpatches.Patch(color=color_dict[observation_order[1]],
                            label=observation_order[1] + " (n=" + str(len(set(group2["cid"]))) + ")")

    fig.legend(title=name.split("_")[0],
               handles=[patch1, patch2],
               loc='lower right',
               bbox_to_anchor=(0.97, 0.05),
               ncol=1,
               fontsize=8,
               title_fontsize=8
               )

    plt.tight_layout()

    if name:
        plt.savefig(out_dir / f"{name}_cell_niche_proportions_panel.png", dpi=500)
    else:
        plt.savefig(out_dir / "cell_niche_proportions_panel.png", dpi=500)

    plt.clf()


def run_gmm_for_marker_PN(adata, marker_list, cluster_key):
    """
    Running 2-component Gaussian mixture model (GMM) to classify cells into positive and negative groups by the marker
    :param adata: AnnData
    :param marker_list: subset of markers for GMM, type: list[str]
    :param cluster_key: indicate which column in obs to store the result, type: str
    :param out_dir: output directory, type: Path
    :param name: extra name for the output file, type: str
    :return:
    """
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    # check which is epithelial and which is non-epithelial in the classification result
    if adata[np.where(adata.obs[cluster_key] == 0)[0], marker_list].X.mean() > adata[
        np.where(adata.obs[cluster_key] == 1)[0], marker_list].X.mean():
        # 0: positive, 1: negative
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: f'{marker_list[0]}+', 1: f'{marker_list[0]}-'})
    else:
        # 0: negative, 1: positive
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: f'{marker_list[0]}-', 1: f'{marker_list[0]}+'})


if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}/summary")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / "TNBC_adata_pretreatment.h5ad")

    # Remove the images with no or less than 10% of epithelial cells
    keep_img = []
    UNIT_KEY = "unique_core_id"
    CLUSTER_KEY = 'layer1'
    for img in pd.unique(adata.obs[UNIT_KEY]):
        p_adata = adata[adata.obs[UNIT_KEY] == img]
        if p_adata.obs[CLUSTER_KEY].value_counts()["Epithelial"] / p_adata.shape[0] > 0.1:
            keep_img.append(img)  # keep 83 images, there were 90 initially

    adata_analysis = adata[np.where(adata.obs[UNIT_KEY].isin(keep_img))[0], :]
    adata_nat = adata_analysis[adata_analysis.obs["chemo_type"] == 'NAT']

    # Compare the marker intensity between different observations
    compare_intensity_panel(adata=adata_nat,
                            cluster_key=CLUSTER_KEY,
                            out_dir=out_dir,
                            color_dict=PCR_TF_COLOR_DICT,
                            observation_order=["Yes", "No"], name="PathCR")

    # Compare marker intensity of each cell type across different observations
    CLUSTER_KEY = 'layer2'
    CELL_TYPE_LIST = ['Epithelial CK5+', 'Epithelial CK5+CK7+', 'Epithelial CK7+', 'Epithelial CK8-18+',
                      'Epithelial CK_low', 'Epithelial H3K27me3+', 'Myoepithelial', 'B Cell', 'T Cell',
                      'Macrophage', 'Blood Endothelial', 'Fibroblast', 'Smooth Muscle Cell',
                      'Vimentin+']
    OBSERVATION_KEY = "pCR"
    for m in adata_analysis.var_names:
        print(m)
        compare_intensity_panel(adata=adata_nat,
                                marker=m,
                                cluster_key=CLUSTER_KEY,
                                cell_type_list=CELL_TYPE_LIST,
                                observation_key=OBSERVATION_KEY,
                                out_dir=out_dir,
                                color_dict=PCR_TF_COLOR_DICT,
                                observation_order=["Yes", "No"], name="PathCR")

    # Compare cell type proportions between different observations (cell type count/esi counts)
    TARGET_CT_GROUP_DICT = dict(zip(['Epithelial CK5+', 'Epithelial CK5+CK7+', 'Epithelial CK7+', 'Epithelial CK8-18+',
                                     'Epithelial CK_low', 'Epithelial H3K27me3+', 'Myoepithelial', 'B Cell', 'T Cell',
                                     'Macrophage', 'Blood Endothelial', 'Fibroblast', 'Smooth Muscle Cell',
                                     'Vimentin+'],
                                    ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial",
                                     "Epithelial", "TME", "TME", "TME", "TME", "TME", "TME", "TME"]))

    CLUSTER_KEY = "layer2"
    UNIT_KEY = "cid"
    OBSERVATION_KEY = "pCR"
    compare_cell_type_proportion(adata=adata_nat, unit_key=UNIT_KEY,
                                 target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                 cluster_key=CLUSTER_KEY, observation_key=OBSERVATION_KEY,
                                 observation_order=["Yes", "No"],
                                 out_dir=out_dir, color_dict=PCR_TF_COLOR_DICT, name="PathCR")

    OBSERVATION_KEY = "relapse"
    OBSERVATION_KEY = "chemo_type"
    compare_cell_type_proportion(adata=adata, unit_key=UNIT_KEY,
                                 target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                 cluster_key=CLUSTER_KEY, observation_key=OBSERVATION_KEY,
                                 observation_order=["ADJ", "NAT"],
                                 out_dir=out_dir, color_dict=CHEMOTYPE_COLOR_DICT, name="Treatment")

    # Cell niches proportion comparison
    TARGET_CT_GROUP_DICT = dict(zip(['CK5_Basal', 'CK5_CK7_Mp', 'CK7_Lum', 'CK8-Lum_Myo', 'CK8_Lum', 'H3K27me3_Epi',
                                     'B', 'T_B', 'T_Mp_Endo', 'Fibro'],
                                    ["Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial", "Epithelial",
                                     "TME", "TME", "TME", "TME"]))

    OBSERVATION_KEY = "chemo_type"
    CLUSTER_KEY = "cn_cluster_name"
    compare_cell_niche_proportion(adata=adata, unit_key=UNIT_KEY,
                                  target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                  cluster_key=CLUSTER_KEY, observation_key=OBSERVATION_KEY,
                                  observation_order=["ADJ", "NAT"],
                                  out_dir=out_dir, color_dict=CHEMOTYPE_COLOR_DICT, name="Treatment")

    OBSERVATION_KEY = "pCR"
    CLUSTER_KEY = "cn_cluster_name"
    compare_cell_niche_proportion(adata=adata_nat, unit_key=UNIT_KEY,
                                  target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                  cluster_key=CLUSTER_KEY, observation_key=OBSERVATION_KEY,
                                  observation_order=["Yes", "No"],
                                  out_dir=out_dir, color_dict=PCR_TF_COLOR_DICT, name="PathCR")

    # Plot the marker intensity across different cell types (not separate observations)
    CELL_TYPE_LIST = ['Epithelial CK5+', 'Epithelial CK5+CK7+', 'Epithelial CK7+', 'Epithelial CK8-18+',
                      'Epithelial CK_low', 'Epithelial H3K27me3+', 'Myoepithelial', 'B Cell', 'T Cell',
                      'Macrophage', 'Blood Endothelial', 'Fibroblast', 'Smooth Muscle Cell',
                      'Vimentin+']
    CLUSTER_KEY = "layer2"

    for marker in adata_analysis.var_names:
        idx = np.where(adata_analysis.var_names == marker)[0][0]
        exp = adata_analysis.X[:, idx]
        # Make a df of the marker to plot (cell_type, expression)
        df = pd.DataFrame({"cell_type": list(adata_analysis.obs[CLUSTER_KEY]),
                           "expression": list(exp)})

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        sns.violinplot(data=df, x="cell_type", y="expression", order=CELL_TYPE_LIST, ax=ax)
        plt.xlabel("Cell type", fontsize=20)
        plt.xticks(fontsize=15, rotation=90)
        plt.ylabel(f"{marker} expression", fontsize=20)

        plt.tight_layout()
        plt.savefig(out_dir / f"{marker}_exp_cell_type.png", dpi=300)
        plt.clf()

    t_adata = adata_analysis.copy()
    run_gmm_for_marker_PN(t_adata, ["H3K27me3"], "H3K27me3")

    # Calculate the proportions of H3K27me3+ cells in recurrence and non-recurrence groups



    t_adata_neo = t_adata[t_adata.obs["chemo_type"] == "NAT"]
    compare_cell_type_proportion(adata=t_adata_neo,
                                 unit_key=UNIT_KEY,
                                 target_ct_group_dict=TARGET_CT_GROUP_DICT,
                                 cluster_key=CLUSTER_KEY,
                                 observation_key=OBSERVATION_KEY,
                                 observation_order=["Yes", "No"],
                                 out_dir=out_dir,
                                 color_dict=PCR_TF_COLOR_DICT, name="PathCR_Ki67", marker_focus_key="Ki67")

    # Get the epithelial h3k27me3 focus
    idx = list(np.where(adata_analysis.obs.layer1 == "TME")[0]) + list(
        np.where(adata_analysis.obs.layer2 == "Epithelial H3K27me3+")[0])
    adata_h3k27me3 = adata_analysis[idx, :].copy()
