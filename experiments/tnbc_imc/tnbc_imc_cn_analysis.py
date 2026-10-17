"""
@author: Tina Hsu
Running cell type niches across cells
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr, entropy
import anndata as ad
import scanpy as sc
from src.annotate import update_labels
from src.visualize import (plot_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap,
                           plot_spatial_cluster_panel, plot_spatial_cluster_distribution, COLOR_CODE)
from scipy.stats import mannwhitneyu


def compute_entropy(x):
    return entropy(x.value_counts(normalize=True), base=2)


if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    XP = 'all_8_tma_merge'

    in_dir = Path(f"experiments/{TYPE}/final/")
    out_dir = Path(f"experiments/{TYPE}/final/result")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in adata
    FULL_ADATA = read_h5ad(out_dir / "TNBC_adata_pretreatment.h5ad")

    """
    Validating CN clusters
    """
    cn_tbl = pd.read_csv(out_dir / "cn_tbl.csv")

    core_by_cluster_tbl = cn_tbl.groupby(['core', 'temp']).size().unstack(fill_value=0)

    core_by_cluster_tbl.to_csv(out_dir / "core_by_cluster_tbl.csv")

    # How evenly distributed (diverse) are round_2_cluster within each core?
    entropy_by_core = cn_tbl.groupby('core')['temp'].apply(compute_entropy)
    entropy_by_core.to_csv(out_dir / "entropy_by_core.csv")

    plt.figure(figsize=(4, 4))
    plt.boxplot(entropy_by_core)
    plt.title("Entropy by core")
    plt.xticks([])
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_by_core.png")

    # How diverse are cores within each round_2_cluster?
    entropy_by_cluster = cn_tbl.groupby('temp')['core'].apply(compute_entropy)
    entropy_by_cluster.to_csv(out_dir / "entropy_by_cluster.csv")

    # Plotting the bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(entropy_by_cluster.index, entropy_by_cluster.values)
    plt.xticks(ticks=entropy_by_cluster.index, rotation=45, labels=['T_Mp_Endo', 'CK8-Lum_Myo', 'CK5_CK7_Mp', 'CK5_Basal', 'H3K27me3_Epi', 'T_B', 'Fibro', 'CK8_Lum', 'CK7_Lum', 'B'])

    # Adding title and labels
    plt.title("Entropy by cluster")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "entropy_by_cluster.png")


    # Markers of interest
    FULL_ADATA.obs["H3K27me3_intensity"] = np.array(FULL_ADATA[:, "H3K27me3"].X.flatten())

    markers = ['unique_core_id', 'cid', 'core_id', 'chemo_type', 'pCR', 'relapse', 'cn_cluster_name', 'ESI', 'layer2', 'H3K27me3_intensity']

    FULL_ADATA.obs[markers].to_csv(out_dir / "full_adata_pretreatment.csv")

    """
    Relapse
    """
    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")

    # Split the data based on relapse condition
    df_no_recurrence = df[df['relapse'] == 'No recurrence']
    df_recurrence = df[df['relapse'] == 'Recurrence']

    # df_no_recurrence.cn_cluster_name = df_no_recurrence.cn_cluster_name.astype("category")
    # df_recurrence.cn_cluster_name = df_recurrence.cn_cluster_name.astype("category")

    # Calculate proportions of cn_clusters for each cid under each relapse condition
    prop_no_recurrence = df_no_recurrence.groupby(['cid', 'cn_cluster_name']).size() / df_no_recurrence.groupby('cid').size()
    prop_recurrence = df_recurrence.groupby(['cid', 'cn_cluster_name']).size() / df_recurrence.groupby('cid').size()

    # Convert proportions to DataFrame for easier comparison
    prop_no_recurrence = prop_no_recurrence.reset_index(name='proportion')
    prop_no_recurrence["relapse"] = "No recurrence"
    prop_recurrence = prop_recurrence.reset_index(name='proportion')
    prop_recurrence["relapse"] = "Recurrence"

    # Merge the two DataFrames on cid and cn_cluster to compare proportions
    merged_props = pd.concat([prop_no_recurrence, prop_recurrence])

    df = merged_props

    """
    Plot
    """

    # Set up the FacetGrid to facet by 'cn_cluster_name'
    g = sns.FacetGrid(df, col="cn_cluster_name", col_wrap=5, height=4)

    # Define a color palette for relapse status
    palette = {"No recurrence": "lightblue", "Recurrence": "salmon"}

    # Map the boxplot to each facet, color-coded by relapse status
    g.map(sns.violinplot, 'relapse', 'proportion', order=['No recurrence', 'Recurrence'], palette=palette)

    # Adjust plot to make it more readable
    g.set_axis_labels("Relapse Status", "Proportion")
    g.set_titles("{col_name}")

    # Calculate Mann-Whitney U test p-values and annotate on each plot
    for ax in g.axes.flat:
        # Get the cluster name
        cluster_name = ax.get_title()

        # Extract data for the current cluster
        cluster_data = df[df['cn_cluster_name'] == cluster_name]

        # Perform Mann-Whitney U test between 'No recurrence' and 'Recurrence' groups
        data_no_recurrence = cluster_data[cluster_data['relapse'] == 'No recurrence']['proportion']
        data_recurrence = cluster_data[cluster_data['relapse'] == 'Recurrence']['proportion']

        if len(data_no_recurrence) > 0 and len(data_recurrence) > 0:  # Ensure we have data to compare
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(data_no_recurrence, data_recurrence, alternative='two-sided')

            # Annotate p-value on the plot
            ax.annotate(f'p = {p_value:.3f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center',
                        fontsize=12)

    # Rotate x-axis labels to avoid overlap if needed
    for ax in g.axes.flat:
        ax.set_xticks([0, 1])  # Set x-ticks explicitly to match the order of 'relapse'
        ax.set_xticklabels(['No recurrence', 'Recurrence'], rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()

    plt.savefig(out_dir / "full_adata_pretreatment_recurrence.png")

    """
    pCR
    """
    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")
    df = df[df.chemo_type == "NAT"]

    # Split the data based on relapse condition
    df_no_recurrence = df[df['pCR'] == 'Yes']
    df_recurrence = df[df['pCR'] == 'No']

    df_no_recurrence.cn_cluster_name = df_no_recurrence.cn_cluster_name.astype("category")
    df_recurrence.cn_cluster_name = df_recurrence.cn_cluster_name.astype("category")

    # Calculate proportions of cn_clusters for each cid under each relapse condition
    prop_no_recurrence = df_no_recurrence.groupby(['cid', 'cn_cluster_name']).size() / df_no_recurrence.groupby(
        'cid').size()
    prop_recurrence = df_recurrence.groupby(['cid', 'cn_cluster_name']).size() / df_recurrence.groupby('cid').size()

    # Convert proportions to DataFrame for easier comparison
    prop_no_recurrence = prop_no_recurrence.reset_index(name='proportion')
    prop_no_recurrence["pCR"] = "No"
    prop_recurrence = prop_recurrence.reset_index(name='proportion')
    prop_recurrence["pCR"] = "Yes"

    # Merge the two DataFrames on cid and cn_cluster to compare proportions
    merged_props = pd.concat([prop_no_recurrence, prop_recurrence])

    df = merged_props

    """
    Plot
    """
    # Set up the FacetGrid to facet by 'cn_cluster_name'
    g = sns.FacetGrid(df, col="cn_cluster_name", col_wrap=5, height=4)

    # Define a color palette for relapse status
    palette = {"No": "lightblue", "Yes": "salmon"}

    # Map the boxplot to each facet, color-coded by relapse status
    g.map(sns.violinplot, 'pCR', 'proportion', order=['No', 'Yes'], palette=palette)

    # Adjust plot to make it more readable
    g.set_axis_labels("pCR Status", "Proportion")
    g.set_titles("{col_name}")

    # Calculate Mann-Whitney U test p-values and annotate on each plot
    for ax in g.axes.flat:
        # Get the cluster name
        cluster_name = ax.get_title()

        # Extract data for the current cluster
        cluster_data = df[df['cn_cluster_name'] == cluster_name]

        # Perform Mann-Whitney U test between 'No recurrence' and 'Recurrence' groups
        data_no_recurrence = cluster_data[cluster_data['pCR'] == 'No']['proportion']
        data_recurrence = cluster_data[cluster_data['pCR'] == 'Yes']['proportion']

        if len(data_no_recurrence) > 0 and len(data_recurrence) > 0:  # Ensure we have data to compare
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(data_no_recurrence, data_recurrence, alternative='two-sided')

            # Annotate p-value on the plot
            ax.annotate(f'p = {p_value:.3f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center',
                        fontsize=12)

    # Rotate x-axis labels to avoid overlap if needed
    for ax in g.axes.flat:
        ax.set_xticks([0, 1])  # Set x-ticks explicitly to match the order of 'relapse'
        ax.set_xticklabels(['No', 'Yes'], rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()

    plt.savefig(out_dir / "full_adata_pretreatment_pcr.png")

    """
    Differences in the immune content between epithelial dominant niches for relapse
    """
    celltype_of_interest = "Macrophage"
    palette = {"No recurrence": "lightblue", "Recurrence": "salmon"}


    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")

    interest_cell_counts = df[df['layer2'] == celltype_of_interest].groupby(['cn_cluster_name', 'relapse']).size()
    total_counts = df.groupby(['cn_cluster_name', 'relapse']).size()

    # Calculate the proportion of B-cells in each group
    proportion_interest_cells = interest_cell_counts / total_counts
    proportion_interest_cells = proportion_interest_cells.reset_index(name=f'{celltype_of_interest}_proportion')

    # Step 2: Create the facet plot
    p_values = []

    for cluster in proportion_interest_cells['cn_cluster_name'].unique():
        # Get the proportions for the current cluster
        group_data = df[df['cn_cluster_name'] == cluster]

        # Proportions of B-cells for "No recurrence" and "Recurrence"
        no_recurrence = group_data[group_data['relapse'] == 'No recurrence']['layer2'] == celltype_of_interest
        recurrence = group_data[group_data['relapse'] == 'Recurrence']['layer2'] == celltype_of_interest

        # Mann-Whitney U test between the two groups
        stat, p_value = mannwhitneyu(no_recurrence, recurrence, alternative='two-sided')
        p_values.append({'cluster': cluster, 'p_value': p_value})

    # Convert p-values into a DataFrame
    p_values_df = pd.DataFrame(p_values)

    # Merge p-values with the proportion data for annotation
    proportion_b_cells = proportion_interest_cells.merge(p_values_df, left_on='cn_cluster_name', right_on='cluster',
                                                  how='left')

    # Step 3: Create the facet plot for each `cn_cluster_name`
    plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate 2x5 grid
    g = sns.FacetGrid(proportion_b_cells, col="cn_cluster_name", height=5, col_wrap=5, margin_titles=True)

    # Map the barplot to each facet (plotting the proportion of B-cells for each relapse status)
    g.map(sns.barplot, 'relapse', f'{celltype_of_interest}_proportion', palette=palette)

    # Annotate the p-value on each plot
    for ax, p_value, cluster in zip(g.axes.flat, proportion_b_cells['p_value'], proportion_b_cells['cn_cluster_name']):
        ax.annotate(f'P-value: {p_value:.3e}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', va='center',
                    fontsize=12)

    # Adjust plot appearance
    g.set_axis_labels("Relapse Status", "Proportion of B-cells")
    g.set_titles("{col_name}")

    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_dir / f"full_adata_pretreatment_{celltype_of_interest}.png")

    """
    Differences in the H3K27me3 intensity between epithelial dominant niches for relapse
    """
    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")

    df = df[df["ESI"] == "Epithelial"]

    p_values = []

    for cluster in df['cn_cluster_name'].unique():
        # Get the H3K27me3_intensity values for the current cluster, split by relapse status
        group_data = df[df['cn_cluster_name'] == cluster]

        # Values of H3K27me3_intensity for "No recurrence" and "Recurrence"
        no_recurrence = group_data[group_data['relapse'] == 'No recurrence']['H3K27me3_intensity']
        recurrence = group_data[group_data['relapse'] == 'Recurrence']['H3K27me3_intensity']

        # Mann-Whitney U test between the two groups
        stat, p_value = mannwhitneyu(no_recurrence, recurrence, alternative='two-sided')
        p_values.append({'cluster': cluster, 'p_value': p_value})

    # Convert p-values into a DataFrame
    p_values_df = pd.DataFrame(p_values)

    # Step 2: Create the facet plot for each `cn_cluster_name`
    plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate 2x5 grid
    g = sns.FacetGrid(df, col="cn_cluster_name", height=5, col_wrap=5, margin_titles=True)

    # Map the boxplot to each facet (plotting the H3K27me3_intensity for each relapse status)
    g.map(sns.boxplot, 'relapse', 'H3K27me3_intensity', palette=palette)

    # Annotate the p-value on each plot
    for ax, p_value, cluster in zip(g.axes.flat, p_values_df['p_value'], p_values_df['cluster']):
        ax.annotate(f'P-value: {p_value:.3e}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', va='center',
                    fontsize=12)

    # Adjust plot appearance
    g.set_axis_labels("Relapse Status", "H3K27me3 Intensity")
    g.set_titles("{col_name}")

    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_dir / f"full_adata_pretreatment_H3K27me3.png")

    """
    Differences in the immune content between epithelial dominant niches for pcR
    """
    celltype_of_interest = "Macrophage"
    palette = {"No": "lightblue", "Yes": "salmon"}

    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")
    df = df[df.chemo_type == "NAT"]

    interest_cell_counts = df[df['layer2'] == celltype_of_interest].groupby(['cn_cluster_name', 'pCR']).size()
    total_counts = df.groupby(['cn_cluster_name', 'pCR']).size()

    # Calculate the proportion of B-cells in each group
    proportion_interest_cells = interest_cell_counts / total_counts
    proportion_interest_cells = proportion_interest_cells.reset_index(name=f'{celltype_of_interest}_proportion')

    # Step 2: Create the facet plot
    p_values = []

    for cluster in proportion_interest_cells['cn_cluster_name'].unique():
        if cluster == 'CK7_Lum':
            continue

        # Get the proportions for the current cluster
        group_data = df[df['cn_cluster_name'] == cluster]

        # Proportions of B-cells for "No recurrence" and "Recurrence"
        no_recurrence = group_data[group_data['pCR'] == 'No']['layer2'] == celltype_of_interest
        recurrence = group_data[group_data['pCR'] == 'Yes']['layer2'] == celltype_of_interest

        # Mann-Whitney U test between the two groups
        stat, p_value = mannwhitneyu(no_recurrence, recurrence, alternative='two-sided')
        p_values.append({'cluster': cluster, 'p_value': p_value})

    # Convert p-values into a DataFrame
    p_values_df = pd.DataFrame(p_values)

    from statsmodels.stats.multitest import multipletests
    reject, pvals_corrected, _, _ = multipletests(p_values_df['p_value'], method='fdr_bh')
    p_values_df['p_value'] = pvals_corrected

    # Merge p-values with the proportion data for annotation
    proportion_b_cells = proportion_interest_cells.merge(p_values_df, left_on='cn_cluster_name', right_on='cluster',
                                                         how='left')

    # Step 3: Create the facet plot for each `cn_cluster_name`
    plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate 2x5 grid
    g = sns.FacetGrid(proportion_b_cells, col="cn_cluster_name", height=5, col_wrap=5, margin_titles=True)

    # Map the barplot to each facet (plotting the proportion of B-cells for each relapse status)
    g.map(sns.barplot, 'pCR', f'{celltype_of_interest}_proportion', palette=palette)

    # Annotate the p-value on each plot
    for ax, p_value, cluster in zip(g.axes.flat, proportion_b_cells['p_value'], proportion_b_cells['cn_cluster_name']):
        ax.annotate(f'P-value: {p_value:.3e}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', va='center',
                    fontsize=12)

    # Adjust plot appearance
    g.set_axis_labels("pCR Status", "Proportion")
    g.set_titles("{col_name}")

    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_dir / f"full_adata_pretreatment_{celltype_of_interest}_pcr.png")

    """
    Differences in the H3K27me3 intensity between epithelial dominant niches for pCR
    """
    df = pd.read_csv(out_dir / "full_adata_pretreatment.csv")

    df = df[df["ESI"] == "Epithelial"]
    df = df[df.chemo_type == "NAT"]

    p_values = []

    for cluster in df['cn_cluster_name'].unique():
        if cluster == 'CK7_Lum':
            continue
        # Get the H3K27me3_intensity values for the current cluster, split by relapse status
        group_data = df[df['cn_cluster_name'] == cluster]

        # Values of H3K27me3_intensity for "No recurrence" and "Recurrence"
        no_recurrence = group_data[group_data['pCR'] == 'No']['H3K27me3_intensity']
        recurrence = group_data[group_data['pCR'] == 'Yes']['H3K27me3_intensity']

        # Mann-Whitney U test between the two groups
        stat, p_value = mannwhitneyu(no_recurrence, recurrence, alternative='two-sided')
        p_values.append({'cluster': cluster, 'p_value': p_value})

    # Convert p-values into a DataFrame
    p_values_df = pd.DataFrame(p_values)

    from statsmodels.stats.multitest import multipletests
    reject, pvals_corrected, _, _ = multipletests(p_values_df['p_value'], method='fdr_bh')
    p_values_df['p_value'] = pvals_corrected

    # Step 2: Create the facet plot for each `cn_cluster_name`
    plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate 2x5 grid
    g = sns.FacetGrid(df, col="cn_cluster_name", height=5, col_wrap=5, margin_titles=True)

    # Map the boxplot to each facet (plotting the H3K27me3_intensity for each relapse status)
    g.map(sns.boxplot, 'pCR', 'H3K27me3_intensity', palette=palette)

    # Annotate the p-value on each plot
    for ax, p_value, cluster in zip(g.axes.flat, p_values_df['p_value'], p_values_df['cluster']):
        ax.annotate(f'P-value: {p_value:.3e}', xy=(0.5, 0.85), xycoords='axes fraction', ha='center', va='center',
                    fontsize=12)

    # Adjust plot appearance
    g.set_axis_labels("pCR Status", "H3K27me3 Intensity")
    g.set_titles("{col_name}")

    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.savefig(out_dir / f"full_adata_pretreatment_H3K27me3_pCR.png")

