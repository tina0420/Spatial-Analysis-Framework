"""
@author: Tina Hsu
Functions for running squidpy functions.
"""
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.stats import entropy


def run_squidpy_statistics(adata, cluster_key):
    """
    Run squidpy functions.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :return:
    """
    # compute a connectivity matrix from spatial coordinates
    # stored in `adata.obsp["spatial_connectivities"]` & `adata.obsp["spatial_distances"]`
    # plot by `sq.pl.spatial_scatter(adata, shape=None, color="leiden", size=10, save=Path(os.getcwd() + "/scatter"))`

    sq.gr.spatial_neighbors(adata)

    """
    Compute centrality scores per cluster or cell type.

    Returns: [# cell types, 3 measures]
    - closeness centrality - measure of how close the group is to other nodes.
    - clustering coefficient - measure of the degree to which nodes cluster together.
    - degree centrality - fraction of non-group members connected to group members.
    """
    # define key as cluster_key in pandas series
    # using adata.obs[cluster_key] which indicates all cell types
    # stored in `adata.uns["cell type_centrality_scores"]`
    # plot by `sq.pl.centrality_scores(adata, cluster_key, save=Path(os.getcwd() + "/centrality"))`

    sq.gr.centrality_scores(adata, cluster_key)

    """
    Compute co-occurrence probability of clusters.
    It calculates a co-occurrence score where p(obs|cond) is the probability of observing a cluster
    conditioned on a cluster, whereas p(obs) is the probability of observing obs in the radius size of interest.

    Returns:[[# cell types, # cell types, distance thresholds],
             [distance thresholds]]
    """
    # stored in `adata.uns["cell type_co_occurrence"]` with ["occ"] and ["interval"]
    # plot by `sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters="0", save=Path(os.getcwd() + "/co_occur0"))`

    sq.gr.co_occurrence(adata, cluster_key=cluster_key)

    """
    Compute neighborhood enrichment by permutation test.
    It calculates an enrichment score based on proximity on the connectivity graph of cell clusters.
    The number of observed events is compared against permutations and a z-score is computed.

    Returns:[[# cell types, # cell types],
             [# cell types, # cell types]]
    """
    # stored in `adata.uns["cell type_nhood_enrichment"]` with ["zscore"] and ["count"]
    # plot by `sq.pl.nhood_enrichment(adata, cluster_key=cluster_key, save=Path(os.getcwd() + "/nhood_enrichment"))`

    # the output of ["count"] is the same as sq.gr.interaction_matrix(adata, cluster_key=cluster_key)
    # plot by `sq.pl.interaction_matrix(adata, cluster_key=cluster_key, save=Path(os.getcwd() + "/interaction_mtx"))`

    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)

    """
    Ripley's K describes the expected number of neighbours of any point in a point pattern across different radii.
    You can use it to analyse point patterns in the space, whether they are aggregated, random, or systematic.
    The only information you use is the coordinates (or distances between them).

    The Ripley’s L is a variance-normalized version of the Ripley’s K statistic.

    n_simulations to calc p-values (default:100) n_steps for steps for the support (default:50)

    Returns:[[# cell types * n_steps, 3 measures]
             [n_simulations * n_steps, 3 measures]
             [n_steps],
             [# cell types, n_steps]]
    ’stats’ - Ripley’s L statistic.
    ’bins` - steps for the support.
    """
    # stored in `adata.uns["cell type_ripley_L"]` with ["L_stat"] ["sims_stat"] ["bins"] and ["pvalues"]
    # plot by `sq.pl.ripley(adata, cluster_key=cluster_key, mode="L", save=Path(os.getcwd() + "/ripleyL"))`

    sq.gr.ripley(adata, cluster_key=cluster_key, mode="L")

    """
    Moran's I analyses how a measured variable is structured in space.
    It uses a spatial neighbourhood matrix between the observations and a measured variable for each observation.
    It has a global and a local variant; with the latter, you can step across different distance classes to reveal
    scale-dependency in the spatial correlation.
    If reasonable null-hypotheses are provided, both can be used for hypothesis testing.

    Returns:[# markers, 9 measures]
    ’I’ - Moran’s I statistic.
    ’pval_norm’ - p-value under normality assumption.
    ’var_norm’ - variance of ‘score’ under normality assumption.
    ’pval_z_sim’ - p-value based on standard normal approximation from permutations.
    ’pval_sim’ - p-value based on permutations.
    ’var_sim’ - variance of ‘score’ from permutations.
    ’{p_val_norm / p_val_z_sim, pval_sim}_fdr_bh’ - the corrected p-values under Benjamini/Hochberg multiple testing.

    """
    # stored in `adata.uns["moranI"]`
    # plot by `sq.pl.spatial_scatter(adata, shape=None, color=[markers], save=Path(os.getcwd() + "/markers"))`

    # sq.gr.spatial_autocorr(adata, mode="moran", n_perms=100)


def get_squidpy_data(adata, cluster_key):
    """
    Get a dictionary of squidpy results
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in uns, type: str
    :return: squidpy results, type: dict
    """
    return {"centrality": adata.uns[f"{cluster_key}_centrality_scores"],
            "co_occur": adata.uns[f"{cluster_key}_co_occurrence"]["occ"],
            "co_occur_interval": adata.uns[f"{cluster_key}_co_occurrence"]["interval"],
            "nhood_zscore": adata.uns[f"{cluster_key}_nhood_enrichment"]["zscore"],
            "nhood_count": adata.uns[f"{cluster_key}_nhood_enrichment"]["count"],
            "ripley_L": adata.uns[f"{cluster_key}_ripley_L"]["L_stat"],
            # "moran_I": adata.uns["moranI"]
            }


def get_interaction_entropy(adata, cluster_key):
    """
    Get interaction entropy.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :return: interaction entropies, type: pd.Dataframe
    """

    interaction_count = get_squidpy_data(adata=adata, cluster_key=cluster_key)["nhood_count"]
    all_label_idxs = adata.obs[cluster_key].value_counts(sort=False).index
    entropies = []
    for lbl in all_label_idxs:
        type_idx = np.where(all_label_idxs == lbl)[0][0]
        entropies.append((entropy(interaction_count[:, type_idx]) + entropy(interaction_count[type_idx, :])) / 2)
    entropies = pd.DataFrame([list(all_label_idxs), entropies]).T
    entropies.columns = ["Label", "Entropy"]

    return entropies


def get_spatially_variable_genes(adata, cluster_key):
    """
    Get spatially variable genes using Moran's I.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :return: spatially variable genes, type: pd.Dataframe
    """
    moran_i = get_squidpy_data(adata=adata, cluster_key=cluster_key)["moran_I"]
    svg = moran_i[moran_i.pval_sim_fdr_bh < 0.05]
    svg = svg["I"].reset_index()
    svg.columns = ["Gene", "I"]

    return svg
