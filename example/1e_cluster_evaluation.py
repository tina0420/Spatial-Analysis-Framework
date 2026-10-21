"""
@author: Tina Hsu
Tutorial 1e: Evaluating different clustering methods.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import v_measure_score
from src.annotate import update_labels
from src.annotate_evaluation import *


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "example"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    in_dir = Path(f"example/output/{run_name}")
    out_dir = Path(f"example/output/{run_name}/evaluation")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(in_dir / f"{run_name}.h5ad")

    """
    We want to define the column that defines the annotated cell types.
    """
    CLUSTER_KEY = "celltype"

    """
    This step is optional in most analyses. Here we are evaluating between two clustering method.
    
    For simplicity, we will cluster all cells into four cell types and update the labels as bgm.
    """
    n_components = len(adata.obs[CLUSTER_KEY].cat.categories)
    if issparse(adata.X):
        labels = BayesianGaussianMixture(n_components=n_components, random_state=0).fit_predict(adata.X.toarray())
    else:
        labels = BayesianGaussianMixture(n_components=n_components, random_state=0).fit_predict(adata.X)

    update_labels(adata, "bgm", labels)

    """
    We can then perform evaluation using metrics such as DB coherence and v-measure similarity between labels.
    """
    pd.DataFrame([[run_name, "DB", "leiden_subset", davies_bouldin(adata.X, adata.obs[CLUSTER_KEY])],
                  [run_name, "DB", "bgm", davies_bouldin(adata.X, adata.obs["bgm"])],
                  [run_name, "V", "leiden_subset:bgm", v_measure_score(adata.obs[CLUSTER_KEY],
                                                                       adata.obs["bgm"])]],
                 columns=["Experiment", "Measure", "Method", "Score"]).to_csv(out_dir / "result.csv",
                                                                              index=False)

    """
    We can look further by finding indices of cells that have the same or different labels.
    
    Note here that the cell types should match, but this isn't taken care of when doing denovo clustering such as below.
    """
    agree_idxs, disagree_idxs = get_agreement_indices(adata.obs[CLUSTER_KEY], adata.obs["bgm"])

    """
    We can visualize our results in various ways.
    """
    # Plot cells that agree in labels for each cluster
    plot_cluster_agreement_panel(adata, cluster_key=CLUSTER_KEY, agree_idxs=agree_idxs, out_dir=out_dir, dotsize=1)

    # Plot cells that agree and disagree
    plot_cluster_agreement_summary(adata, agree_idxs=agree_idxs, disagree_idxs=disagree_idxs,
                                   out_dir=out_dir, figsize=(3.5, 2))

    # Plot confusion matrix of two clustering methods
    plot_cluster_disagreement_matrix(adata.obs[CLUSTER_KEY], adata.obs["bgm"], out_dir=out_dir)
