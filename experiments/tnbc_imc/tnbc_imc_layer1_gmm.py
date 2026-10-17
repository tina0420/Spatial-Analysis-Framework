"""
@author: Tina Hsu
Running 2-component GMM for classifying epithelial and non-epithelial cells.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels
from src.visualize import plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel
import scanpy as sc
from experiments.tnbc_imc.tnbc_imc_setup_analysis import E_TME_COLOR_DICT


def run_gmm_for_epithelial(adata, marker_list, cluster_key, out_dir, name=None):
    """
    Running 2-component Gaussian mixture model (GMM) to classify epithelial cells and non-epithelial cells
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
        # 0: epithelial, 1: non-epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'Epithelial', 1: 'TME'})
    else:
        # 0: non-epithelial, 1: epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'TME', 1: 'Epithelial'})

    adata[np.where(adata.obs[cluster_key] == 'Epithelial')[0], :].write(out_dir / f"{name}_epi.h5ad")
    adata[np.where(adata.obs[cluster_key] == 'TME')[0], :].write(out_dir / f"{name}_tme.h5ad")


if __name__ == '__main__':
    # Setup disease type and experiment to run
    TYPE = 'tnbc_imc'
    PURPOSE = 'final_thesis'
    XP = 'TNBC_pretreatment'

    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    CLUSTER_KEY = 'layer1'

    in_dir = Path(f'data/{TYPE}/{PURPOSE}')
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata = read_h5ad(in_dir / f"{XP}.h5ad")

    run_gmm_for_epithelial(adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=XP)

    # check epithelial and non-epithelial zscore and UMAP
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(12, 12),
                                        rotation=35, color_dict=E_TME_COLOR_DICT, name=f"{XP}_layer1_default", v_max=4, v_min=-4)
    # Run PCA & neighbors
    # sc.tl.pca(adata, svd_solver='arpack')
    # sc.pp.neighbors(adata, use_rep='X_pca')
    # sc.tl.umap(adata)
    #
    # plot_umap(adata, CLUSTER_KEY, out_dir, color_dict=E_TME_COLOR_DICT,
    #           figsize=(4.5, 4.5), dotsize=0.5, name=f"{XP}_layer1")
    # plot_umap_panel(adata, CLUSTER_KEY, out_dir, color_dict=E_TME_COLOR_DICT,
    #                 dotsize=0.1, name=f"{XP}_layer1")

    adata.write(in_dir / f"{XP}_layer1_finalize.h5ad")

    # Store the adata
    adata.write(in_dir / f"{XP}_finalize.h5ad")

    # check if it ran well and store correctly
    adata = read_h5ad(in_dir / f"{XP}_finalize.h5ad")

