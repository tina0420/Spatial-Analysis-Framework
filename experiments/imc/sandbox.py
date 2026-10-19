import os
from pathlib import Path
from anndata import read_h5ad
from IPython.display import Image
from segmentation import run_mesmer
from marker_quality_control import run_qc
from marker_quality_control import plot_marker_nuclear_per_image
import scanpy as sc
from visualization import *
from clustering import compare_clustering_resolutions
from visualization import plot_compare_clustering_line
from utils import update_labels
from analysis_setup import *
import numpy as np

if __name__ == '__main__':
    # Directory setting
    EXPERIMENT = "imc_data"

    IN_DIR = Path("data/")
    OUT_DIR = Path(f"output/{EXPERIMENT}")
    if os.path.exists(OUT_DIR) == 0:
        os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset
    adata = read_h5ad(IN_DIR / f"{EXPERIMENT}_layer2.h5ad")

    # # Segmentation (processing with .TIFFs)
    # NUCLEAR_CHANNELS = [31]
    # CYTOPLASM_CHANNELS = [3, 29]
    # OUT_DIR = Path(f"output/{EXPERIMENT}/segmentation")
    # if os.path.exists(OUT_DIR) == 0:
    #     os.makedirs(OUT_DIR, exist_ok=True)
    #
    # DEEPCELL_TOKEN = "gLJAn8ZY.OmHKrKnth0mqobIljEi6o16StVeqhFVn"
    # MES_INTERIOR_THRESHOLD = 0.1
    # MES_MAXIMA_THRESHOLD = 0.58
    # MES_IMAGE_MPP = 1
    # MASK = False
    # OVERLAY = False
    # PRED_DF = False
    # run_mesmer(nuclear_channel=NUCLEAR_CHANNELS,
    #            cytoplasm_channels=CYTOPLASM_CHANNELS,
    #            in_dir=IN_DIR,
    #            out_dir=OUT_DIR,
    #            deepcell_token=DEEPCELL_TOKEN,
    #            mes_image_mpp=MES_IMAGE_MPP,
    #            mes_interior_threshold=MES_INTERIOR_THRESHOLD,
    #            mes_maxima_threshold=MES_MAXIMA_THRESHOLD,
    #            plot_mask=MASK,
    #            plot_overlay=OVERLAY,
    #            output_pred_df=PRED_DF)
    #
    #
    # # Quality Control-correlation: exp_based, loc_based QC
    # EXP_MATCH = True
    # EXP_PLOT = True
    # LOC_MATCH = False
    # LOC_PLOT = True
    # INTERSECTION_PLOT = True
    #
    # OUT_DIR = Path(f"output/{EXPERIMENT}/qc")
    # if os.path.exists(OUT_DIR) == 0:
    #     os.makedirs(OUT_DIR, exist_ok=True)
    #
    # run_qc(adata=adata, out_dir=OUT_DIR, name="tnbc",
    #        exp_match=EXP_MATCH, exp_plot=EXP_PLOT,
    #        loc_match=LOC_MATCH, loc_plot=LOC_PLOT,
    #        intersection_plot=INTERSECTION_PLOT)
    #
    # # Quality Control-marker staining: marker-nuclei staining plot
    # OUT_DIR = Path(f"output/{EXPERIMENT}/qc/markers")
    # if os.path.exists(OUT_DIR) == 0:
    #     os.makedirs(OUT_DIR, exist_ok=True)
    #
    # plot_marker_nuclear_per_image(marker_list=adata.var_names,
    #                               nuclei_name="Nuclei",
    #                               in_dir=IN_DIR,
    #                               out_dir=OUT_DIR)

    # # Clustering for cell types
    # # First-round clustering: ESI
    # OUT_DIR = Path(f"output/{EXPERIMENT}/cell_type")
    # if os.path.exists(OUT_DIR) == 0:
    #     os.makedirs(OUT_DIR, exist_ok=True)
    #
    # METHOD = 'leiden'
    # MARKER_LIST = ['CK5', 'CK7', 'CK8-18', 'panCK',
    #                'CD3e', 'CD20', 'CD45', 'CD68',
    #                'CD31', 'SMA', 'Vimentin']
    #
    # RESOLUTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #
    # adata_layer1 = adata[:, MARKER_LIST].copy()
    # sc.tl.pca(adata_layer1, n_comps=6, svd_solver='arpack')
    # sc.pp.neighbors(adata_layer1, n_neighbors=15, use_rep='X_pca')
    #
    # # Grid search for clustering with different resolution
    # for res in RESOLUTIONS:
    #     CLUSTER_KEY = f'layer_1_{res:.1f}'
    #     print(f"Resolution: {res:.1f}")
    #
    #     # sc.tl.leiden(adata_layer1, resolution=res, key_added=CLUSTER_KEY)
    #
    #     print("Plotting")
    #     plot_zscore_cluster_mean_heatmap(
    #         adata=adata_layer1,
    #         column_names=MARKER_LIST,
    #         cluster_key=CLUSTER_KEY,
    #         out_dir=OUT_DIR,
    #         vmax=3,
    #         vmin=-1,
    #         figsize=(7, 7),
    #         name=f'layer_1_{res:.1f}'
    #     )
    #
    # # 0.6 showed the best clustering performance so we give cell annotation for the clusters from this resolution
    # METHOD = 'davies_bouldin'
    # compare_clustering_resolutions(adata=adata_layer1,
    #                                resolutions=RESOLUTIONS,
    #                                method=METHOD)
    #
    # plot_compare_clustering_line(adata=adata_layer1,
    #                              method=METHOD,
    #                              out_dir=OUT_DIR,
    #                              figsize=(7, 5),
    #                              name="layer_1")
    #
    # # Layer1 annotation-ESI
    # # Layer 1 cell type annotation
    # CLUSTER_KEY = 'layer_1_0.6'
    # NEW_CLUSTER_KEY = "layer_1"
    # esi = ['Immune', 'Epithelial', 'Stromal', 'Immune', 'Epithelial',
    #        'Epithelial', 'Epithelial', 'Immune', 'Stromal', 'Undefined',
    #        'Stromal', 'Stromal', 'Immune', 'Immune', 'Undefined']
    #
    # # Update the labels to Anndata object
    # esi_map = dict(zip(range(len(esi)), esi))
    # adata_layer1.obs[CLUSTER_KEY] = adata_layer1.obs[CLUSTER_KEY].astype(int)
    # update_labels(adata_layer1, NEW_CLUSTER_KEY, [esi_map[i] for i in adata_layer1.obs[CLUSTER_KEY]])
    # adata.obs[NEW_CLUSTER_KEY] = adata_layer1.obs[NEW_CLUSTER_KEY]
    # adata = adata[adata.obs[NEW_CLUSTER_KEY] != "Undefined"]
    #
    # # Validation of first round cell type annotation
    # plot_zscore_cluster_mean_heatmap(adata, MARKER_LIST, NEW_CLUSTER_KEY,
    #                                  OUT_DIR, vmax=1, vmin=-0.5, figsize=(7, 7), name=f"{NEW_CLUSTER_KEY}")
    #
    # # ESI cell type distributions
    # for roi in set(adata.obs.unique_img_key):
    #     temp_adata = adata[adata.obs["unique_img_key"] == roi]
    #     plot_spatial_cluster_distribution(adata=temp_adata, cluster_key=NEW_CLUSTER_KEY, out_dir=OUT_DIR,
    #                                       color=ESI_CELL_TYPE_COLOR_DICT,
    #                                       figsize=(3.5, 3.5), dotsize=0.5, name=roi + "_L1")
    #
    # adata.write(IN_DIR / f"{EXPERIMENT}_layer1.h5ad")
    #
    # # Second-round clustering
    # CLUSTER_KEY = 'layer_1'
    # NEW_CLUSTER_KEY = 'layer_2'
    LAYER_2_E_MARKERS = ['CK5', 'CK7', 'CK8-18']
    LAYER_2_TME_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68',
                           'CD31', 'SMA', 'Vimentin']
    #
    # adata_layer2_e = adata[:, LAYER_2_E_MARKERS].copy()
    # adata_layer2_e = adata_layer2_e[adata_layer2_e.obs[CLUSTER_KEY] == "Epithelial"]
    # adata_layer2_tme =adata[:, LAYER_2_TME_MARKERS].copy()
    # adata_layer2_tme = adata_layer2_tme[adata_layer2_tme.obs[CLUSTER_KEY] != "Epithelial"]
    #
    # # layer 2 clustering: grid search on resolution
    # RESOLUTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #
    # sc.pp.neighbors(adata_layer2_e, n_neighbors=15, use_rep='X_pca')
    #
    # # Grid search for clustering with different resolution
    # for res in RESOLUTIONS:
    #     CLUSTER_KEY = f'layer_2_{res:.1f}'
    #     print(f"Resolution: {res:.1f}")
    #
    #     sc.tl.leiden(adata_layer2_e, resolution=res, key_added=CLUSTER_KEY)
    #
    #     print("Plotting")
    #     plot_zscore_cluster_mean_heatmap(
    #         adata=adata_layer2_e,
    #         column_names=LAYER_2_E_MARKERS,
    #         cluster_key=CLUSTER_KEY,
    #         out_dir=OUT_DIR,
    #         vmax=1.5,
    #         vmin=-0.5,
    #         figsize=(7, 7),
    #         name=f'layer_2_{res:.1f}'
    #     )
    #
    # METHOD = 'davies_bouldin'
    # compare_clustering_resolutions(adata=adata_layer2_e,
    #                                resolutions=RESOLUTIONS,
    #                                method=METHOD)
    #
    # plot_compare_clustering_line(adata=adata_layer2_e,
    #                              method=METHOD,
    #                              out_dir=OUT_DIR,
    #                              figsize=(7, 5),
    #                              name="layer_2")
    # # resolution 0.6 shows the best clustering result, so here we annotate the clusters from under that resolution
    # e_labels = ["CK7+ Luminal", "CK5+ Basal", "CK+ Epithelial", "CK5+ Basal", "CK7+ Luminal",
    #             "CK+ Epithelial", "CK5+ Basal", "CK+ Epithelial", "CK+ Epithelial", "CK5+ Basal",
    #             "CK5+ Basal", "CK8/18+ Luminal"]
    #
    # # Update the labels to Anndata object
    # CLUSTER_KEY = "layer_2_0.6"
    # NEW_CLUSTER_KEY = "layer_2"
    # e_map = dict(zip(range(len(e_labels)), e_labels))
    # adata_layer2_e.obs[CLUSTER_KEY] = adata_layer2_e.obs[CLUSTER_KEY].astype(int)
    # update_labels(adata_layer2_e, NEW_CLUSTER_KEY, [e_map[i] for i in adata_layer2_e.obs[CLUSTER_KEY]])
    #
    # # Validation of first round cell type annotation
    # plot_zscore_cluster_mean_heatmap(adata_layer2_e, LAYER_2_E_MARKERS, NEW_CLUSTER_KEY,
    #                                  OUT_DIR, vmax=1, vmin=-0.5, figsize=(7, 7), name=f"{NEW_CLUSTER_KEY}_e")
    #
    # # layer 2 clustering for tme group
    # sc.pp.neighbors(adata_layer2_tme, n_neighbors=15, use_rep='X_pca')
    #
    # # Grid search for clustering with different resolution
    # for res in RESOLUTIONS:
    #     CLUSTER_KEY = f'layer_2_{res:.1f}'
    #     print(f"Resolution: {res:.1f}")
    #
    #     #sc.tl.leiden(adata_layer2_tme, resolution=res, key_added=CLUSTER_KEY)
    #
    #     print("Plotting")
    #     plot_zscore_cluster_mean_heatmap(
    #         adata=adata_layer2_tme,
    #         column_names=LAYER_2_TME_MARKERS,
    #         cluster_key=CLUSTER_KEY,
    #         out_dir=OUT_DIR,
    #         vmax=2,
    #         vmin=-1,
    #         figsize=(7, 7),
    #         name=f'layer_2_{res:.1f}'
    #     )
    #
    # METHOD = 'davies_bouldin'
    # compare_clustering_resolutions(adata=adata_layer2_tme,
    #                                resolutions=RESOLUTIONS,
    #                                method=METHOD)
    #
    # plot_compare_clustering_line(adata=adata_layer2_tme,
    #                              method=METHOD,
    #                              out_dir=OUT_DIR,
    #                              figsize=(7, 5),
    #                              name="layer_2")
    # # resolution 0.5 shows the best clustering result, so here we annotate the clusters from under that resolution
    # tme_labels = ["Macrophage", "SMA+ Stromal", "Stromal", "Macrophage", "SMA+ Stromal",
    #               "SMA+ Vim+ Stromal", "B-cell", "Stromal", "Endothelial", "Macrophage",
    #               "T-cell"]
    #
    # # Update the labels to Anndata object
    # CLUSTER_KEY = "layer_2_0.5"
    # NEW_CLUSTER_KEY = "layer_2"
    # tme_map = dict(zip(range(len(tme_labels)), tme_labels))
    # adata_layer2_tme.obs[CLUSTER_KEY] = adata_layer2_tme.obs[CLUSTER_KEY].astype(int)
    # update_labels(adata_layer2_tme, NEW_CLUSTER_KEY, [tme_labels[i] for i in adata_layer2_tme.obs[CLUSTER_KEY]])
    #
    # # Validation of first round cell type annotation
    # plot_zscore_cluster_mean_heatmap(adata_layer2_tme, LAYER_2_TME_MARKERS, NEW_CLUSTER_KEY,
    #                                  OUT_DIR, vmax=1, vmin=-0.5, figsize=(7, 7), name=f"{NEW_CLUSTER_KEY}_tme")
    #
    # # Update both epithelial and tmw cells into the overall adata
    # temp = np.zeros(len(adata)).astype(str)
    # temp[adata.obs.layer_1 == "Epithelial"] = adata_layer2_e.obs.layer_2
    # temp[adata.obs.layer_1 != "Epithelial"] = adata_layer2_tme.obs.layer_2
    # update_labels(adata, "layer_2", temp)
    #
    # Visualization
    CLUSTER_KEY = "layer_2"
    plot_zscore_cluster_mean_heatmap(adata, LAYER_2_E_MARKERS + LAYER_2_TME_MARKERS, CLUSTER_KEY,
                                     OUT_DIR, vmax=2, vmin=-0.25, figsize=(7, 7), name=CLUSTER_KEY)

    # Need to remove the cells in the cluster of "CK8/18+ Luminal" since the z-score heatmap
    adata_new = adata.copy()
    adata_new = adata_new[adata_new.obs.layer_2 != "CK8/18+ Luminal"]

    adata_new.write(IN_DIR / f"{EXPERIMENT}_layer2.h5ad")

    for roi in set(adata.obs.unique_img_key):
        temp_adata = adata[adata.obs["unique_img_key"] == roi]
        plot_spatial_cluster_distribution(adata=temp_adata, cluster_key=CLUSTER_KEY, out_dir=OUT_DIR,
                                          color=LAYER_2_CELL_TYPE_COLOR_DICT,
                                          figsize=(3.5, 3.5), dotsize=0.5, name=roi + "_L2")

    # After defining the cell types, we then move on to EDA (abundance)
    # Cell Counts
    from src.visualize import plot_cell_area_count, plot_cluster_cell_count, plot_cell_area

    CLUSTER_KEY = "layer_1"
    plot_cluster_cell_count(adata, CLUSTER_KEY, OUT_DIR, figsize=(3.5, 2), name="ct_L1", dpi=150)
    CLUSTER_KEY = "layer_2"
    plot_cluster_cell_count(adata, CLUSTER_KEY, OUT_DIR, figsize=(3.5, 2), name="ct_L2", dpi=150)

    # Cell Area
    plot_cell_area_count(adata, OUT_DIR, figsize=(3.5, 2), name=None)

    # Abundance of cell types
    # Bars showing abundance across patients
    from visualization import plot_cell_type_prop_per_img_bar
    # layer_1 (ESI)
    CLUSTER_KEY = "layer_1"
    UNIT_KEY = 'CID'
    CLUSTER_KEY_ORDER = list(ESI_CELL_TYPE_COLOR_DICT.keys())[::-1]
    plot_cell_type_prop_per_img_bar(adata=adata, unit_key=UNIT_KEY, cluster_key=CLUSTER_KEY,
                                    color_dict=ESI_CELL_TYPE_COLOR_DICT, out_dir=OUT_DIR,
                                    cluster_key_order=CLUSTER_KEY_ORDER, figsize=(25, 15), name="L1_ct", dpi=150)

    # layer_2
    CLUSTER_KEY = "layer_2"
    UNIT_KEY = 'CID'
    CLUSTER_KEY_ORDER = list(LAYER_2_CELL_TYPE_COLOR_DICT.keys())[::-1]
    plot_cell_type_prop_per_img_bar(adata=adata, unit_key=UNIT_KEY, cluster_key=CLUSTER_KEY,
                                    color_dict=LAYER_2_CELL_TYPE_COLOR_DICT, out_dir=OUT_DIR,
                                    cluster_key_order=CLUSTER_KEY_ORDER, figsize=(25, 15), name="L2_ct", dpi=150)

    # Shannon Diversity
    from spatial_analysis import run_shannon_entropy
    UNIT_KEY = "CID"
    CLUSTER_KEY = "layer_2"
    entropy_score = run_shannon_entropy(adata=adata, unit_key=UNIT_KEY, cluster_key=CLUSTER_KEY)

    # # Cliff Delta (run and plot)
    # from spatial_analysis import compare_effect_size
    # from visualization import plot_effect_size
    #
    # CLUSTER_KEY = "layer_2"
    # UNIT_KEY = "CID"
    # OBSERVATION_KEY = "Recurrence status"
    # OBSERVATION_ORDER = [1, 0]
    # CELL_TYPE_LIST = ['CK5+ Basal',  # Epithelial
    #                   'CK7+ Luminal',
    #                   'CK+ Epithelial',
    #                   'T-cell',  # Immune
    #                   'B-cell',
    #                   'Macrophage',
    #                   'Endothelial',  # Stromal
    #                   'SMA+ Stromal',
    #                   'SMA+ Vim+ Stromal',
    #                   'Stromal'
    #                   ]
    # ESI_CELL_TYPE_LENGTH = [3, 7]
    # ESI_CELL_TYPE_LENGTH = [3, 3, 4]
    #
    # ct_mtx, delta_df = compare_effect_size(adata=adata, cluster_key=CLUSTER_KEY, unit_key=UNIT_KEY,
    #                                        observation_key=OBSERVATION_KEY, observation_order=OBSERVATION_ORDER,
    #                                        cell_type_list=CELL_TYPE_LIST, cell_type_len=ESI_CELL_TYPE_LENGTH)
    #
    # # plotting
    # df = adata.obs[[OBSERVATION_KEY, "CID"]]
    # plot_effect_size(delta_results_df=delta_df,
    #                  cluster_key=CLUSTER_KEY,
    #                  calculate_key='cliff_delta',
    #                  observation_order=OBSERVATION_ORDER,
    #                  observation_counts=pd.Series(
    #                      dict(zip(df.CID.values, df[OBSERVATION_KEY].values)).values()).value_counts(),
    #                  observation_dict={0: 'No recurrence', 1: 'Recurrence'},
    #                  color_dict=LAYER_2_CELL_TYPE_COLOR_DICT,
    #                  out_dir=OUT_DIR,
    #                  figsize=(8, 8),
    #                  name="3types_Recurrence status_cliff_cell_type_cid")



