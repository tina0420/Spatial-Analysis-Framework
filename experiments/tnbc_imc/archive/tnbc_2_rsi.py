"""
@author: Tina Hsu
RSI cell types identification for TNBC IMC.
"""
import os
from pathlib import Path
from tnbc_data_info import EXPERIMENT_LIST, EXPERIMENT_MARKERS
from anndata import read_h5ad
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import run_leiden, update_labels
from src.visualize import plot_zscore_cell_expression_heatmap, plot_umap, plot_umap_panel, \
    plot_zscore_cluster_mean_heatmap, plot_cluster_cell_count
import scanpy as sc


def run_gmm_for_epi(adata, marker_list, cluster_key, out_dir, name=None):
    """
    Running 2-component GMM to classify epithelial and non-epithelial cells
    :param adata: AnnData
    :param marker_list: markers used for GMM, type: list[str]
    :param cluster_key: indicate the column used in adata.obs, type: str
    :param out_dir: output directory, type: Path
    :param name: name for the output file, type: str
    :return:
    """
    print("run gmm")
    markers = np.sum(adata[:, marker_list].X, axis=1).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=2, random_state=42).fit(markers)
    bgm_labels = bgm.predict(markers)
    update_labels(adata, cluster_key=cluster_key, labels=bgm_labels)

    print("finish updating labels")
    # check which is epithelial and which is non-epithelial in the classification result
    if adata[np.where(adata.obs[CLUSTER_KEY] == 0)[0], marker_list].X.mean() > adata[
        np.where(adata.obs[CLUSTER_KEY] == 1)[0], marker_list].X.mean():
        # 0: epithelial, 1: non-epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'non-epithelial'})
    else:
        # 0: non-epithelial, 1: epithelial
        adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'non-epithelial', 1: 'epithelial'})

    # store the result
    adata.write(out_dir / f"{name}_annotation.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0], :].write(out_dir / f"{name}_epithelial.h5ad")
    adata[np.where(adata.obs[CLUSTER_KEY] == 'non-epithelial')[0], :].write(out_dir / f"{name}_non-epithelial.h5ad")
    plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name=f"{name}_find_epithelial")
    print("start running pca")
    # running PCA to check the accuracy of classification (plot)
    temp_adata = adata.copy()
    sc.pp.pca(temp_adata)
    sc.pp.neighbors(temp_adata)
    sc.tl.umap(temp_adata)
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, figsize=(4.5, 4.5), dotsize=0.5, name=f"{name}_find_epithelial")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, dotsize=0.1, name=f"{name}_find_epithelial")


def run_leiden_for_clusters(adata, marker_list, cluster_key, out_dir, name):
    run_leiden(adata=adata, columns=marker_list, n_neighbors=150, name=cluster_key)  # 13:33 starts
    print("plotting")
    plot_zscore_cluster_mean_heatmap(tme_full_adata, marker_list, cluster_key,
                                     out_dir=out_dir, figsize=(7, 7), name=name)
    print("storing the h5ad file")
    tme_full_adata.write(out_dir / name)


def update_rsi_labels(adata, ):
    pass


if __name__ == '__main__':
    # running GMM first to see the classification of epithelial and non-epithelial cells
    # common_markers = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'CD3e', 'CD20', 'CD45',
    #                       'CD68', 'CD31', 'SMA', 'Vimentin']
    TYPE = 'tnbc_imc'
    CLUSTER_KEY = 'rsi'
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    """
    Running all the experiments in the first step (epithelial)
    """
    for EXPERIMENT in EXPERIMENT_LIST:
        # testing for one xp
        print("___________________________________________________")
        print(EXPERIMENT)
        if EXPERIMENT != 'XP12274r34':

            in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
            out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)

            # adata = read_h5ad(in_dir / f"{EXPERIMENT}_filter.h5ad")
            #
            # run_gmm_for_epi(adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=EXPERIMENT)
            # # if change the v_mzx and v_min to make it interpretable, rerun it
            # plot_zscore_cell_expression_heatmap(adata, adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
            #                                     rotation=35, name=f"{EXPERIMENT}_2_find_epithelial", v_max=2, v_min=-2)

            # read the epithelial and non-epithelial h5ad individually and more separate immune and stromal and then integrate
            # all the labels
            epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")
            tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")

            # (non-epithelial) identify immune and stromal cells by leiden clustering
            IMMUNE_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

            NEW_CLUSTER_KEY = 'leiden'
            run_leiden(adata=tme_adata, columns=IMMUNE_MARKERS, n_neighbors=150, name=NEW_CLUSTER_KEY)  # 10:04 starts
            plot_zscore_cluster_mean_heatmap(tme_adata, IMMUNE_MARKERS, NEW_CLUSTER_KEY,
                                             out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_ime_str")
            tme_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")

    tme_labels = tme_adata.obs[NEW_CLUSTER_KEY].to_numpy()
    new_labels = np.zeros(adata.shape[0], dtype=int)

    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'epithelial')[0]] = [int(i) for i in epi_labels]

    new_labels[np.where(adata.obs[CLUSTER_KEY] == 'non-epithelial')[0]] = [int(i) + 21 for i in tme_labels]

    # ref: adata.obs[cluster_key] = adata.obs[cluster_key].map({0: 'epithelial', 1: 'non-epithelial'
    # ref: label_name_dict = dict(zip(range(len(label_names)), label_names))
    # update_labels(adata, NEW_CLUSTER_KEY, [label_name_dict[i] for i in adata.obs[CLUSTER_KEY]])

    update_labels(adata, "leiden", new_labels)

    """
    To identify which is immune and which is stromal by the heatmap
    """
    IMM_STR_LABELS = {}
    # need to keep the rule, if all the colors showing in similar blue, we identify as stromal and will take a note
    # cluster 1, 10
    IMM_STR_LABELS['XP12446'] = ['stromal', 'stromal', 'immune', 'immune', 'stromal',
                                 'stromal', 'immune', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'immune', 'immune', 'stromal',
                                 'immune', 'immune', 'stromal']
    # cluster 9
    IMM_STR_LABELS['XP13085'] = ['immune', 'immune', 'immune', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'stromal', 'immune', 'immune']
    # cluster 4, 8, 9
    IMM_STR_LABELS['XP12813'] = ['stromal', 'immune', 'immune', 'stromal', 'stromal',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'stromal', 'immune', 'stromal',
                                 'stromal', 'immune', 'immune', 'immune', 'immune']

    IMM_STR_LABELS['XP13083'] = ['stromal', 'immune', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'stromal', 'stromal',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'immune', 'stromal']
    # cluster 3
    IMM_STR_LABELS['XP13087'] = ['immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'immune', 'immune',
                                 'stromal', 'stromal', 'immune', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'stromal']
    # cluster 7
    IMM_STR_LABELS['XP13084'] = ['stromal', 'stromal', 'stromal', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'stromal', 'immune', 'stromal', 'immune', 'stromal',
                                 'immune']
    # cluster 6, 14, 18
    IMM_STR_LABELS['XP12814'] = ['immune', 'stromal', 'stromal', 'immune', 'immune',
                                 'stromal', 'stromal', 'stromal', 'immune', 'immune',
                                 'immune', 'immune', 'immune', 'stromal', 'stromal',
                                 'stromal', 'stromal', 'immune', 'stromal']
    # cluster 7
    IMM_STR_LABELS['XP13086'] = ['immune', 'immune', 'stromal', 'stromal', 'immune',
                                 'immune', 'immune', 'stromal', 'immune', 'stromal',
                                 'immune', 'stromal', 'stromal', 'immune', 'immune',
                                 'stromal', 'immune']
    # cluster 2
    IMM_STR_LABELS['XP12274r1'] = ['stromal', 'stromal', 'stromal', 'immune', 'stromal',
                                   'stromal', 'stromal', 'immune', 'immune', 'immune',
                                   'immune', 'immune', 'immune', 'stromal', 'stromal',
                                   'immune', 'immune', 'stromal']
    # cluster 0, 4, 6
    IMM_STR_LABELS['XP12274r23'] = ['stromal', 'immune', 'stromal', 'stromal', 'stromal',
                                    'stromal', 'stromal', 'immune', 'immune', 'immune',
                                    'immune', 'stromal', 'stromal', 'stromal', 'stromal',
                                    'stromal', 'immune']
    # cluster 0
    IMM_STR_LABELS['XP12274r34'] = ['stromal', 'stromal', 'immune', 'stromal', 'stromal',
                                    'immune', 'stromal', 'immune', 'immune', 'immune',
                                    'immune', 'stromal', 'stromal', 'stromal', 'stromal',
                                    'immune']




    # update rsi in each anndata
    for EXPERIMENT in EXPERIMENT_LIST:
        # testing for one xp
        print("___________________________________________________")
        print(EXPERIMENT)
        if EXPERIMENT != 'XP12274r34':

            in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
            out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)
            print("read file")
            # read the annotation.h5ad
            adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
            tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
            epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

            print("run leiden")
            # run leiden for epithelial cells
            EPITHELIAL_CLUSTER_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67']
            NEW_CLUSTER_KEY = 'leiden'
            run_leiden(adata=epi_adata, columns=EPITHELIAL_CLUSTER_MARKERS, n_neighbors=150, name=NEW_CLUSTER_KEY)  # 16:13 starts
            plot_zscore_cluster_mean_heatmap(epi_adata, EPITHELIAL_CLUSTER_MARKERS, NEW_CLUSTER_KEY,
                                             out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_epi")
            print("store the file")
            epi_adata.write(out_dir / f"{EXPERIMENT}_epithelial.h5ad")


    for EXPERIMENT in EXPERIMENT_LIST:
        # testing for one xp
        print("___________________________________________________")
        print(EXPERIMENT)
        if EXPERIMENT != 'XP12274r34':
            in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
            out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)

            print("read the files")
            # tidy and update the labels, including rsi and leiden (one xp testing with XP12274r34)
            adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
            tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
            epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

            print("start mergeing leiden clusters")
            # for leiden (based on leiden clusters stored in each adata)
            CLUSTER_KEY = 'rsi'
            NEW_CLUSTER_KEY = 'leiden'
            epi_labels = epi_adata.obs[NEW_CLUSTER_KEY].to_numpy()
            tme_labels = tme_adata.obs[NEW_CLUSTER_KEY].to_numpy()

            new_labels = np.zeros(adata_anno.shape[0], dtype=int)
            new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'epithelial')[0]] = [int(i) for i in epi_labels]
            new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'non-epithelial')[0]] = [int(i) + len(set(epi_labels))-1 for i in tme_labels]

            print("update and storage")
            update_labels(adata_anno, NEW_CLUSTER_KEY, new_labels)
            adata_anno.write(out_dir / f"{EXPERIMENT}_annotation.h5ad")




    # for rsi (obtain immune and stromal cells from the clusters merged manually (IMM_STR_LABELS)

    for EXPERIMENT in EXPERIMENT_LIST:
        # testing for one xp
        print("___________________________________________________")
        print(EXPERIMENT)
        if EXPERIMENT != 'XP12274r34':
            in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
            out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

            if os.path.exists(out_dir) == 0:
                os.makedirs(out_dir, exist_ok=True)

            print("read the files")
            # tidy and update the labels, including rsi and leiden (one xp testing with XP12274r34)
            adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
            tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
            epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

            print("start mapping")
            label_name_dict = dict(zip(range(len(IMM_STR_LABELS[EXPERIMENT])), IMM_STR_LABELS[EXPERIMENT]))
            CLUSTER_KEY = 'rsi'
            NEW_CLUSTER_KEY = 'leiden'
            update_labels(tme_adata, CLUSTER_KEY, [label_name_dict[i] for i in tme_adata.obs[NEW_CLUSTER_KEY]])
            tme_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")



    """
    Full anndata
    """
    in_dir = Path(f"data/{TYPE}")
    adata_list = [read_h5ad(in_dir / xp / f"{xp}_filter.h5ad")[:, common_markers] for xp in EXPERIMENT_LIST]

    EXPERIMENT = 'full_11_xp'
    in_dir = Path(f"analysis/{TYPE}/original_data_results")
    out_dir = Path(f"analysis/{TYPE}/original_data_results")

    # all 2038297 cells across 11 XPs
    # full_adata = read_h5ad(in_dir / f"{EXPERIMENT}_epi_tme_gmm.h5ad")

    # focus on TME to further separate immune and stromal cells (by z-score heatmap)
    tme_full_adata = read_h5ad(in_dir / f"{EXPERIMENT}_tme.h5ad")

    IMMUNE_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']

    NEW_CLUSTER_KEY = 'leiden'
    run_leiden(adata=tme_full_adata, columns=IMMUNE_MARKERS, n_neighbors=150, name=NEW_CLUSTER_KEY)  # 13:33 starts
    print("plotting")
    plot_zscore_cluster_mean_heatmap(tme_full_adata, IMMUNE_MARKERS, NEW_CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name=f"{EXPERIMENT}_ime_str")
    print("storing the h5ad file")
    tme_full_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")

    # identify the clusters for immune or stromal
    # cluster 17 not sure but follow the color in z-score
    IMM_STR_LABELS['full_11_xp'] = ['stromal', 'immune', 'immune', 'stromal', 'immune',
                                    'immune', 'immune', 'immune', 'stromal', 'immune',
                                    'stromal', 'immune', 'immune', 'immune', 'stromal',
                                    'stromal', 'stromal', 'stromal', 'immune', 'immune',
                                    'stromal']

    # merge the esi labels
    # adata_anno.obs['gmm'] = adata_anno.obs['gmm'].map({'epithelial': 'epithelial', 'TME': 'non-epithelial'})
    print("read the files")
    # tidy and update the labels, including rsi and leiden (one xp testing with XP12274r34)
    adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_epi_tme_gmm.h5ad")
    tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
    epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

    print("start mergeing leiden clusters")
    # for leiden (based on leiden clusters stored in each adata)
    # CLUSTER_KEY = 'gmm'
    # NEW_CLUSTER_KEY = 'rsi'
    # epi_labels = epi_adata.obs[CLUSTER_KEY].to_numpy()
    # tme_labels = tme_adata.obs[CLUSTER_KEY].to_numpy()
    #
    # new_labels = np.zeros(adata_anno.shape[0], dtype=int)
    # new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'epithelial')[0]] = [int(i) for i in epi_labels]
    # new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'non-epithelial')[0]] = [int(i) + len(set(epi_labels)) - 1 for i
    #                                                                             in tme_labels]

    print("start mapping")
    label_name_dict = dict(zip(range(len(IMM_STR_LABELS[EXPERIMENT])), IMM_STR_LABELS[EXPERIMENT]))
    CLUSTER_KEY = 'rsi'
    NEW_CLUSTER_KEY = 'leiden'
    update_labels(tme_adata, CLUSTER_KEY, [label_name_dict[i] for i in tme_adata.obs[NEW_CLUSTER_KEY]])
    tme_adata.write(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")

    # compare the results from individual slide and all slides

    # get the list of the cell types for all
    epi_labels = epi_adata.obs['gmm'].to_numpy()
    tme_labels = tme_adata.obs['rsi'].to_numpy()

    new_labels = np.zeros(adata_anno.shape[0], dtype=int)
    new_labels = np.zeros(adata_anno.shape[0], dtype=object)
    new_labels[np.where(adata_anno.obs['gmm'] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(adata_anno.obs['gmm'] == 'TME')[0]] = tme_labels

    full_rsi_labels = new_labels

    new_labels_all = []
    CLUSTER_KEY = 'rsi'
    NEW_CLUSTER_KEY = 'leiden'
    for EXPERIMENT in EXPERIMENT_LIST:
        print("------------------------------")
        print(EXPERIMENT)

        in_dir = Path(f"output/{TYPE}/{EXPERIMENT}")
        out_dir = Path(f"output/{TYPE}/{EXPERIMENT}")

        adata_anno = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
        tme_adata = read_h5ad(out_dir / f"{EXPERIMENT}_non-epithelial.h5ad")
        epi_adata = read_h5ad(out_dir / f"{EXPERIMENT}_epithelial.h5ad")

        print("get labels")
        epi_labels = epi_adata.obs[CLUSTER_KEY].to_numpy()
        tme_labels = tme_adata.obs[CLUSTER_KEY].to_numpy()
        tme_labels = tme_adata.obs[CLUSTER_KEY].to_numpy()
        print("check if the tme labels updated in tme_adata")

        print("merging")
        new_labels = np.zeros(adata_anno.shape[0], dtype=object)
        new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'epithelial')[0]] = epi_labels
        new_labels[np.where(adata_anno.obs[CLUSTER_KEY] == 'non-epithelial')[0]] = tme_labels

        print("shape of the anndata")
        print(adata_anno.shape[0])
        print("length of the labels")
        print(len(new_labels))

        new_labels_all.append(new_labels)

    new_type_labels_all = np.concatenate(new_labels_all)

    # finish preparation and start doing comparison
    # 1.merged individual result, new_type_labels_all
    # 2. results from full dataset
    EXPERIMENT = 'full_11_xp'
    CLUSTER_KEY = 'rsi'
    in_dir = Path(f"analysis/{TYPE}/original_data_results")
    full_adata = read_h5ad(in_dir / f"{EXPERIMENT}_annotation.h5ad")
    full_labels = full_adata.obs[CLUSTER_KEY].to_numpy()
    diff_num = np.sum(new_type_labels_all != full_labels)








