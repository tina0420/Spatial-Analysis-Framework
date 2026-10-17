"""
@author: Tina Hsu
TNBC IMC layer 1 cell types (ESI)
"""
import pandas as pd
from experiments.tnbc_imc.tnbc_imc_setup_xp import *
# from experiments.tnbc_imc.tnbc_tma_1_batch_effect import *
import os
from pathlib import Path
from anndata import read_h5ad
from experiments.tnbc_imc.tnbc_tma_2_gmm import *
import scanpy as sc
from src.visualize import plot_zscore_cluster_mean_heatmap
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# tidy up the code with updating labels here and can import it directly


def update_manual_labels(adata, old_cluster_key, new_cluster_key, new_labels, out_dir, name):
    """
    Update manual annotation to the immune, stromal and unknown cell types
    :param adata: Anndata
    :param old_cluster_key: indicate the column with the reference for mapping new labels, type: str
    :param new_cluster_key: indicate the column we need to update with the manual annotations, type: str
    :param new_labels: new labels for update, type: np.Array
    :param out_dir: output directory, type: Path
    :param name: extra name for the output file, type: str
    :return:
    """
    temp_adata = adata.copy()
    label_name_dict = dict(zip(range(len(new_labels)), new_labels))

    update_labels(temp_adata, new_cluster_key, [label_name_dict[int(i)] for i in temp_adata.obs[old_cluster_key]])

    temp_adata.write(out_dir / f"{name}_non-epithelial.h5ad")


def update_all_labels(experiment, cluster_key, out_dir):
    """
    Update all the annotation together into the merged one
    :param experiment: experiment name, type: str
    :param cluster_key: indicate the column in obs, type: str
    :param out_dir: output directory, type: Path
    :return:
    """
    temp_adata = read_h5ad(out_dir / f"{experiment}_merge.h5ad")
    temp_epi = read_h5ad(out_dir / f"{experiment}_epithelial.h5ad")
    temp_tme = read_h5ad(out_dir / f"{experiment}_non-epithelial.h5ad")

    new_labels = np.zeros(temp_adata.shape[0], dtype=object)
    epi_labels = np.array(['epithelial'] * temp_epi.shape[0])
    tme_labels = temp_tme.obs[cluster_key].to_numpy()

    new_labels[np.where(temp_adata.obs[cluster_key] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(temp_adata.obs[cluster_key] != 'epithelial')[0]] = tme_labels

    update_labels(temp_adata, cluster_key, new_labels)

    temp_adata.write(out_dir / f"{experiment}_merge.h5ad")


# we need to have leiden clustering with xp1+xp2 for each TMA individually
if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    # TMA = 'full_4_tma'
    TMA_LIST = ['BOB', '22-010A', '22-010C', '23-009']

    common_markers_record = {}
    # merge XPs from the same TMA
    for tma in TMA_LIST:
        print("________________")
        print(tma)
        EXPERIMENT_1 = TMA_XP[tma][0]
        EXPERIMENT_2 = TMA_XP[tma][1]

        print(EXPERIMENT_1)
        print(EXPERIMENT_2)

        in_dir = Path(f"data/{TYPE}/{EXPERIMENT_1}")
        adata_1 = read_h5ad(in_dir / f"{EXPERIMENT_1}_filter.h5ad")
        in_dir = Path(f"data/{TYPE}/{EXPERIMENT_2}")
        adata_2 = read_h5ad(in_dir / f"{EXPERIMENT_2}_filter.h5ad")

        out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        # get common markers and store it
        common_markers = list(adata_1.var_names.intersection(adata_2.var_names))
        common_markers_record[tma] = common_markers

        # merge and output
        adata_1 = adata_1[:, common_markers]
        adata_2 = adata_2[:, common_markers]

        adata = adata_1.concatenate(adata_2, batch_key='batch')
        adata.obs['batch'] = adata.obs['batch'].map({'0': EXPERIMENT_1, '1': EXPERIMENT_2})

        adata.write(out_dir / f"{tma}_combined.h5ad")

    # running gmm with each tma_combined.h5ad
    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    CLUSTER_KEY = 'esi'
    for tma in TMA_LIST:
        print(tma)
        adata = read_h5ad(in_dir / f"{tma}_combined.h5ad")
        temp_adata = adata.copy()
        run_gmm_for_epithelial(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=f"{tma}_combined")

        # check epithelial and non-epithelial zscore and UMAP
        plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                            rotation=35, name=f"{tma}_combined_find_epithelial")

    # running leiden cluster with non-epithelial h5ad for each TMA_combined_non-epithelial
    CLUSTER_KEY = 'leiden'
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']
    """
    23-009 needs to run itself because of the tme markers it has are different
    """

    for tma in ['BOB', '22-010A', '22-010C']:
        print(tma)
        adata_tme = read_h5ad(in_dir / f"{tma}_combined_non-epithelial.h5ad")
        temp_adata = adata_tme[:, IMMUNE_CLUSTER_MARKERS].copy()
        sc.tl.pca(temp_adata, svd_solver='arpack')
        sc.pp.neighbors(temp_adata, use_rep='X_pca')
        sc.tl.leiden(temp_adata)
        plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                         out_dir=out_dir, figsize=(7, 7), name=f"{tma}_combined_non-epithelial")
        temp_adata.write(out_dir / f"{tma}_combined_non-epithelial.h5ad")

    # for 23-009
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin']
    tma = '23-009'
    adata_tme = read_h5ad(in_dir / f"{tma}_combined_non-epithelial.h5ad")
    temp_adata = adata_tme[:, IMMUNE_CLUSTER_MARKERS].copy()
    sc.tl.pca(temp_adata, svd_solver='arpack')
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    sc.tl.leiden(temp_adata)
    plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name=f"{tma}_combined_non-epithelial")
    temp_adata.write(out_dir / f"{tma}_combined_non-epithelial.h5ad")

    # select one of the experiments in TMA with more relpase-related cores and run the preprocessing again
    xp_selected_dict = dict(zip(TMA_LIST, ['XP12446', 'XP13083', 'XP13087', 'XP13086']))
    for tma, xp in xp_selected_dict.items():
        print("_______________________________")
        print(tma, xp)
        in_dir = Path(f"data/{TYPE}/{xp}")
        adata = read_h5ad(in_dir / f"{xp}_filter.h5ad")
        out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
        adata.write(out_dir / f"{tma}_{xp}.h5ad")

    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    # run gmm with each tma_xp.h5ad
    EPITHELIAL_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK']
    CLUSTER_KEY = 'esi'
    for tma, xp in xp_selected_dict.items():
        print(tma, xp)
        adata = read_h5ad(in_dir / f"{tma}_{xp}.h5ad")
        temp_adata = adata.copy()
        run_gmm_for_epithelial(temp_adata, EPITHELIAL_MARKERS, CLUSTER_KEY, out_dir, name=f"{tma}_{xp}")

        # check epithelial and non-epithelial zscore and UMAP
        plot_zscore_cell_expression_heatmap(temp_adata, temp_adata.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                            rotation=35, name=f"{tma}_{xp}_find_epithelial")

    CLUSTER_KEY = 'leiden'
    IMMUNE_CLUSTER_MARKERS = ['CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']
    for tma, xp in xp_selected_dict.items():
        print(tma, xp)
        adata_tme = read_h5ad(in_dir / f"{tma}_{xp}_non-epithelial.h5ad")
        temp_adata = adata_tme[:, IMMUNE_CLUSTER_MARKERS].copy()
        sc.tl.pca(temp_adata, svd_solver='arpack')
        sc.pp.neighbors(temp_adata, use_rep='X_pca')
        sc.tl.leiden(temp_adata)
        plot_zscore_cluster_mean_heatmap(temp_adata, IMMUNE_CLUSTER_MARKERS, CLUSTER_KEY,
                                         out_dir=out_dir, figsize=(7, 7), name=f"{tma}_{xp}_non-epithelial")
        temp_adata.write(out_dir / f"{tma}_{xp}_non-epithelial.h5ad")

    # update the manual labels and merge them together

    IMM_STR_LABELS = {}
    IMM_STR_LABELS['BOB_combined'] = ['unknown', 'stromal', 'stromal', 'stromal', 'immune',
                                      'immune', 'stromal', 'immune', 'immune', 'stromal',
                                      'stromal', 'immune', 'immune', 'stromal', 'immune',
                                      'unknown', 'unknown', 'stromal', 'immune', 'unknown',
                                      'unknown', 'stromal', 'stromal']

    IMM_STR_LABELS['22-010A_combined'] = ['immune', 'stromal', 'stromal', 'immune', 'immune',
                                          'unknown', 'stromal', 'immune', 'immune', 'unknown',
                                          'immune', 'stromal', 'unknown', 'stromal', 'immune',
                                          'immune', 'unknown', 'stromal', 'unknown', 'stromal',
                                          'stromal', 'stromal', 'unknown', 'immune', 'unknown']

    IMM_STR_LABELS['22-010C_combined'] = ['immune', 'immune', 'immune', 'stromal', 'unknown',
                                          'stromal', 'immune', 'stromal', 'unknown', 'immune',
                                          'stromal', 'immune', 'stromal', 'stromal', 'unknown',
                                          'immune', 'stromal', 'stromal', 'unknown', 'stromal',
                                          'immune', 'stromal', 'immune', 'stromal', 'stromal',
                                          'unknown']

    IMM_STR_LABELS['23-009_combined'] = ['immune', 'unknown', 'immune', 'immune', 'immune',
                                         'stromal', 'immune', 'immune', 'unknown', 'stromal',
                                         'stromal', 'stromal', 'stromal', 'immune', 'immune',
                                         'unknown', 'stromal', 'immune', 'unknown', 'stromal',
                                         'stromal', 'stromal', 'unknown', 'immune', 'stromal',
                                         'stromal', 'immune', 'unknown', 'immune', 'unknown']

    IMM_STR_LABELS['BOB_XP12446'] = ['unknown', 'immune', 'unknown', 'immune', 'immune',
                                     'stromal', 'unknown', 'stromal', 'stromal', 'stromal',
                                     'immune', 'stromal', 'immune', 'stromal', 'stromal',
                                     'immune', 'immune', 'unknown', 'immune', 'stromal',
                                     'unknown', 'immune']

    IMM_STR_LABELS['22-010A_XP13083'] = ['unknown', 'immune', 'immune', 'unknown', 'immune',
                                         'stromal', 'stromal', 'immune', 'immune', 'unknown',
                                         'immune', 'stromal', 'unknown', 'unknown', 'stromal',
                                         'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                         'stromal', 'stromal', 'stromal', 'immune', 'stromal']

    IMM_STR_LABELS['22-010C_XP13087'] = ['immune', 'unknown', 'immune', 'unknown', 'immune',
                                         'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                         'stromal', 'immune', 'immune', 'stromal', 'stromal',
                                         'stromal', 'stromal', 'stromal', 'unknown', 'stromal',
                                         'immune', 'unknown', 'unknown']

    IMM_STR_LABELS['23-009_XP13086'] = ['immune', 'immune', 'immune', 'immune', 'stromal',
                                        'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                        'stromal', 'stromal', 'stromal', 'immune', 'unknown',
                                        'unknown', 'stromal', 'stromal', 'stromal', 'immune',
                                        'immune', 'immune', 'stromal', 'stromal', 'immune']

    # update non-epithelial labels and then update all the labels
    for tma, xp in xp_selected_dict.items():
        print(tma)
        print(f"{tma}_{xp}")

        # update non-epithelial labels
        OLD_CLUSTER_KEY = 'leiden'
        NEW_CLUSTER_KEY = 'esi'

        adata = read_h5ad(in_dir / f"{tma}_combined_non-epithelial.h5ad")
        labels = IMM_STR_LABELS[f"{tma}_combined"]
        update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, f"{tma}_combined")

        # update and merge all labels together into..._merge.h5ad
        CLUSTER_KEY = 'esi'
        update_all_labels(f"{tma}_combined", CLUSTER_KEY, out_dir)

        # CHECK
        adata = read_h5ad(out_dir / f"{tma}_combined_merge.h5ad")

    # for the individual xp
    for tma, xp in xp_selected_dict.items():
        print(tma)
        print(f"{tma}_{xp}")

        # update non-epithelial labels
        OLD_CLUSTER_KEY = 'leiden'
        NEW_CLUSTER_KEY = 'esi'

        adata = read_h5ad(in_dir / f"{tma}_{xp}_non-epithelial.h5ad")
        labels = IMM_STR_LABELS[f"{tma}_{xp}"]
        update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, f"{tma}_{xp}")

        # update and merge all labels together into..._merge.h5ad
        CLUSTER_KEY = 'esi'
        update_all_labels(f"{tma}_{xp}", CLUSTER_KEY, out_dir)

        # CHECK
        adata = read_h5ad(out_dir / f"{tma}_{xp}_merge.h5ad")


    # get the full h5ad (2: 1 in common markers and 1 with all markers and nan)
    # common markers first (tma one by one)
    def tidy_h5ad(tma, xp, in_dir):
        adata_indi = read_h5ad(in_dir / f"{tma}_{xp}_merge.h5ad")
        adata_merged = read_h5ad(in_dir / f"{tma}_combined_merge.h5ad")

        adata_merged = adata_merged[np.where(adata_merged.obs.batch == xp)[0], :]
        if adata_merged.shape[0] != adata_indi.shape[0]:
            print("The lengths of adata are not matched!")
        adata_indi.obs.columns = ['image_id', 'area', 'layer1']
        adata_indi.obs['layer1_combined'] = list(adata_merged.obs['esi'].values)

        temp_adata = adata_indi.copy()
        temp_df = pd.DataFrame({'layer1': temp_adata.obs['layer1'].map({'epithelial': 'epithelial',
                                                                        'immune': 'immune',
                                                                        'stromal': 'stromal',
                                                                        'unknown': 'stromal'}).astype('category'),
                                'layer1_combined': temp_adata.obs['layer1_combined'].map({'epithelial': 'epithelial',
                                                                                          'immune': 'immune',
                                                                                          'stromal': 'stromal',
                                                                                          'unknown': 'stromal'}).astype(
                                    'category')})
        consensus_labels = []
        for i in range(temp_df.shape[0]):
            if temp_df.iloc[i, 0] == temp_df.iloc[i, 1]:
                consensus_labels.append(temp_df.iloc[i, 0])
            else:
                consensus_labels.append('unknown')

        temp_adata.obs['layer1_consensus'] = consensus_labels

        return temp_adata

    tma = TMA_LIST[0]
    xp = xp_selected_dict[tma]
    adata_1 = tidy_h5ad(tma, xp, in_dir)

    tma = TMA_LIST[1]
    xp = xp_selected_dict[tma]
    adata_2 = tidy_h5ad(tma, xp, in_dir)

    tma = TMA_LIST[2]
    xp = xp_selected_dict[tma]
    adata_3 = tidy_h5ad(tma, xp, in_dir)

    tma = TMA_LIST[3]
    xp = xp_selected_dict[tma]
    adata_4 = tidy_h5ad(tma, xp, in_dir)

    # merge with common markers
    common_markers = list(
        set(adata_1.var_names) & set(adata_2.var_names) & set(adata_3.var_names) & set(adata_4.var_names))
    common_markers = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3',
                      'CD3e', 'CD20', 'CD45', 'CD68',
                      'CD31', 'SMA', 'Vimentin', 'vWF']

    adata = adata_1[:, common_markers].concatenate(adata_2[:, common_markers], adata_3[:, common_markers],
                                                   adata_4[:, common_markers], batch_key="batch")
    adata.obs['batch'] = adata.obs['batch'].map({'0': TMA_LIST[0], '1': TMA_LIST[1],
                                                 '2': TMA_LIST[2], '3': TMA_LIST[3]})

    # run logistic regression model to predict the mismatched labels
    esi_df = adata.obs.copy()
    # get the indices matched and mismatched
    matched_esi_idx = np.where(esi_df['layer1_consensus'] != 'unknown')[0]
    mismatched_esi_idx = np.where(esi_df['layer1_consensus'] == 'unknown')[0]

    # randomly select half of the matched labels as training data
    training_idx = random.sample(list(matched_esi_idx), len(matched_esi_idx) // 2)
    testing_idx = list(mismatched_esi_idx)

    # make training and testing dataframe to input model
    training_df = pd.DataFrame(adata.X[training_idx, :], columns=adata.var_names)
    training_df['cell_type'] = list(esi_df['layer1_consensus'][training_idx].values)

    testing_df = pd.DataFrame(adata.X[testing_idx, :], columns=adata.var_names)
    testing_df['cell_type'] = list(esi_df['layer1_consensus'][testing_idx].values)

    # train the model
    model = LogisticRegression()
    model.fit(training_df.iloc[:, :-1], training_df['cell_type'].values)

    # predict with the testing data
    y_pred = model.predict(testing_df.iloc[:, :-1])

    # update the prediction with the original matched labels together
    new_labels = []
    key = 0
    for i in range(esi_df.shape[0]):
        if esi_df['layer1_consensus'][i] == 'unknown':
            new_labels.append(y_pred[key])
            key += 1
        else:
            new_labels.append(esi_df['layer1_consensus'][i])

    esi_df['layer1_classify'] = new_labels
    adata.obs['layer1_classify'] = new_labels

    adata.write(out_dir / "full_4_tma.h5ad")
    adata.write(out_dir / "full_4_tma_finalize.h5ad")

    print("Accuracy:", accuracy_score(list(adata.obs['layer1_combined'][testing_idx].values), y_pred))

# including all the markers later
