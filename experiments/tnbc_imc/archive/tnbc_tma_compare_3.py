"""
@author: Tina Hsu
Compare the experiments from the same TNBC TMA, comparing ESI results in the same cores from two XPs and then
comparing with the full adata results (TMA_merge)
"""
import os
from pathlib import Path
from tnbc_xp_setup import *
from anndata import read_h5ad
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import *
from src.visualize import COLOR_CODE, plot_spatial_cluster_distribution, COLORS
from src.annotate import update_labels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_logistic_to_predict():
    pass


if __name__ == '__main__':
    # parameter setting
    TYPE = 'tnbc_imc'
    TMA = '22-010C'

    EXPERIMENT_1 = TMA_XP[TMA][0]
    EXPERIMENT_2 = TMA_XP[TMA][1]

    in_dir = Path(f"analysis/{TYPE}/{TMA}")
    out_dir = Path(f"analysis/{TYPE}/{TMA}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # compare ESI distribution in the common core on the slides
    adata_1 = read_h5ad(in_dir / f"{EXPERIMENT_1}_merge.h5ad")
    adata_2 = read_h5ad(in_dir / f"{EXPERIMENT_2}_merge.h5ad")
    adata = read_h5ad(in_dir / f"{TMA}_merge.h5ad")

    # get the common cores from
    matched_cores = [k for k in CORE_INFO[EXPERIMENT_1] if 'roi' in CORE_INFO[EXPERIMENT_1][k]
                     and 'roi' in CORE_INFO[EXPERIMENT_2][k]]

    # get the roi# on each slide
    CLUSTER_KEY = 'image_id'
    corr_list = []
    proportions = []
    for i in range(len(matched_cores)):
        # one core comparison
        print("-----------------------------------------------------")

        cores = matched_cores[i]
        print(cores)

        mask = adata_1.obs[CLUSTER_KEY].str.contains(CORE_INFO[EXPERIMENT_1][cores]['roi'])
        matching_rows = np.where(mask)[0]
        print(len(matching_rows))
        prop_1 = adata_1[matching_rows, :].obs['esi'].value_counts(normalize=True).sort_index()

        mask = adata_2.obs[CLUSTER_KEY].str.contains(CORE_INFO[EXPERIMENT_2][cores]['roi'])
        matching_rows = np.where(mask)[0]
        print(len(matching_rows))
        prop_2 = adata_2[matching_rows, :].obs['esi'].value_counts(normalize=True).sort_index()

        proportions_df = pd.DataFrame({EXPERIMENT_1: prop_1, EXPERIMENT_2: prop_2})

        proportions.append([prop_1[j] for j in range(prop_1.shape[0])] +
                           [prop_2[j] for j in range(prop_2.shape[0])])
        corr_list.append(spearmanr(proportions_df[EXPERIMENT_1], proportions_df[EXPERIMENT_2])[0])

    proportions_record = pd.DataFrame(proportions, columns=['E_1', 'I_1', 'S_1', 'U_1', 'E_2', 'I_2', 'S_2', 'U_2'])
    proportions_record['spearman'] = corr_list
    proportions_record['core_id'] = matched_cores
    proportions_record.to_csv(out_dir / "esi_proportions_compare.csv", index=False)

    adata = read_h5ad(in_dir / f"{TMA}_merge.h5ad")
    # get the same cells, training set
    CLUSTER_KEY = 'esi'
    training_idx = np.array([], dtype=object)
    testing_idx = np.array([], dtype=object)

    # merge the esi results from xp1 and xp2
    individual_esi = list(adata_1.obs[CLUSTER_KEY].values) + list(adata_2.obs[CLUSTER_KEY].values)
    full_esi = list(adata.obs[CLUSTER_KEY].values)

    # store the matched index in training_idx and unmatched index in testing_idx
    training_idx = [i for i, (a, b) in enumerate(zip(full_esi, individual_esi)) if a == b]
    testing_idx = [i for i, (a, b) in enumerate(zip(full_esi, individual_esi)) if a != b]

    # make the dataframe
    training_df = pd.DataFrame(adata.X[training_idx, :], columns=adata.var_names)
    training_df['cell_type'] = list(adata.obs[CLUSTER_KEY].iloc[training_idx].values)

    testing_df = pd.DataFrame(adata.X[testing_idx, :], columns=adata.var_names)
    testing_df['cell_type'] = list(adata.obs[CLUSTER_KEY].iloc[testing_idx].values)

    # train the logistic regression model
    model = LogisticRegression()
    model.fit(training_df.iloc[:, :-1], training_df['cell_type'].values)

    # predict with the test data
    y_pred = model.predict(testing_df.iloc[:, :-1])

    # check the performance
    print("Accuracy:", accuracy_score(testing_df['cell_type'].values, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(testing_df['cell_type'].values, y_pred))
    print("Classification Report:\n", classification_report(testing_df['cell_type'].values, y_pred))

    # compare the prediction results to the original results from individual xp
    # there are 3 lists now about the cell types of the cells (xp1 or xp2 result, TMA result and prediction result)
    # specifically for the unmatched rows (testing_idx)
    result_1 = [individual_esi[i] for i in testing_idx]
    result_2 = [full_esi[i] for i in testing_idx]
    result_3 = list(y_pred)

    esi_compare_df = pd.DataFrame({'individual': result_1, 'tma': result_2, 'lr_predict': result_3})
    indi_win = []
    tma_win = []
    both_wrong = []
    for i in range(esi_compare_df.shape[0]):
        prediction = esi_compare_df['lr_predict'][i]
        if prediction == esi_compare_df['individual'][i]:
            indi_win.append(i)
        elif prediction == esi_compare_df['tma'][i]:
            tma_win.append(i)
        else:
            both_wrong.append(i)

    # visualize the result (pie chart)
    category = ['individual', 'tma', 'both_wrong']
    counts = [len(indi_win), len(tma_win), len(both_wrong)]

    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=category, autopct='%1.1f%', startangle=90)
    plt.legend(category, title="Percentages of correct classification")

    plt.title("Cell type proportions", weight='bold', fontsize=8)
    plt.axis('equal')
    plt.tight_layout()

    # test for replacing all the unknown into stromal
    esi_un_stromal = esi_compare_df.copy()
    esi_un_stromal['lr_predict'] = esi_un_stromal['lr_predict'].map({'epithelial': 'epithelial',
                                                                     'immune': 'immune',
                                                                     'stromal': 'stromal',
                                                                     'unknown': 'stromal'})

    indi_win_2 = []
    tma_win_2 = []
    both_wrong_2 = []
    for i in range(esi_un_stromal.shape[0]):
        prediction = esi_un_stromal['lr_predict'][i]
        if prediction == esi_un_stromal['individual'][i]:
            indi_win_2.append(i)
        elif prediction == esi_un_stromal['tma'][i]:
            tma_win_2.append(i)
        else:
            both_wrong_2.append(i)

    CLUSTER_KEY = 'image_id'
    EPI_COLOR_DICT = dict(
        zip(['epithelial', 'immune', 'stromal', 'unknown'], ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e']))
    for i in range(len(matched_cores)):
        # one core comparison
        print("-----------------------------------------------------")

        cores = matched_cores[i]
        print(cores)

        # mask = adata_1.obs[CLUSTER_KEY].str.contains(CORE_INFO[EXPERIMENT_1][cores]['roi'])
        # matching_rows = np.where(mask)[0]
        # temp_adata = adata_1[matching_rows, :].copy()
        # temp_adata.obs['esi'] = temp_adata.obs['esi'].map({'epithelial': 'epithelial',
        #                                                    'immune': 'immune',
        #                                                    'stromal': 'stromal',
        #                                                    'unknown': 'stromal'}).astype('category')
        #
        # plot_spatial_cluster_distribution(temp_adata, 'esi', out_dir, figsize=(5, 5), dotsize=1, color=EPI_COLOR_DICT,
        #                                   name=f"{EXPERIMENT_1}_{cores}")
        #
        # mask = adata_2.obs[CLUSTER_KEY].str.contains(CORE_INFO[EXPERIMENT_2][cores]['roi'])
        # matching_rows = np.where(mask)[0]
        # temp_adata = adata_2[matching_rows, :].copy()
        # temp_adata.obs['esi'] = temp_adata.obs['esi'].map({'epithelial': 'epithelial',
        #                                                    'immune': 'immune',
        #                                                    'stromal': 'stromal',
        #                                                    'unknown': 'stromal'}).astype('category')
        #
        # plot_spatial_cluster_distribution(temp_adata, 'esi', out_dir, figsize=(5, 5), dotsize=1, color=EPI_COLOR_DICT,
        #                                   name=f"{EXPERIMENT_2}_{cores}")

        xp1_roi = CORE_INFO[EXPERIMENT_1][cores]['roi']
        xp2_roi = CORE_INFO[EXPERIMENT_2][cores]['roi']
        # for the first xp
        temp_adata = adata[np.where(adata.obs['batch'] == EXPERIMENT_1)[0], :].copy()
        mask = temp_adata.obs[CLUSTER_KEY].str.contains(xp1_roi)
        matching_rows = np.where(mask)[0]
        temp_adata = temp_adata[matching_rows, :]
        temp_adata.obs['esi'] = temp_adata.obs['esi'].map({'epithelial': 'epithelial',
                                                           'immune': 'immune',
                                                           'stromal': 'stromal',
                                                           'unknown': 'stromal'}).astype('category')

        plot_spatial_cluster_distribution(temp_adata, 'esi', out_dir, figsize=(5, 5), dotsize=1, color=EPI_COLOR_DICT,
                                          name=f"{TMA}_xp1_{cores}")

        temp_adata = adata[np.where(adata.obs['batch'] == EXPERIMENT_2)[0], :].copy()
        mask = temp_adata.obs[CLUSTER_KEY].str.contains(xp2_roi)
        matching_rows = np.where(mask)[0]
        temp_adata = temp_adata[matching_rows, :]
        temp_adata.obs['esi'] = temp_adata.obs['esi'].map({'epithelial': 'epithelial',
                                                           'immune': 'immune',
                                                           'stromal': 'stromal',
                                                           'unknown': 'stromal'}).astype('category')

        plot_spatial_cluster_distribution(temp_adata, 'esi', out_dir, figsize=(5, 5), dotsize=1, color=EPI_COLOR_DICT,
                                          name=f"{TMA}_xp2_{cores}")

    # get the matched esi percentages from individual xp and full tma
    individual_esi = list(adata_1.obs['esi'].values) + list(adata_2.obs['esi'].values)
    tma_esi = list(adata.obs['esi'].values)

    matched_esi_idx = [i for i, (a, b) in enumerate(zip(individual_esi, tma_esi)) if a == b]
    mismatched_esi_idx = [i for i, (a, b) in enumerate(zip(individual_esi, tma_esi)) if a != b]

    """
    Confusion matrix for mismatched rows
    """
    # run the confusion matrix with the mismatched cells from individual and TMA
    cm = confusion_matrix([individual_esi[i] for i in mismatched_esi_idx], [tma_esi[i] for i in mismatched_esi_idx])
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_percent = cm / cm_sum * 100
    labels = ['epithelial', 'immune', 'stromal']

    # prepare annotation (counts + per)
    annotation = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if count == 0:
                annotation[i, j] = str(count)
            else:
                percent = cm_percent[i, j]
                annotation[i, j] = f"{count}\n({percent:.1f}%)"

    # plot
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=annotation, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('TMA_combined')
    plt.ylabel('TMA_individual')
    per = round(len(mismatched_esi_idx) / len(individual_esi), 2)
    plt.title(f'Mismatched ESI annotations ({per} of all)')
    plt.tight_layout()
    plt.savefig(out_dir / f"{TMA}_esi_mismatched_confusion_matrix.png", dpi=500)
    plt.clf()

    esi_compare_df = pd.DataFrame({'layer1': individual_esi, 'layer1_combined': tma_esi})
    esi_compare_df.to_csv(in_dir / f"{TMA}_esi_comparison.csv", index=False)

    # calculate the concensus with the replace the unknown as stromal but not change the original df
    # change the unknown cell types to stromal cell types
    update_esi = {'epithelial': 'epithelial',
                  'immune': 'immune',
                  'stromal': 'stromal',
                  'unknown': 'stromal'}
    cal_individual_esi = [update_esi[i] for i in individual_esi]
    cal_tma_esi = [update_esi[i] for i in tma_esi]

    consensus_esi = []
    for i in range(len(cal_individual_esi)):
        if cal_individual_esi[i] == cal_tma_esi[i]:
            consensus_esi.append(cal_individual_esi[i])
        else:
            consensus_esi.append('unknown')

    esi_compare_df['layer1_consensus'] = consensus_esi
    # get the index of matched and mismatched results
    matched_esi_idx = np.where(esi_compare_df['layer1_consensus'] != 'unknown')[0]
    mismatched_esi_idx = np.where(esi_compare_df['layer1_consensus'] == 'unknown')[0]

    import random

    training_idx = random.sample(list(matched_esi_idx), len(matched_esi_idx) // 2)
    testing_idx = list(mismatched_esi_idx)

    # make the dataframe
    training_df = pd.DataFrame(adata.X[training_idx, :], columns=adata.var_names)
    training_df['cell_type'] = list(esi_compare_df['layer1_consensus'][training_idx].values)

    testing_df = pd.DataFrame(adata.X[testing_idx, :], columns=adata.var_names)
    testing_df['cell_type'] = list(esi_compare_df['layer1_consensus'][testing_idx].values)

    # train the logistic regression model
    model = LogisticRegression()
    model.fit(training_df.iloc[:, :-1], training_df['cell_type'].values)

    # predict with the test data
    y_pred = model.predict(testing_df.iloc[:, :-1])
    y_pred

    prediction_compare_df = esi_compare_df.iloc[testing_idx, :-1]
    prediction_compare_df['lg_prediction'] = list(y_pred)


    mis_individual = [cal_individual_esi[i] for i in testing_idx]
    mis_tma = [cal_tma_esi[i] for i in testing_idx]

    prediction_compare_df = pd.DataFrame({'layer1': mis_individual, 'layer1_combined': mis_tma})
    prediction_compare_df['lr_pred'] = list(y_pred)

    indi_win = []
    merged_win = []
    both_wrong = []
    for i in range(prediction_compare_df.shape[0]):
        pred = prediction_compare_df['lr_pred'][i]
        if pred == prediction_compare_df['layer1'][i]:
            indi_win.append(i)
        elif pred == prediction_compare_df['layer1_combined'][i]:
            merged_win.append(i)
        else:
            both_wrong.append(i)

    # need to tidy up the dataframe
    new_labels = np.zeros(esi_compare_df.shape[0], dtype=object)
    matched_labels = [esi_compare_df['layer1'][i] for i in matched_esi_idx]
    mismatched_labels = [esi_compare_df['layer1'][i] for i in mismatched_esi_idx]







