"""
@author: Tina Hsu
Spatial analysis for ctDNA TNBC project (start from the original experiments)
"""
import os
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import pandas as pd
from experiments.tnbc_imc.tnbc_imc_setup_xp import COHORT_TMA_XP, CORE_INFO
from experiments.tnbc_imc.tnbc_imc_setup_analysis import TMA_COLOR_DICT
from experiments.tnbc_imc.tnbc_imc_plotting import plot_marker_expression_distribution_violin

if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    IMAGE_KEY = 'image_key'
    PATIENT_KEY = 'patient_key'

    in_dir = Path(f"analysis/{TYPE}/cohort_specific")
    out_dir = Path(f"analysis/{TYPE}/cohort_specific")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in all the adata files (from 4 TMAs)
    adata_list = [read_h5ad(Path(f"data/{TYPE}") / xp / f"{xp}_filter.h5ad") for xp in list(COHORT_TMA_XP.values())]

    # Modify the format of 'image_id' in adata.obs (from ...tiff to ROI_000)
    # Convert image_id to the format of ROI_XXX one xp by one xp because of different format stored in 22-010A
    # (will do a change on previous python code for filtering markers and change image_id)
    adata_list[0].obs['image_id'] = adata_list[0].obs['image_id'].astype(str)
    adata_list[0].obs['image_id'] = adata_list[0].obs['image_id'].str.split('_').str[1] + '_' + \
                                    adata_list[0].obs['image_id'].str.split('_').str[2].str[:3]
    adata_list[0].obs['image_id'] = adata_list[0].obs['image_id'].astype('category')

    # second adata
    adata_list[1].obs['image_id'] = adata_list[1].obs['image_id'].astype(str)
    adata_list[1].obs['image_id'] = adata_list[1].obs['image_id'].str.split('_').str[2] + '_' + \
                                    adata_list[1].obs['image_id'].str.split('_').str[3].str[:3]
    adata_list[1].obs['image_id'] = adata_list[1].obs['image_id'].astype('category')

    # the third adata
    adata_list[2].obs['image_id'] = adata_list[2].obs['image_id'].astype(str)
    adata_list[2].obs['image_id'] = adata_list[2].obs['image_id'].str.split('_').str[1] + '_' + \
                                    adata_list[2].obs['image_id'].str.split('_').str[2].str[:3]
    adata_list[2].obs['image_id'] = adata_list[2].obs['image_id'].astype('category')

    # the fourth adata
    adata_list[3].obs['image_id'] = adata_list[3].obs['image_id'].astype(str)
    adata_list[3].obs['image_id'] = adata_list[3].obs['image_id'].str.split('_').str[1] + '_' + \
                                    adata_list[3].obs['image_id'].str.split('_').str[2].str[:3]
    adata_list[3].obs['image_id'] = adata_list[3].obs['image_id'].astype('category')

    # Merge them together as a new adata
    adata = adata_list[0].concatenate(adata_list[1], adata_list[2], adata_list[3], batch_key='tma')
    adata.obs['tma'] = adata.obs['tma'].map({'0': list(COHORT_TMA_XP.keys())[0],
                                             '1': list(COHORT_TMA_XP.keys())[1],
                                             '2': list(COHORT_TMA_XP.keys())[2],
                                             '3': list(COHORT_TMA_XP.keys())[3]})

    # Add 'image_key' to adata.obs as the unique key of image
    adata.obs[IMAGE_KEY] = [adata.obs.tma.values[row] + "_" + adata.obs.image_id.values[row]
                            for row in range(adata.shape[0])]

    # To add 'patient_key' to adata.obs as the unique key of patient (with the block/tissue resource included)
    # Read in the patient-block mapping file first (with 8 TMAs) and tidy up and store in the adata.uns ()
    patient_metadata_all = pd.read_csv(Path(f"data/{TYPE}") / "ctdna_patients_tma.csv")
    patient_metadata_all = patient_metadata_all[
        patient_metadata_all['TMA'].isin(['BOB', '22-010A', '22-010C', '23-009'])]

    # Remove the information we don't need here
    patient_metadata_all = patient_metadata_all[['CID', 'Path ID/Accession', 'Block #', 'TMA', 'Core ID']]

    # Rename the columns
    patient_metadata_all.columns = ['cid', 'accession', 'block_num', 'tma', 'core_id']

    # Sort the dataframe by CID number
    patient_metadata_all = patient_metadata_all.sort_values(by='cid')
    patient_metadata_all.reset_index(drop=True, inplace=True)

    # Add 'patient_key' to the new column
    patient_metadata_all[PATIENT_KEY] = [
        str(patient_metadata_all.cid[row]) + "_" + str(patient_metadata_all.accession[row]) + "_" + str(
            patient_metadata_all.block_num[row]) for row in range(patient_metadata_all.shape[0])]

    # Add core_loc, img_num and image_key (TMA_ROI#)
    core_loc = []
    for i in range(patient_metadata_all.shape[0]):
        xp = COHORT_TMA_XP[patient_metadata_all['tma'][i]]
        core_id = patient_metadata_all['core_id'][i]
        temp = []
        for loc, info in CORE_INFO[xp].items():
            if info.get('core_id') == core_id:
                temp.append(loc)
        core_loc.append(temp)

    # img number
    img_1_key = []
    img_2_key = []
    for i in range(patient_metadata_all.shape[0]):
        # check the first core
        img = CORE_INFO[COHORT_TMA_XP[patient_metadata_all['tma'][i]]][core_loc[i][0]].get('roi', np.nan)
        if pd.isna(img):
            img_1_key.append(img)
        else:
            img_1_key.append(patient_metadata_all['tma'][i] + "_" + img)

        # Check the second core
        img = CORE_INFO[COHORT_TMA_XP[patient_metadata_all['tma'][i]]][core_loc[i][1]].get('roi', np.nan)
        if pd.isna(img):
            img_2_key.append(img)
        else:
            img_2_key.append(patient_metadata_all['tma'][i] + "_" + img)

    # Insert core_loc and image to df
    patient_metadata_all['loc_1'] = [i[0] for i in core_loc]
    patient_metadata_all['loc_2'] = [i[1] for i in core_loc]
    patient_metadata_all['image_1'] = img_1_key
    patient_metadata_all['image_2'] = img_2_key

    # Remove the rows without images (check image_1 and image_2)
    patient_metadata_all = patient_metadata_all.drop([i for i in range(patient_metadata_all.shape[0]) if
                                                      pd.isna(patient_metadata_all['image_1'][i]) and pd.isna(
                                                          patient_metadata_all['image_2'][i])])

    patient_metadata_all.reset_index(inplace=True, drop=True)
    patient_metadata_all.to_csv(out_dir / "cohort_specific_metadata_imc.csv", index=False)

    # Read in patient clinical data (imc_pat_info.csv)
    in_dir = out_dir
    metadata = pd.read_csv(in_dir / "imc_pat_info.csv")

    # Get the patients marked as TNBC
    tnbc_patient = metadata.iloc[np.where(metadata['BiomarkerCheck'] == 'TNBC')[0], :].reset_index(drop=True)
    pids = tnbc_patient['CID'].values

    # Extract the patient metadata with only TNBC patients
    patient_metadata_all = patient_metadata_all[patient_metadata_all.cid.isin(pids)]
    patient_metadata_all.reset_index(inplace=True, drop=True)
    patient_metadata_all.to_csv(out_dir / "cohort_specific_metadata_imc_tnbc.csv", index=False)

    # check for pid:
    # for i in pids:
    #     if i in list_adata:
    #         continue
    #     else:
    #         print(i)

    # Add other clinical info to patient_metadata_all
    patient_metadata_final = patient_metadata_all.merge(tnbc_patient, left_on='cid', right_on='CID', how='left').drop(
        columns='CID')
    patient_metadata_final = patient_metadata_final.drop("recurrence", axis=1)

    patient_metadata_final.to_csv(out_dir / "cohort_specific_metadata_imc_tnbc.csv", index=False)

    """
    Patient information csv tidied up finished!
    """
    # Get the subset of adata with only TNBC patients
    adata_tnbc = adata[np.where((adata.obs.image_key.isin(patient_metadata_all.image_1)) | (
        adata.obs.image_key.isin(patient_metadata_all.image_2)))[0], :].copy()

    # Add 'patient_key' to adata_tnbc.obs and then other clinical information (e.g. recurrence, stage, pCR,...)
    # Creat a dictionary to map the image_key to patient_key effectively
    img_pid = {}
    for i in range(patient_metadata_final.shape[0]):
        if pd.isna(patient_metadata_final.image_1[i]):
            print(f"no image_1: {i}")
        else:
            img_pid[patient_metadata_final.image_1[i]] = patient_metadata_final.patient_key[i]

        if pd.isna(patient_metadata_final.image_2[i]):
            print(f"no image_2: {i}")
        else:
            img_pid[patient_metadata_final.image_2[i]] = patient_metadata_final.patient_key[i]

    # Check if any images missed
    # for img in set(adata_tnbc.obs.image_key.values):
    #     if img not in img_pid:
    #         print(img)

    adata_tnbc.obs['patient_key'] = [img_pid[img] for img in adata_tnbc.obs.image_key.values]

    adata_tnbc.write(out_dir / "cohort_tnbc_initial.h5ad")

    """
    Finish adata and patient info (csv) for cohort specific TNBC dataset (marked as TNBC)
    """
    adata = read_h5ad(out_dir / "cohort_tnbc_initial.h5ad")

    # run umap
    import scanpy as sc
    from src.visualize import plot_umap, plot_umap_panel
    from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT

    temp_adata = adata.copy()

    CLUSTER_KEY = 'tma'
    XP = XP = 'cohort_tnbc'

    sc.tl.pca(temp_adata, svd_solver='arpack')
    print("NEIGHBORS")
    sc.pp.neighbors(temp_adata, use_rep='X_pca')
    print("umap")
    sc.tl.umap(temp_adata)

    tma_color_dict = dict(zip(['BOB', '22-010A', '22-010C', '23-009'],
                              ['tab:blue', 'tab:pink', 'tab:orange', 'tab:green']))
    plot_umap(temp_adata, CLUSTER_KEY, out_dir, color_dict=tma_color_dict,
              figsize=(4.5, 4.5), dotsize=0.5, name=f"{XP}_tma")
    plot_umap_panel(temp_adata, CLUSTER_KEY, out_dir, color_dict=tma_color_dict,
                    dotsize=0.1, name=f"{XP}_tma")

    # plot violin
    CLUSTER_KEY = 'tma'
    plot_marker_expression_distribution_violin(temp_adata, CLUSTER_KEY, out_dir, TMA_COLOR_DICT,
                                               set_count=4, figsize=(8, 4), name=f"{XP}_4_TMA")

    # Extract the 3 TMA
    adata_3_tma = adata[np.where(adata.obs.tma.isin(['22-010A', '22-010C', '23-009']))[0], :].copy()
    adata_3_tma.write(out_dir / "cohort_tnbc_initial_3_tma.h5ad")

    """
    For downstream analysis preparation
    """
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific'
    XP = 'cohort_tnbc'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/tils")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad with tme cells only
    adata = read_h5ad(in_dir / f"{XP}_finalize.h5ad")
    adata_epi = read_h5ad(in_dir / f"{XP}_epi.h5ad")
    adata_tme = read_h5ad(in_dir / f"{XP}_tme.h5ad")

    # Add layer2 immune cell types to the original adata as a new tils h5ad
    adata_tils = adata.copy()
    CLUSTER_KEY = 'layer1'
    new_labels = np.zeros(adata_tils.shape[0], dtype='object')
    new_labels[np.where(adata_tils.obs[CLUSTER_KEY] == 'Epithelial')[0]] = adata_epi.obs[CLUSTER_KEY].to_numpy()

    new_labels[np.where(adata_tils.obs[CLUSTER_KEY] == 'Immune')[0]] = \
    adata_tme.obs.iloc[np.where(adata_tme.obs[CLUSTER_KEY] == 'Immune')[0], :]['layer2'].to_numpy()

    new_labels[np.where(adata_tils.obs[CLUSTER_KEY] == 'Stromal')[0]] = \
        adata_tme.obs.iloc[np.where(adata_tme.obs[CLUSTER_KEY] == 'Stromal')[0], :][CLUSTER_KEY].to_numpy()

    adata_tils.obs['tils'] = new_labels
    adata_tils.write(in_dir / f"{XP}_tils.h5ad")

    # need to add patient clinic info to the h5ad.obs
    patient_info = pd.read_csv(in_dir / "cohort_specific_metadata_imc_tnbc.csv")

    # Add all patient info (metadata) into adata.obs
    columns = ['core_id', 'Death', 'DiagnosisDate',
               'ChemoType', 'pCR', 'RecurrenceType', 'hrd_type', 'hr_status',
               'BiomarkerCheck', 'Biomarkers', 'Stage', 'Tumour size (cm)', 'Grade',
               'Node status', 'LVI', 'Margin status']

    for col in columns:
        print(col)
        # Construct a map which can reduce time of process (dict)
        CLUSTER_KEY = col
        info_map = {}
        for i in range(patient_info.shape[0]):
            # print(patient_info.patient_key.values[i])
            info_map[patient_info.patient_key.values[i]] = patient_info[CLUSTER_KEY].values[i]

        # get the values in the corresponding column
        information = [info_map[key] for key in adata.obs.patient_key.values]
        adata.obs[CLUSTER_KEY] = information

    adata.write(in_dir / f"{XP}_finalize.h5ad")






