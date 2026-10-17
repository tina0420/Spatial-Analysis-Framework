"""
@author: Tina Hsu
Organization of cohort data, including IMC dataset and patient metadata
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from experiments.tnbc_imc.tnbc_imc_setup_xp import COHORT_TMA_XP, CORE_INFO

from experiments.tnbc_imc.tnbc_imc_plotting import plot_marker_expression_distribution_violin


if __name__ == '__main__':
    # Set the disease type and TMAs
    TYPE = 'tnbc_imc'
    TMA_list = ['BOB', '22-010A', '22-010C', '23-009']

    # Get the list of experiments
    xp_list = [COHORT_TMA_XP[tma] for tma in TMA_list]

    # Extract the experiments and merge them together (xp_filter.h5ad under different folders)
    adata_list = [read_h5ad(Path(f"data/{TYPE}") / xp / f"{xp}_filter.h5ad") for xp in xp_list]

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

    # Merge the experiments together as a new adata and tell which TMA is from
    adata = adata_list[0].concatenate(adata_list[1], adata_list[2], adata_list[3], batch_key='tma')
    adata.obs['tma'] = adata.obs['tma'].map({'0': TMA_list[0], '1': TMA_list[1], '2': TMA_list[2], '3': TMA_list[3]})

    # Read in patient metadata
    in_dir = out_dir = Path(f"data/{TYPE}")
    patient_df = pd.read_csv(in_dir / "ctdna_patients_relapse.csv")[['CID', 'Path ID/Accession', 'Block #', 'TMA',
                                                                     'Core ID', 'Core loc_1', 'Core loc_2', 'relapse']]

    # Create primary keys for the corresponding images of each patient and store them in image_1 and image_2 by core_id
    img_1_key = []
    img_2_key = []
    for i in range(patient_df.shape[0]):
        print("_________________________")
        if patient_df['TMA'][i] in TMA_list:
            print(patient_df['TMA'][i])
            # Get the core location (r_c_) of the core ID
            core_location = [loc for loc, subdict in CORE_INFO[COHORT_TMA_XP[patient_df['TMA'][i]]].items()
                             if subdict.get('core_id') == patient_df['Core ID'][i]]
            print(core_location)

            # Get the roi of the core, if no image then set as NaN
            # Check the first core
            img = CORE_INFO[COHORT_TMA_XP[patient_df['TMA'][i]]][core_location[0]].get('roi', np.nan)
            if pd.isna(img):
                img_1_key.append(img)
            else:
                img_1_key.append(patient_df['TMA'][i] + "_" + img)

            # Check the second core
            img = CORE_INFO[COHORT_TMA_XP[patient_df['TMA'][i]]][core_location[1]].get('roi', np.nan)
            if pd.isna(img):
                img_2_key.append(img)
            else:
                img_2_key.append(patient_df['TMA'][i] + "_" + img)

        else:
            img_1_key.append(np.nan)
            img_2_key.append(np.nan)

    patient_df['image_1'] = img_1_key
    patient_df['image_2'] = img_2_key

    # Create primary key for patients
    patient_key = []
    for i in range(patient_df.shape[0]):
        # Combine 'CID', 'Path ID/Accession' and 'Block #'
        patient_key.append(patient_df.iloc[i, :]["CID"] + "_" +
                           patient_df.iloc[i, :]["Path ID/Accession"] + "_" +
                           patient_df.iloc[i, :]["Block #"])

    # Add the patient_key to patient_df and store a new csv
    patient_df['patient_key'] = patient_key

    # Change the column names in patient_df
    patient_df.columns = ['cid', 'accession', 'block_num', 'tma', 'core_id', 'core_loc_1', 'core_loc_2',
                          'relapse', 'image_1', 'image_2', 'patient_key']

    # Keep the patients in the 4 TMAs only
    patient_df = patient_df[patient_df['tma'].isin(TMA_list)].reset_index(drop=True)

    patient_df.to_csv(out_dir / "ctdna_patients_relapse_primary.csv", index=False)
    patient_df.to_csv(Path(f"analysis/{TYPE}/ctdna_cohort") / "ctdna_patients_relapse_primary.csv", index=False)

    # Read in the patient information again
    in_dir = Path(f"analysis/{TYPE}/ctdna_cohort")
    patient_df = pd.read_csv(in_dir / "ctdna_patients_relapse_primary.csv")

    # Create primary image key for each cell, tma_ROI_xxx (adata.obs)
    adata.obs['image_key'] = [adata.obs.tma.values[row] + "_" + adata.obs.image_id.values[row]
                              for row in range(adata.shape[0])]

    """
    Notice: Cannot store image_1 and image_2 in the adata.uns due to NaN
    """
    # Check the data type of each column
    for col in patient_df.columns:
        print(f"{col}: {patient_df[col].apply(type).unique()}")

    # Convert all NaN into empty string due to the storage rule from h5ad
    patient_df['image_1'] = patient_df['image_1'].fillna("")
    patient_df['image_2'] = patient_df['image_2'].fillna("")

    # Store the patient_df in the h5ad
    adata.uns['ctdna_cohort'] = patient_df

    # Store the h5ad with all cell data from 4 xps
    out_dir = Path(f"analysis/{TYPE}/ctdna_cohort")
    adata.write(out_dir / "tnbc_imc_all_4_xp.h5ad")

    # Check if the adata stored well
    adata = read_h5ad(in_dir / "tnbc_imc_all_4_xp.h5ad")

    # Get the image numbers related to the cohort by image_1 and image_2 in adata.uns['ctdna_cohprt']
    cohort_img = ([x for x in adata.uns['ctdna_cohort'].image_1.values if x != ""] +
                  [x for x in adata.uns['ctdna_cohort'].image_2.values if x != ""])

    # Get the subset of adata with the data from cohort images
    adata_cohort = adata[np.where(adata.obs.image_key.isin(cohort_img))[0], :]

    # Store the cohort related h5ad
    adata_cohort.write(out_dir / "tnbc_imc_ctdna_cohort.h5ad")

    # Check if there's any image in the cohort has no IMC data in the adata_cohort
    for i in range(len(cohort_img)):
        if cohort_img[i] not in list(set(adata_cohort.obs.image_key)):
            print((i, cohort_img[i]))

    # Plot distribution of cell expression in all markers across 4 TMAs
    plot_marker_expression_distribution_violin(adata_cohort, 'tma', out_dir,
                                               ['tab:blue', 'tab:pink', 'tab:orange', 'tab:green'],
                                               set_count=4, figsize=(8, 4), name="cohort_only")
    # Get the subset of cohort_h5ad without BOB
    TMA_list = ['22-010A', '22-010C', '23-009']
    adata_cohort_3_tma = adata_cohort[np.where(adata_cohort.obs.tma != 'BOB')[0], :]
    adata_cohort_3_tma.write(out_dir / "tnbc_imc_cohort_3_tma.h5ad")


