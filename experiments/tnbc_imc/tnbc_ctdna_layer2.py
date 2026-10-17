"""
@author: Tina Hsu
Merging layer 2 cell types for analysis in TNBC IMC by BM-panel
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np


if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    TMA = 'full_4_tma'

    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get the layer 2 adata and patient information
    adata_epi = read_h5ad(in_dir / f"{TMA}_layer2_epi.h5ad")
    adata_tme = read_h5ad(in_dir / f"{TMA}_layer2_imm_str.h5ad")

    original_all_adata = read_h5ad(Path(f"analysis/{TYPE}/ctdna_4_tma") / f"{TMA}_finalize.h5ad")

    """
    Because the layer2_epi.h5ad and layer2_imm_str.h5ad both have their markers only, it would be better extracting 
    the 'leiden' and ('layer2_epi' + 'layer2_is') from each of them and adding into the original adata with all the 
    cells (full_4_tma_finalize.h5ad) 
    1. modify the numbers stored in the adata.obs.leiden and merge the numbers together to make it continues (plus the 
       number of clusters from the first adata)
    2. create the lists for leiden and layer2_classify (destination: adata.obs)
    """
    # modify the numbers in adata.obs.leiden to make it continues, transfer to integer first then calculation and merge
    # modify adata.tme only
    length = len(set(adata_epi.obs.leiden))

    adata_tme.obs.leiden = adata_tme.obs.leiden.astype(int)
    adata_tme.obs.leiden = adata_tme.obs.leiden + length
    adata_tme.obs.leiden = adata_tme.obs.leiden.astype(str)

    adata_tme.write(in_dir / f"{TMA}_layer2_imm_str.h5ad")

    # index of epithelial and tme in the whole adata
    epi_idx = np.where(original_all_adata.obs.layer1_classify == 'epithelial')[0]
    tme_idx = np.where(original_all_adata.obs.layer1_classify != 'epithelial')[0]

    # merge leiden numbers and store
    new_col = np.zeros(original_all_adata.shape[0], dtype=object)
    new_col[epi_idx] = adata_epi.obs.leiden.to_numpy()
    new_col[tme_idx] = adata_tme.obs.leiden.to_numpy()

    original_all_adata.obs['layer2_leiden'] = new_col

    # merge layer2 cell type results
    new_col = np.zeros(original_all_adata.shape[0], dtype=object)
    new_col[epi_idx] = adata_epi.obs.layer2_epi.to_numpy()
    new_col[tme_idx] = adata_tme.obs.layer2_is.to_numpy()

    original_all_adata.obs['layer2_classify'] = new_col

    original_all_adata.write(in_dir / f"{TMA}_layer2_finalize.h5ad")

    """
    Task 5: Differential abundance analysis
    creat a new h5ad with only patients in the cohort and then calculate the correlations between cell types 
    proportions and relapse
    1. extract the patients and ROIs in the cohort and store it into a new h5ad
    2. 
    """
    adata = read_h5ad(in_dir / f"{TMA}_layer2_finalize.h5ad")
    patient_df = pd.read_csv(in_dir / "ctdna_patients_relapse.csv")

    # get the subset of patient_df with patients have images only
    patient_df_with_image = patient_df.iloc[np.where(~pd.isna(patient_df['Core img_1']) | ~pd.isna(patient_df['Core img_2']))[0], :]
    patient_df_with_image.to_csv(in_dir / "ctdna_patients_relapse_image_only.csv", index=False)

    temp_adata = adata.copy()
    # for better matching, convert the image_id to the format i want (ROI_XXX)
    temp_adata.obs['image_id'] = temp_adata.obs['image_id'].astype(str)
    # temp_adata.obs['image_id'].str.extract(r'(ROI_\d{3})')
    temp_adata.obs['image_id'] = temp_adata.obs['image_id'].str.split('_').str[1] + '_' + temp_adata.obs['image_id'].str.split('_').str[2].str[:3]
    temp_adata.obs['image_id'] = temp_adata.obs['image_id'].astype('category')

    t_obs = temp_adata.obs

    # subset of h5ad with only patients having images
    # 'batch' in adata.obs & 'TMA' in patient_df_with_image are the same ('22-010A',...)
    t_df = patient_df_with_image.reset_index(drop=True)
    rows = []
    for i in range(t_df.shape[0]):
        print(i)
        tma = t_df['TMA'][i]
        img_1 = t_df['Core img_1'][i]
        img_2 = t_df['Core img_2'][i]
        if pd.isna(img_1) is False:
        if pd.isna(img_1) is False:
            rows += list(np.where((temp_adata.obs.batch == tma) & (temp_adata.obs.image_id == img_1))[0])

        if pd.isna(img_2) is False:
            rows += list(np.where((temp_adata.obs.batch == tma) & (temp_adata.obs.image_id == img_2))[0])

    # get subset h5ad (adata) for image only
    adata_img = adata[rows, :]
    adata_img.write(in_dir / f"{TMA}_cohort_patient_only.h5ad")
    t_df.to_csv(in_dir / f"{TMA}_cohort_patient_only_reset_index.csv")

    # using adata_img
    patient_df = t_df.copy()
    relpase_p = patient_df.iloc[np.where(patient_df['relapse'] == 'relapse')[0], :]
    non_relapse_p =patient_df.iloc[np.where(patient_df['relapse'] == 'non-relapse')[0], :]

    # cell type proportions calculation
    cell_types = ['luminal', 'myoepithelial', # epithelial
                  'T cell', 'B cell', 'Macrophage', # immune
                  'Blood endothelial', 'Smooth muscle cell', 'Vimentin'] # stromal
    cell_types_pro_dict = dict(zip(['epithelial', 'immune', 'stromal'], [['luminal', 'myoepithelial'],
                                                                         ['T cell', 'B cell', 'Macrophage'],
                                                                         ['Blood endothelial', 'Smooth muscle cell', 'Vimentin']]))
    t_obs = t_obs.iloc[rows, :]
    # patient by patient for the cell type proportions
    ct_prop = []
    CLUSTER_KEY = 'layer2_classify'
    for i in range(patient_df.shape[0]):
        print("__________________________")
        print(i)
        tma = patient_df['TMA'][i]
        cid = patient_df['CID'][i]
        # choose the first image, if none then the second one
        img = patient_df['Core img_1'][i]
        if pd.isna(img):
            img = patient_df['Core img_2'][i]

        temp_df = t_obs.iloc[np.where((t_obs['batch'] == tma) & (t_obs['image_id'] == img))[0], :]
        # start calculation
        sub_prop = []
        for esi, types in cell_types_pro_dict.items():
            # ct_prop = temp_df[temp_df[CLUSTER_KEY].isin(types)][CLUSTER_KEY].value_counts()
            print(esi)
            subset = temp_df[temp_df[CLUSTER_KEY].isin(types)]
            sub_prop += list(subset[CLUSTER_KEY].value_counts(normalize=True).reindex(types, fill_value=0))

        ct_prop.append(sub_prop)

    # create a df for cell type proportions and then add with the relapse information
    ct_df = pd.DataFrame(ct_prop, columns=cell_types)




    # temp code for plotting distribution
    patient_adata = read_h5ad(in_dir / "full_4_tma_cohort_patient_only.h5ad")
    temp_adata = patient_adata




