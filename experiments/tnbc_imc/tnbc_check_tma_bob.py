import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from experiments.tnbc_imc.tnbc_imc_plotting import plot_marker_expression_distribution_violin
from experiments.tnbc_imc.tnbc_imc_setup_xp import COHORT_TMA_XP

if __name__ == '__main__':
    # To check if the distributions of cell expression in GATA3 are different across TNBC patients and other patients
    # in BOB TMA
    TYPE = 'tnbc_imc'
    TMA = 'BOB_XP12446'

    # Data directory
    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/{TMA}")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in patient information and h5ad
    adata = read_h5ad(in_dir / f"{TMA}_merge.h5ad")
    patient_df = pd.read_csv(in_dir / "ctdna_patients_relapse.csv")[['TMA', 'Core loc_1', 'Core loc_2', 'Core img_1',
                                                                     'Core img_2']]

    # Get the image name from the patient in the cohort
    bob_df = patient_df.iloc[np.where(patient_df.TMA == 'BOB')[0], :][['Core img_1', 'Core img_2']]
    imgs = [i for i in bob_df['Core img_1'].values if pd.isna(i) is False]
    imgs += [i for i in bob_df['Core img_2'].values if pd.isna(i) is False]

    # Add a new column with TNBC and non-TNBC to the adata and select the column to plot violin plot
    adata.obs['patient_type'] = ['TNBC' if i in imgs else 'non-TNBC' for i in adata.obs.image_id.values]

    # Plot the violin plot
    CLUSTER_KEY = 'patient_type'
    color_list = ['tab:blue', 'tab:pink']
    plot_marker_expression_distribution_violin(adata, CLUSTER_KEY, out_dir, color_list, set_count=6,
                                               name="BOB_TNBC_non-TNBC")

    # As the result is not ideal, we then need to plot the UMAP with cohort patients only across 4 TMAs to see of BOB
    # really different

    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/")
    # Read in h5ad with all patients, including non-cohort patients
    adata = read_h5ad(in_dir / "full_4_tma.h5ad")
    patient_df = pd.read_csv(in_dir / "ctdna_patients_relapse.csv")

    # Create primary key for the patient image (TMA_ROI_xxx)
    key_id = []
    for i in range(patient_df.shape[0]):
        # Use first image, if NaN then use the second image
        img = patient_df.iloc[i, :]["Core img_1"]
        if pd.isna(img):
            img = patient_df.iloc[i, :]["Core img_2"]

        # If there's no images from that patient, store nan
        if pd.isna(img):
            key_id.append(img)
        else:
            key_id.append(patient_df.iloc[i, :]["TMA"] + "_" + img)

    patient_df['image_key'] = key_id

    # Modify the image_id to only ROI_#
    adata.obs['image_id'] = adata.obs['image_id'].astype(str)
    adata.obs['image_id'] = adata.obs['image_id'].str.split('_').str[1] + '_' + \
                            adata.obs['image_id'].str.split('_').str[2].str[:3]
    adata.obs['image_id'] = adata.obs['image_id'].astype('category')

    # Create primary key for each cell (adata.obs)
    adata.obs['image_key'] = [adata.obs.batch.values[row] + "_" + adata.obs.image_id.values[row] for row in range(adata.shape[0])]

    # Get the subset of h5ad by key_id
    adata_cohort = adata[np.where(adata.obs.image_key.isin(list(set(patient_df.image_key.values))))[0], :]
    for i in set(patient_df.image_key):
        if i not in list(set(adata_cohort.obs.image_key)):
            print(i)




