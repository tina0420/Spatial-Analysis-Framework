"""
@author: Tina Hsu
Extracting subset of original whole adata for different purposes
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import read_h5ad


if __name__ == '__main__':
    """
    Purpose: Extract one image for one patient
    """
    # Get the subset adata that one image per image
    image_selected = []

    for pid in np.unique(adata.obs.patient_key):
        image_selected.append(list(set(adata[np.where(adata.obs.patient_key == pid)[0], :].obs.image_key.values))[0])

    subset_idx = []
    for img in image_selected:
        subset_idx = subset_idx + list(np.where(adata.obs.image_key == img)[0])

    # 284858 cells
    sub_adata = adata[subset_idx, :]
    sub_adata.write(out_dir / f"{XP}_one_img_per_pid.h5ad")


    """
    Purpose: Excluding the patients/images with less than 5% epithelial cells
    output adata: effective_adata, 
    """

    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    XP = '22-23_3_tma'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/tils")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in adata for tils analysis
    adata = read_h5ad(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")
    pid_to_exclude = []
    for pid in np.unique(adata.obs.patient_key):
        p_adata = adata[adata.obs.patient_key == pid, :]
        if p_adata.obs.layer1.value_counts()['Epithelial'] / p_adata.shape[0] < 0.05:
            print(pid)
            pid_to_exclude.append(pid)

    # extract a new adata excluding the patients with less than 5 % epithelial cells
    effective_adata = adata[~adata.obs.patient_key.isin(pid_to_exclude)].copy()

    """
    Purpose: Randomly select non-relapse patients in the same size of relapse patients (n=14 here)
    output adata: balanced_adata, f"{XP}_balanced.adata"
    """


    # randomly select the same number of the patients for non-relapse
    import random

    re_pid = list(set(effective_adata[effective_adata.obs.relapse == 'relapse'].obs.patient_key))
    non_pid = list(set(effective_adata[effective_adata.obs.relapse == 'non-relapse'].obs.patient_key))
    # Randomly select 14 non-relapse patient_key (same number to relapse patients)
    non_pid_random = random.sample(non_pid, 14)
    balanced_pid = re_pid + non_pid_random

    balanced_adata = effective_adata[effective_adata.obs.patient_key.isin(balanced_pid)].copy()

    """
    Purpose: Getting the subset of the patients marked as TNBC based on the file from Elena (col: BiomarkerCheck==TNBC)
    output adata: effective_adata, 
    """
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    XP = '22-23_3_tma'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/interactions")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in the patient metadata from Elena

    patient_info = pd.read_csv(in_dir / "imc_pat_info.csv")

    # get the patient list with pCR data
    pcr_available = patient_info[~pd.isna(patient_info.pCR.values)][['CID', 'pCR']]
    pcr_pid = pcr_available.CID.values

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/intermediate")
    adata = read_h5ad(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")

    pcr_adata = adata[np.where(adata.obs['patient_key'].str.startswith(tuple(pcr_pid)))[0], :].copy()



    # Read in adata for tils analysis
    adata = read_h5ad(in_dir / f"{XP}_one_image_per_patient_tils.h5ad")
    pid_to_exclude = []
    for pid in np.unique(adata.obs.patient_key):
        p_adata = adata[adata.obs.patient_key == pid, :]
        if p_adata.obs.layer1.value_counts()['Epithelial'] / p_adata.shape[0] < 0.05:
            print(pid)
            pid_to_exclude.append(pid)

    # extract a new adata excluding the patients with less than 5 % epithelial cells
    effective_adata = adata[~adata.obs.patient_key.isin(pid_to_exclude)].copy()


