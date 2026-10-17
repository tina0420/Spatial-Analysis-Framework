"""
@author: Tina Hsu
Experiments focuses on intermediate covariances which are not for all patients.
For pCR and
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import read_h5ad


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    XP = '22-23_3_tma'

    # in_dir = out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/intermediate")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in patient metadata
    patient_info = pd.read_csv(in_dir / "imc_pat_info.csv")

    # For pCR
    patient_pcr = [patient_info.CID.values[x] for x in range(patient_info.shape[0])
                   if pd.isna(patient_info.pCR.values[x]) is False]

    # Read in h5ad with tme cells only
    adata = read_h5ad(Path(f"analysis/{TYPE}/{PURPOSE}/{XP}") / f"{XP}_one_img_per_pid.h5ad")

    # Unique patients from the adata
    uniq_patient_cid = list(
        pd.Series(list([x.split("_")[0] for x in set(adata.obs.patient_key.values)])).value_counts().index)

    # Compare the patients CID from metadata and the patients here with images


    pcr_adata = adata[np.where(adata.obs.patient_key.str.startswith(tuple(patient_pcr)))[0], :].copy()

    t_list = list(set(pcr_adata.obs.patient_key))


    # Check the full dataset (4 tmas of cohort patients)
    adata = read_h5ad(Path(f"analysis/{TYPE}/{PURPOSE}") / "tnbc_imc_ctdna_cohort.h5ad")

    # On this
    adata = read_h5ad(Path(f"analysis/{TYPE}/{PURPOSE}/{XP}") / f"{XP}_finalize.h5ad")

    # adata = read_h5ad(Path(f"analysis/{TYPE}/{PURPOSE}") / "tnbc_imc_cohort_3_tma.h5ad")

    # Add the patient_key to the adata.obs
    img_pat_map = {}
    for i in range(adata.uns['ctdna_cohort'].shape[0]):
        if pd.isna(adata.uns['ctdna_cohort']['image_1'].values[i]):
            continue
        else:
            img_pat_map[adata.uns['ctdna_cohort']['image_1'].values[i]] = \
            adata.uns['ctdna_cohort']['patient_key'].values[i]

        if pd.isna(adata.uns['ctdna_cohort']['image_2'].values[i]):
            continue
        else:
            img_pat_map[adata.uns['ctdna_cohort']['image_2'].values[i]] = \
            adata.uns['ctdna_cohort']['patient_key'].values[i]

    adata.obs["patient_key"] = [img_pat_map[img] for img in adata.obs.image_key.values]


