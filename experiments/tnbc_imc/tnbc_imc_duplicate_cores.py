"""
@author: Tina Hsu
Investigating the similarity of the original core and duplicate core (same location)
"""
import os
from pathlib import Path
from experiments.tnbc_imc.tnbc_imc_setup_xp import TMA_XP
from anndata import read_h5ad


if __name__ == '__main__':
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'ctdna_cohort'
    XP = '22-23_3_tma'
    IMAGE_KEY = 'image_key'
    PATIENT_KEY = 'patient_key'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/22-23_3_tma_finalize.h5ad")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{XP}/multiple_cores")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # # Read in h5ad file for each individual xp
    # xp_1 = TMA_XP[TMA][0]
    # xp_2 = TMA_XP[TMA][1]
    # adata_1 = read_h5ad((in_dir / f"{xp_1}" / f"{xp_1}_filter.h5ad"))
    # adata_2 = read_h5ad((in_dir / f"{xp_2}" / f"{xp_2}_filter.h5ad"))

    # Another way, I can use the cohort h5ad to get the same cores on the duplicate xp
    adata = read_h5ad(in_dir / )





    # Getting the same core locations from both experiments by CORE_INFO


    # Extracting the patients in the cohort








