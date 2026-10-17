import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from experiments.tnbc_imc.tnbc_imc_setup_analysis import RELAPSE_TF_COLOR_DICT, PCR_TF_COLOR_DICT, CHEMOTYPE_COLOR_DICT
from sklearn.mixture import BayesianGaussianMixture
from src.annotate import update_labels

if __name__ == '__main__':

    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'final_thesis'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    patient_df = pd.read_csv(in_dir / 'patient_imc_final_metadata.csv')
    patient_df = patient_df[patient_df['collection_type'] == 'post_treatment']

    # select 5 recurrent and 5 non-recurrent post samples
    img = ['22-010A_ROI_070', '22-010A_ROI_060', '22-010A_ROI_042', '22-010C_ROI_008', '22-010A_ROI_015',
           '22-010A_ROI_032', '22-010A_ROI_053', '22-010C_ROI_056', '22-010C_ROI_071', '23-009_ROI_006']

    # Get the h5ad with these patients/images
    adata = read_h5ad(in_dir / "cohort_all_img.h5ad")

    sub_adata = adata[np.where(adata.obs.unique_img_key.isin(img))[0], :]

    sub_adata.write(out_dir / "imc_testing.h5ad")
