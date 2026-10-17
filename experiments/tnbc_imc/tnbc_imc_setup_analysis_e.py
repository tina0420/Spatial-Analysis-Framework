"""
@author: Tina Hsu
Setup for TNBC IMC analysis
"""
from pathlib import Path
from anndata import read_h5ad
import pandas as pd

TYPE = 'tnbc_imc'
PURPOSE = 'final_thesis'

IN_DIR = Path(f"data/{TYPE}/{PURPOSE}")
OUT_DIR = Path(f"analysis/{TYPE}/thesis_all/figures/Revision")

# Read in adata
METADATA = pd.read_csv(IN_DIR / "TNBC_patient_imc_final_metadata_category.csv")
FULL_ADATA = read_h5ad(IN_DIR / f"TNBC_pretreatment.h5ad")

"""
QC
Weird Mp: "22-010A_ROI_024", "22-010A_ROI_026"
Weird Endo: "22-010C_ROI_029", "22-010C_ROI_030"
No-epithelial cells: "22-010A_ROI_068", "23-010_ROI_002"
"""
# Remove bad images
FULL_ADATA = FULL_ADATA[~FULL_ADATA.obs.unique_img_key.isin(["22-010A_ROI_024", "22-010A_ROI_068", "23-010_ROI_002"])]

# Extract the patient receiving Neo-adjuvant chemotherapy as a new adata
# NAT_ADATA = FULL_ADATA[FULL_ADATA.obs["chemo_type"] == "NAT"]

# Fixed colors
TMA_COLOR_DICT = {'BOB': 'tab:blue',
                  '22-010A': 'tab:pink',
                  '22-010B': 'tab:yellow',
                  '22-010C': 'tab:orange',
                  '23-005': 'tab:brown',
                  '23-009': 'tab:green',
                  '23-010': 'tab:purple',
                  'Brian': 'tab:cyan'}

RELAPSE_TF_COLOR_DICT = {0: 'tab:blue', 1: 'tab:pink'}

PCR_TF_COLOR_DICT = {0: 'tab:blue', 1: 'tab:pink'}

E_TME_COLOR_DICT = {'Epithelial': '#b41f77', 'TME': '#1f77b4'}

ESI_CELL_TYPE_COLOR_DICT = {'Epithelial': '#b41f77', 'Immune': '#77b41f', 'Stromal': '#1f77b4'}

ESI_TILS_COLOR_DICT = {'Epithelial': '#b41f77', 'Stromal': '#1f77b4',
                       'Infiltrating': '#ef2516', 'Bystanding': '#2e82f1', 'Ignored': '#c1dafb'}

# colour dict for showing the epithelial affected distribution
EPI_TILS_COLOR_DICT = {'T-cell': '#64D60E',
                       'Stromal': '#ae8362',
                       'Yes': '#ef2516',
                       'No': '#2e82f1',
                       'Macrophage': '#ae8ae7',
                       'B-cell': '#296B58',
                       'Others': '#e0e0e0'}

# Cell type colors
CELL_TYPE_COLOR_DICT = {'CK5+ Basal': '#f10024',
                        'CK7+ Luminal': '#ffb6c1',
                        'CK8/18+ Luminal': '#ef38aa',
                        'CK+ Epithelial': '#ae8ae7',
                        'T-cell': '#c3e78a',
                        'B-cell': '#178867',
                        'Macrophage': '#9ac4a0',
                        'Endothelial': '#5ea3cd',
                        'SMA+ Stromal': '#01588d',
                        'Stromal': '#bbe4fc',
                        }

LAYER2_CELL_TYPE_COLOR_DICT = {'CK5+ Basal': '#f10024',
                               'CK7+ Luminal': '#ffb6c1',
                               'CK8/18+ Luminal': '#ef8616',
                               'CK+ Epithelial': '#ae8ae7',
                               'T-cell': '#c3e78a',
                               'B-cell': '#178867',
                               'Macrophage': '#9ac4a0',
                               'Endothelial': '#5ea3cd',
                               'SMA+ Stromal': '#01588d',
                               'Stromal': '#bbe4fc',
                               }

CELL_NICHE_COLOR_DICT = {'B_Patch': '#77b41f',
                         'Endothelial_Predominant': '#6B4E1E',
                         'Epithelial_Predominant': '#1f77b4',
                         'Stromal_Predominant': '#01588d',
                         'T_Infiltration_Basal': '#ff9896',
                         'T_Infiltration_Luminal': '#bd67bb',
                         'T_Mp_Aggregation': '#178867',
                         'T_Resident_Stromal': '#ef8616'}
