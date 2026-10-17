"""
@author: Tina Hsu
Segmentation for tiff files.
"""
import os
from pathlib import Path
from src.extract import extract_mcd_for_imc
from src.segment import run_mesmer


if __name__ == '__main__':
    # Setting Mesmer parameters
    NUCLEAR_CHANNELS = [31]
    CYTOPLASM_CHANNELS = [3, 29]
    INPUT_DIR = Path("data/tnbc_imc/XP12274/")
    OUTPUT_DIR = Path("output/tnbc_imc/XP12274")
    if os.path.exists(OUTPUT_DIR) == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    DEEPCELL_TOKEN = "gLJAn8ZY.OmHKrKnth0mqobIljEi6o16StVeqhFVn"
    MES_INTERIOR_THRESHOLD = 0.1
    MES_MAXIMA_THRESHOLD = 0.58
    MES_IMAGE_MPP = 1
    MASK = False
    OVERLAY = False
    PRED_DF = False

    """
    Extract tiff files from MCD
    """
    extract_mcd_for_imc(in_dir=INPUT_DIR,
                        out_dir=OUTPUT_DIR)

    """
    Start running Mesmer
    """
    run_mesmer(nuclear_channel=NUCLEAR_CHANNELS,
               cytoplasm_channels=CYTOPLASM_CHANNELS,
               in_dir=INPUT_DIR,
               out_dir=OUTPUT_DIR,
               deepcell_token=DEEPCELL_TOKEN,
               mes_image_mpp=MES_IMAGE_MPP,
               mes_interior_threshold=MES_INTERIOR_THRESHOLD,
               mes_maxima_threshold=MES_MAXIMA_THRESHOLD,
               plot_mask=MASK,
               plot_overlay=OVERLAY,
               output_pred_df=PRED_DF)
