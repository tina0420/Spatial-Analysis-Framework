"""
@author: Tina Hsu
Run mesmer for segmentation
"""
import os
from pathlib import Path
from src.extract import __extract_panel_from_mcd
from src.segment import run_mesmer

if __name__ == '__main__':
    # parameter setting
    TYPE = "tnbc_imc_fluidigm"
    EXPERIMENT = "22-010B"
    DEEPCELL_TOKEN = "PRSsl9I1.f1GFmLjfURTQEdXCenrawh3IfykzxA2S"
    NUCLEAR_CHANNEL = [33]
    # cytoplasm channels with CD68, SMA, CD45, CD31 and panCK
    # CYTOPLASM_CHANNELS = [6, 8, 18, 27, 29]
    CYTOPLASM_CHANNELS = [16, 37, 10, 9, 5]
    MES_IMAGE_MPP = 1

    print(f"Processing {EXPERIMENT}")
    # check the path
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # get panel file
    file_list = list(in_dir.rglob("*.mcd"))
    for mcd_path in file_list:
        file_name = os.path.basename(mcd_path)
        print(f"Processing {file_name}")

        __extract_panel_from_mcd(mcd_path, in_dir)

    # run mesmer for segmentation
    run_mesmer(nuclear_channel=NUCLEAR_CHANNEL,
               cytoplasm_channels=CYTOPLASM_CHANNELS,
               in_dir=in_dir,
               out_dir=out_dir,
               deepcell_token=DEEPCELL_TOKEN,
               mes_image_mpp=MES_IMAGE_MPP,
               plot_overlay=True,
               output_pred_df=False
               )

