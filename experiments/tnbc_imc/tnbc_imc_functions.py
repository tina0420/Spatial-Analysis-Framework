"""
Functions used for TNBC IMC analysis in the thesis
"""
import os
from pathlib import Path
from anndata import read_h5ad
import tifffile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import squidpy as sq
# from src.segment import remove_salt_pepper_noise_normalize
from deepcell.utils.plot_utils import create_rgb_image
import skimage.io as skio


"""

"""



"""
Specific marker exp distribution plot (need to fix the 3-channel issue)
"""
# def plot_target_marker_expression(panel_dict, image_path, marker_colour_dict, out_dir, kernel_size=3,
#                                   norm_method="minmax", resize=False,
#                                   name=None):
#     # Read image
#     img = skio.imread(image_path)
#
#     # Remove noise
#     nuclei, cytoplasm = image_preprocess(img, nuclear_channel, cytoplasm_channels, kernel_size, norm_method)
#
#     x = stack_for_mesmer(nuclei, cytoplasm)
#
#     for marker in marker_colour_dict.keys():
#         channel_idx = panel_dict[marker]
#
#         # nuclei = img[panel[nuclei_name], :, :]
#         cytoplasm = img[channel_idx, :, :]
#         x = stack_for_mesmer(nuclei, cytoplasm)
#
#         plot_marker_nuclear(x, out_dir, name=file_name.split("_")[0] + "_" + marker)
#
#     # Remove noise
#     nuclei, cytoplasm = image_preprocess(img, nuclear_channel, cytoplasm_channels, kernel_size, norm_method)
#
#
# # Main used for testing the functions above
# if __name__ == '__main__':
#     # Directory setting
#     TYPE = 'tnbc_imc'
#     PURPOSE = 'final_thesis'
#     SUB_PURPOSE = 'marker_exp_distribution'
#     XP = 'TNBC_pretreatment'
#
#     in_dir = Path(f'data/{TYPE}/{PURPOSE}')
#     out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/{SUB_PURPOSE}")
#     if os.path.exists(out_dir) == 0:
#         os.makedirs(out_dir, exist_ok=True)
#
#     MARKER_DICT = {"panCK": "red",
#                    "CD3e": "blue",
#                    "CD20": "green"}
#
#     # Read in h5ad with tme cells only
#     metadata = pd.read_csv(in_dir / "TNBC_patient_imc_final_metadata_category.csv")
#     metadata = metadata[metadata["collection_type"] == 0]
#
#     # Go through all the images in metadata (pre-treatment only)
#     img_df = metadata[["CID", "Recurrence status", "TMA", "Experiment", "Core location", "Image ID"]]
#     img_df = img_df[pd.notna(img_df["Image ID"])].reset_index(drop=True)
#
#     # Get all marker info from panel.csv
#     panel_df = pd.read_csv(in_dir / "panel.csv")
#     panel = {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}
#
#     for i in range(img_df.shape[0]):
#         relapse = img_df["Recurrence status"].values[i]
#         tma = img_df["TMA"].values[i]
#         xp = img_df["Experiment"].values[i]
#         img_key = img_df["Image ID"].values[i]
#         print(relapse, xp, img_key)
#         # For those are the
#         temp_in_dir = Path(f'data/{TYPE}/{xp}')
#         if tma in ["BOB", "22-010A", "22-010C", "23-009"]:
#             img = img_df["Image ID"].values[i].split("_")[1] + "_" + img_df["Image ID"].values[i].split("_")[2]
#             image_path = list(temp_in_dir.rglob(f"*{img}*.tif*"))[0]
#             if ".png" in str(list(temp_in_dir.rglob(f"*{img}*.tif*"))[0]):
#                 image_path = list(temp_in_dir.rglob(f"*{img}*.tif*"))[1]
#
#             print(image_path)
#         elif tma in ["22-010B", "23-005", "23-010"]:
#             pass
#         img = skio.imread(image_path)
#
#         channel_idxs = [panel[m] for m in list(MARKER_DICT.keys())]
#
#         # not doing this noice removl for visualization (because here is only for visualization, for mesmer we did)
#         # [remove_salt_pepper_noise_normalize(img, kernel_size=3, norm_method="minmax", idx) for idx in channel_idxs]
#
#         x = np.stack([img[idx, :, :] for idx in channel_idxs], axis=-1)
#         x = np.expand_dims(np.stack([img[idx, :, :] for idx in channel_idxs], axis=-1), axis=0)
#
#         rgb_images = create_rgb_image(x, channel_colors=list(MARKER_DICT.values()))
#         plt.figure(figsize=(7, 7))
#         plt.imshow(rgb_images[0])
#         plt.axis('off')
#         plt.tight_layout()
#
#         if relapse == 0:
#             temp_out_dir = ""
#         elif relapse == 1:
#             temp_out_dir = ""
#
#         plt.savefig(temp_out_dir / f"{name}.png", dpi=500)
#         plt.clf()
#
#     img_data = tifffile.imread(image_path)
#     img_container = sq.im.ImageContainer(img_data, layer="image")
#
#     fig, ax = plt.subplots(figsize=(10, 10))
#
#     img_container.show(
#         channel=channel_idxs,
#         channel_colors=list(MARKER_DICT.values()),
#         norm=None,
#         ax=ax
#     )
#
#     ax.set_title("TNBC TME: Pan-CK(R), CD8(G), DNA(B)")
#     plt.show()

