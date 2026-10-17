"""
@author: Tina Hsu
Using transformer to differentiate cell types in an image.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import read_h5ad
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader


if __name__ == "__main__":
    """
    Parameters setting
    """
    TYPE = "tnbc_imc"
    EXPERIMENT = "XP12446"

    # Set paths
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}/transformer")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read AnnData
    adata = read_h5ad(path_dir / f"{EXPERIMENT}_clustering.h5ad")

    """
    Test for the first image
    """
    image_1 = adata[np.where(adata.obs.image_id ==
                            '2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff')[0], :]
    # 1782 cells

    COLOR_MAP = ['#77b41f', '#b41f77', '#1f77b4', '#ff7f0e', '#ff9896', '#bd67bb', '#8c564b', '#fff80e', '#1fb45c',
                 '#b41f2d', '#1f2db4', '#ff430e', '#ff96c9', '#9467bd', '#8c4b61', '#c1bb00', '#c7e8ae', '#e8aec7',
                 '#aec7e8', '#ffbb78', '#ff96fd', '#c5b0d5', '#8c774b', '#090909', '#4B0082', '#FF1493', '#00CED1']

    """
    For each marker distribution
    """

    for i in range(image_1.X.shape[1]):
        plt.figure(figsize=(12, 9))
        color = LinearSegmentedColormap.from_list("colorbar", ["lightgray", COLOR_MAP[i]])
        scatter = plt.scatter(x=image_1.obsm["spatial"][:, 0], y=image_1.obsm["spatial"][:, 1], c=image_1.X[:, i],
                              cmap=color)
        plt.colorbar(scatter, label='Intensity')
        marker = image_1.var_names[i]
        plt.title(marker, fontsize=25)
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        plt.tight_layout()
        plt.savefig(out_dir / f"im1_{marker}_intensity.png", dpi=500)
        plt.clf()

    """
    Overlap the plotting with all the markers
    """

    # All markers
    plt.figure(figsize=(12, 9))
    for j in range(image_1.X.shape[1]):
        color = LinearSegmentedColormap.from_list("colorbar", ["lightgray", COLOR_MAP[j]])
        plt.scatter(x=image_1.obsm["spatial"][:, 0], y=image_1.obsm["spatial"][:, 1], c=image_1.X[:, j], cmap=color,
                    edgecolors='none', alpha=0.5, label=image_1.var_names[j])
    plt.title("All markers", fontsize=25)
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y", fontsize=15)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir / "im1_all_markers_intensity.png", dpi=500)
    plt.clf()

    ### acrross all the images
    for pid in set(adata.obs.image_id):
        p_adata = adata[np.where(adata.obs.image_id == pid)[0], :]

        plt.figure(figsize=(12, 9))
        for j in range(p_adata.X.shape[1]):
            color = LinearSegmentedColormap.from_list("colorbar", ["lightgray", COLOR_MAP[j]])
            plt.scatter(x=p_adata.obsm["spatial"][:, 0], y=p_adata.obsm["spatial"][:, 1], c=p_adata.X[:, j], cmap=color,
                        edgecolors='none', alpha=0.5, label=p_adata.var_names[j])

        plt.title("All markers", fontsize=25)
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        image = pid.split("_")[-2]
        print(image)
        plt.savefig(out_dir / f"{image}_all_markers_intensity.png", dpi=500)
        plt.clf()


    """
    Overlap the marker distributions with the specific markers (epithelial, immune and stromal)
    """

    LIST_MARKER_COMB_DICT = {"epithelial": ["CK5", "CK7", "CK8-18", "CK19", "panCK",
                                            "B-Catenin", "H3K27me3", "EZH2", "Ki67", "EGFR",
                                            "GATA3", "CA-IX", "Slug", "Prostaglandin", "p53",
                                            "ER-D8H8", "ER-SP1", "PR-SP2", "erbB2"],
                             "immune": ["CD3e", "CD20", "CD45", "CD68"],
                             "stromal": ["CD31", "SMA", "Vimentin", "vWF"]}

    # set a dict for the markers corresponding index (all markers)
    marker_index = dict(zip(image_1.var_names, range(len(image_1.var_names))))

    for type, marker_comb in LIST_MARKER_COMB_DICT.items():
        marker_idx = [marker_index[x] for x in marker_comb]
        plt.figure(figsize=(12, 9))
        for k in range(len(marker_idx)):
            color = LinearSegmentedColormap.from_list("colorbar", ["lightgray", COLOR_MAP[marker_idx[k]]])
            plt.scatter(x=image_1.obsm["spatial"][:, 0], y=image_1.obsm["spatial"][:, 1], c=image_1.X[:, marker_idx[k]],
                        cmap=color, edgecolors='none', alpha=0.5, label=marker_comb[k])
        plt.title(f"{type} markers", fontsize=25)
        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        #plt.show()
        plt.savefig(out_dir / f"im1_{type}_markers_intensity.png", dpi=500)

    """
    Transformers (input: image, output: cell type of each cell)
    """
    # Prepare the image (resize or cut)
    import cv2
    img = cv2.imread(out_dir / "a51_all_markers_intensity.png")

    x, y, w, h = 250, 100,
    # test_cut_img = img[100: 4400, 200: 5800]
    test_cut_img = img[150: 4000, 300: 5500]

    test_cut_img = cv2.cvtColor(test_cut_img, cv2.COLOR_BGR2RGB)
    plt.imshow(test_cut_img)
    plt.axis('off')
    plt.show()

    # cv2.imshow("test cut img", test_cut_img)

    # PILLOW
    from PIL import Image
    img = Image.open(out_dir / "a51_all_markers_intensity.png")
    width, height = img.size
    crop_box = (width * 0.08, height * 0.055, width * 0.855, height * 0.92)
    test_cut_img = img.crop(crop_box)
    test_cut_img.show()

    # ok! --> extracting the central part of the plotting








    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match ViT input size
        transforms.ToTensor(),  # Convert images to tensors
    ])

    dataset = datasets.ImageFolder(root=out_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load the Vision Transformer model without pre-trained weights
    model = vit_b_16(weights=None)  # Initialize model randomly
    model = vit_b_16(pretrained=True) # Pre-trained model








