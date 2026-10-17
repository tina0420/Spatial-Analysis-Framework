"""
@author: Tina Hsu
Quantify measurement and clean data with the segmentation result
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from src.preprocess import quantify_expression, normalize
import numpy as np
import skimage


def merge_images(adata, upper_img, lower_img, upper_dim=None, lower_dim=None, in_dir=None, plot=False):
    """
    Merging the images info which are from the same core.
    :param adata: Anndata
    :param upper_img: name of the upper ROI, type: str
    :param lower_img: name the lower ROI, type: str
    :param upper_dim: dimension (width, height) of the upper ROI, type: list[int]
    :param lower_dim: dimension (width, height) of the lower ROI, type: list[int]
    :param in_dir: directory with the ROIs, type: Path
    :param plot: plot to check the merging accuracy, type: bool
    :return: Anndata with the merged data
    """
    # merge location info, only change y-coordinate of the lower image
    upper_idx = np.where(adata.obs.image_id == upper_img)[0]
    lower_idx = np.where(adata.obs.image_id == lower_img)[0]
    adata.obsm['spatial'][lower_idx, :]

    if upper_dim is None:
        upper_dim = get_image_size(in_dir / upper_img)

    if lower_dim is None:
        lower_dim = get_image_size(in_dir / lower_img)

    # plot
    if plot == True:
        # check the original image by plotting the location
        plot_check_merge(adata[upper_idx, :], (12, 9), in_dir, f"{upper_img}_upper", img_dim=upper_dim)
        plot_check_merge(adata[lower_idx, :], (12, 9), in_dir, f"{lower_img}_lower", img_dim=lower_dim)

    # change the y-coordinate of lower image
    if abs(upper_dim[0] - lower_dim[0]) < 10:
        for idx in lower_idx:
            adata.obsm["spatial"][idx, 1] = adata.obsm["spatial"][idx, 1] + upper_dim[1]
    else:
        # large difference in the dimension
        print(f"Dimensions are much different: {upper_dim} and {lower_dim}.")

    # merge expression info (keep the original image name)
    adata.obs.iloc[lower_idx, 0] = upper_img

    if plot == True:
        plot_check_merge(adata[np.where(adata.obs.image_id == upper_img)[0], :],
                         (7, 7), in_dir, f"{upper_img}_merged")


def plot_check_merge(adata, fig_size, out_dir, name, img_dim=None):
    """
    Plot cell locations from an image.
    :param adata: Anndata
    :param fig_size: figure size, type:tuple
    :param out_dir: output directory, type: Path
    :param name: name of the figure, type: str
    :param img_dim: dimension of the image, type: tuple[int, int]
    :return:
    """
    plt.figure(figsize=fig_size)
    plt.scatter(x=adata.obsm["spatial"][:, 0], y=-adata.obsm["spatial"][:, 1], s=1.5)
    if img_dim:
        plt.title(f"{name} ({str(img_dim[0])}x{str(img_dim[1])} pixels)", size=15)
    else:
        plt.title(name, size=15)
    plt.xlabel("X", size=15)
    plt.ylabel("Y", size=15)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=100)
    plt.clf()


def get_image_size(img_path):
    """
    Getting size (width, height) of the image
    :param img_path: path of the image, type:Path
    :return: width and height of the image, type: tuple[int, int]
    """
    image = skimage.io.imread(img_path)
    size = (image.shape[2], image.shape[1])
    """
    !!TroubleShooting: cannot read the specific images, already tried pillow, matplotlib.pyplot, cv2 and skimage  
    first three tools failed with two no reason and 1 error message from cv2 that it couldn't read the images with 64-bit.
    skimage read the image successfully.
    """

    return size


if __name__ == '__main__':
    # parameter setting (for every single experiment)
    TYPE = "tnbc_imc_fluidigm"
    EXPERIMENT = "22-010B"

    # check the path
    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # quantify measurement
    adata = quantify_expression(out_dir, is_sparse=False)

    # chose the columns we want (in the order of epithelial, immune and stromal markers)
    BM_markers = ['Cell: CK5 mean', 'Cell: CK7 mean', 'Cell: CK8-18 mean', 'Cell: CK19 mean', 'Cell: panCK mean',
                  'Cell: B-Catenin mean', 'Cell: H3K27me3 mean', 'Cell: EZH2 mean', 'Cell: Ki67 mean',
                  'Cell: EGFR mean', 'Cell: GATA3 mean', 'Cell: CA-IX mean', 'Cell: Slug mean',
                  'Cell: Prostaglandin mean', 'Cell: p53 mean', 'Cell: ER-D8H8 mean', 'Cell: ER-SP1 mean',
                  'Cell: PR-SP2 mean', 'Cell: erbB2 mean',
                  'Cell: CD3e mean', 'Cell: CD20 mean', 'Cell: CD45 mean', 'Cell: CD68 mean',
                  'Cell: CD31 mean', 'Cell: SMA mean', 'Cell: Vimentin mean', 'Cell: vWF mean']

    Fluidigm_markers = ['Cell: Collagen1 mean', 'Cell: CD20 mean', 'Cell: PanCK mean', 'Cell: CD163 mean',
                        'Cell: Vimentin mean', 'Cell: Ki-67 mean', 'Cell: CD31 mean', 'Cell: CD45 mean',
                        'Cell: CD44 mean', 'Cell: CD11c mean', 'Cell: FoxP3 mean', 'Cell: CD4 mean',
                        'Cell: E-cadherin mean', 'Cell: CD68 mean', 'Cell: CD66b mean', 'Cell: FAP mean',
                        'Cell: CD8a mean', 'Cell: CD57 mean', 'Cell: Podoplanin mean', 'Cell: PD-1 mean',
                        'Cell: PD-L1 mean', 'Cell: B-catenin mean', 'Cell: CD3e mean', 'Cell: Fibronectin mean',
                        'Cell: EpCAM mean', 'Cell: CD45RO mean', 'Cell: HLA-DR mean', 'Cell: CD14 mean',
                        'Cell: GranzymeB mean', 'Cell: aSMA mean']

    # adata = adata[:, BM_markers]
    adata = adata[:, Fluidigm_markers]

    # rename the columns
    adata.var_names = [x.split(" ")[1] for x in adata.var_names]

    # Do normalization by
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    """
    If there are images from the same core, need to merge the information by the name of the images.
    """
    # XP12813 has 2 continues images, "row4-5_ROI73_ROI_050_contd.ome.tiff" from "row4-5_ROI50_ROI_050.ome.tiff"
    #                                 "row4-5_ROI74_ROI_58_contd.ome.tiff" from "row4-5_ROI58_ROI_058.ome.tiff"

    # XP13087 has 2 continues images, "ROI89_ROI_044-contd.ome.tiff" and "ROI44_ROI_044.ome.tiff"
    #                                 "ROI90_ROI_085-contd.ome.tiff" from "ROI85_ROI_085.ome.tiff"

    # XP12814 has 2 continues images, "ROI56_ROI_001_contd.ome.tiff" and "ROI1_ROI_001.ome.tiff"
    #                                 "ROI57_ROI_018-cont'd.ome.tiff" from "ROI18_ROI_018.ome.tiff"

    # merge_images(adata=adata,
    #              upper_img="row4-5_ROI50_ROI_050.ome.tiff",
    #              lower_img="row4-5_ROI73_ROI_050_contd.ome.tiff",
    #              in_dir=in_dir,
    #              plot=True)
    # merge_images(adata=adata,
    #              upper_img="row4-5_ROI58_ROI_058.ome.tiff",
    #              lower_img="row4-5_ROI74_ROI_58_contd.ome.tiff",
    #              in_dir=in_dir,
    #              plot=True)

    adata.write(out_dir / f"{EXPERIMENT}.h5ad")
