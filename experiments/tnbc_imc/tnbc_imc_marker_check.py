"""
Check quality of the markers
"""
import os
from pathlib import Path
from src.segment import *
import shutil


def plot_marker_nuclear_per_image(marker_list, nuclei_name, in_dir, out_dir):
    """
    Plot the marker and nuclei images to check marker quality by each image
    :param marker_list: markers to plot, type: list[str]
    :param nuclei_name: marker name of nuclei, type: str
    :param in_dir: directory of the tiff file and panel information, type: Path
    :param out_dir: directory to store the output images, type: Path
    :return:
    """
    # Get all marker info from panel.csv
    panel_df = pd.read_csv(in_dir / "panel.csv")
    panel = {name: int(index) for name, index in zip(panel_df['Target'], panel_df.index)}

    for roi, image_path in enumerate(list(in_dir.rglob("*.tif")) + list(in_dir.rglob("*.tiff")), start=1):
        file_name = os.path.basename(image_path)
        print(file_name.split("_")[0], file_name)

        # Read image
        img = skio.imread(image_path)

        for marker in marker_list:
            channel_idx = panel[marker]

            nuclei = img[panel[nuclei_name], :, :]
            cytoplasm = img[channel_idx, :, :]
            x = stack_for_mesmer(nuclei, cytoplasm)

            plot_marker_nuclear(x, out_dir, name=file_name.split("_")[0] + "_" + marker)


def plot_marker_nuclear(x, out_dir, name):
    """
    Plot the nuclear-and-one-marker-only image
    :param x: Pixel intensity of the image in 4-dimension, type: np.array
    :param out_dir: Directory to store the figure, type: str
    :param name: Name for the image, type: str
    :return:
    """
    rgb_images = create_rgb_image(x, channel_colors=['green', 'red'])

    plt.figure(figsize=(7, 7))

    plt.imshow(rgb_images[0])

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(out_dir / f"{name}.png", dpi=500)
    plt.clf()


if __name__ == '__main__':
    TYPE = "tnbc_imc_fluidigm"
    XP = "22-010B"
    # MARKER_LIST = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK', 'B-Catenin', 'H3K27me3',
    #                'EZH2', 'Ki67', 'EGFR', 'GATA3', 'CA-IX', 'Slug', 'Prostaglandin',
    #                'p53', 'ER-D8H8', 'ER-SP1', 'PR-SP2', 'erbB2', 'CD3e', 'CD20', 'CD45',
    #                'CD68', 'CD31', 'SMA', 'Vimentin', 'vWF']

    MARKER_LIST = ['Collagen1', 'CD20', 'PanCK', 'CD163', 'Vimentin', 'Ki-67', 'CD31',
                   'CD45', 'CD44', 'CD11c', 'FoxP3', 'CD4', 'E-cadherin', 'CD68', 'CD66b',
                   'FAP', 'CD8a', 'CD57', 'Podoplanin', 'PD-1', 'PD-L1', 'B-catenin',
                   'CD3e', 'Fibronectin', 'EpCAM', 'CD45RO', 'HLA-DR', 'CD14', 'GranzymeB',
                   'aSMA']

    MARKER_LIST = ['PanCK', 'E-cadherin', 'EpCAM', 'Ki-67', 'B-catenin', 'PD-1', 'PD-L1',
                   'CD3e', 'CD4', 'CD8a', 'FoxP3', 'CD45RO', 'GranzymeB', 'CD20',
                   'CD44', 'CD45', 'CD68', 'CD163', 'CD14', 'CD57', 'CD11c', 'HLA-DR', 'CD66b',
                   'Collagen1', 'Vimentin', 'CD31', 'FAP', 'Fibronectin', 'aSMA']



    NUCLEI_NAME = "DNA-191"

    # generate the nuclear-marker-only images
    in_dir = Path(f"data/{TYPE}/{XP}")
    out_dir = Path(f"qc/{TYPE}/{XP}/marker")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    plot_marker_nuclear_per_image(marker_list=MARKER_LIST,
                                  nuclei_name=NUCLEI_NAME,
                                  in_dir=in_dir,
                                  out_dir=out_dir)

    # Organize the images to the correct folder
    in_dir = Path(f"qc/{TYPE}/{XP}/marker")
    for marker in MARKER_LIST:
        out_dir = in_dir / marker
        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

    for image_path in list(in_dir.rglob("*.png")):
        file_name = os.path.basename(image_path).split("_")[1].split(".")[0]
        out_dir = in_dir / file_name
        shutil.move(str(image_path), str(out_dir))
