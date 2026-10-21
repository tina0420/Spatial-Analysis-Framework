"""
@author: Tina Hsu
Tutorial 1a: Processing multiplexed images and writing to AnnData.
"""
import os
from pathlib import Path
import scanpy as sc
from src.extract import extract_mcd_for_imc
from src.segment import run_mesmer
from src.preprocess import quantify_expression, normalize


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "example"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    in_dir = Path(f"example/data/tonsil_imc")
    out_dir = Path(f"example/output/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    In this example, we are using tonsil IMC data.
    
    To acquire multiplexed protein images in tiff format, we need to process the MCD file. 
    Sakura has functions to directly process MCDs. We can extract the panel and tiff files in the MCD file.
    
    In the output directory, we can find tiff files and a panel file which are required for segmentation.
    For ease, we are outputting to the same directory.
    """
    extract_mcd_for_imc(in_dir=in_dir, out_dir=in_dir)

    """
    Next, we will begin segmenting the tiff files with Mesmer.

    Segmentation will require tiff files, a panel file, and a Deepcell token which can be obtained on the Deepcell site.
    
    The channels here are 0-indexed.
    
    For this dataset, we can see from the panel file that the nuclear channel is 8 (DNA1), 
    and the cytoplasm channels are 4 (CD20) and 5 (CD3).
    
    The mes_image_mpp parameter is to indicate microns per pixel for the image of interest.
    For this dataset, we use IMC data which is 1. Other datasets like CosMx, use 0.18.
    
    The last two threshold parameters control the size of cells and cell count.
    Here, we use default parameters copied from the function.
    Tuning the parameters for segmentation can be done by re-running the function and visualizing the mask and overlay.

    This function will output an expression file called measurements.
    It can also optionally output a cell masks and an overlay image with cells and their boundaries.
    """
    DEEPCELL_TOKEN = "gLJAn8ZY.OmHKrKnth0mqobIljEi6o16StVeqhFVn"

    run_mesmer(nuclear_channel=[8], cytoplasm_channels=[4, 5],
               in_dir=in_dir, out_dir=out_dir,
               deepcell_token=DEEPCELL_TOKEN,
               mes_image_mpp=1, mes_interior_threshold=0.2, mes_maxima_threshold=0.075,
               plot_mask=True, plot_overlay=True, output_pred_df=True)

    """
    After Mesmer finishes, we can process the measurements file into an AnnData file 
    for further processing and analysis.
    This is streamlined by using the following function.
    Since our input is proteomics data, there is no need to use a sparse matrix for storing expression.
    """
    adata = quantify_expression(out_dir, is_sparse=False)

    """
    Optionally, we can subset columns that we are interested in.
    """
    col_of_interest = ['Cell: CD68 mean', 'Cell: CD20 mean', 'Cell: CD3 mean', 'Cell: BCL6 mean']
    adata = adata[:, col_of_interest]

    """
    Next, we will normalize the expression data according to the modality.
    For spatial proteomics such as IMC, we commonly use a combination of arcsinh and clipping.
    For spatial transcriptomics such as CosMx, we use library size normalization or cell area normalization.
    
    For this dataset, we use a common normalization technique for IMC.
    """
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    """
    Commonly for spatial transcriptomics and sometimes for spatial proteomics, we might want to reduce the dimensions.
    This can be for checking batch effects, finding variable genes, or just visualization.
    """
    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata)

    """
    We can now save the AnnData to a h5ad file for clustering.
    """
    adata.write(out_dir / f"{run_name}.h5ad")
