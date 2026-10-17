"""
@author: Tina Hsu
Preprocess the measurements.csv from segmentation.
"""
import os
from pathlib import Path
import pandas as pd
from src.preprocess import read_to_anndata_wrapper, normalize
from src.extract import quantify_from_qupath

# from experiments.tnbc_imc.archive.check_batch_effect import *
# from experiments.tnbc_imc.archive.preprocessing import *


if __name__ == '__main__':

    # parameters for the codes
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"

    # Setting the paths for reading data and output figures
    path_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"figures/{TYPE}/{EXPERIMENT}")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Tidy the measurements.csv to the format we want
    # Process measurements.csv from QuPath
    exp_df, loc_df = quantify_from_qupath(path_dir)

    # Choose the columns (markers)
    exp_df = pd.concat([exp_df.iloc[:, 0], exp_df.iloc[:, 4:-4]], axis=1)

    # Rename columns names
    exp_df.columns = [exp_df.columns[0]] + [x.split(" ")[1] for x in exp_df.columns[1:]]

    # Output to the input structure for Sakura
    exp_df.to_csv(path_dir / "y.csv", index=False)
    loc_df.to_csv(path_dir / "loc.csv", index=False)

    # Preprocess
    # Read from y.csv and loc.csv to Anndata object
    adata = read_to_anndata_wrapper(path_dir=path_dir)

    # Normalization
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    # Output the h5ad file
    adata.write(path_dir / f"{EXPERIMENT}.h5ad")

    """
    Function for preprocessing.
    """

    # def preprocess(experiment_name, disease_data_type):
    #     """
    #     Tidy up the measurement file into the format we want for Sakura.
    #     :param experiment_name: The name of the experiment, type: str
    #     :param disease_data_type: Type of the disease and data used for the path name, type: str
    #     :return:
    #     """
    #     # Setting the paths for reading data and output figures
    #     path_dir = Path(f"data/{disease_data_type}/{experiment_name}")
    #     out_dir = Path(f"figures/{disease_data_type}/{experiment_name}")
    #
    #     if os.path.exists(out_dir) == 0:
    #         os.makedirs(out_dir, exist_ok=True)
    #
    #     # Tidy the measurements.csv to the format we want
    #     # Process measurements.csv from QuPath
    #     exp_df, loc_df = process_qupath(path_dir)
    #
    #     # Choose the columns (markers)
    #     exp_df = pd.concat([exp_df.iloc[:, 0], exp_df.iloc[:, 4:-4]], axis=1)
    #
    #     # Rename columns names
    #     exp_df.columns = [exp_df.columns[0]] + [x.split(" ")[1] for x in exp_df.columns[1:]]
    #
    #     # Output to the input structure for Sakura
    #     exp_df.to_csv(path_dir / "y.csv", index=False)
    #     loc_df.to_csv(path_dir / "loc.csv", index=False)
    #
    #     # Preprocess
    #     # Read from y.csv and loc.csv to Anndata object
    #     adata = read_to_anndata_wrapper(path_dir=path_dir)
    #
    #     # Normalization
    #     normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    #     normalize(adata, method="clip", clip_percentile=99.5)
    #
    #     # Output the h5ad file
    #     adata.write(path_dir / f"{experiment_name}.h5ad")
    #
    # preprocess(EXPERIMENT, TYPE)

