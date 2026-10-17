"""
@author: Tina Hsu
Functions for preprocessing and normalizing segmented data.
"""
import os
import re
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore
from scipy.sparse import issparse, csr_matrix
import warnings


def quantify_expression(path_dir, is_sparse=False):
    """
    Process segmented expression.
    :param path_dir: input path
    :param is_sparse: if sparse matrix, for transcriptomics, type: bool
    :return: AnnData
    """
    # Read segmented expression
    df = pd.read_parquet(path_dir / "measurements.parquet")

    channels = ['Image', 'Cell: Area']

    for col in df.columns:
        if re.search('Centroid', col):
            channels.append(col)

    for col in df.columns:
        if re.search('Cell', col):
            if re.search('mean', col):
                channels.append(col)
    df = df.loc[:, channels]

    # Subset expression columns
    exp_df = pd.concat([df.iloc[:, 0], df.iloc[:, 4:]], axis=1)

    # Subset location columns
    loc_df = df.loc[:, channels[:4]]
    loc_df.columns = ["Image", "Area", "X", "Y"]

    # Read to anndata
    adata = __read_to_anndata(exp_df, loc_df, is_sparse)

    return adata


def __read_to_anndata(exp_df, loc_df, is_sparse):
    """
    Reading to AnnData.
    :param exp_df: expression matrix, type: pd.DataFrame
    :param loc_df: location matrix, type: pd.DataFrame
    :param is_sparse: if sparse matrix, for transcriptomics, type: boolean
    :return: AnnData
    """
    # Create AnnData object
    if is_sparse:
        adata = AnnData(csr_matrix(exp_df.iloc[:, 1:].values), var=pd.DataFrame([], index=exp_df.columns[1:]))
        adata.obs["image_id"] = csr_matrix(exp_df.iloc[:, 0].values)
    else:
        adata = AnnData(exp_df.iloc[:, 1:].values, var=pd.DataFrame([], index=exp_df.columns[1:]))
        adata.obs["image_id"] = exp_df.iloc[:, 0].values

    adata.obsm["spatial"] = loc_df[["X", "Y"]].values
    adata.obs["area"] = loc_df["Area"].values

    return adata


def normalize(adata, method="arcsinh", arcsinh_cofactor=0.8, clip_percentile=99.5):
    """
    Normalize.
    :param adata: AnnData
    :param method: normalization method, one of ['arcsinh', 'log'], type: str
    :param arcsinh_cofactor: cofactor for arcsinh, type: float
    :param clip_percentile: percentile to clip, type: float
    :return:
    """
    if issparse(adata.X):
        adata.X = csr_matrix(__normalize(adata.X.toarray(), method, arcsinh_cofactor, clip_percentile))
    else:
        adata.X = __normalize(adata.X, method, arcsinh_cofactor, clip_percentile)

    adata.uns[method] = {}


def __normalize(X, method, arcsinh_cofactor=0.8, clip_percentile=99.5):
    """
    Normalize.
    :param X: expression, type: np.array
    :param method: normalization method, one of ['arcsinh', 'log', 'clip', 'zscore'], type: str
    :param arcsinh_cofactor: cofactor for arcsinh, type: float
    :param clip_percentile: percentile to clip, type: float
    :return: normalized expression, type: np.array
    """
    for col in range(X.shape[1]):
        # check for divide by zero
        if X[:, col].sum() == 0.0:
            continue
        else:
            if method == "arcsinh":
                X[:, col] = __arcsinh_transformation(X[:, col], cofactor=arcsinh_cofactor)

            elif method == "clip":
                X[:, col] = __percentile_clip(X[:, col], percentile=clip_percentile)

            elif method == "zscore":
                X[:, col] = zscore(X[:, col])

            elif method == "log":
                X[:, col] = __log_transformation(X[:, col])

            else:
                raise Exception("Choose from arcsinh, clip, zscore, log")

    return X


def __arcsinh_transformation(arr, cofactor=0.8):
    """
    Arcsinh transformation.
    :param arr: single column of expression, type: np.array
    :param cofactor: for cofactor for arcsinh, type: float
    :return: arcsinh transformation expression, type: np.array
    """
    arr = np.arcsinh(arr.T / cofactor)

    return arr.T


def __log_transformation(arr):
    """
    Log transformation.
    :param arr: single column of expression, type: np.array
    :return: log transformation expression, type: np.array
    """
    return np.log(arr + 1)


def __percentile_clip(arr, percentile=99.5):
    """
    Clipping upper percentile.
    :param arr: single column of expression, type: np.array
    :param percentile: percentile to clip, type: float
    :return: clipped expression, type: np.array
    """
    cap = np.percentile(arr, percentile)

    return np.where(arr > cap, cap, arr)


def read_neighbors(adata, rel_path="rel.csv", threshold=30):
    """
    Read neighbors.
    :param adata: AnnData
    :param rel_path: path to relation matrix, type:str
    :param threshold: threshold, type: float
    :return:
    """
    if os.path.exists(rel_path):
        rel_df = pd.read_csv(rel_path)
        adata.uns["relation"] = rel_df.values
    else:
        adata.uns["relation"] = __construct_relations(adata.obsm["spatial"], threshold=threshold)
        adata.uns["threshold"] = threshold
        warnings.warn("Constructing relations since no rel file was found.")


def __construct_relations(loc, threshold):
    """
    Constructs relations based on pairwise distance thresholding
    :param loc: location, type: numpy matrix
    :param threshold: threshold for pairwise distance, type: float
    :return: matrix of indices of cells neighbouring each other, type: numpy matrix
    """
    pairwise_distance = squareform(pdist(loc))
    adjacency_matrix = np.where(pairwise_distance > threshold, 0, pairwise_distance)
    edge_indices = np.array(np.nonzero(np.triu(adjacency_matrix))).T

    return edge_indices


def read_to_anndata_wrapper(path_dir, is_sparse=False):
    """
    Wrapper for reading to AnnData.
    :param path_dir: input path, type: Path
    :param is_sparse: if sparse matrix, for transcriptomics, type: boolean
    :return: AnnData
    """
    y_df = pd.read_csv(path_dir / "y.csv")
    loc_df = pd.read_csv(path_dir / "loc.csv")

    adata = __read_to_anndata(y_df, loc_df, is_sparse)

    return adata
