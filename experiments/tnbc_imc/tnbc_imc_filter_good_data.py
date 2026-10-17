"""
@author: Tina Hsu
Filtering out the markers marked as fail or unclear by pathology and modifying data for effective downstream analysis
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import normalize

if __name__ == '__main__':
    # Set disease type and experiment to process
    TYPE = 'tnbc_imc'
    XP = 'XP.....'
    print(f"Running {XP}")

    # Set the directory for reading files and storage
    in_dir = out_dir = Path(f"data/{TYPE}/{XP}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in h5ad
    adata = read_h5ad(in_dir / f"{XP}.h5ad")

    # Get the subset of adata with common marker list
    # Store the processed h5ad
    adata.write(in_dir / f"{XP}_filter.h5ad")

