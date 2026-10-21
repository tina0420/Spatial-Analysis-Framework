"""
@author: Tina Hsu
Tutorial 1b: Clustering cells.
"""
from pathlib import Path
from anndata import read_h5ad
from src.annotate import run_leiden


if __name__ == '__main__':
    """
    We specify the experiment name here. This is to ease organizing files and folders.
    """
    run_name = "example"
    print(f"Processing {run_name}")

    """
    We start by setting paths to read and write.
    """
    out_dir = Path(f"example/output/{run_name}")

    """
    We want to read the AnnData object from the h5ad file.
    """
    adata = read_h5ad(out_dir / f"{run_name}.h5ad")

    """
    We can cluster cells based on expression. This will overcluster which we can process in the next step.
    """
    run_leiden(adata, columns=adata.var_names, n_neighbors=150, n_pcs=None, name="cluster")

    """
    We can now save the results by overwriting the h5ad file for annotation.
    """
    adata.write(out_dir / f"{run_name}.h5ad")
