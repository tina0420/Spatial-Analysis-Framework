"""
@author: Tina Hsu
Quality control for TNBC IMC dataset.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from experiments.tnbc_imc.archive.preprocessing import preprocess
from experiments.tnbc_imc.clustering import clustering
from experiments.tnbc_imc.archive.expression_based import check_correlation

if __name__ == "__main__":
    experiments = []
    for experiment in experiments:
        print(f"Processing {experiment}")

        path_dir = Path(f"data/tnbc_imc/{experiment}")
        out_dir = Path(f"qc_results/tnbc_imc/{experiment}")
        if os.path.exists(out_dir) == 0:
            os.makedirs(out_dir, exist_ok=True)

        print("Step1: Preprocessing")
        preprocess(experiment)

        experiment = "XP12446"

        print("Step2: Clustering for cell types")
        adata = read_h5ad(path_dir / f"{experiment}.h5ad")

        # Differentiate cell types by the specific  (leiden clustering)
        markers = ['CD20', 'CD31', 'CD3e', 'CD68', 'SMA', 'panCK']
        neighbors_to_consider = 200

        clustering(experiment, adata)
        #identify_cell_types()

        print("Step3: Expression-based exploration")
        CLUSTER_KEY = "3_types_labels"
        check_correlation()








