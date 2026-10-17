"""
@author: Tina Hsu
Label transfer
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from anndata import read_h5ad
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import plot_spatial_cluster_distribution, plot_spatial_cluster_panel, plot_cluster_mean_heatmap, \
    plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific'
    XP = 'cohort_tnbc_finalize_binary_relapse'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/8_tma_cohort")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in adata for tils analysis
    adata = read_h5ad(in_dir / f"{XP}.h5ad")

    """
    Code starts from here
    """
    # Sample cells from each cell type for classifier training data
    # Dict for cell type and its corresponding cell index
    cell_types = list(set(adata.obs.layer2))
    cell_type_dict = {}
    for ct in cell_types:
        cell_type_dict[ct] = list(np.where(adata.obs.layer2 == ct)[0])

    sampled_cells = {}
    np.random.seed(42)
    for celltype, cell_indices in cell_type_dict.items():
        sampled_cells[celltype] = np.random.choice(cell_indices, size=4000, replace=False)

    # Subset to lineage markers

    X = []
    y = []

    for celltype, cell_indices in sampled_cells.items():
        print(celltype)
        expr_data = adata[cell_indices, :].X.toarray()
        X.append(expr_data)
        y.extend([celltype] * len(cell_indices))

    X = np.vstack(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Train a classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Predict the cell types for the rest of the cells
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    max_probabilities = np.max(probabilities, axis=1)

    medium_score = np.median(max_probabilities)  # 0.85

    # Predict new data
    out_dir = Path('analysis/tnbc_imc/cohort_specific_8_tma/label_transfer')

    adata_all = read_h5ad(out_dir / "new_4_tma_all.h5ad")
    predictions = clf.predict(adata_all.X)
    adata_all.obs["layer2"] = predictions

    probabilities = clf.predict_proba(adata_all.X)
    max_probabilities = np.max(probabilities, axis=1)
    medium_score = np.median(max_probabilities)  # 0.77

    adata_all.uns['rf_labels_prob'] = max_probabilities

    """
    Plot
    """
    plot_out_dir = out_dir / "label_transfer"
    if os.path.exists(plot_out_dir) == 0:
        os.makedirs(plot_out_dir, exist_ok=True)

    # Plot current training dataset
    training_df = pd.DataFrame(adata_all.X, columns=adata_all.var_names)
    training_df.to_csv(plot_out_dir / "new_4_tma_classify.csv", index=False)

    plt.figure(figsize=(24, 16))
    sns.heatmap(training_df, cmap="coolwarm", vmax=5, vmin=-5)
    plt.xlabel("Cell Type")
    plt.ylabel("Protein Marker")
    plt.savefig(plot_out_dir / "new_classified_heatmap.png")

    CLUSTER_KEY = 'layer2'
    LAYER2_CELL_TYPE_COLOR_DICT = {'Epithelial CK5+': '#f10024',
                                   'Epithelial CK5+CK7+': '#ffb6c1',
                                   'Epithelial CK5+CK8-18+': '#ef38aa',
                                   'Epithelial CK7+': '#ae8ae7',
                                   'Epithelial CK8-18+': '#ef8616',
                                   'Epithelial CK_low': '#7639d6',
                                   'Epithelial H3K27me3+': '#d0be6b',
                                   'Ki67+': '#f9f1a4',
                                   'Myoepithelial': '#8e5112',
                                   'Myoepithelial H3K27me3+': '#f6d646',
                                   'TME Cluster 9': '#666666',
                                   'B Cell': '#178867',
                                   'T Cell': '#c3e78a',
                                   'Macrophage': '#9ac4a0',
                                   'Blood Endothelial': '#5ea3cd',
                                   'Fibroblast': '#bbe4fc',
                                   'Smooth Muscle Cell': '#01588d',
                                   'Vimentin+': '#92b6cb'
                                   }

    adata_all.obs[CLUSTER_KEY] = adata_all.obs[CLUSTER_KEY].astype("category")
    plot_zscore_cluster_mean_heatmap(adata_all, adata_all.var_names, CLUSTER_KEY, plot_out_dir, vmax=None, vmin=None,
                                     figsize=(7, 7), name=None)

    plot_zscore_cell_expression_heatmap(adata_all, adata_all.var_names, CLUSTER_KEY, plot_out_dir, figsize=(7, 7),
                                        rotation=35, color_dict=LAYER2_CELL_TYPE_COLOR_DICT,
                                        name="new_4_tma_classification",
                                        v_max=None, v_min=None)

    adata_all.write(Path('analysis/tnbc_imc/cohort_specific_8_tma') / "new_4_tma_all_layer2_classification.h5ad")
    adata_all.write(Path('analysis/tnbc_imc/cohort_specific_8_tma') / "new_4_tma_all.h5ad")
    adata_all.write(Path('analysis/tnbc_imc/summary') / "new_4_tma_all.h5ad")
