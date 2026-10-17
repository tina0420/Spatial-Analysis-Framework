"""
@author: Tina Hsu
Example run TNBC files for preprocessing QuPath measurements and checking batch effect.
"""
import os
from pathlib import Path
from src.preprocess import read_to_anndata_wrapper, normalize
from src.extract import quantify_from_qupath
from src.annotate import run_leiden
from experiments.tnbc_imc.archive.check_batch_effect import *
from experiments.tnbc_imc.archive.preprocessing import *
from anndata import read_h5ad
from src.visualize import *
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # Which sample to run
    run_name = "XP12446"
    print(f"Processing {run_name}")

    # Set reading and writing paths
    path_dir = Path(f"data/tnbc_imc/{run_name}")
    out_dir = Path(f"figures/tnbc_imc/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Tidy the measurenemts.csv to the format we want
    # Process measurements.csv from QuPath
    exp_df, loc_df = quantify_from_qupath(path_dir)

    # Choose the columns
    columns = ['Image', 'Cell: CK5 mean',
               'Cell: CK19 mean', 'Cell: CK8-18 mean',
               'Cell: CD68 mean', 'Cell: vWF mean',
               'Cell: SMA mean', 'Cell: Vimentin mean',
               'Cell: erbB2 mean', 'Cell: CD3e mean',
               'Cell: Slug mean', 'Cell: EZH2 mean',
               'Cell: Prostaglandin mean', 'Cell: p53 mean',
               'Cell: H3K27me3 mean', 'Cell: PR-SP2 mean',
               'Cell: CD45 mean', 'Cell: GATA3 mean',
               'Cell: CD20 mean', 'Cell: B-Catenin mean',
               'Cell: CA-IX mean', 'Cell: ER-D8H8 mean',
               'Cell: Ki67 mean', 'Cell: EGFR mean',
               'Cell: ER-SP1 mean', 'Cell: CD31 mean',
               'Cell: CK7 mean', 'Cell: panCK mean',
               ]

    # Extract the columns (markers) you need
    exp_df = exp_df[columns]

    # Rename columns names
    exp_df.columns = [columns[0]] + [x.split(" ")[1] for x in columns[1:]]

    # Output to the input structure for Sakura
    exp_df.to_csv(path_dir / "y.csv", index=False)
    loc_df.to_csv(path_dir / "loc.csv", index=False)

    # Preprocess
    # Read from y.csv and loc.csv to Anndata object
    adata = read_to_anndata_wrapper(path_dir=path_dir)

    # Normalization
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    # Check batch effect across all the images by PCA
    pca_plot_batch_effect(out_dir, run_name + "_all", adata, exp_df["Image"])

    # Check batch effect across all the images by UMAP
    umap_plot_batch_effect(out_dir, run_name + "_all", adata, exp_df["Image"])

    # Get the index of each image
    image_index_dict = get_index_of_each_image(exp_df)

    # Differentiate cell types by the specific markers
    markers = ['CD20', 'CD31', 'CD3e', 'CD68', 'CK5', 'CK7', 'SMA', 'panCK']
    neighbors_to_consider = 150
    run_leiden(adata, columns=markers, n_neighbors=neighbors_to_consider, n_pcs=None)

    # Output the h5ad file
    adata.write(path_dir / f"{run_name}.h5ad")

    # Read the h5ad file and start to do qc
    adata = read_h5ad(path_dir / f"{run_name}.h5ad")

    # Plot the distribution of the clusters before doing classification (around 30 clusters)
    CLUSTER_KEY = "leiden_subset"
    plot_spatial_cluster_distribution(adata=adata, cluster_key=CLUSTER_KEY, out_dir=out_dir,
                                      dotsize=1, figsize=(12, 9), name="all")

    plot_zscore_cluster_mean_heatmap(adata, markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")

    # Classification for cell types
    def plot_classification_three_classes_scatter(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None):
        # given columns, get column index
        idxs = []
        for i in column_names:
            for j in range(len(adata.var_names)):
                if i == adata.var_names[j]:
                    idxs.append(j)

        # create zscore dataframe
        if issparse(adata.X):
            exp_mtx = adata.X[:, idxs].toarray()
        else:
            exp_mtx = adata.X[:, idxs]

        for col in range(exp_mtx.shape[1]):
            # check for divide by zero
            if exp_mtx[:, col].sum() == 0.0:
                continue
            exp_mtx[:, col] = zscore(exp_mtx[:, col])

        #print(exp_mtx)
        #print(exp_mtx.shape)

        labels = adata.obs[cluster_key].values
        # print(labels)
        # print(set(labels))
        label_categories = sorted(list(pd.unique(labels)))
        cluster_mean = np.array([np.mean(exp_mtx[np.where(labels == i)[0]], axis=0) for i in label_categories])
        cluster_df = pd.DataFrame(cluster_mean, columns=adata.var_names[idxs], index=label_categories)
        print(cluster_df)

        # Classification
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_df['Cluster'] = kmeans.fit_predict(cluster_df)
        # AttributeError: 'NoneType' object has no attribute 'split' ### Not solved yet
        print(cluster_df)

    plot_classification_three_classes_scatter(adata, markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")





















