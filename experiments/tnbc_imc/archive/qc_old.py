"""
@author: Tina Hsu
Example run TNBC files for preprocessing QuPath measurements and checking batch effect.
"""
import os
from pathlib import Path
from src.preprocess import read_to_anndata_wrapper, normalize, read_neighbors
from src.extract import quantify_from_qupath
from src.annotate import run_leiden
from experiments.tnbc_imc.archive.check_batch_effect import *
from experiments.tnbc_imc.archive.preprocessing import *
from experiments.tnbc_imc.archive.expression_and_intensity import *
from scipy import stats
from sklearn.mixture import GaussianMixture
from anndata import read_h5ad
from src.visualize import *


if __name__ == '__main__':
    # Which sample to run
    run_name = "XP12446"
    print(f"Processing {run_name}")

    # Set reading and writing paths
    path_dir = Path(f"data/tnbc_imc/{run_name}")
    out_dir = Path(f"figures/tnbc_imc/{run_name}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    """
    Process the measurements.csv to the format we want, y.csv and loc.csv
    """

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

    # Column names for the second naming type (for the segmented files by Yukta)
    # columns = ['Image', 'Cell: CK5(Pr141Di) mean',
    #    'Cell: CK19(Nd143Di) mean', 'Cell: CK8-18(Nd144Di) mean',
    #    'Cell: CD68(Nd146Di) mean', 'Cell: vWF(Sm147Di) mean',
    #    'Cell: SMA(Nd148Di) mean', 'Cell: Vimentin(Sm149Di) mean',
    #    'Cell: erbB2(Eu151Di) mean', 'Cell: CD3e(Sm152Di) mean',
    #    'Cell: Slug(Gd155Di) mean', 'Cell: EZH2(Gd156Di) mean',
    #    'Cell: Prostaglandin(Gd158Di) mean', 'Cell: p53(Tb159Di) mean',
    #    'Cell: H3K27me3(Gd160Di) mean', 'Cell: PR-SP2(Dy161Di) mean',
    #    'Cell: CD45(Dy162Di) mean', 'Cell: GATA3(Dy163Di) mean',
    #    'Cell: CD20(Dy164Di) mean', 'Cell: B-Catenin(Ho165Di) mean',
    #    'Cell: CA-IX(Er166Di) mean', 'Cell: ER-D8H8(Er167Di) mean',
    #    'Cell: Ki67(Er168Di) mean', 'Cell: EGFR(Tm169Di) mean',
    #    'Cell: ER-SP1(Yb171Di) mean', 'Cell: CD31(Yb172Di) mean',
    #    'Cell: CK7(Yb174Di) mean', 'Cell: panCK(Lu175Di) mean',
    #    ]

    # Extract the columns (markers) you need
    exp_df = exp_df[columns]

    # Rename columns names
    exp_df.columns = [columns[0]] + [x.split(" ")[1] for x in columns[1:]]

    # Output to the input structure for Sakura
    exp_df.to_csv(path_dir / "y.csv", index=False)
    loc_df.to_csv(path_dir / "loc.csv", index=False)

    """
    Start to do data preprocessing. (With y.csv and loc.csv)
    """

    # Read from y.csv and loc.csv to Anndata object
    adata = read_to_anndata_wrapper(path_dir=path_dir)

    xxx = read_to_anndata(path_dir=path_dir)

    # Normalization
    normalize(adata, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata, method="clip", clip_percentile=99.5)

    # Normalization here
    normalization(xxx, method="arcsinh", value_for_method=0.8)
    normalization(xxx, method="clip", value_for_method=99.5)

    # Check the results with different preprocess codes are the same or not
    for i in range(adata.X.shape[0]):
        for j in range(adata.X.shape[1]):
            if adata.X[i, j] != xxx.X[i, j]:
                print(i, j)

    # Check batch effect across all the images by PCA
    # pca_plot_batch_effect(out_dir, run_name+"_all", adata.X, image_df)
    pca_plot_batch_effect(out_dir, run_name + "_all", adata, exp_df["Image"])

    # Check batch effect across all the images by UMAP
    umap_plot_batch_effect(out_dir, run_name+"_all", adata, exp_df["Image"])

    # Get the index of each image
    image_index_dict = get_index_of_each_image(exp_df)

    # New order for the markers (start from the pass markers)
    cols = ['B-Catenin', 'CD31', 'CD45', 'CD68', 'CK7', 'CK8-18', 'H3K27me3',
            'Ki67', 'SMA', 'Vimentin', 'panCK', 'CD20', 'CD3e', 'CK5', 'CA-IX',
            'EGFR', 'EZH2', 'Prostaglandin', 'Slug', 'ER-D8H8', 'ER-SP1',
            'PR-SP2', 'erbB2', 'CK19', 'GATA3', 'vWF', 'p53']

    # Differentiate cell types by the specific markers
    markers = ['CD20', 'CD31', 'CD3e', 'CD68', 'CK5', 'CK7', 'SMA', 'panCK']
    neighbors_to_consider = 150
    run_leiden(adata, columns=markers, n_neighbors=neighbors_to_consider, n_pcs=None)
    #leiden_subset(xxx, columns=markers, n_neighbors=neighbors_to_consider, n_pcs=None)
    CLUSTER_KEY = "leiden_subset"
    adata.write(path_dir / f"{run_name}_here.h5ad")

    ##### Read the h5ad file #####

    adata = read_h5ad(path_dir / f"{run_name}_here.h5ad")

    # Define which labels to use as basis
    CLUSTER_KEY = "leiden_subset"

    run_leiden()
    zscore()

    plot_cluster_mean_heatmap(adata, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")

    plot_zscore_cluster_mean_heatmap(adata, markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="all")
    plot_zscore_cluster_mean_heatmap(xxx, markers, CLUSTER_KEY, out_dir=path_dir, figsize=(7, 7), name="all")

    ##############################################################
    #################### Test the first image ####################
    ##############################################################

    image_1 = '2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff'
    adata_a1 = adata[np.where(exp_df["Image"] == image_1)[0], :]
    plot_zscore_cluster_mean_heatmap(adata_a1, markers, CLUSTER_KEY, out_dir=out_dir, figsize=(7, 7), name="a1")

    # Test for more clustering (from 33 clusters)
    def more_clustering_for_cell_types(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None):
        pass

    # Expression DataFrame includes only expression data
    exp_df = pd.DataFrame(adata.X, columns=list(adata.var_names))
    exp_df = exp_df[cols]

    # The first image
    image_1 = "2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff"
    image_1_exp_df = exp_df.iloc[image_index_dict[image_1], :]     # (1782, 27)

    # Clustering and Annotation for cell types






    # Correlation for each cell type





    # All markers all images
    plot_all_markers_expression(out_dir=out_dir, exp_df=exp_df)






    # for each marker across different images (82 images)
    MARKER = "B-Catenin"
    marker_exp = pd.concat([test[MARKER], exp_df["Image"]], axis=1)
    ### useless: group_test = marker_exp.groupby("Image")


    plt.figure(figsize=(20,15))
    #sns.scatterplot(data=marker_exp, x="Image", y="B-Catenin")
    sns.boxplot(data=exp_df, x="Image", y="B-Catenin")
    plt.xticks([x for x in range(0, len(set(exp_df["Image"])))])
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()

    image_1 = "2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff"
    image_2 = "2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a2_ac.ome.tiff"

    rho, p_value = stats.spearmanr(np.array(exp_df[exp_df["Image"] == image_1][MARKER]), np.array(exp_df[exp_df["Image"] == image_2][MARKER]), axis=1)


    # for image 1
    image_index_dict = get_index_of_each_image(exp_df)
    image_1_exp = test.iloc[image_index_dict[image_1], :]
    # FOR IMAGE 2

    image_2_exp = test.iloc[image_index_dict[image_2], :]

    # corr/heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(image_1_exp_wo_nor.corr(method="spearman"), cmap="YlGnBu", annot=True)
    plt.title("Correlation across all markers", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.savefig(out_dir / "Correlation_heatmap_all_markers_all_images.png", dpi=500)
    # plt.clf()
    plt.show()


    ################## For testing ############################

    # distribution of each marker
    MARKER = "B-Catenin"
    testtest = np.array(image_1_exp[MARKER])
    sd = np.std(testtest)
    mean = np.mean(testtest)
    sd / mean *100

    for i in range(image_1_exp.shape[1]):
        MARKER = image_1_exp.columns[i]
        print(MARKER)
        testtest = np.array(image_1_exp[MARKER])
        sd = np.std(testtest)
        mean = np.mean(testtest)
        print(sd / mean * 100)
        print("---------------------------------------------------")

    plt.figure(figsize=(20, 15))
    sns.violinplot(data=test)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)
    plt.xlabel("Marker", fontsize=25)
    plt.ylabel("Intensity", fontsize=25)
    plt.title("Expression of all the markers", fontsize=30)
    plt.grid(alpha=0.1)
    plt.tight_layout()
    plt.show()

    # Hierachical Clustering
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    X = np.array(image_1_exp["B-Catenin"])
    clustering = AgglomerativeClustering().fit(X)
    clustering
    AgglomerativeClustering()
    clustering.labels_


    # Gaussian mixture metric
    gmm = GaussianMixture(n_components=1)
    gmm.fit(image_1_exp["B-Catenin"])

    # predictions from gmm
    labels = gmm.predict(image_1_exp["B-Catenin"])
    frame = pd.DataFrame(image_1_exp["B-Catenin"])
    frame['cluster'] = labels
    frame.columns = ['Weight', 'Height', 'cluster']

    color = ['blue', 'green', 'cyan', 'black']
    for k in range(0, 4):
        data = frame[frame["cluster"] == k]
        plt.scatter(data["Weight"], data["Height"], c=color[k])
    plt.show()



    testtest = pd.melt(test, var_name="Marker", value_name="Intensity")
    plt.grid(alpha=0.1)
    plt.show()

    # Separate the file of individual image
    # Get the index



    indices_2 = get_index_of_each_image(loc_df)

    # Create Adata with individual image
    ### test: image a1




    # For loop for each image, extract AnnData by index (image name -> index -> data)
    for image in set(loc_df.Image):
        print(image)

    # Read spatial neighbours if exists, otherwise create neighbors using threshold defined
    read_neighbors(adata, rel_path=path_dir / "rel.csv", threshold=30)




    # test for pca with more n_component

    def pca_plot_batch_effect(out_dir, file_name, exp_df, image_name_df):
        """
        Run and plot PCA for checking batch effect
        :param out_dir: Directory to store the output image, type: Path
        :param file_name: Name of the file, type: Str
        :param exp_df: Expression data to run, type: pd.DataFrame
        :param image_name_df: Image name to concat, type: pd.DataFrame
        :return: Output an image
        """
        # Prepare data
        pca = PCA(n_components=6, random_state=42)
        transform_df = pca.fit_transform(exp_df)
        transform_df = pd.DataFrame(transform_df)
        pca_df = pd.concat([image_name_df, transform_df], axis=1)
        pca_df.columns = ["Image", "PC1", "PC2"]

        # Plot
        plt.figure(figsize=(12, 9))
        plt.title("PCA for batch effect", fontsize=30)
        plt.xlabel("PC1", fontsize=20)
        plt.ylabel("PC2", fontsize=20)
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Image")
        plt.tight_layout()


        # using adata.X
        pca = PCA(n_components=6, random_state=42)
        transform_df = pca.fit_transform(adata.X)
        transform_df = pd.DataFrame(transform_df)
        pca_df = pd.concat([image_df, transform_df], axis=1)
        pca_df.columns = [""]

        plt.figure(figsize=(12, 9))
        plt.title("PCA for batch effect", fontsize=30)
        plt.xlabel("PC5", fontsize=20)
        plt.ylabel("PC6", fontsize=20)
        sns.scatterplot(data=pca_df, x="PC5", y="PC6", hue="Image")
        plt.tight_layout()
        plt.show()


    ########## Expression ##########

    # Mean of each marker
    markers_mean = adata.X.mean

    test = pd.DataFrame(adata.X)
    for column in test:
        plt.figure()
        test.boxplot([column])
        plt.show()


    # New order for the markers (start from the pass markers)
    cols = ['B-Catenin', 'CD31', 'CD45', 'CD68', 'CK7', 'CK8-18', 'H3K27me3', 'Ki67',
            'SMA', 'Vimentin', 'panCK', 'CD20', 'CD3e', 'CK5', 'CA-IX',
            'EGFR', 'EZH2', 'Prostaglandin', 'Slug',  'ER-D8H8', 'ER-SP1',
            'PR-SP2', 'erbB2', 'CK19', 'GATA3', 'vWF', 'p53']

    test = pd.DataFrame(adata.X, columns=list(adata.var_names))
    test = test[cols]


    plot_all_markers_expression(out_dir=out_dir, exp_df=test)

    # However, each marker has its own intensity


    image_index_dict = get_index_of_each_image(exp_df)


    # test for the first image
    test_a1 = test.iloc[image_index_dict["2024-07-18_TNBC-TMA-AT19296_BOB_TNBC_Brian_BMplus-FP_s0_a1_ac.ome.tiff"], :]

    # test expression across different images (focus one marker, B-Catenin)
    test_across = test

    test_across_beta = test_across[["B-Catenin", "Image"]]
    plt.figure(figsize=(20, 15))
    sns.boxplot(data=test_across_beta, x="Image", y="B-Catenin")
    plt.show()


    rho, p_value = stats.spearmanr(np.array(exp_df[["ER-SP1", "PR-SP2", "erbB2"]].T), axis=1)
    plt.figure(figsize=(12, 9))
    sns.heatmap(pd.DataFrame(rho, columns=["ER-SP1", "PR-SP2", "erbB2"], index=["ER-SP1", "PR-SP2", "erbB2"]),
                cmap="YlGnBu", annot=True, annot_kws={'size': 25})
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #plt.show()
    plt.tight_layout()
    plt.savefig(out_dir / "three_markers_heatmap.png", dpi=500)


    sns.heatmap(image_1_exp.corr(method="spearman"), cmap="YlGnBu", annot=True, vmax=1, vmin=-1)
    plt.title("Correlation across all markers", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.savefig(out_dir / "Correlation_heatmap_all_markers_all_images.png", dpi=500)
    # plt.clf()
    plt.show()


    ############################# Testing starts from here #############################








