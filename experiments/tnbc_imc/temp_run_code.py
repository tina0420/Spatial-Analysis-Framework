import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np

if __name__ == '__main__':
    TYPE = 'tnbc_imc'
    TMA = 'full_4_tma'

    # Data directory
    in_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    out_dir = Path(f"analysis/{TYPE}/ctdna_4_tma/results")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    patient_df = pd.read_csv(in_dir / "full_4_tma_cohort_patient_only_reset_index.csv")[
        ["TMA", "CID", 'Core img_1', 'Core img_2', 'relapse']]

    # Read in cohort only h5ad
    adata = read_h5ad(in_dir / "full_4_tma_layer2_finalize.h5ad")

    # get the subset of the patient cohort only and store in a new h5ad
    # add new column for the adata.obs, cell_id (TMA_ROI#) and CID
    cell_ids = []
    cid_nums = []
    # for i in range(adata.shape[0]):
    #     img = patient_df.iloc[i, :]["Core img_1"]
    #     if pd.isna(img):
    #         img = patient_df.iloc[i, :]["Core img_2"]
    #
    #     patient_obs = adata[(adata.obs.batch == patient_df.iloc[i, :]["TMA"]) &
    #                         (adata.obs['image_id'] == img)].obs[CLUSTER_KEY]
    #
    #     cell_ids.append(patient_df.iloc[i, :]["TMA"] + "_" + img)

    [patient_df.iloc[i, :]["TMA"] + "_" + patient_df.iloc[i, :]["Core img_1"] for i in range]

    # generate the figures first
    adata_patient = read_h5ad(in_dir / "full_4_tma_cohort_patient_only.h5ad")

    # Plot the heatmap of cell types in layer 1 and layer 2
    from src.visualize import plot_zscore_cell_expression_heatmap, plot_zscore_cluster_mean_heatmap

    # modify the name of the epithelial cell type, luminal -> Luminal
    ct_name_dict = dict(zip(['B cell',
                             'Blood endothelial',
                             'Macrophage',
                             'Smooth muscle cell',
                             'T cell',
                             'Undefined',
                             'Undefined immune',
                             'Undefined stromal',
                             'Vimentin',
                             'luminal',
                             'myoepithelial',
                             'undefined_epithelial'],
                            ['B cell',
                             'Blood endothelial',
                             'Macrophage',
                             'Smooth muscle cell',
                             'T cell',
                             'Undefined',
                             'Undefined immune',
                             'Undefined stromal',
                             'Vimentin',
                             'Luminal',
                             'Myoepithelial',
                             'Undefined epithelial']))

    ct_name_dict = dict(zip(['epithelial', 'immune', 'stromal'],
                            ['Epithelial', 'Immune', 'Stromal']))

    adata_patient.obs.layer2_classify = [ct_name_dict[ct] for ct in adata_patient.obs.layer2_classify]
    adata_patient.obs.layer1_classify = [ct_name_dict[ct] for ct in adata_patient.obs.layer1_classify]

    adata_patient.write(in_dir / "full_4_tma_cohort_patient_only.h5ad")

    # plot heatmap for cell types
    CLUSTER_KEY = 'layer1_classify'
    plot_zscore_cell_expression_heatmap(adata_patient, adata_patient.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name="layer1_cells_exp")
    plot_zscore_cluster_mean_heatmap(adata_patient, adata_patient.var_names, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer1_cell_type_exp")

    CLUSTER_KEY = 'layer2_classify'
    plot_zscore_cell_expression_heatmap(adata_patient, adata_patient.var_names, CLUSTER_KEY, out_dir, figsize=(7, 7),
                                        rotation=35, name="layer2_cells_exp")
    plot_zscore_cluster_mean_heatmap(adata_patient, adata_patient.var_names, CLUSTER_KEY,
                                     out_dir=out_dir, figsize=(7, 7), name="layer2_cell_type_exp")

    # plot cell type distribution
    from src.visualize import plot_spatial_cluster_distribution, COLOR_CODE

    cell_ids = []
    for i in range(patient_df.shape[0]):
        # Use first image, if NaN then use the second image
        img = patient_df.iloc[i, :]["Core img_1"]
        if pd.isna(img):
            img = patient_df.iloc[i, :]["Core img_2"]

        cell_ids.append(patient_df.iloc[i, :]["TMA"] + "_" + img)

    patient_df['img_key'] = cell_ids

    from experiments.tnbc_imc.tnbc_imc_setup_analysis import ESI_CELL_TYPE_COLOR_DICT, LAYER2_CELL_TYPE_COLOR_DICT
    CLUSTER_KEY = 'layer1_classify'
    for i in range(patient_df.shape[0]):
        pid = patient_df.iloc[i, -1]
        print(pid)
        tma = pid.split("_")[0]
        roi = pid.split("_")[1] + "_" + pid.split("_")[2]

        temp_adata = adata_patient[
                     np.where((adata_patient.obs.batch == tma) & (adata_patient.obs['image_id'] == roi))[0], :]
        # plot distribution

        plot_spatial_cluster_distribution(temp_adata, CLUSTER_KEY, out_dir, figsize=(5, 5), dotsize=2,
                                          color=ESI_CELL_TYPE_COLOR_DICT,
                                          name=f"{tma}_{roi}_layer1")

    # for layer 2
    CLUSTER_KEY = 'layer2_classify'
    for i in range(patient_df.shape[0]):
        pid = patient_df.iloc[i, -1]
        print(pid)
        tma = pid.split("_")[0]
        roi = pid.split("_")[1] + "_" + pid.split("_")[2]

        temp_adata = adata_patient[
                     np.where((adata_patient.obs.batch == tma) & (adata_patient.obs['image_id'] == roi))[0], :]
        # plot distribution

        plot_spatial_cluster_distribution(temp_adata, CLUSTER_KEY, out_dir / "22-010A", figsize=(5.2, 5), dotsize=1,
                                          color=LAYER2_CELL_TYPE_COLOR_DICT,
                                          name=f"{tma}_{roi}_layer2")


    from src.visualize import plot_cluster_cell_count

    plot_cluster_cell_count(adata_patient[np.where(adata_patient), :], "layer2_classify", out_dir / "22-010A", figsize=(3.5, 2), name=None)


