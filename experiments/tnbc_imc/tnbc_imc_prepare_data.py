"""
Preparing data for downstream analysis, e.g. merging several experiments together.
"""
import os
from pathlib import Path
from anndata import read_h5ad
from src.preprocess import normalize
import pandas as pd
import numpy as np

if __name__ == "__mian__":
    # Set disease type and which experiment to process
    TYPE = "tnbc_imc"
    ORIGINAL_XP = "cohort_specific"
    XP = 'cohort_specific_5_tma'

    in_dir = Path(f"analysis/{TYPE}/{ORIGINAL_XP}")
    out_dir = Path(f"analysis/{TYPE}/{XP}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Merging the original h5ad with the new images scanned

    # Original one
    adata_4_tma = read_h5ad(in_dir / "cohort_tnbc_initial_4_tma.h5ad")

    # new generated experiment (modify and add new columns in obs)
    adata_new_all = read_h5ad(Path("data/tnbc_imc/XP17579") / "XP17579.h5ad")
    adata_new_all.obs.image_id = [x.split(".")[0] for x in adata_new_all.obs.image_id.values]
    adata_new_all.obs["tma"] = ["23-005"] * adata_new_all.shape[0]
    adata_new_all.obs["image_key"] = ["23-005_" + adata_new_all.obs.image_id.values[i].split("_")[1] for i in
                                      range(adata_new_all.shape[0])]

    # normalization
    normalize(adata_new_all, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata_new_all, method="clip", clip_percentile=99.5)

    adata_new_all.write(Path("data/tnbc_imc/XP17579") / "XP17579_filter.h5ad")

    # Use image name to find the target rois in the all new experiment
    # necessary patients in the original adata (TNBC and pre-treatment)
    # Read in the patient info
    patient_df = pd.read_csv(Path("analysis/tnbc_imc/summary") / "ctdna_patient_info_5_tmas_finalize.csv")

    adata_new = adata_new_all[
        np.where(adata_new_all.obs["image_id"].isin(list(patient_df.image_1)))[0], adata_4_tma.var_names]

    new_xp_df = patient_df[patient_df["tma"] == "23-005"]
    mapping = dict(zip(list(new_xp_df.image_1.values),
                       [new_xp_df.cid.values[i] + "_" + new_xp_df.accession.values[i] for i in
                        range(new_xp_df.shape[0])]))
    adata_new.obs["patient_key"] = [mapping[x] for x in adata_new.obs.image_id.values]

    adata_new.write(out_dir / "XP17579_to_add.h5ad")

    # Get the target patients' data from the original adata (36 --> 24)
    original_rois = [x for x in patient_df.image_1.values if pd.isna(x) == False]
    original_rois += [x for x in patient_df.image_2.values if pd.isna(x) == False]

    # 53 rois from the previous tmas
    adata_original = adata_4_tma[np.where(adata_4_tma.obs.image_key.isin(original_rois))[0], :]

    # if the precious one didn't do normalization then do
    normalize(adata_original, method="arcsinh", arcsinh_cofactor=0.8)
    normalize(adata_original, method="clip", clip_percentile=99.5)

    adata_original.write(out_dir / "previous_4_tma.h5ad")

    # Merge them together
    adata = adata_original.concatenate(adata_new)
    adata.obs.drop(columns=["batch"], inplace=True)

    adata.write(out_dir / f"{XP}_new_7.h5ad")

    adata = read_h5ad(out_dir / f"{XP}_new_7.h5ad")
    # Add come clinical info to the adata.obs
    adata.obs["cid"] = [x.split("_")[0] for x in adata.obs.patient_key.values]

    mapping = {}
    for i in range(patient_df.shape[0]):
        mapping[patient_df.cid.values[i]] = list(patient_df.iloc[i, 10:25]) + [patient_df.iloc[i, 29]]

    to_add = []
    for i in range(adata.shape[0]):
        pid = adata.obs.cid.values[i]
        to_add.append(mapping[pid])

    to_add_df = pd.DataFrame(to_add, columns=['death', 'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
                                              'hr_status', 'biomarke_check', 'biomarkers', 'stage', 'tumour_size_cm',
                                              'grade', 'node_status', 'lvi', 'margin_status', 'collection_type'])

    # Add them into adata.obs
    for col in to_add_df.columns:
        print(col)
        adata.obs[col] = list(to_add_df[col].values)

    adata.write(out_dir / f"{XP}_new_7.h5ad")
    adata.write(out_dir / f"{XP}_new_7_finalize.h5ad")

    # Batch correction
    import scanpy as sc
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)

    from src.visualize import plot_umap, plot_umap_panel
    CLUSTER_KEY = "tma"
    TMA_COLOR_DICT = {'BOB': 'tab:blue',
                      '22-010A': 'tab:pink',
                      '22-010C': 'tab:orange',
                      '23-009': 'tab:green',
                      '23-005': 'tab:brown'}
    plot_umap(adata, CLUSTER_KEY, out_dir, color_dict=TMA_COLOR_DICT,
              figsize=(4.5, 4.5), dotsize=0.5, name=XP)
    plot_umap_panel(adata, CLUSTER_KEY, out_dir, color_dict=TMA_COLOR_DICT,
                    dotsize=0.1, name=XP)

    plot_marker_expression_distribution_violin(adata, CLUSTER_KEY, out_dir, TMA_COLOR_DICT,
                                               set_count=4, figsize=(8, 4), name=XP)

    # Extra work: rename the image_key to TMA_RnCn based on the info of core_loc_1 and 2
    # Wait not sure if this is good cuz the previous experiment stored with different format
    # first make the core_loc to upper case
    patient_df['core_loc_1'] = patient_df['core_loc_1'].str.upper()
    patient_df['core_loc_2'] = patient_df['core_loc_2'].str.upper()

    patient_df.to_csv(Path("analysis/tnbc_imc/summary") / "ctdna_patient_info_5_tmas_finalize.csv", index=False)

    mapping = {}
    for idx in range(patient_df.shape[0]):
        if pd.isna(patient_df.image_1.values[idx]) == False:
            mapping[patient_df.image_1.values[idx]] = patient_df.core_loc_1.values[idx]



