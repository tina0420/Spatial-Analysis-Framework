"""
New data merging
"""
from experiments.tnbc_imc.tnbc_imc_setup_xp import COHORT_TMA_XP
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import scanpy as sc

if __name__ == "__main__":
    TYPE = "tnbc_imc"

    in_dir = Path(f"data/{TYPE}/")
    out_dir = Path(f'data/{TYPE}/final_thesis')
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Get all the xps and load
    xp_list = list(COHORT_TMA_XP.values())
    adata_list = [read_h5ad(in_dir / xp / f"{xp}.h5ad") for xp in xp_list]

    #  Extract the common for each xp
    COMMON_MARKERS = ['CK5', 'CK7', 'CK8-18', 'panCK', 'B-Catenin', 'H3K27me3', 'Ki67',
                      'GATA3', 'CD3e', 'CD20', 'CD45', 'CD68', 'CD31', 'SMA', 'Vimentin',
                      'vWF']

    for i in range(len(adata_list)):
        adata_list[i] = adata_list[i][:, COMMON_MARKERS]

    # Modify the image_id to a consistent naming convention (ROI_XXX), different adata needs different ways to split
    for i in range(len(adata_list)):
        # print(i)
        # print(list(set(adata_list[i].obs.image_id)))
        if (i == 0) or (i == 2) or (i == 3):
            adata_list[i].obs.image_id = ["ROI_" + x.split(".")[0].split("_")[2] for x in
                                          adata_list[i].obs.image_id.values]
        elif i == 1:
            adata_list[i].obs.image_id = ["ROI_" + x.split(".")[0].split("_")[3] for x in
                                          adata_list[i].obs.image_id.values]
        elif (i == 4) or (i == 5) or (i == 6) or (i == 7):
            adata_list[i].obs.image_id = ["ROI_" + x.split(".")[0].split("_")[0][3:].zfill(3) for x in
                                          adata_list[i].obs.image_id.values]

    """
    Merge all experiments together, need to add or delete based on the experiment number
    """
    merge_adata = adata_list[0].concatenate(adata_list[1], adata_list[2], adata_list[3], adata_list[4],
                                            adata_list[5], adata_list[6], adata_list[7], batch_key="TMA")

    merge_adata.obs["TMA"] = merge_adata.obs["TMA"].astype(int)
    mapping_dict = dict(zip(range(len(xp_list)), COHORT_TMA_XP.keys()))
    merge_adata.obs["TMA"] = [mapping_dict[x] for x in merge_adata.obs["TMA"].values]

    # Add "Image ID" to adata.obs (TMA_image_id)
    merge_adata.obs["unique_img_key"] = [merge_adata.obs["TMA"].values[i] + "_" + merge_adata.obs["image_id"].values[i]
                                         for i in range(merge_adata.shape[0])]

    merge_adata.write(out_dir / "all_xp_8_tma_merge.h5ad")

    # Select the images in the cohort
    # Read in the metadata
    metadata = pd.read_csv(out_dir / "patient_imc_final_metadata.csv")

    imc_imgs = [x for x in metadata["Image ID"].values if pd.isna(x) == False]

    cohort_adata = merge_adata[merge_adata.obs["unique_img_key"].isin(imc_imgs)].copy()

    # Add the metadata to the adata.obs
    original = cohort_adata.obs.copy()
    new_obs = pd.merge(original, metadata.iloc[:, [0, 1, 2, 3, 4, 7, 9, 11]], on="unique_img_key", how='left')

    cohort_adata.obs = new_obs
    cohort_adata.write(out_dir / "cohort_all_img.h5ad")
    # Select the images in the cohort which are from pre-treatment tissues
    pre_adata = cohort_adata[cohort_adata.obs["collection_type"] == 0].copy()

    pre_adata.write(out_dir / "cohort_pretreatment_img.h5ad")

    # # Make the values into categories (0, 1, 2)
    # mapping_dict = {'ADJ': 0, "NAT": 1}
    # metadata["Chemo type"] = [mapping_dict[x] for x in metadata["Chemo type"].values]
    #
    # mapping_dict = {'No': 0, "Yes": 1}
    # metadata["pCR"] = metadata["pCR"].replace(mapping_dict).fillna(pd.NA)
    #
    # mapping_dict = {'Local': 1,
    #                 'Local and metastatic': 1,
    #                 'Metastatic': 1,
    #                 'No recurrence': 0,
    #                 'new cancer': 1}
    # metadata["Recurrence status"] = [mapping_dict[x] for x in metadata["Recurrence status"].values]
    #
    # mapping_dict = {"pre_treatment": 0, "post_treatment": 1}
    # metadata["collection_type"] = metadata["collection_type"].replace(mapping_dict).fillna(pd.NA)
    #
    # metadata.to_csv(out_dir / "patient_imc_final_metadata_category.csv", index=False)
    # metadata.columns = ['CID', 'Chemo type', 'pCR', 'Recurrence status', 'collection_type',
    #                     'Accession', 'TMA', 'Experiment', 'Core ID', 'Core location',
    #                     'Core image', 'unique_img_key']
