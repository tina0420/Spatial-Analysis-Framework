"""
@author: Tina Hsu
Summary for the data (for thesis and ctDNA TNBC paper)
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from anndata import read_h5ad

if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = "tnbc_imc"
    PURPOSE = 'cohort_specific'

    in_dir = Path(f"analysis/{TYPE}")
    out_dir = Path(f"analysis/{TYPE}/summary")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in the cohort information and metadata
    # cohort information (all pid related to recurrence and naj)
    rd_df = pd.read_csv(in_dir / "RD_cases.csv")
    # tma info with which pid on which tma
    tma_info = pd.read_csv(in_dir / "ctDNA_cohort.csv")
    # clinical metadata of the patients (status and any date info, ~10 for only cid and dates)
    metadata = pd.read_csv(in_dir / "For_imc.csv")

    # How many patients on/not on the TMAs?
    # A: 117 on the TMAs and 11 are not.
    on_tma_list = []
    not_on_tma_list = []
    for pid in rd_df.CID:
        if pid in tma_info.Name.values:
            on_tma_list.append(pid)
        else:
            not_on_tma_list.append(pid)

    rd_df[rd_df.CID.isin(not_on_tma_list)].to_csv(out_dir / "pid_not_on_tma.csv", index=False)

    # Check how many patients have IMC now?
    # A:                Current answer is 51  or 49

    # need to start form h5ad file (XPs).
    # image_id -> image_key -> core_id -> pid|accession (sectormap) -> patient_key
    # then compute the number of unique pid in patient_key and that's the answer

    # image_id to image_key, and suddently found there's already image_key and patient_key.
    # need to check 22-010A since we changed the experiment before (to XP12813)
    adata_all_4_tma = read_h5ad(in_dir / "ctdna_cohort/tnbc_imc_all_4_xp.h5ad")  # 540308 x 16
    patient_info = pd.read_csv(in_dir / "ctdna_cohort/ctdna_patients_relapse_primary.csv")
    patient_clinical_data = pd.read_csv(
        in_dir / "ctdna_cohort/imc_pat_info.csv")  # From Elena, metadata from the imc patient list

    # # for checking 22-010A
    # # check image_key, correct!
    # set(adata_all_4_tma.obs.apply(lambda r: str(r["image_key"]).startswith(str(r["tma"])), axis=1))
    # sub_adata = adata_all_4_tma[adata_all_4_tma.obs.tma == "22-010A"].obs.iloc[:, :4].reset_index(drop=True)
    # sub_adata["image_key"].str.startswith(sub_adata["tma"])

    # Update the recurrence status to the patient info table because we're filtering out the patients have no images,
    # in case some recurrence patients were filtered out
    patient_info = patient_info.merge(patient_clinical_data, left_on="cid", right_on="CID", how="left")
    patient_info = patient_info.drop(columns=["relapse", "CID"])

    patient_info.columns = ['cid', 'accession', 'block_num', 'tma', 'core_id', 'core_loc_1',
                            'core_loc_2', 'image_1', 'image_2', 'patient_key', 'death',
                            'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
                            'hr_status', 'biomarke_check', 'biomarkers', 'stage',
                            'tumour_size_cm', 'grade', 'node_status', 'lvi', 'margin_status']
    patient_info.to_csv(in_dir / "ctdna_cohort/ctdna_patients_relapse_primary.csv", index=False)

    # add date info
    collection_df = pd.read_excel(in_dir / "20251103_tnbc_ffpe_collection_dates.xlsx")
    collection_date = []
    for i in range(patient_info.shape[0]):
        # get pid and accession
        pid = patient_info.iloc[i, 0]
        path_id = patient_info.iloc[i, 1]
        if path_id in collection_df["Path case"].values:
            collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                 (collection_df["Path case"] == path_id)]["Collection date"].values[
                                       0])
        else:
            collection_date.append("")
            print(i)

    patient_info["collection_date"] = collection_date
    # others
    to_add_dates = []
    for pid in patient_info.cid.values:
        print(pid)
        to_add_dates.append(metadata[metadata.CID == pid].iloc[0, 7:10].values)

    patient_info[["primary_surgery_date", "primary_chemo_start", "last_primary_chemo"]] = np.vstack(to_add_dates)

    patient_info["collection_date"] = pd.to_datetime(patient_info["collection_date"], format='%Y-%m-%d')
    patient_info["primary_surgery_date"] = pd.to_datetime(patient_info["primary_surgery_date"], format='%Y-%m-%d')
    patient_info["primary_chemo_start"] = pd.to_datetime(patient_info["primary_chemo_start"], format='%Y-%m-%d')
    patient_info["last_primary_chemo"] = pd.to_datetime(patient_info["last_primary_chemo"], format='%Y-%m-%d')

    # Compare the time of therapy start date and the date of tissue collection
    pre_treat_idx = np.where(patient_info["collection_date"] <= patient_info["primary_chemo_start"])[0]
    collection_type = np.zeros(patient_info.shape[0], dtype='object')
    collection_type[:] = np.nan
    collection_type[pre_treat_idx] = np.repeat("pre_treatment", len(pre_treat_idx))

    # not pre-treatment
    post_treat_idx = np.where(patient_info["collection_date"] > patient_info["primary_chemo_start"])[0]
    collection_type[post_treat_idx] = np.repeat("post_treatment", len(post_treat_idx))

    # add the pre_treatment and post_treatment info to the df
    patient_info["collection_type"] = collection_type

    patient_info.to_csv(out_dir / "ctdna_patients_info_4_tmas_all.csv", index=False)

    # filter out the patients with no images (they should have at least one image)
    imc_patient_df = patient_info[
        (pd.isna(patient_info["image_1"]) == False) | (pd.isna(patient_info["image_2"]) == False)].reset_index(
        drop=True)

    imc_patient_df.to_csv(out_dir / "ctdna_patients_info_4_tmas_has_at_least_one_image.csv", index=False)

    # extract TNBC patients only # 40 rows
    imc_patient_df = imc_patient_df[imc_patient_df["biomarke_check"] == "TNBC"].reset_index(drop=True)
    imc_patient_df.to_csv(out_dir / "ctdna_patients_info_4_tmas_has_at_least_one_image_tnbc.csv", index=False)

    # then filter out the patients with only post-treatment tissue
    keep_pid = []
    for pid in set(imc_patient_df.cid):
        types = list(set(imc_patient_df[imc_patient_df["cid"] == pid]["collection_type"]))
        if len(types) > 1:
            keep_pid.append(pid)
        else:
            if types[0] == "pre_treatment":
                keep_pid.append(pid)

    imc_pre_tre_patient_df = imc_patient_df[imc_patient_df.cid.isin(keep_pid)]
    out_dir = Path("analysis/tnbc_imc/summary")
    imc_pre_tre_patient_df.to_csv(out_dir / "ctdna_patients_info_4_tmas_has_at_least_one_image_tnbc_pretreat.csv",
                                  index=False)
    imc_pre_tre_patient_df.to_csv(out_dir / "ctdna_patient_info_4_tmas_finalize.csv", index=False)
    imc_post_tre_patient_df = imc_patient_df[~imc_patient_df.cid.isin(keep_pid)]
    imc_post_tre_patient_df.to_csv(out_dir / "ctdna_patients_info_4_tmas_has_at_least_one_image_tnbc_posttreat.csv",
                                   index=False)

    # Merging the patient information for the pending scanned file
    need_img = pd.read_csv(in_dir / "imc_tnbc_to_img_bm_panel_collection.csv")

    # add therapy and response info
    to_add_info = []
    for pid in need_img.CID.values:
        print(pid)
        to_add_info.append(rd_df[rd_df.CID == pid].iloc[0, 1:3].values)

    need_img[["Chemo type", "pCR"]] = np.vstack(to_add_info)

    need_img.to_csv(in_dir / "imc_tnbc_to_img_bm_panel_collection.csv", index=False)

    # add the type of the patient (TNBC or other type need to be validataed)
    to_img_pid_info = pd.read_csv(Path("analysis/tnbc_imc") / "forIMC.csv")
    # read in the df after removing the patients with only post-treatment
    new_to_img_df = pd.read_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel_pre_post_treatment.csv")

    mapping_dict = dict(zip(list(to_img_pid_info["CID"].values),
                            list(to_img_pid_info["Biomarker check"].values)))

    new_to_img_df["BiomarkerCheck"] = ["TNBC" if pd.isna(mapping_dict[pid]) else mapping_dict[pid] for pid in
                                       new_to_img_df.CID.values]

    new_to_img_df.to_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel_pre_post_treatment.csv", index=False)

    new_to_img_df = new_to_img_df[new_to_img_df["BiomarkerCheck"] == "TNBC"]
    new_to_img_df.reset_index(drop=True, inplace=True)

    # extract the necessary columns for Amelia and Esther
    new_to_img_df = new_to_img_df[['CID', 'TMA AT', 'TMA Name', 'Path ID/Accession', 'Core ID_1',
                                   'Core ID_2', 'Core loc_1', 'Core loc_2', 'Core loc_3', 'Core loc_4',
                                   'collection_type', 'Selected_core', 'IMC']]

    new_to_img_df.to_csv(Path("analysis/tnbc_imc") / "imc_bm_panel_to_img_tnbc_only.csv", index=False)

    """
    When there are new images need to be added into the current patient info summary, use the following codes 
    """

    # Add the pending cores/patients info into ctdna_patient_info_4_tmas_finalize.csv
    imc_pre_tre_patient_df = pd.read_csv(out_dir / "ctdna_patient_info_4_tmas_finalize.csv")

    # get the new images from the pending cores info and find their corresponding info
    to_add_df = new_to_img_df[pd.isna(new_to_img_df["IMC"]) == False]
    tma_info = pd.read_csv(in_dir / "ctDNA_cohort.csv")

    to_add_df = to_add_df[
        ["CID", "Path ID/Accession", 'Core ID_1', 'Core loc_1', 'Core loc_2', 'collection_type', 'IMC']]

    to_add_list = []
    for pid in set(to_add_df.CID):
        # core and img info
        sub_list = list(to_add_df[to_add_df.CID == pid].iloc[0, 0:2].values)
        sub_list += ["", "23-005"]
        sub_list += list(to_add_df[to_add_df.CID == pid].iloc[0, 2:5].values)
        sub_list.append(to_add_df[to_add_df.CID == pid].iloc[0, 6])
        sub_list += ["", f"{pid}_{sub_list[1]}"]

        # patient clinical metadata
        sub_list += list(to_img_pid_info[to_img_pid_info.CID == pid].iloc[0, 1:].values)

        # date info, if not found then do it manually

        sub_list.append(need_img[need_img.CID == pid].iloc[0, 12])

        sub_list += list(metadata[metadata.CID == pid].iloc[0, 7:10].values)
        sub_list.append(to_add_df[to_add_df.CID == pid].iloc[0, 5])

        to_add_list.append(sub_list)

    new_img_df = pd.DataFrame(to_add_list)

    new_img_df.columns = imc_pre_tre_patient_df.columns
    new_df = pd.concat([imc_pre_tre_patient_df, new_img_df], ignore_index=True)

    new_df.to_csv(out_dir / "ctdna_patient_info_5_tmas_finalize.csv", index=False)

    """
    Necessary cores for TNBC/pre-treatment patients 
    """
    tma_4_cohort = pd.read_csv(
        Path("analysis/tnbc_imc/summary") / "ctdna_patients_info_4_tmas_has_at_least_one_image_tnbc_pretreat.csv")

    # Check if all of them are either recurrent or neo-adjuvant
    for i in range(tma_4_cohort.shape[0]):
        if (tma_4_cohort["chemo_type"].values[i] == "ADJ") & (tma_4_cohort["recurrence"].values[i] == "No recurrence"):
            print(i, tma_4_cohort["cid"].values[i])

    new_add_df = pd.read_csv(Path("analysis/tnbc_imc/summary") / "TNBC_BMpanel_To_Image_20251120.csv")

    # Get the patient list we newly have images
    new_pids = list(set(new_add_df.cid))

    # Modify the format we want to for merging
    new_add_df["tma"] = [x.split("_")[0] if x != "TNBC_Brian" else x for x in new_add_df.tma]
    new_add_df["tma"] = [x.split(" ")[0] if x == "23-010 WGS" else x for x in new_add_df.tma]
    new_add_df["tma"] = [x.split("_")[1] if x == "TNBC_Brian" else x for x in new_add_df.tma]

    new_add_df = new_add_df[
        ['cid', 'accession', 'tma', 'core_id', 'core_loc_1', 'cor_ loc_2', 'image_1']]

    to_add = []
    for i in range(new_add_df.shape[0]):
        sub_add = []
        sub_add += list(new_add_df.iloc[i, :2].values)
        sub_add.append("")
        sub_add += list(new_add_df.iloc[i, 2:].values)
        sub_add.append("")
        sub_add.append(new_add_df.iloc[i, 0] + new_add_df.iloc[i, 1])

        to_add.append(sub_add)

    tidy_new_add_df = pd.DataFrame(to_add, columns=['cid', 'accession', 'block_num', 'tma', 'core_id', 'core_loc_1',
                                                    'core_loc_2', 'image_1', 'image_2', 'patient_key'])

    # Add clinical data from Elena
    patient_clinical_data = pd.read_csv(out_dir / "forIMC.csv")
    tidy_new_add_df = tidy_new_add_df.merge(patient_clinical_data, left_on="cid", right_on="CID", how="left")
    tidy_new_add_df = tidy_new_add_df.drop(columns=["CID"])

    tidy_new_add_df.columns = ['cid', 'accession', 'block_num', 'tma', 'core_id', 'core_loc_1',
                               'core_loc_2', 'image_1', 'image_2', 'patient_key', 'death',
                               'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
                               'hr_status', 'biomarke_check', 'biomarkers', 'stage',
                               'tumour_size_cm', 'grade', 'node_status', 'lvi', 'margin_status']
    tidy_new_add_df.to_csv(out_dir / "new_add_patient_info.csv", index=False)
    tidy_new_add_df = pd.read_csv(Path("analysis/tnbc_imc/summary") / "new_add_patient_info.csv")

    # add collection data
    collection_df = pd.read_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel_pre_post_treatment.csv")

    to_add_dates = []
    for pid in tidy_new_add_df.accession.values:
        print(pid)
        to_add_dates.append(collection_df[collection_df["Path ID/Accession"] == pid].iloc[0, 12:17].values)

    tidy_new_add_df[["collection_date", "primary_surgery_date", "primary_chemo_start", "last_primary_chemo",
                     "collection_type"]] = np.vstack(to_add_dates)

    tidy_new_add_df.to_csv(Path("analysis/tnbc_imc/summary") / "new_add_patient_info_finalize.csv", index=False)

    # Merge them all together
    new_merge = pd.concat([tma_4_cohort, tidy_new_add_df], axis=0)

    new_merge.to_csv(Path("analysis/tnbc_imc/summary") / "all_8_tma_to_analysis_patient_info.csv", index=False)

    # repeat patient pid: 300, ADJ, Recurrent
    # 305, ADJ, Recurrent
    # 1710, ADJ, Recurrent
    # 5629, NAT, No recurrent
    # 6435, ADJ, No recurrent
    new_merge.recurrence.value_counts()
    # -3 recurrence, -2 no recurrence
    # So we have 17 recurrent and 30 non-recurrent

    new_merge.chemo_type.value_counts()
    # -4 ADJ, -1 NAT
    # So we have 29 ADJ and 18 NAT
    new_merge[new_merge.chemo_type == "NAT"].pCR.value_counts()
    # Among neo-adjuvant patients, there are 8 have pCR and 10 don't have pCR

    in_dir = Path("analysis/tnbc_imc/summary")
    out_dir = Path("analysis/tnbc_imc/cohort_specific_8_tma")

    # Original adata (4 previous to compare the structure and merge), need to remove 23-005 to prevent overlap
    original_adata = read_h5ad(in_dir / "5_tma_new_7_analysis.h5ad")
    prev_4_adata = original_adata[np.where(original_adata.obs.tma != "23-005")[0], :].copy()
    # prev_4_adata.write(in_dir / "prev_4_tma_analysis.h5ad")

    # new adata (4 new adata, not add Brian first, need to wait the collection date)
    new_adata = read_h5ad(out_dir / "new_4_tma_all.h5ad")

    # all patient info
    patient_info_all = pd.read_csv(Path("analysis/tnbc_imc/summary") / "all_8_tma_to_analysis_patient_info.csv")

    # For new adata (tidy and add information)
    new_adata.obs["image_id"] = [x.split(".")[0] for x in new_adata.obs["image_id"].values]
    new_adata.obs["image_id"] = new_adata.obs["image_id"].str.upper()
    img_key = []
    for x in range(new_adata.obs.shape[0]):
        if len(new_adata.obs.image_id.values[x].split("_")) == 2:
            img_key.append(new_adata.obs.tma.values[x] + "_" + new_adata.obs.image_id.values[x].split("_")[1])
        elif len(new_adata.obs.image_id.values[x].split("_")) == 3:
            img_key.append(new_adata.obs.tma.values[x] + "_" + new_adata.obs.image_id.values[x].split("_")[2])

    new_adata.obs["image_key"] = img_key
    new_adata.write(out_dir / "new_4_tma_all.h5ad")
    new_adata = read_h5ad(Path("analysis/tnbc_imc/confirmed") / "new_4_tma_all.h5ad")

    # Get patient_key from patient info and other info
    keep_img = []
    for img in set(new_adata.obs.image_key):
        if (img in patient_info_all["image_1"].values) | (img in patient_info_all["image_2"].values):
            keep_img.append(img)
        else:
            continue

    new_adata_analysis = new_adata[new_adata.obs.image_key.isin(keep_img)].copy()
    new_adata_analysis.write(Path("analysis/tnbc_imc/confirmed") / "new_4_tma_on_cohort.h5ad")

    to_add = []
    for i in range(new_adata_analysis.obs.shape[0]):
        img = new_adata_analysis.obs.image_key.values[i]
        if img in patient_info_all["image_1"].values:
            to_add.append(np.append(patient_info_all[patient_info_all.image_1 == img].iloc[0, 9:25].values,
                                    patient_info_all[patient_info_all.image_1 == img].iloc[0, 29]))

        elif img in patient_info_all["image_2"].values:
            to_add.append(np.append(patient_info_all[patient_info_all.image_2 == img].iloc[0, 9:25].values,
                                    patient_info_all[patient_info_all.image_2 == img].iloc[0, 29]))
        else:
            print(img, "no info!")

    new_adata_analysis.obs[['patient_key', 'death', 'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
                            'hr_status', 'biomarke_check', 'biomarkers', 'stage', 'tumour_size_cm',
                            'grade', 'node_status', 'lvi', 'margin_status', 'collection_type']] = np.vstack(to_add)

    # got error when outputing, so
    new_adata_analysis.obs['diagnosis_date'] = pd.to_datetime(new_adata_analysis.obs['diagnosis_date']).dt.strftime(
        '%Y-%m-%d')
    new_adata_analysis.obs['grade'] = new_adata_analysis.obs['grade'].astype(int)

    new_adata_analysis.write(Path("analysis/tnbc_imc/confirmed") / "new_4_tma_on_cohort.h5ad")

    # new_adata_analysis.write(out_dir / "new_4_tma_analysis.h5ad")
    # new_adata_analysis.write(in_dir / "new_4_tma_analysis.h5ad")

    # Merge previous 4 and new 4 together, columns needed 'layer1', 'esi', 'image_roi', 'layer2', 'relapse'
    new_adata_analysis.obs["cid"] = [x[:10] for x in new_adata_analysis.obs.patient_key.values]
    mapping_dict = {'Epithelial CK5+': 'Epithelial',
                    'Epithelial CK5+CK7+': 'Epithelial',
                    'Epithelial CK5+CK8-18+': 'Epithelial',
                    'Epithelial CK7+': 'Epithelial',
                    'Epithelial CK8-18+': 'Epithelial',
                    'Epithelial CK_low': 'Epithelial',
                    'Epithelial H3K27me3+': 'Epithelial',
                    'Ki67+': 'Epithelial',
                    'Myoepithelial': 'Epithelial',
                    'Myoepithelial H3K27me3+': 'Epithelial',
                    'B Cell': 'Immune',
                    'T Cell': 'Immune',
                    'Macrophage': 'Immune',
                    'Blood Endothelial': 'Stromal',
                    'Fibroblast': 'Stromal',
                    'Smooth Muscle Cell': 'Stromal',
                    'Vimentin+': 'Stromal'
                    }

    new_adata_analysis.obs["esi"] = [mapping_dict[x] for x in new_adata_analysis.obs.layer2.values]
    new_adata_analysis.obs["layer1"] = [mapping_dict[x] for x in new_adata_analysis.obs.layer2.values]

    mapping_dict = {'Local and metastatic': 'Recurrence',
                    'Metastatic': 'Recurrence',
                    'No recurrence': 'No recurrence',
                    'new cancer': 'Recurrence'}

    new_adata_analysis.obs["relapse"] = [mapping_dict[x] for x in new_adata_analysis.obs.recurrence.values]
    to_add = []
    for i in range(new_adata_analysis.shape[0]):
        roi = new_adata_analysis.obs.image_id.values[i].split("_")[0]
        tma = new_adata_analysis.obs.tma.values[i]
        if len(roi.split("I")[1]) == 1:
            to_add.append(tma + "_ROI_00" + roi.split("I")[1])
        elif len(roi.split("I")[1]) == 2:
            to_add.append(tma + "_ROI_0" + roi.split("I")[1])
        elif len(roi.split("I")[1]) == 3:
            print("roi")
            to_add.append(tma + "_ROI_" + roi.split("I")[1])
            continue

    new_adata_analysis.obs["image_roi"] = to_add
    # add core_id info
    to_add = []
    for i in range(new_adata_analysis.obs.shape[0]):
        img = new_adata_analysis.obs.image_key.values[i]
        if img in patient_info_all["image_1"].values:
            to_add.append(patient_info_all[patient_info_all.image_1 == img].iloc[0, 4])

        elif img in patient_info_all["image_2"].values:
            to_add.append(patient_info_all[patient_info_all.image_2 == img].iloc[0, 4])
        else:
            print(img, "no info!")
    new_adata_analysis.obs["core_id"] = to_add

    # reorder new_adata_analysis.obs
    new_order_adata = new_adata_analysis.copy()
    new_order_adata.obs = new_order_adata.obs[
        ['image_id', 'area', 'tma', 'core_id', 'image_key', 'image_roi', 'patient_key', 'cid', 'death',
         'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
         'hr_status', 'biomarke_check', 'biomarkers', 'stage', 'tumour_size_cm',
         'grade', 'node_status', 'lvi', 'margin_status', 'collection_type',
         'layer1', 'esi', 'layer2', 'relapse']]

    new_order_adata.write(Path("analysis/tnbc_imc/confirmed") / "new_4_tma_on_cohort.h5ad")

    # merge

    # with images less or no epithelial cells (before filtering)
    new_merge = pretreatment.concatenate(new_order_adata, batch_key="batch")
    new_merge.obs.drop(columns="batch", inplace=True)

    new_merge.obs['diagnosis_date'] = pd.to_datetime(new_merge.obs['diagnosis_date']).dt.strftime(
        '%Y-%m-%d')

    # all cohort images but no filtering for epithelial cells
    new_merge.write(out_dir / "8_tma_all_cohort_img.h5ad")
    new_merge.write(in_dir / "8_tma_all_cohort_img.h5ad")

    # plotting
    # out_dir = out_dir / "layer2"

    testing = read_h5ad(Path("analysis/tnbc_imc/cohort_specific") / "cohort_tnbc_finalize_binary_relapse.h5ad")

    testing.obs["cid"] = [x.split("_")[0] for x in testing.obs.patient_key.values]

    # rename the columns first
    testing.obs.columns = ['image_id', 'area', 'tma', 'image_key', 'patient_key', 'esi',
                           'layer2', 'core_id', 'death', 'diagnosis_date', 'chemo_type', 'pCR',
                           'recurrence', 'hrd_type', 'hr_status', 'biomarke_check',
                           'biomarkers', 'stage', 'tumour_size_cm', 'grade', 'node_status',
                           'lvi', 'margin_status', 'relapse', 'immune_ct', 'cid']
    testing.obs.drop(columns='immune_ct', inplace=True)
    # need: 'collection_type', 'layer1', 'image_roi'
    testing.obs["layer1"] = [mapping_dict[x] for x in testing.obs.layer2.values]
    testing.obs["image_roi"] = testing.obs.image_key.values
    # extract the patient in our cohort (in patient_info)
    pid_target = list(patient_info_all.cid)
    testing_target = testing[np.where(testing.obs.cid.isin(pid_target))[0], :]

    # collection date
    to_add = []
    for i in range(testing_target.obs.shape[0]):
        img = testing_target.obs.image_key.values[i]
        if img in patient_info_all["image_1"].values:
            to_add.append(patient_info_all[patient_info_all.image_1 == img].iloc[0, 29])

        elif img in patient_info_all["image_2"].values:
            to_add.append(patient_info_all[patient_info_all.image_2 == img].iloc[0, 29])
        else:
            print(img, "no info!")
    testing_target.obs["collection_type"] = to_add
    # reorder the columns
    testing_new_order = testing_target.copy()
    testing_new_order.obs = testing_new_order.obs[
        ['image_id', 'area', 'tma', 'core_id', 'image_key', 'image_roi', 'patient_key', 'cid', 'death',
         'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
         'hr_status', 'biomarke_check', 'biomarkers', 'stage', 'tumour_size_cm',
         'grade', 'node_status', 'lvi', 'margin_status', 'collection_type',
         'layer1', 'esi', 'layer2', 'relapse']]

    new_tma_new_order = new_adata_analysis.copy()
    new_tma_new_order.obs = new_tma_new_order.obs[
        ['image_id', 'area', 'tma', 'core_id', 'image_key', 'image_roi', 'patient_key', 'cid', 'death',
         'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
         'hr_status', 'biomarke_check', 'biomarkers', 'stage', 'tumour_size_cm',
         'grade', 'node_status', 'lvi', 'margin_status', 'collection_type',
         'layer1', 'esi', 'layer2', 'relapse']]

    # filtering epithelial
    to_remove = []
    for img in set(new_tma_new_order.obs.image_roi):
        p_adata = new_order_adata[np.where(new_order_adata.obs.image_roi == img)[0], :]
        if p_adata.obs.esi.value_counts()['Epithelial'] / p_adata.shape[0] < 0.1:
            to_remove.append(img)
            print(img)

    out_dir = Path("analysis/tnbc_imc/summary")
    testing_new_order.write(out_dir / "pre_4_tma_cohort_tnbc_analysis.h5ad")

    new_tma_new_order.write(out_dir / "new_4_tma_cohort_tnbc.h5ad")
    new_adata_analysis_filter = new_adata_analysis[~new_adata_analysis.obs.image_roi.isin(to_remove)].copy()
    new_adata_analysis_filter.write(out_dir / "new_4_tma_cohort_tnbc_filter_epi_analysis.h5ad")

    merge_adata = testing_new_order.concatenate(new_adata_analysis_filter, batch_key="batch")
    merge_adata.obs.drop(columns="batch", inplace=True)

    merge_adata.obs['diagnosis_date'] = pd.to_datetime(merge_adata.obs['diagnosis_date']).dt.strftime(
        '%Y-%m-%d')
    merge_adata.write(out_dir / "8_tma_cohort_analysis.h5ad")

    post_treat_img = ["22-010A_ROI_013", "22-010A_ROI_014"]
    pre_treat_only_adata = merge_adata[~merge_adata.obs.image_roi.isin(post_treat_img)]

    pre_treat_only_adata.write(out_dir / "8_tma_cohort_analysis_pretreat.h5ad")

    """
    Summary the whole metadata (tidy up info for obs)
    """
    TYPE = "tnbc_imc"
    PURPOSE = 'metadata_summary'

    in_dir = out_dir = Path(f"data/{TYPE}/{PURPOSE}")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    basic_df = pd.read_csv(in_dir / "ctdna_patients_tma.csv")
    basic_df.columns = ['cid', 'accession', 'core_id', 'tma', 'core_loc_1',
                        'core_loc_2', 'image_1', 'image_2']

    elena_imc_1 = pd.read_csv(in_dir / "imc_pat_info.csv")
    elena_imc_2 = pd.read_csv(in_dir / "imc_pat_info_2.csv")

    collection_df = pd.read_csv(in_dir / "collection_data.csv")
    metadata = pd.read_csv(in_dir / "for_IMC.csv")

    new_img_df = pd.read_csv(in_dir / "new_scan_df.csv")

    # merge two df from elena and add it into merging
    elena_imc_1.columns = ['cid', 'death', 'diagnosis_date', 'chemo_type', 'pCR', 'recurrence',
                           'hrd_type', 'hr_status', 'biomarke_check', 'biomarkers', 'stage',
                           'tumour_size_cm', 'grade', 'node_status', 'lvi', 'margin_status']

    elena_imc_2.columns = ['cid', 'death', 'diagnosis_date', 'chemo_type', 'pCR', 'recurrence',
                           'hrd_type', 'hr_status', 'biomarke_check', 'biomarkers', 'stage',
                           'tumour_size_cm', 'grade', 'node_status', 'lvi', 'margin_status']

    to_add_pid_info = pd.concat([elena_imc_1, elena_imc_2], axis=0)
    to_add_pid_info.to_csv(in_dir / "patient_info_from_elena.csv", index=False)
    merging_df = basic_df.merge(to_add_pid_info, on="cid", how="left")

    # add collection date and other date info
    collection_date = []
    for i in range(merging_df.shape[0]):
        # get pid and accession
        pid = merging_df.iloc[i, 0]
        path_id = merging_df.iloc[i, 1]
        if path_id in collection_df["Path case"].values:
            collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                 (collection_df["Path case"] == path_id)]["Collection date"].values[
                                       0])
        else:
            collection_date.append("")
            print(i)

    merging_df["collection_date"] = collection_date
    # other date info
    to_add_dates = []
    for pid in merging_df.cid.values:
        if pid not in metadata.CID.values:
            print(pid)
            to_add_dates.append(np.array(["", "", ""]))
        else:
            to_add_dates.append(metadata[metadata.CID == pid].iloc[0, 7:10].values)

    merging_df[["primary_surgery_date", "primary_chemo_start", "last_primary_chemo"]] = np.vstack(to_add_dates)

    merging_df["collection_date"] = pd.to_datetime(merging_df["collection_date"], format='%Y-%m-%d')
    merging_df["primary_surgery_date"] = pd.to_datetime(merging_df["primary_surgery_date"], format='%Y-%m-%d')
    merging_df["primary_chemo_start"] = pd.to_datetime(merging_df["primary_chemo_start"], format='%Y-%m-%d')
    merging_df["last_primary_chemo"] = pd.to_datetime(merging_df["last_primary_chemo"], format='%Y-%m-%d')

    # Compare the time of therapy start date and the date of tissue collection
    pre_treat_idx = np.where(merging_df["collection_date"] <= merging_df["primary_chemo_start"])[0]
    collection_type = np.zeros(merging_df.shape[0], dtype='object')
    collection_type[:] = np.nan
    collection_type[pre_treat_idx] = np.repeat("pre_treatment", len(pre_treat_idx))

    # not pre-treatment
    post_treat_idx = np.where(merging_df["collection_date"] > merging_df["primary_chemo_start"])[0]
    collection_type[post_treat_idx] = np.repeat("post_treatment", len(post_treat_idx))

    # add the pre_treatment and post_treatment info to the df
    merging_df["collection_type"] = collection_type

    merging_df.columns = ['cid', 'accession', 'core_id', 'tma', 'core_loc_1', 'core_loc_2',
                          'image_1', 'image_2', 'death', 'diagnosis_date', 'chemo_type', 'pCR',
                          'recurrence', 'hrd_type', 'hr_status', 'biomarker_check', 'biomarkers',
                          'stage', 'tumour_size_cm', 'grade', 'node_status', 'lvi',
                          'margin_status', 'collection_date', 'primary_surgery_date',
                          'primary_chemo_start', 'last_primary_chemo', 'collection_type']

    merging_df.to_csv(out_dir / "master_metadata.csv", index=False)

    new_img_df.columns = ['cid', 'TMA AT', 'tma', 'accession', 'core_id_1',
                          'core_id_2', 'core_loc_1', 'core_loc_2', 'core_loc_3', 'core_loc_4',
                          'collection_type', 'selected_core']

    new_img_df["tma"] = [x.split("_")[0] if x != "TNBC_Brian" else x for x in new_img_df.tma]
    new_img_df["tma"] = [x.split(" ")[0] if x == "23-010 WGS" else x for x in new_img_df.tma]
    new_img_df["tma"] = [x.split("_")[1] if x == "TNBC_Brian" else x for x in new_img_df.tma]
    new_img_df.to_csv(in_dir / "new_scan_df.csv")

    # add new images
    for i in range(merging_df.shape[0]):
        pass
    for i in range(new_img_df.shape[0]):
        pid = new_img_df.cid.values[i]
        accession = new_img_df.accession.values[i]
        tma = new_img_df.accession.values[i]
        selected_core = new_img_df.selected_core.values[i]
        p_adata = merge_adata[(merging_df.cid == pid) & (merging_df.accession == accession)]

    """
    Use the master metadata to extract cores for new panel
    """
    TYPE = "tnbc_imc"
    PURPOSE = 'final_zscore'

    in_dir = Path(f"data/{TYPE}/{PURPOSE}")
    out_dir = Path(f"data/{TYPE}/{PURPOSE}")

    df = pd.read_csv(in_dir / "final_metadata.csv")

    keep_pid = []
    for pid in set(df.cid):
        types = list(set(df[df["cid"] == pid]["collection_type"]))
        if len(types) > 1:
            print(pid)
            keep_pid.append(pid)
        else:
            if types[0] == "pre_treatment":
                keep_pid.append(pid)

    # extract the cores for first 4 TMAs
    tmas = ["22-010A", "22-010C", "23-009", "BOB"]
    sub_df = metadata[metadata.tma.isin(tmas)]

    keep_pid = []
    for pid in set(sub_df.cid):
        types = list(set(sub_df[sub_df["cid"] == pid]["collection_type"]))
        if len(types) > 1:
            print(pid)
            keep_pid.append(pid)
        else:
            if types[0] == "pre_treatment":
                keep_pid.append(pid)

    pid_to_have = sub_df[sub_df.cid.isin(keep_pid)]
    pid_to_have = pid_to_have[['cid', 'accession', 'core_id', 'tma', 'core_loc',
                               'collection_type']]

    to_merge = []
    for i in set(pid_to_have.cid):
        p_df = pid_to_have[pid_to_have.cid == i]
        pid_to_add = []
        if len(set(p_df.accession)) == 1 and len(set(p_df.tma)) == 1:
            pid_to_add += [p_df.cid.values[0], p_df.accession.values[0], p_df.core_id.values[0],
                           p_df.tma.values[0]]
            pid_to_add += list(p_df.core_loc.values)
            pid_to_add.append(p_df.collection_type.values[0])
            to_merge.append(pid_to_add)

        elif len(set(p_df.accession)) == 1 and len(set(p_df.tma)) != 1:
            print("more than one tma", i)
            p_df = p_df.sort_values(by="tma", ascending=True)
            # for the first one
            pid_to_add += [p_df.cid.values[0], p_df.accession.values[0], p_df.core_id.values[0],
                           p_df.tma.values[0]]
            pid_to_add += list(p_df.core_loc.values)[:2]
            pid_to_add.append(p_df.collection_type.values[0])
            to_merge.append(pid_to_add)

            # for the second one
            pid_to_add = []
            pid_to_add += [p_df.cid.values[2], p_df.accession.values[2], p_df.core_id.values[2],
                           p_df.tma.values[2]]
            pid_to_add += list(p_df.core_loc.values)[2:]
            pid_to_add.append(p_df.collection_type.values[2])
            to_merge.append(pid_to_add)

        elif len(set(p_df.accession)) != 1:
            print("more than one accession", i)
            p_df = p_df.sort_values(by="accession", ascending=True)
            # for the first one
            pid_to_add += [p_df.cid.values[0], p_df.accession.values[0], p_df.core_id.values[0],
                           p_df.tma.values[0]]
            pid_to_add += list(p_df.core_loc.values)[:2]
            pid_to_add.append(p_df.collection_type.values[0])
            to_merge.append(pid_to_add)

            # for the second one
            pid_to_add = []
            pid_to_add += [p_df.cid.values[2], p_df.accession.values[2], p_df.core_id.values[2],
                           p_df.tma.values[2]]
            pid_to_add += list(p_df.core_loc.values)[2:]
            pid_to_add.append(p_df.collection_type.values[2])
            to_merge.append(pid_to_add)

        else:
            print("none of above", i)
            print(p_df)

    whole_merge = []
    for i in range(len(to_merge)):
        if len(to_merge[i]) == 7:
            sub = to_merge[i][:-1] + ["", ""]
            sub.append(to_merge[i][-1])
            whole_merge.append(sub)
        elif len(to_merge[i]) == 6:
            sub = to_merge[i][:-1] + ["", "", ""]
            sub.append(to_merge[i][-1])
            whole_merge.append(sub)
        elif len(to_merge[i]) == 9:
            whole_merge.append(to_merge[i])
        else:
            print("wrong!")

    testing = pd.DataFrame(whole_merge, columns=["cid", "accession", "core_id", "tma", "core_loc_1", "core_loc_2",
                                                 "core_loc_3", "core_loc_4", "collection_type"])

    testing.to_csv(in_dir / "first_4_tma_cores_to_img_info.csv", index=False)

    # for wet lab
    testing = testing.iloc[:, :-1]

    # FOR TMA MAPPING
    TMA_list = ['Brian',
                'BOB',
                '22-010A', '22-010B', '22-010C', '23-005',
                '23-009', '23-010']
    TMA_AT_list = ['AT3905', 'AT19296', 'AT19813', 'AT21453', 'AT21454', 'AT24020', 'AT26328', 'AT26329']
    tma_mapping = dict(zip(TMA_list, TMA_AT_list))
    testing["tma_at"] = [tma_mapping[x] for x in testing.tma.values]

    testing = testing[['cid', 'accession', 'tma_at', 'core_loc_1', 'core_loc_2', 'core_loc_3', 'core_loc_4']]

    testing.to_csv(in_dir / "TNBC_prev_4_TMA_cores_for_fluidigm.csv", index=False)

    """
    Summary for all the patients in the ctDNA cohort (with images or without images)
    1. patient ID (cid)
    2. With images or not here in the IMC analysis (Y/N)
    3. the reason why we didn't use it (not on the TMAs, not TNBC, not pre-treatment collected tissues or 
       images with less than 10% epithelial cells)
    """
    TYPE = "tnbc_imc"
    PURPOSE = 'metadata'

    in_dir = out_dir = Path(f"data/{TYPE}/{PURPOSE}")

    # Read in the patient list, metadata we summarized and anndata with all images info
    patient_list = pd.read_csv(in_dir / "RD_cases.csv")
    metadata_df = pd.read_csv(in_dir / "final_metadata.csv")

    # Anndata from TNBC patient and pre-treatment Anndata specificaaly
    adata = read_h5ad(in_dir / "TNBC_adata.h5ad")
    adata_pre = read_h5ad(in_dir / "TNBC_adata_pretreatment.h5ad")

    miss_cid = []
    for i in range(patient_list.shape[0]):
        if patient_list["CID"].values[i] not in list(metadata_df["cid"]):
            miss_cid.append(patient_list["CID"].values[i])
        else:
            continue

    # Create a new df (new_df)
    new_df = pd.DataFrame({"CID": list(patient_list["CID"])})

    pre_sum_df = pd.read_csv(in_dir / "ctdna_patients_tma.csv")
    scan_df = pd.read_csv(in_dir / "update_new_scan.csv")
    imc_metadata_db = pd.read_csv(in_dir / "for_IMC_from_airtable.csv")

    miss_cid = []
    for i in range(patient_list.shape[0]):
        if patient_list["CID"].values[i] not in list(pre_sum_df["CID"]):
            miss_cid.append(patient_list["CID"].values[i])
        else:
            continue

    tma_list = ["BOB", "22-010A", "22-010C", "23-009"]
    len(set(pre_sum_df[pre_sum_df.TMA.isin(tma_list)].CID))
    cohort_all_df = pd.read_csv(in_dir / "ctDNA_cohort.csv")

    for i in pre_sum_df.CID:
        if i in list(cohort_all_df.Name):
            continue
        else:
            print(i)

    for i in scan_df.cid:
        if i not in list(metadata_df.cid):
            print(i)

    new_df = pd.merge(new_df, pre_sum_df.iloc[:, [0, 2, 3, 6, 7]], how="left", on=["CID"])
    # Merge the collection date
    metadata_df.columns = ['CID', 'accession', 'Core ID', 'TMA', 'core_loc', 'image', 'death',
                           'diagnosis_date', 'chemo_type', 'pCR', 'recurrence', 'hrd_type',
                           'hr_status', 'biomarker_status', 'biomarker_check', 'biomarkers',
                           'stage', 'tumour_size_cm', 'grade', 'node_status', 'lvi',
                           'margin_status', 'collection_date', 'primary_surgery_date',
                           'primary_chemo_start', 'last_primary_chemo', 'collection_type']
    metadata_df["Core ID"].astype("Int64")
    df['column_name'] = df['column_name'].fillna(0).astype(np.int64)
    new_df["Core ID"].astype("Int64")
    new_df = pd.merge(new_df, metadata_df.iloc[:, [0, 2, 3, 14, 26]], how="left", on=["CID", "Core ID", "TMA"])

    # Another thought, update the new scanned file to the pre_sum df and merge at one time
    scan_df.columns = ['CID', 'TMA AT', 'TMA', 'accession', 'core_id_1',
                       'core_id_2', 'core_loc_1', 'core_loc_2', 'core_loc_3', 'core_loc_4',
                       'collection_type', 'Core ID', 'image_1', 'image_2']

    # Read in the current organization table and half-baked table
    pid_df = pd.read_csv(in_dir / "pid_imc.csv")
    record = []
    for pid in pid_df.CID.values:
        types = set(metadata_df[metadata_df.CID == pid].collection_type)
        if len(types) == 0:
            record.append(pd.NA)
        elif len(types) == 1:
            record.append(list(types)[0])
        elif len(types) == 2:
            record.append(f"{list(types)[0]}, {list(types)[1]}")
        else:
            print(pid)
            print("wrong")

    pid_df["collection_type"] = record

    # Go back to the for IMC db and collection df to check the tissue collection type
    # pid without collection_type info yet (need accession as well from airtable)
    to_check = []
    for i in range(pid_df.shape[0]):
        if pd.isna(pid_df.iloc[i, 5]):
            to_check.append(pid_df.iloc[i, 0])
        else:
            continue

    tma_info = pd.read_csv(in_dir / "ctDNA_cohort.csv")
    accession = []
    for pid in to_check:
        print(pid)
        acc = set(tma_info[tma_info.Name == pid]["Path ID/Accession"])
        print(acc)
        if len(acc) == 0:
            accession.append(pd.NA)
        elif len(acc) == 1:
            accession.append(list(acc)[0])
        elif len(acc) == 2:
            accession.append(f"{list(acc)[0]}, {list(acc)[1]}")
        else:
            print("Wrong")

    check_dates = pd.DataFrame({'CID': to_check, 'Path case': accession})

    collection_df = pd.read_csv(in_dir / "collection_data.csv")
    collection_date = []
    for i in range(check_dates.shape[0]):
        # get pid and accession
        pid = check_dates.iloc[i, 0]
        path_id = check_dates.iloc[i, 1]

        if pd.isna(path_id):
            collection_date.append(pd.NA)
        else:
            if len(path_id.split(", ")) > 1:
                temp = []
                temp_pid = path_id.split(", ")[0]
                if temp_pid in collection_df["Path case"].values:
                    temp.append(collection_df[(collection_df["Case ID"] == pid) &
                                              (collection_df["Path case"] == temp_pid)][
                                    "Collection date"].values[
                                    0])
                else:
                    temp.append(pd.NA)

                temp_pid = path_id.split(", ")[1]
                if temp_pid in collection_df["Path case"].values:
                    temp.append(collection_df[(collection_df["Case ID"] == pid) &
                                              (collection_df["Path case"] == temp_pid)][
                                    "Collection date"].values[
                                    0])
                else:
                    temp.append(pd.NA)
                    print(i)

                collection_date.append(f"{temp[0]}, {temp[1]}")
            else:

                if path_id in collection_df["Path case"].values:
                    collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                         (collection_df["Path case"] == path_id)][
                                               "Collection date"].values[
                                               0])
                else:
                    collection_date.append(pd.NA)
                    print(i)

    check_dates["collection_date"] = collection_date

    # others
    to_add_dates = []
    for pid in check_dates.CID.values:
        if pid not in list(imc_metadata_db.CID):
            print(pid)
            to_add_dates.append(np.array([pd.NA, pd.NA, pd.NA]))
        else:
            to_add_dates.append(imc_metadata_db[imc_metadata_db.CID == pid].iloc[0, 7:10].values)

    check_dates[["primary_surgery_date", "primary_chemo_start", "last_primary_chemo"]] = np.vstack(to_add_dates)

    # Output first to check missed collection info manually (due to different format of CID and Accession)
    check_dates.to_csv(in_dir / "check_dates.csv", index=False)
    check_dates = pd.read_csv(in_dir / "check_dates.csv")

    # check_dates["collection_date"] = pd.to_datetime(check_dates["collection_date"], format='%Y-%m-%d')
    # patient_info["primary_surgery_date"] = pd.to_datetime(patient_info["primary_surgery_date"], format='%Y-%m-%d')
    # patient_info["primary_chemo_start"] = pd.to_datetime(patient_info["primary_chemo_start"], format='%Y-%m-%d')
    # patient_info["last_primary_chemo"] = pd.to_datetime(patient_info["last_primary_chemo"], format='%Y-%m-%d')

    # Compare the time of therapy start date and the date of tissue collection
    types = []
    for i in range(check_dates.shape[0]):
        if (pd.isna(check_dates.iloc[i, 2])) or (pd.isna(check_dates.iloc[i, 4])):
            types.append(pd.NA)
        elif check_dates.iloc[i, 2] > check_dates.iloc[i, 4]:
            types.append("post_treatment")
        elif check_dates.iloc[i, 2] < check_dates.iloc[i, 4]:
            types.append("pre_treatment")
        else:
            print("Wrong!")

    check_dates["collection_type"] = types

    # add these check collection date rows into the original df, pid_df
    all_types = list(pid_df.collection_type)
    to_add_types = list(check_dates.collection_type)
    new_types = []
    add_idx = 0
    for i in all_types:
        if pd.isna(i) == True:
            new_types.append(to_add_types[add_idx])
            add_idx += 1
        elif pd.isna(i) == False:
            new_types.append(i)
        else:
            print("Wrong!")

    pid_df["collection_type"] = new_types
    pid_df.to_csv(in_dir / "patient_imc.csv", index=False)

    pid_df = pd.read_csv(in_dir / "patient_imc.csv")
    pid_df_all = pd.read_csv(in_dir / "patient_imc.csv")
    # Make the values into categories (0, 1, 2)
    mapping_dict = {'ADJ': 0, "NAT": 1}
    pid_df["Chemo type"] = [mapping_dict[x] for x in pid_df["Chemo type"].values]

    mapping_dict = {'No': 0, "Yes": 1}
    pid_df["pCR"] = pid_df["pCR"].replace(mapping_dict).fillna(pd.NA)

    mapping_dict = {'Local': 1,
                    'Local and metastatic': 1,
                    'Metastatic': 1,
                    'No recurrence': 0,
                    'new cancer': 1}
    pid_df["Recurrence status"] = [mapping_dict[x] for x in pid_df["Recurrence status"].values]

    mapping_dict = {"pre_treatment": 0, "post_treatment": 1, "post_treatment, pre_treatment": 2}
    pid_df["collection_type"] = pid_df["collection_type"].replace(mapping_dict).fillna(pd.NA)

    # Adding TMA and core location for image ID to the df

    # Adding the biomarkercheck result to the table
    new_meta_1_df = pd.read_csv(in_dir / "new_metadata_1_imc_pat_info.csv")
    new_meta_2_df = pd.read_csv(in_dir / "new_metadata_2_forIMC.csv")
    new_meta_merge = pd.concat([new_meta_1_df, new_meta_2_df], axis=0)
    new_meta_merge = new_meta_merge[["CID", "BiomarkerCheck", "Biomarkers"]]

    pid_df_all = pd.merge(pid_df_all, new_meta_merge, on='CID', how='left')

    # adding the rescanned images from the rest four TMAs to pre_sum_df
    # pre_sum_df from scan_df
    # Modify the scan_df first to fit the format we want for the merging
    scan_df = scan_df.iloc[:, [0, 2, 3, 10, 11, 12, 14]]
    scan_df = scan_df.iloc[:, [0, 2, 6, 1, 4, 5, 3]]
    merge_try = pd.merge(pre_sum_df, scan_df, how="left", on=["CID", "accession", "Core ID", "TMA"])

    df_a = pre_sum_df.iloc[:, :4].sort_values(by=['CID', 'accession', 'TMA']).reset_index(drop=True)
    df_b = merge_try.iloc[:, :4].sort_values(by=['CID', 'accession', 'TMA']).reset_index(drop=True)
    for i in range(104, df_a.shape[0]):
        print(i)
        print(set(df_a.iloc[i, :].values == df_b.iloc[i + 1, :].values))

    # The conclusion: in the merge_try, there's an extra row for CID0003061 since we have newly scanned the cores for it

    merge_try = merge_try[['CID', 'accession', 'Core ID', 'TMA', 'Core loc_1_x', 'Core img_1_x', 'Core loc_2',
                           'Core img_2', 'Core loc_1_y', 'Core img_1_y', 'collection_type']]
    tidy_list = []
    for i in range(merge_try.shape[0]):
        if (pd.isna((merge_try.iloc[i, 4]))) and (pd.isna((merge_try.iloc[i, 8]))):
            tidy_list.append(np.array([pd.NA, pd.NA, pd.NA, pd.NA]))
        elif (pd.isna((merge_try.iloc[i, 4])) == False) and (pd.isna((merge_try.iloc[i, 8]))):
            # with only the previous recorded info
            tidy_list.append(merge_try.iloc[i, 4:8].values)

        elif (pd.isna((merge_try.iloc[i, 4]))) and (pd.isna((merge_try.iloc[i, 8])) == False):
            # only the latest record
            tidy_list.append(np.concatenate((merge_try.iloc[i, 8:10].values, np.array([pd.NA, pd.NA]))))
        else:
            continue

    temp_merge_df = merge_try.iloc[:, :4].copy()
    temp_merge_df[["Loc_1", "Img_1", "Loc_2", "Img_2"]] = np.vstack(tidy_list)

    # adding the revised temp_merge_df info to the pid_df_all
    t_df = pid_df_all.copy()
    t_df = pd.merge(t_df, temp_merge_df, on="CID", how="left")

    pid_df_all = pd.merge(pid_df_all, new_meta_merge, on='CID', how='left')

    temp_merge_df.to_csv(in_dir / "imc_img_to_pid.csv", index=False)

    final_df = pd.read_csv(in_dir / "patient_imc.csv")

    # Warning!!! the image record of 22-010A was wrong because we selected the different experiment to use but i forgot to update
    imgs = []
    for i in range(final_df.shape[0]):
        if final_df["TMA"].values[i] == "22-010A":
            row_val = final_df["Core location"].values[i]
            sub_dict = CORE_INFO['XP12813'][row_val]
            if "roi" in sub_dict.keys():
                img = sub_dict['roi']
            else:
                img = np.nan

            imgs.append(img)
        else:
            imgs.append(final_df["Core image"].values[i])

    final_df["Core image_new"] = imgs
    # good!
    final_df = final_df[['CID', 'Chemo type', 'pCR', 'Recurrence status',
                         'collection_type', 'accession', 'TMA', 'Experiment', 'Core ID', 'Core location',
                         'Core image_new']]
    final_df.columns = ['CID', 'Chemo type', 'pCR', 'Recurrence status',
                        'collection_type', 'accession', 'TMA', 'Experiment', 'Core ID', 'Core location',
                        'Core image']

    # Add the image id (tma_roi)
    img_ids = []
    for i in range(final_df.shape[0]):
        if pd.isna(final_df["Core image"].values[i]):
            # No image
            img_ids.append(np.nan)
        else:
            img_ids.append(final_df["TMA"].values[i] + "_" + final_df["Core image"].values[i])

    final_df["Image ID"] = img_ids

    # Add experiments used for each TMA
    COHORT_TMA_XP = {}
    COHORT_TMA_XP['BOB'] = 'XP12446'
    COHORT_TMA_XP['22-010A'] = 'XP12813'
    COHORT_TMA_XP['22-010C'] = 'XP13087'
    COHORT_TMA_XP['23-009'] = 'XP13086'
    COHORT_TMA_XP['23-005'] = "XP17579"
    COHORT_TMA_XP['22-010B'] = 'XP13364'
    COHORT_TMA_XP['23-010'] = 'XP17584'
    COHORT_TMA_XP['Brian'] = 'XP17583'

    final_df["Experiment"] = [COHORT_TMA_XP[x] if pd.notna(x) else np.nan for x in final_df["TMA"].values]

    final_df['Core location'] = final_df['Core location'].str.upper()

    final_df.to_csv(in_dir / "patient_imc.csv", index=False)


