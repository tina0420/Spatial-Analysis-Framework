"""
@author: Tina Hsu
Tidy up metadata for analysis
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from anndata import read_h5ad

"""
Data organization fot TNBC patients data.
metadata we have:
1. All TNBC patients data (for_IMC.csv)
2. ctDNA TNBC cohort patients data (ctDNA_cohort.csv)

What to organize:
1. total patient number
2. how many patient experience relapse
3. patient numbers of neoadjuvant and adjuvant therapy
4. survival situation



"""

if __name__ == '__main__':
    TYPE = "tnbc_imc"
    # organize the patient info and core mapping from SectorMap
    in_dir = Path(f"data/{TYPE}")
    # sector_map = pd.read_excel(in_dir / "BOB_TNBC_Brian_AT19296_sectormap.xls", sheet_name="Core ID",
    #                            engine="xlrd", header=2)

    """
    All TNBC patients (for_IMC.csv)
    """
    metadata = pd.read_csv(in_dir / "for_IMC.csv")
    patient_all_id = metadata['CID']

    # change all time info to the date format
    metadata['Last clinical visit'] = pd.to_datetime(metadata['Last clinical visit'], format='%Y-%m-%d')
    metadata['Death'] = pd.to_datetime(metadata['Death'], format='%Y-%m-%d')
    metadata['Diagnosis date'] = pd.to_datetime(metadata['Diagnosis date'], format='%Y-%m-%d')
    metadata['Core Bx date'] = pd.to_datetime(metadata['Core Bx date'], format='%Y-%m-%d')
    metadata['Primary surgery date'] = pd.to_datetime(metadata['Primary surgery date'], format='%Y-%m-%d')
    metadata['Primary chemo start'] = pd.to_datetime(metadata['Primary chemo start'], format='%Y-%m-%d')
    metadata['Last primary chemo'] = pd.to_datetime(metadata['Last primary chemo'], format='%Y-%m-%d')
    metadata['Hormone/ZA start'] = pd.to_datetime(metadata['Hormone/ZA start'], format='%Y-%m-%d')
    metadata['Last primary chemo'] = pd.to_datetime(metadata['Last primary chemo'], format='%Y-%m-%d')
    # metadata['Last primary chemo'] = pd.to_datetime(metadata['Last primary chemo'], format='%Y-%m-%d'), there's in progress
    metadata['Local recurrence diagnosis'] = pd.to_datetime(metadata['Local recurrence diagnosis'], format='%Y-%m-%d')
    metadata['Mets diagnosis'] = pd.to_datetime(metadata['Mets diagnosis'], format='%Y-%m-%d')
    metadata['Earlier BOB or\nTTR Blood'] = pd.to_datetime(metadata['Earlier BOB or\nTTR Blood'], format='%Y-%m-%d')
    # metadata['TTR tissue'] = pd.to_datetime(metadata['TTR tissue'], format='%Y-%m-%d'), there's "no tissue"

    # therapy
    adjuvant_idx = np.where(metadata['Primary chemo start'] > metadata['Primary surgery date'])[0]  # n=152
    neoadjuvant_idx = np.where(metadata['Primary chemo start'] < metadata['Primary surgery date'])[0]  # n=77

    # cannot know the therapy
    none_in_either_therapy = []
    for i in range(metadata.shape[0]):
        if i in neoadjuvant_idx:
            continue
        elif i in adjuvant_idx:
            continue
        else:
            print(i)
            none_in_either_therapy.append(i)

    # n=6, [46, 48, 155, 157, 164, 182]
    # 46, 48, 157, 164 and 182: had primary surgery but no chemotherapy
    # 155: had chemotherapy but no primary surgery

    # relapse info
    relapse_idx = metadata[pd.isna(metadata['First site of relapse if known']) == False].index
    # n=63, 63 patients happened relapse

    # survival outcome (death)
    death_idx = metadata[pd.isna(metadata['Death']) == False].index
    # n=45, 45 patients who were dead

    """
    Comparison between two databases, all TNBC and ctDNA TNBC cohort 
    """
    # AT26330 which is the non_WGS TMA, is not in the ctDNA cohort
    # patients in the TMA (AT26330)
    len(np.where(metadata['Block AT (from Link FFPE Samples)'] == 'AT26330')[0])
    # n=14, [17,  32,  39,  41,  60,  74, 107, 112, 157, 164, 176, 187, 192, 233]
    len(np.where(metadata['Block AT (from Link FFPE Samples)'] == 'AT26330,AT26330')[0])
    # n=5, [31, 44, 204, 214, 218]
    len(np.where(metadata['Block AT (from Link FFPE Samples)'] == 'AT26330,AT26330,AT26330')[0])
    # n=2, [155, 189]

    # check patients
    pid_in_all = {}
    for i in range(len(patient_id)):
        for j in range(len(patient_all_id)):
            if patient_id[i] == patient_all_id[j]:
                pid_in_all[i] = j
            else:
                continue
    # check which one doesn't have patient info
    for i in range(len(patient_id)):
        if i in pid_in_all:
            continue
        else:
            print(i)
            # 90, 'CID0000313' --> with a blank at the end of str! :((, it is included!!

    # add 'CID0000313' (idx=90) to the dict
    pid_in_all[90] = 36

    """
    ctDNA cohort patients (ctDNA_cohort.csv), no AT26330 
    """
    TMA_list = ['TNBC_Brian',
                'BOB_TNBC_Brian',
                '22-010A_TNBC ctDNA', '22-010B_TNBC ctDNA', '22-010C_TNBC ctDNA',
                '23-005_TNBC ctDNA',
                '23-009 Damian/Esther_TNBC ctDNA', '23-010 WGS_cohort_TNBC ctDNA']
    TMA_AT_list = ['AT3905', 'AT19296', 'AT19813', 'AT21453', 'AT21454', 'AT24020', 'AT26328', 'AT26329']
    total_core_num = [84, 82, 72, 102, 90, 106, 58, 44, 60]
    total_patient_num = [29, 37, 36, 50, 42, 41, 23, 15, 21]

    tnbc_df = pd.DataFrame({"TMA": TMA_list, "TMA_AT": TMA_AT_list,
                            "Number of core": total_core_num, "Number of patient:": total_patient_num})

    ctdna_df = pd.read_csv(in_dir / "ctDNA_cohort.csv")
    patient_id = ctdna_df["Name"]
    tma_id = ctdna_df["TMA AT"]

    rel_core = [9, 8, 46, 60, 54,
                60, 12, 32, 0]
    rel_patient = [7, 4, 23, 30, 25,
                   24, 4, 10, 0]
    tnbc_df["Relapse number of core"] = rel_core
    tnbc_df["Relapse number of patient"] = rel_patient

    set(patient_id)
    subset = ctdna_df[ctdna_df["TMA AT"] == 'AT24020']
    len(set(subset["Name"]))  # PATIENT NUMBER
    len(set(subset["Path ID/Accession"]))

    # read in the sectormap
    patient_list = [set(subset["Name"])]

    sector_map = pd.read_excel(in_dir / "23-005_sectormap.xlsx",
                               sheet_name="CoreID", header=2)
    counts = 0
    for i in set(subset["Path ID/Accession"]):
        count = (sector_map['Accession #'] == i).sum()
        print(i)
        print(count)
        print(sector_map[sector_map['Accession #'] == i]['Core ID'])
        print("____________________________")
        counts = counts + count

    # Tidying up patient information starts from here
    """
    Add core locations of patients in the metadata (ctdna TNBC cohort)
    """
    from experiments.tnbc_imc.tnbc_imc_setup_xp import *

    patient_info = pd.read_csv(in_dir / "ctdna_patients_tma.csv")
    TMA_dict = dict(zip(TMA_list, TMA_AT_list))
    xp_1 = TMA_XP['BOB'][0]
    xp_2 = TMA_XP['BOB'][1]
    # BOB, XP12446, XP13085
    # 22-010A,
    subset = patient_info.iloc[np.where(patient_info['TMA AT'] == TMA_dict['BOB_TNBC_Brian'])[0], :]
    # loc_1 = []
    # loc_2 = []
    idx = []
    for i in range(subset.shape[0]):
        core_id = subset['Core ID'].values[i]
        print(core_id)
        # check first xp
        # matching_locations = [loc for loc, subdict in CORE_INFO[xp_1].items() if subdict.get('core_id') == core_id]
        matching_locations = [loc for loc, subdict in CORE_INFO[xp_2].items() if subdict.get('core_id') == core_id]

        print(matching_locations)
        # print("for xp1:")
        if 'roi' not in CORE_INFO[xp_2][matching_locations[0]]:
            print("No images, ", matching_locations[0])
        else:
            print(CORE_INFO[xp_2][matching_locations[0]]['roi'])
            idx.append(i)

        if 'roi' not in CORE_INFO[xp_2][matching_locations[1]]:
            print("No images, ", matching_locations[1])
        else:
            print(CORE_INFO[xp_2][matching_locations[1]]['roi'])
            idx.append(i)
        print("__________________________________")

    len(set(subset.iloc[idx, :]["Path ID/Accession"].values))
    # subset['core_1'] = loc_1
    # subset['core_2'] = loc_2

    """
    Task: Adding patient information into ctDNA TNBC cohort (ctdna_patients_tma.csv)
    1. adding the core and image information (core location and the ROI number based on core_id)
       columns: [(original)'Core ID', 'Core Loc_1', 'Core Loc_2'], maybe ROI # is not necessary here stored in the table.
    2. 
    """
    from experiments.tnbc_imc.tnbc_imc_setup_xp import *

    TYPE = "tnbc_imc"
    in_dir = Path(f"data/{TYPE}")

    patient_df = pd.read_csv(in_dir / "ctdna_patients_tma.csv")

    # tma and experiment info
    TMA_list = ['Brian', 'BOB', '22-010A', '22-010B', '22-010C', '23-005', '23-009', '23-010']
    TMA_AT_list = ['AT3905', 'AT19296', 'AT19813', 'AT21453', 'AT21454', 'AT24020', 'AT26328', 'AT26329']
    tma_at_list = dict(zip(TMA_AT_list, TMA_list))

    # add 4 more columns: dup1_core1, dup1_core2, dup2_core1, dup2_core2 (dup1->original experiment, dup2->duplicates)
    patient_df.columns = ['CID', 'TMA AT', 'Path ID/Accession', 'Block #', 'Block AT',
                          'Block type', 'H&E AT', 'RNA AT', 'DNA AT', 'Proteomics AT', 'Core ID',
                          'TMA Name']
    patient_df['TMA'] = [tma_at_list[i] for i in list(patient_df['TMA AT'].values)]

    core_loc = []
    dup1 = []
    for i in range(patient_df.shape[0]):
        if patient_df['TMA'][i] in TMA_XP:
            # first experiment
            xp = TMA_XP[patient_df['TMA'][i]][0]
            core_id = patient_df['Core ID'][i]
            temp = []
            for loc, info in CORE_INFO[xp].items():
                if info.get('core_id') == core_id:
                    temp.append(loc)
                    # dup1.append(info.get('roi', np.nan))
            core_loc.append(temp)
        else:
            core_loc.append(np.nan)

    # add core_loc into df (if there's a list for one core_id, separate them into two elements
    first_core = []
    second_core = []
    for i in range(len(core_loc)):
        if type(core_loc[i]) == list:
            first_core.append(core_loc[i][0])
            second_core.append(core_loc[i][1])
        else:
            first_core.append(np.nan)
            second_core.append(np.nan)

    # add the new columns
    patient_df['Core loc_1'] = first_core
    patient_df['Core loc_2'] = second_core

    patient_df.to_csv(in_dir / "ctdna_patients_tma.csv", index=False)

    # check if the core_id is 36 in the xp, 'XP12813'
    for loc, info in CORE_INFO[xp].items():
        if info.get('core_id') == 36:
            print(loc, info.get('roi', np.nan))
            dup1.append([loc, info.get('roi', np.nan)])

    first_core = []
    second_core = []
    tma_xp_dict = dict(zip(['BOB', '22-010A', '22-010C', '23-009'],
                           ['XP12446', 'XP13083', 'XP13087', 'XP13086']))
    for i in range(patient_df.shape[0]):
        if patient_df['TMA'][i] in tma_xp_dict:
            # get tma and xp
            xp = tma_xp_dict[patient_df['TMA'][i]]
            # get the core location
            if pd.isna(patient_df['Core loc_1'][i]) is False:
                first_core.append(CORE_INFO[xp][patient_df['Core loc_1'][i]].get('roi', np.nan))
            else:
                first_core.append(np.nan)

            if pd.isna(patient_df['Core loc_2'][i]) is False:
                second_core.append(CORE_INFO[xp][patient_df['Core loc_2'][i]].get('roi', np.nan))
            else:
                second_core.append(np.nan)
        else:
            print(patient_df['TMA'][i])
            first_core.append(np.nan)
            second_core.append(np.nan)

    # add the new columns (image name of the core)
    patient_df['Core img_1'] = first_core
    patient_df['Core img_2'] = second_core

    # patient_df.drop(patient_df.columns[0], axis=1, inplace=True)
    patient_df.to_csv(in_dir / "ctdna_patients_tma.csv", index=False)

    patient_df = pd.read_csv(in_dir / "ctdna_patients_tma.csv")

    # there's a CID has a space at the end of the string ('CID0000313 ')
    patient_df['CID'][137] = 'CID0000313'
    # add relapse information (relapse or not relapse)
    patient_idx = []
    for i in range(patient_df.shape[0]):
        print("_____________________________")
        print(i)
        print(patient_df['CID'][i])
        idx = int(np.where(metadata['CID'] == patient_df['CID'][i])[0])
        print(idx)
        patient_idx.append(idx)

    relapse_info = [metadata['First site of relapse if known'].values[i] for i in patient_idx]
    relapse_binary = ['non-relapse' if pd.isna(x) else 'relapse' for x in relapse_info]
    patient_df['relapse'] = relapse_binary

    patient_df.to_csv(in_dir / "ctdna_patients_relapse.csv", index=False)

    # Add more patient information into adata.obs, e.g. pCR and neo-adjuvant and adjuvant

    # for basic CID
    cid_df = pd.read_csv(in_dir / "RD_cases.csv")

    metadata = pd.read_csv(in_dir / "for_IMC.csv")

    cid_path = pd.read_csv(in_dir / "ctDNA_cohort.csv").iloc[:, :3]

    imc_current = pd.read_csv(Path('analysis/tnbc_imc/cohort_specific') / "cohort_specific_metadata_imc.csv").iloc[
                  :, :5]  # 51 patients in total have IMC

    # CIDs don't have images
    imc_to_image = []
    for pid in set(cid_df.CID):
        if pid not in list(imc_current.cid):
            imc_to_image.append(pid)

    # Get these images (subset) from the whole cohort patients (from Elena)

    # idx=83, cid="CID00000309" with Adjuvant therapy but has "No" response data
    target_patients = cid_df[cid_df['CID'].isin(imc_to_image)].copy().reset_index(drop=True)
    target_patients.to_csv(Path("analysis/tnbc_imc") / "cohort_pid_without_imc.csv", index=False)

    adj_df = target_patients[target_patients["Chemo type"] == "ADJ"]
    # 15 with recurrence
    to_add_recurr = list(adj_df[adj_df["Recurrence status"] != "No recurrence"]["CID"])

    # all the patients with neoadjuvant therapy
    neo_df = target_patients[target_patients["Chemo type"] == "NAT"]
    to_add_neoadj = list(neo_df["CID"])

    # Merge all the patients need images together and get the subset of the whole core info (from which TMA)
    cid_to_image_total = to_add_recurr + to_add_neoadj

    to_img_df = metadata[metadata["CID"].isin(cid_to_image_total)].iloc[:, :2]
    to_img_df = cid_path[cid_path["Name"].isin(cid_to_image_total)]

    to_img_df = cid_path[cid_path["Name"].isin(with_info)]

    # focus on the rest TMAs
    rest_tma = ["AT3905", "AT21453", "AT24020", "AT26329"]

    # for the no info patients
    no_info = []
    with_info = []
    for i in cid_to_image_total:
        if (i not in list(cid_path.Name)) and (i not in metadata.CID):
            no_info.append(i)
        else:
            with_info.append(i)

    no_info_df = cid_df[cid_df["CID"].isin(no_info)]

    patients_data_here = pd.read_csv(Path("data/tnbc_imc") / "ctdna_patients_tma.csv")

    # df with CID, TMA and Accession for the patients need to image
    to_img_df = patients_data_here.copy()
    # Cannot limit to the rest of the TMAs since there might be some cores in the previous TMAs
    # to_img_df = to_img_df[to_img_df["TMA AT"].isin(rest_tma)]
    # to_img_df.reset_index(drop=True, inplace=True)

    to_img_df = to_img_df[to_img_df["CID"].isin(with_info)]
    to_img_df.reset_index(drop=True, inplace=True)

    # Double check
    adj_recu = list(
        cid_df[(cid_df["Chemo type"] == "ADJ") & (cid_df["Recurrence status"] != "No recurrence")].CID.values)  # 21

    naj = list(cid_df[cid_df["Chemo type"] == "NAT"].CID.values)  # 57, 17 with recurrence

    all_needed_patients = adj_recu + naj  # 78

    # Remove the patients not on TMAs
    not_on_tma = []
    pid_on_tma = []
    for pid in all_needed_patients:
        if pid not in list(set(cid_path.Name)):
            not_on_tma.append(pid)
        else:
            pid_on_tma.append(pid)

    pid_on_tma_df = cid_df[cid_df["CID"].isin(pid_on_tma)]
    pid_on_tma_df.reset_index(drop=True, inplace=True)
    pid_on_tma_df.to_csv(Path("analysis/tnbc_imc") / "cohort_patients_necessary.csv", index=False)

    # get the current patients with IMC
    patients_have_imc = list(set(imc_current.cid.values))

    # which patients already have IMC then ignore
    need_to_image_pid = []
    i = 0
    for pid in pid_on_tma:
        if pid in patients_have_imc:
            i = i + 1
        else:
            need_to_image_pid.append(pid)

    # Check with the previous csv
    output_csv = pd.read_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel.csv")
    output_csv = output_csv[pd.isna(output_csv["CID"]) == False]
    output_csv.to_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel.csv", index=False)

    # not_in_output = []
    # for pid in need_to_image_pid:
    #     if pid not in list(set(output_csv.CID)):
    #         not_in_output.append(pid)

    not_in_output = []
    for pid in list(set(output_csv.CID)):
        if pid not in need_to_image_pid:
            not_in_output.append(pid)

    # check each pid
    pid = "CID0003699"
    testing = to_img_df[to_img_df.CID == pid]

    # Extract the information from the patients in need_to_image_pid from to_img_df (patient_data info I summarized)
    new_patients_df = to_img_df[to_img_df["CID"].isin(need_to_image_pid)].reset_index(drop=True)

    all_need = need_to_image_pid.copy()
    subset_set = list(set(new_patients_df.CID))

    for i in all_need:
        if i not in subset_set:
            print(i)

    pid = "CID0001720"
    testing = to_img_df[to_img_df.CID == pid]

    # Check the h5ad file we used in analysis
    from anndata import read_h5ad

    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific'
    XP = 'cohort_tnbc'
    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    in_dir = Path(f"analysis/{TYPE}/ctdna_cohort")

    adata = read_h5ad(in_dir / "tnbc_imc_cohort_3_tma.h5ad")

    testing = adata.uns["ctdna_cohort"]
    testing_2 = testing[(testing["image_1"] != '') | (testing["image_2"] != '')]
    len(testing_2.cid)  # 56
    len(set(testing_2.cid))  # 51
    imc_patients_all = list(set(testing_2.cid))  # 51, total patients have IMC

    # imc_patient_info = pd.read_csv(in_dir / "imc_pat_info.csv") , from Elena

    i = 0
    need = []
    for pid in pid_on_tma:
        if pid in imc_patients_all:
            i += 1
        else:
            need.append(pid)

    new_patients_df = to_img_df[to_img_df["CID"].isin(need)].reset_index(drop=True)

    # output_pid = list(output_csv.CID)
    patients_36 = list(set(new_patients_df.CID))
    original_output_list = list(output_csv.CID.values)

    testing = []
    for i in range(len(patients_36)):
        if patients_36[i] in original_output_list:
            continue
        else:
            testing.append(pid)

    # remove the one in output which is not belong to them
    testing = []
    for i in range(len(original_output_list)):
        if original_output_list[i] in patients_36:
            continue
        else:
            testing.append(pid)

    pid = ['CID0003139']

    """
    Add the (core) collection date to the df
    """
    pid_to_img = pd.read_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel.csv")
    rd_df = pd.read_csv(Path("analysis/tnbc_imc") / "RD_cases.csv")
    metadata = pd.read_csv(Path("analysis/tnbc_imc") / "For_imc.csv").iloc[:, :10]
    tma_info = pd.read_csv(Path("analysis/tnbc_imc") / "ctDNA_cohort.csv")
    # get the relapse data one-by-one
    pid_to_img["recurrence"] = [rd_df[rd_df.CID == pid_to_img.iloc[i, 0]]["Recurrence status"].values[0] for i in
                                range(pid_to_img.shape[0])]

    pid_to_img["recurrence"] = pid_to_img["recurrence"].map({"Local and metastatic": "Recurrence",
                                                             "Metastatic": "Recurrence",
                                                             "new cancer": "Recurrence",
                                                             "No recurrence": "No recurrence"})

    pid_to_img.to_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel_with_recurrence.csv", index=False)

    collection_df = pd.read_excel(Path("analysis/tnbc_imc") / "20251103_tnbc_ffpe_collection_dates.xlsx")

    collection_date = []
    for i in range(pid_to_img.shape[0]):
        pid = pid_to_img.iloc[i, 0]

        path_id = pid_to_img.iloc[i, 1]
        if path_id in collection_df["Path case"].values:

            if pid == "CID0001710":
                path_id_1 = path_id.split("/")[0][:-2]
                collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                     (collection_df["Path case"] == path_id_1)][
                                           "Collection date"].values[0])
                path_id_2 = path_id.split("/")[1][1:]
                collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                     (collection_df["Path case"] == path_id_2)][
                                           "Collection date"].values[0])
            else:
                collection_date.append(collection_df[(collection_df["Case ID"] == pid) &
                                                     (collection_df["Path case"] == path_id)]["Collection date"].values[
                                           0])
        else:
            collection_date.append("")
            print(i)

    pid_to_img["collection_date"] = collection_date
    pid_to_img["collection_date"] = pd.to_datetime(pid_to_img["collection_date"], format='%Y-%m-%d')

    # add the collection date manually for the patients' path ID are wrong
    collection_date = np.array(collection_date)
    collection_date[np.where(pd.isna(pid_to_img["collection_date"]) == True)[0]] = np.array(
        ["", "", "2018-10-15", "2017-09-18", "2017-12-05", "(2017-11-28/2018-04-18)",
         "2018-03-09", "2018-04-05", "2019-01-25", "2019-05-09", "(2018-01-25/2018-02-01)"])

    pid_to_img.to_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel.csv", index=False)

    to_add_dates = []
    for pid in pid_to_img.cid.values:
        print(pid)
        to_add_dates.append(metadata[metadata.CID == pid].iloc[0, 7:10].values)

    pid_to_img[["primary_surgery_date", "primary_chemo_start", "last_primary_chemo"]] = np.vstack(to_add_dates)

    pid_to_img["collection_date"] = pd.to_datetime(pid_to_img["collection_date"], format='%Y-%m-%d')
    pid_to_img["primary_surgery_date"] = pd.to_datetime(pid_to_img["primary_surgery_date"], format='%Y-%m-%d')
    pid_to_img["primary_chemo_start"] = pd.to_datetime(pid_to_img["primary_chemo_start"], format='%Y-%m-%d')
    pid_to_img["last_primary_chemo"] = pd.to_datetime(pid_to_img["last_primary_chemo"], format='%Y-%m-%d')

    # Compare the time of therapy start date and the date of tissue collection
    pre_treat_idx = np.where(pid_to_img["collection_date"] <= pid_to_img["primary_chemo_start"])[0]
    collection_type = np.zeros(pid_to_img.shape[0], dtype='object')
    collection_type[:] = np.nan
    collection_type[pre_treat_idx] = np.repeat("pre_treatment", len(pre_treat_idx))

    # not pre-treatment
    post_treat_idx = np.where(pid_to_img["collection_date"] > pid_to_img["primary_chemo_start"])[0]
    collection_type[post_treat_idx] = np.repeat("post_treatment", len(post_treat_idx))

    # add the pre_treatment and post_treatment info to the df
    pid_to_img["collection_type"] = collection_type

    pid_to_img.to_csv(Path("analysis/tnbc_imc") / "imc_tnbc_to_img_bm_panel_collection.csv", index=False)

    # add the same info to the cohort_specific csv
    pid_to_img = pd.read_csv(Path("analysis/tnbc_imc/cohort_specific") / "cohort_specific_metadata_imc_tnbc.csv")
    pid_to_img.to_csv(Path("analysis/tnbc_imc/cohort_specific") / "cohort_specific_metadata_imc_tnbc.csv", index=False)

    # Re-summarize the patients infomation and images
    out_dir = Path(f"analysis/tnbc_imc/summary")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # 128 in ctdna tnbc cohort (rd_info)
    all_pid_cohort = list(rd_df.CID)

    # how many patients are not on the TMAs
    # 117 are on TMAs, 11 are not
    on_tma_list = []
    not_on_tma_list = []
    for pid in rd_df.CID:
        if pid in tma_info.Name.values:
            on_tma_list.append(pid)
        else:
            not_on_tma_list.append(pid)

    rd_df[rd_df.CID.isin(not_on_tma_list)].to_csv(out_dir / "pid_not_on_tma.csv", index=False)

    #







