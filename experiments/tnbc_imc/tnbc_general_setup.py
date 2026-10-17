"""
Some general setup for TNBC IMC information
"""
from src.visualize import COLOR_CODE
import os
from pathlib import Path
import pandas as pd
import numpy as np

"""
Mapping between ROIs and cores on the slide, storing in the dictionary, CORE_INFO with experiment name the key.

3 components: Core Location (r_c_), Core ID (1, 2, ...), ROI number (ROI_000)
"""
CORE_INFO = {}
EXPERIMENT = ''

# 1. Core Location (r_c_)
# It depends on how many cores per row and how many rpws on the slide.
cores_loc = []
for i in range(1, 7):
    for j in range(1, 17):
        loc = f"r{i}c{j}"
        if i == 6:
            if j >= 7:
                cores_loc.append(loc)
        else:
            cores_loc.append(loc)

# 2. Core ID (1, 2, ...)
# It should follow the design of the TMA (how many cores each row and it starts from right or left).
core_id = [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1,
           7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
           18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13,
           19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
           30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25,
           31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36,
           41, 41, 40, 40, 39, 39, 38, 38, 37, 37]

core_id = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
           16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9,
           17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
           29, 29, 28, 28, 27, 27, 26, 26, 25, 25]

# or row by row if the orders are different each row
core_id = []
for i in range(132, 127, -1):
    core_id = core_id + [i] * 2

# 3. ROI number (ROI_000)
# It combines ROI_ and 3-digit numbers.
to_add_roi = []
for i in range(6, 0, -1):
    key = f"ROI_{i:03d}"
    # if i != 40 and i != 28:
    #     to_add_dict.append(key)
    # else:
    #     print(key)
    to_add_roi.append(key)
    # check each ROI
    print(key)

# integrate all the information into a dictionary for one experiment, key=experiment name
CORE_INFO[EXPERIMENT] = {}

# if most cores have images:
roi = 0
for i in range(len(cores_loc)):
    # should replace the core location without images
    if cores_loc[i] != 'r2c11' and cores_loc[i] != 'r5c12' and cores_loc[i] != 'r6c9' and cores_loc[i] != 'r6c10':

        CORE_INFO[EXPERIMENT][cores_loc[i]] = {'core_id': core_id[i], 'roi': to_add_roi[roi]}
        roi += 1
    else:
        print("no images")
        print(cores_loc[i])
        CORE_INFO[EXPERIMENT][cores_loc[i]] = {'core_id': core_id[i]}

# if there are few images from the slide, can create a dictionary for the few cores and their corresponding images:
# Ex: XP13085, not all cores
to_add_roi = {'r1c12': 'ROI_001', 'r1c9': 'ROI_002', 'r1c4': 'ROI_003',
              'r2c9': 'ROI_004', 'r2c8': 'ROI_005', 'r2c5': 'ROI_006',
              'r3c8': 'ROI_007', 'r3c5': 'ROI_008', 'r3c3': 'ROI_009',
              'r4c11': 'ROI_010', 'r4c4': 'ROI_011', 'r4c2': 'ROI_012',
              'r5c12': 'ROI_013', 'r5c5': 'ROI_014', 'r5c4': 'ROI_015',
              'r6c12': 'ROI_016', 'r6c10': 'ROI_017', 'r6c4': 'ROI_018',
              'r7c9': 'ROI_019', 'r7c8': 'ROI_020', 'r7c6': 'ROI_021'}

for i in range(len(cores_loc)):
    if cores_loc[i] in to_add_roi:
        CORE_INFO[EXPERIMENT][cores_loc[i]] = {'core_id': core_id[i], 'roi': to_add_roi[cores_loc[i]]}
    else:
        CORE_INFO[EXPERIMENT][cores_loc[i]] = {'core_id': core_id[i]}

"""
Creating color dictionary for visualization in the analysis

1 component: a list of the labels (categories)
"""
# replace the dictionary name for different usages
LABEL_LIST = []
COLOR_DICT = dict(zip(LABEL_LIST, COLOR_CODE[:len(LABEL_LIST)]))

"""
Organize the patient-core information in a computational way, an individual dataframe and also add it into the previous 
dictionary of CORE_INFO for better usage

1. Sector Map of each TMA
2. ctDNA TNBC cohort datasheet

Notice: Different sector maps have different key column for mapping!
"""
TYPE = 'tnbc_imc'
in_dir = Path(f"data/{TYPE}")
out_dir = Path(f"data/{TYPE}")

ctdna_df = pd.read_csv(in_dir / "ctDNA_cohort.csv")

tma_list = set(ctdna_df['TMA AT'])

# relapse patient, 119 (CID), 122 (Accession)
len(set(ctdna_df['Name']))

organize_df = ctdna_df.iloc[:, :10].copy()

# all the info are needed
tidy_list = []
check_list = []
# for TMA
# Brian, 'AT3905' & 'AT19813,AT3905'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT3905')[0], :]
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813,AT3905')[0], :]

# core_id = [26, 34, 30, 3, 35, 24, 27, 9, 35, 27, 26, 2]
core_id = [26, 34, 30, 3, 35, 24, 27, 9, 35, 27, 26]
core_id = [2]
# ['CID0001683', 'AT19813,AT3905', '18SS-16699', '1D', 'AT79',
#        'Surgical', 'AT3521', 'AT17936', 'AT13173', 'AT19364']
i = 0
check_list.append(['CID0001683', 'AT3905', '18SS-16699', '1D', 'AT79',
                   'Surgical', 'AT3521', 'AT17936', 'AT13173', 'AT19364'] + [core_id[i]])

# BOB, 'AT19296' & 'AT19813,AT19296'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19296')[0], :]
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813,AT19296')[0], :]

core_id = [40, 16, 23]
core_id = [35]
['CID0000300', 'AT19813,AT19296', 'PS15-26257', 'A6', 'AT3169',
 'Surgical', 'AT3186', 'AT17259', 'AT13895-AT26761', 'AT19346']
i = 0
check_list.append(['CID0000300', 'AT19296', 'PS15-26257', 'A6', 'AT3169',
                   'Surgical', 'AT3186', 'AT17259', 'AT13895-AT26761', 'AT19346'] + [core_id[i]])

# 22-010A, 'AT19813' &  'AT19813,AT19296', 'AT19813,AT3905'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813')[0], :]
core_id = [16, 18, 8, 7, 3, 11, 10, 12, 13, 19, 9, 27, 17, 29, 25, 30, 22, 31, 35, 34, 36]

subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813,AT3905')[0], :]
core_id = [21]
['CID0001683', 'AT19813,AT3905', '18SS-16699', '1D', 'AT79',
 'Surgical', 'AT3521', 'AT17936', 'AT13173', 'AT19364']
i = 0
check_list.append(['CID0001683', 'AT19813', '18SS-16699', '1D', 'AT79',
                   'Surgical', 'AT3521', 'AT17936', 'AT13173', 'AT19364'] + [core_id[i]])

subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813,AT19296')[0], :]
core_id = [4]
['CID0000300', 'AT19813,AT19296', 'PS15-26257', 'A6', 'AT3169',
 'Surgical', 'AT3186', 'AT17259', 'AT13895-AT26761', 'AT19346']
i = 0
check_list.append(['CID0000300', 'AT19813', 'PS15-26257', 'A6', 'AT3169',
                   'Surgical', 'AT3186', 'AT17259', 'AT13895-AT26761', 'AT19346'] + [core_id[i]])

# 22-010B, 'AT21453'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT21453')[0], :]
core_id = [68, 51, 54, 52, 46, 47, 59, 58, 72, 82, 81, 57, 70, 75, 67,
           41, 71, 80, 79, 56, 62, 48, 83, 85, 40, 66, 44, 45, 73, 60]

# 22-010C, 'AT21454'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT21454')[0], :]
core_id = [102, 112, 97, 100, 121, 114, 106, 105, 103, 113, 132, 99, 130, 131,
           115, 123, 88, 104, 91, 98, 119, 118, 117, 116, 94, 90, 129]

#
# sector_map = pd.read_excel(in_dir / "22-010_sectoemap.xlsx",
#                        sheet_name="CoreID", header=2)
#
# need: core_id 94, 90, 129
# auto_core_id = []
# for i in range(subset.shape[0]):
#     acc = subset.iloc[:, 2].values[i]
#     print(acc)
#     auto_core_id.append(sector_map.iloc[np.where(sector_map["Accession #"] == acc)[0], 0])
#     print()


# 23-005, 'AT24020'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT24020')[0], :]
core_id = [39, 38, 37, 26, 23, 21, 1, 2, 5, 30, 9, 41, 40, 24, 11, 12, np.nan, 51, 50, 34, 17, 16,
           27, 28, 22, 8, 44, 45, 25, 35, 36]  # NOT CHECK LENGTH YET

# 23-009. 'AT26328'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT26328')[0], :]
core_id = [2, 9, 15, 16, 23, 24, 9]

# 23-010, 'AT26329'
subset = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT26329')[0], :]
core_id = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 12, 15, 16, 17, 19, 20]

for i in range(subset.shape[0]):
    print("______________________________")
    print(i)
    # tidy_list.append(list(subset.iloc[i, :].values) + [core_id[i]])
    check_list.append(list(subset.iloc[i, :].values) + [core_id[i]])

# general adding
for i in range(subset.shape[0]):
    print("______________________________")
    print(i)
    # tidy_list.append(list(subset.iloc[i, :].values) + [core_id[i]])
    check_list.append(list(subset.iloc[i, :].values) + [core_id[i]])

cohort_patient_cores_on_tma = pd.DataFrame(check_list,
                                           columns=['CID', 'TMA AT', 'Path ID/Accession', 'Block #', 'Block AT',
                                                    'Block type', 'H&E AT', 'RNA AT', 'DNA AT', 'Proteomics AT',
                                                    'Core ID'])

cohort_patient_cores_on_tma.to_csv(out_dir / "ctdna_patients_tma.csv", index=False)

TMA_list = ['TNBC_Brian',
            'BOB_TNBC_Brian',
            '22-010A_TNBC ctDNA', '22-010B_TNBC ctDNA', '22-010C_TNBC ctDNA',
            '23-005_TNBC ctDNA',
            '23-009 Damian/Esther_TNBC ctDNA', '23-010 WGS_cohort_TNBC ctDNA', '23-011 non_WGS_cohort_TNBC ctDNA']
TMA_ATplist = ['AT3905', 'AT19296', 'AT19813', 'AT21453', 'AT21454', 'AT24020', 'AT26328', 'AT26329', 'AT26330']

tma_at_dict = dict(zip(TMA_ATplist, TMA_list))

CLUSTER_KEY = 'TMA AT'
to_add_tma_name = list(cohort_patient_cores_on_tma[CLUSTER_KEY].map(tma_at_dict))

CLUSTER_KEY = 'TMA'
cohort_patient_cores_on_tma[CLUSTER_KEY] = to_add_tma_name

for key, value in tma_at_dict.items():
    print(value)
    subd = cohort_patient_cores_on_tma.iloc[np.where(cohort_patient_cores_on_tma['TMA AT'] == key)[0], :]
    print(len(set(subd['CID'])))
    print("____________________________")


"""
Note for wrong side/info from the ctDNA TNBC datasheet
--> need to double check the patients in more than one TMAs
1. AT3905, CID0001045, Accession needs to remove 'C' at the end
2. VS17-36414, PS17-21861, 18SS-16699 have more than one rows but recorded the same Block
3. for 23-005, BS17-3164 doesn't match to BS17-03164
4. for 23-005, BS17-5576 doesn't match to BS17-05576
5. for 23-005, BS17-6872 doesn't match to BS17-06872
6. for 23-005, BS17-7319 doesn't match to BS17-07319
7. there's no PS18-901 in the TMA, for 23-005, i marked as np.nan [...,..., np.nan, ...]
8. for 23-005, BS18-4952 doesn't match to BS18-04952
9. for 23-005, BS19-519 doesn't match to BS19-00519
10. for 23-009, LS18-12091 has two same rows with the same Block #
11. for 23-010, PS15-26978 has Block AT which cannot be found in the sector map
12. for 23-010, PS15-26978 A1, A2 (Block #) are different to sector map (1, 2)
"""

# CHECK
t_data = organize_df.iloc[np.where(organize_df['TMA AT'] == 'AT19813,AT3905')[0], :]

patient_info = pd.read_csv(in_dir / "ctdna_patients_tma.csv")



