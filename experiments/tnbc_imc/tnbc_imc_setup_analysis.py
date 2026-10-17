"""
@author: Tina Hsu
Setup for TNBC IMC analysis
"""
# fixed colors for each TMA used in the cohort analysis
TMA_COLOR_DICT = {'BOB': 'tab:blue',
                  '22-010A': 'tab:pink',
                  '22-010C': 'tab:orange',
                  '23-009': 'tab:green',
                  '23-005': 'tab:brown',
                  '22-010B': 'tab:cyan',
                  '23-010': 'tab:purple',
                  'Brian': 'tab:gray'}

# fixed colors for relapse and non-relapse groups showing in different formats, e.g. boxplot, violin,...
# RELAPSE_TF_COLOR_DICT = {'No recurrence': 'tab:blue',
#                          'Local': 'tab:pink',
#                          'Local and metastatic': 'tab:orange',
#                          'Metastatic': 'tab:green'}

RELAPSE_TF_COLOR_DICT = {'No recurrence': 'tab:blue',
                         'Recurrence': 'tab:pink'}

# fixed colors for Yes or No of pCR
PCR_TF_COLOR_DICT = {'No': 'tab:blue',
                     'Yes': 'tab:pink'}

CHEMOTYPE_COLOR_DICT = {"ADJ": "tab:blue",
                        "NAT": "tab:pink"}

# fixed colors for the specific cell types (dict), specifically for BM dataset now
ESI_CELL_TYPE_COLOR_DICT = {'Epithelial': '#b41f77',
                            'Immune': '#77b41f',
                            'Stromal': '#1f77b4'}

E_TME_COLOR_DICT = {'Epithelial': '#b41f77',
                    'TME': '#1f77b4'}

# fixed color for tils (Epithelial, Stromal and detailed Immune cells)
ESI_TILS_COLOR_DICT = {'Epithelial': '#64D60E',
                       'Stromal': '#ae8362',
                       'Infiltrating': '#ef2516',
                       'Bystanding': '#2e82f1',
                       'Ignored': '#c1dafb'}

CT_TILS_COLOR_DICT = {'Epithelial': '#64D60E',
                      'Stromal': '#ae8362',
                      'Infiltrating': '#ef2516',
                      'Bystanding': '#2e82f1',
                      'Ignored': '#c1dafb',
                      'Immune': '#fafafa'}

# layer 2 cell types include 'Luminal',
# LAYER2_CELL_TYPE_COLOR_DICT = {'Luminal': '#178867',
#                                'Myoepithelial': '#77b41f',
#                                'Undefined epithelial': '#caed97',
#                                'T cell': '#b41f77',
#                                'B cell': '#ff7f00',
#                                'Macrophage': '#e78ac3',
#                                'Undefined immune': '#fb9a99',
#                                'Blood endothelial': '#64b0e4',
#                                'Smooth muscle cell': '#1f2db4',
#                                'Vimentin': '#ede597',
#                                'Undefined stromal': '#c9cdf6',
#                                'Undefined': '#666666'}

# it is the previous one which includes all patients in the cohort and CLUSTER2 in epithelial group undefined
# LAYER2_CELL_TYPE_COLOR_DICT = {'B-Catenin': '#d0bfeb',
#                                'Basal': '#c3e78a',
#                                'CK5+CK7+Ki67+Basaluminal': '#c2dbab',
#                                'CK7+CK8/18+Luminal': '#8ae7ae',
#                                'CK7+Luminal': '#178867',
#                                'CK8/18+Ki67+Luminal': '#1d8f4a',
#                                'CK8/18+Luminal': '#81aa0c',
#                                'Cluster2': '#b15928',
#                                'Cluster21': '#666666',
#                                'H3K27me3+': '#daebbf',
#                                'H3K27me3+Luminal': '#eafabb',
#                                'Ki67+': '#9974d4',
#                                'Ki67+Myoepithelial': '#f5f784',
#                                'Myoepithelial': '#e1b00f',
#                                'Undefined Epithelial': '#666666',
#                                'B Cell': '#f4ac5f',
#                                'Macrophage': '#ede597',
#                                'SMA+T Cell': '#b41f77',
#                                'T Cell': '#e07880',
#                                'Blood Endothelial': '#5ea3cd',
#                                'Fibroblast': '#ae8ae7',
#                                'Smooth Muscle Cell': '#1f2db4',
#                                'VWF+': '#f2ca47',
#                                'Vimentin+': '#c3dcec'
#                                }

# LAYER2_CELL_TYPE_COLOR_DICT = {'Epithelial CK5+': '#f10024',
#                                'Epithelial CK5+CK7+': '#ffb6c1',
#                                'Epithelial CK5+CK8-18+': '#ef38aa',
#                                'Epithelial CK7+': '#ae8ae7',
#                                'Epithelial CK8-18+': '#ef8616',
#                                'Epithelial CK_low': '#7639d6',
#                                'Epithelial H3K27me3+': '#d0be6b',
#                                'Ki67+': '#f9f1a4',
#                                'Myoepithelial': '#8e5112',
#                                'Myoepithelial H3K27me3+': '#f6d646',
#                                'B Cell': '#178867',
#                                'T Cell': '#c3e78a',
#                                'Macrophage': '#9ac4a0',
#                                'Blood Endothelial': '#5ea3cd',
#                                'Fibroblast': '#bbe4fc',
#                                'Smooth Muscle Cell': '#01588d',
#                                'Vimentin+': '#92b6cb'
#                                }

# LAYER2_CELL_TYPE_COLOR_DICT = {'B Cell': '#178867',
#                                'Blood Endothelial': '#5ea3cd',
#                                'Epithelial CK5+': '#f10024',
#                                'Epithelial CK5+CK7+': '#ffb6c1',
#                                'Epithelial CK7+': '#ae8ae7',
#                                'Epithelial CK8-18+': '#ef8616',
#                                'Epithelial CK_low': '#7639d6',
#                                'Epithelial H3K27me3+': '#d0be6b',
#                                'Fibroblast': '#bbe4fc',
#                                'Macrophage': '#9ac4a0',
#                                'Myoepithelial': '#8e5112',
#                                'Smooth Muscle Cell': '#01588d',
#                                'T Cell': '#c3e78a',
#                                'Vimentin+': '#92b6cb'}

LAYER2_CELL_TYPE_COLOR_DICT = {'B-cell': '#178867',
                               'Endothelial': '#5ea3cd',
                               'CK5+ Basal': '#f10024',
                               'CK7+ Luminal': '#ae8ae7',
                               'CK+ Epithelial': '#7639d6',
                               'Fibroblast': '#bbe4fc',
                               'Macrophage': '#9ac4a0',
                               'Myoepithelial': '#8e5112',
                               'Smooth Muscle Cell': '#01588d',
                               'T Cell': '#c3e78a',
                               'Vimentin+': '#92b6cb'}




CELL_NICHE_COLOR_DICT = {'B': '#178867',
                         'CK5_Basal': '#f10024',
                         'CK5_CK7_Mp': '#ffb6c1',
                         'CK7_Lum': '#ae8ae7',
                         'CK8-Lum_Myo': '#8e5112',
                         'CK8_Lum': '#ef8616',
                         'Fibro': '#bbe4fc',
                         'H3K27me3_Epi': '#d0be6b',
                         'T_B': '#c3e78a',
                         'T_Mp_Endo': '#5ea3cd'}

# LAYER2_CELL_TYPE_COLOR_DICT = {'Epithelial CK5+': '#f10024',
#                                'Epithelial CK5+CK7+': '#ffb6c1',
#                                'Epithelial CK5+CK7+CK8-18+': '#ef38aa',
#                                'Epithelial CK5+CK7+CK8-18+SMA+': '#ff405d',
#                                'Epithelial CK5+CK7+SMA+': '#ff8a7c',
#                                'Epithelial CK5+SMA+': '#a30018',
#                                'Epithelial CK7+': '#ae8ae7',
#                                'Epithelial CK7+CK8-18+': '#7639d6',
#                                'Epithelial CK7+CK8-18+SMA+': '#f9f1a4',
#                                'Epithelial CK8-18+': '#ef8616',
#                                'Epithelial CK_low': '#d0be6b',
#                                'Epithelial SMA+': '#8e5112',
#                                'B Cell': '#178867',
#                                'T Cell': '#c3e78a',
#                                'Macrophage': '#9ac4a0',
#                                'Blood Endothelial': '#5ea3cd',
#                                'Fibroblast': '#bbe4fc',
#                                'Smooth Muscle Cell': '#01588d',
#                                'Vimentin+': '#92b6cb',
#                                'vWF+': '#ace8ed',
#                                }

# previous Vimentin (light blue+green), '#bae8ef'
# previous blue SMA, #1f2db4
# pink, #e78ac3
# light pink, '#e78ac3'
# rose pink, #fb9a99
# dark purple, 6a3d9a
# brown, '#b15928'
#  mid purple, '#7570b3'
# light purple, #cab2d6
