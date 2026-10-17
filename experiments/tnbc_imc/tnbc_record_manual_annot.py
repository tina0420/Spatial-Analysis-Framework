"""
@author: Tina Hsu
It's a file with the record of some manual annotations.
"""
from experiments.tnbc_imc.tnbc_imc_setup_xp import *

# Task: If there's batch effect between the experiments from the same TMA

# manually annotate immune, stromal and unknown cell types for the leiden clusters in non-epithelial group
IMM_STR_LABELS = {}

# 22-010C
IMM_STR_LABELS['22-010C'] = ['stromal', 'stromal', 'unknown', 'unknown', 'unknown',
                             'unknown', 'stromal', 'unknown', 'unknown', 'unknown',
                             'unknown', 'immune', 'immune', 'immune', 'unknown',
                             'unknown', 'immune', 'unknown', 'immune', 'stromal',
                             'unknown', 'immune', 'unknown']

IMM_STR_LABELS[TMA_XP['22-010C'][0]] = ['immune', 'unknown', 'stromal', 'stromal', 'unknown',
                                        'immune', 'stromal', 'unknown', 'unknown', 'unknown',
                                        'unknown', 'unknown', 'unknown', 'immune', 'stromal',
                                        'stromal', 'immune', 'unknown', 'unknown', 'unknown',
                                        'unknown', 'unknown', 'immune', 'unknown']

IMM_STR_LABELS[TMA_XP['22-010C'][1]] = ['unknown', 'immune', 'unknown', 'unknown', 'stromal',
                                        'stromal', 'unknown', 'unknown', 'unknown', 'unknown',
                                        'stromal', 'unknown', 'immune', 'immune', 'immune',
                                        'unknown', 'unknown', 'stromal', 'stromal', 'immune',
                                        'immune']

# 22-010A_test
IMM_STR_LABELS['22-010A'] = ['immune', 'stromal', 'immune', 'unknown', 'unknown',
                             'unknown', 'unknown', 'immune', 'unknown', 'unknown',
                             'unknown', 'immune', 'immune', 'immune', 'unknown',
                             'immune', 'stromal', 'unknown', 'unknown', 'stromal',
                             'stromal', 'stromal', 'unknown', 'unknown', 'unknown',
                             'stromal', 'unknown']

IMM_STR_LABELS[TMA_XP['22-010A'][0]] = ['immune', 'stromal', 'unknown', 'immune', 'immune',
                                        'immune', 'unknown', 'immune', 'stromal', 'unknown',
                                        'stromal', 'stromal', 'unknown', 'unknown', 'immune',
                                        'stromal', 'immune', 'immune', 'stromal', 'stromal',
                                        'unknown', 'stromal', 'immune', 'unknown']

IMM_STR_LABELS[TMA_XP['22-010A'][1]] = ['unknown', 'stromal', 'immune', 'unknown', 'immune',
                                        'stromal', 'immune', 'immune', 'immune', 'immune',
                                        'stromal', 'immune', 'stromal', 'unknown', 'unknown',
                                        'unknown', 'stromal', 'unknown', 'unknown', 'stromal',
                                        'stromal', 'stromal', 'unknown', 'unknown', 'unknown']

# 23 - 009_test
IMM_STR_LABELS['23-009'] = ['immune', 'immune', 'unknown', 'immune', 'immune',
                            'unknown', 'stromal', 'stromal', 'stromal', 'stromal',
                            'immune', 'immune', 'unknown', 'unknown', 'unknown',
                            'immune', 'immune', 'immune', 'unknown', 'unknown',
                            'unknown', 'immune', 'unknown', 'unknown', 'stromal',
                            'stromal', 'unknown', 'stromal', 'unknown', 'stromal',
                            'unknown']

IMM_STR_LABELS[TMA_XP['23-009'][0]] = ['immune', 'stromal', 'stromal', 'unknown', 'immune',
                                       'stromal', 'stromal', 'immune', 'unknown', 'unknown',
                                       'unknown', 'stromal', 'immune', 'unknown', 'immune',
                                       'stromal', 'immune', 'stromal', 'immune', 'immune',
                                       'immune', 'unknown', 'stromal', 'stromal', 'unknown',
                                       'immune', 'unknown']

IMM_STR_LABELS[TMA_XP['23-009'][1]] = ['stromal', 'immune', 'unknown', 'stromal', 'immune',
                                       'unknown', 'stromal', 'unknown', 'immune', 'immune',
                                       'immune', 'stromal', 'unknown', 'stromal', 'immune',
                                       'unknown', 'immune', 'unknown', 'unknown', 'unknown',
                                       'immune', 'stromal', 'immune', 'stromal', 'stromal',
                                       'immune']

# BOB
IMM_STR_LABELS['BOB'] = ['stromal', 'stromal', 'immune', 'unknown', 'stromal',
                         'unknown', 'immune', 'unknown', 'immune', 'unknown',
                         'immune', 'stromal', 'unknown', 'immune', 'unknown',
                         'immune', 'unknown', 'unknown', 'unknown', 'unknown',
                         'immune']

IMM_STR_LABELS[TMA_XP['BOB'][0]] = ['unknown', 'unknown', 'unknown', 'immune', 'unknown',
                                    'stromal', 'unknown', 'immune', 'immune', 'stromal',
                                    'unknown', 'immune', 'immune', 'stromal', 'unknown',
                                    'stromal', 'unknown', 'immune']

IMM_STR_LABELS[TMA_XP['BOB'][1]] = ['immune', 'immune', 'immune', 'unknown', 'immune',
                                    'stromal', 'unknown', 'stromal', 'unknown', 'stromal',
                                    'immune', 'stromal', 'unknown', 'unknown', 'stromal',
                                    'stromal', 'immune', 'unknown', 'immune', 'unknown']

# 23-011 (XP12274), which is not related to the ctDNA TNBC cohort

# Task: Preprocessing the fulL dataset with 4 TMAs (BOB. 22-010A, 22-010C, 23-009)

# manually annotate immune, stromal and unknown cell types for the leiden clusters in non-epithelial group
imm_str_labels = {}

imm_str_labels['full_4_tma'] = ['unknown', 'immune', 'immune', 'unknown', 'unknown',
                                'stromal', 'stromal', 'immune', 'immune', 'stromal',
                                'unknown', 'immune', 'immune', 'immune', 'stromal',
                                'stromal', 'immune', 'unknown', 'immune', 'immune',
                                'unknown', 'immune', 'immune', 'unknown', 'immune',
                                'stromal', 'stromal', 'unknown', 'unknown', 'unknown']

imm_str_labels['BOB'] = ['unknown', 'stromal', 'stromal', 'immune', 'stromal',
                         'stromal', 'immune', 'unknown', 'unknown', 'immune',
                         'unknown', 'unknown', 'immune', 'immune', 'immune',
                         'unknown', 'stromal', 'immune', 'immune', 'unknown',
                         'unknown', 'stromal', 'unknown', 'stromal', 'unknown',
                         'immune', 'immune', 'unknown']

imm_str_labels['22-010A'] = ['unknown', 'immune', 'immune', 'unknown', 'unknown',
                             'stromal', 'unknown', 'immune', 'stromal', 'immune',
                             'stromal', 'stromal', 'unknown', 'stromal', 'immune',
                             'immune', 'immune', 'unknown', 'stromal', 'stromal',
                             'unknown', 'unknown', 'immune', 'immune', 'unknown',
                             'unknown', 'unknown', 'unknown']

imm_str_labels['22-010C'] = ['unknown', 'unknown', 'unknown', 'immune', 'immune',
                             'stromal', 'immune', 'unknown', 'stromal', 'stromal',
                             'immune', 'stromal', 'stromal', 'immune', 'unknown',
                             'immune', 'unknown', 'unknown', 'stromal', 'immune',
                             'unknown', 'stromal', 'stromal', 'immune', 'unknown',
                             'unknown', 'stromal', 'unknown', 'unknown', 'unknown']

imm_str_labels['23-009'] = ['immune', 'stromal', 'immune', 'stromal', 'unknown',
                            'unknown', 'immune', 'stromal', 'stromal', 'immune',
                            'stromal', 'unknown', 'immune', 'unknown', 'immune',
                            'unknown', 'unknown', 'immune', 'stromal', 'immune',
                            'immune', 'unknown', 'unknown', 'unknown', 'stromal',
                            'stromal', 'unknown', 'immune', 'unknown', 'unknown',
                            'immune', 'unknown']

# Task: Picking up the better experiment from each TMA running individually and merge 2 duplicates together and run

# manually annotate immune, stromal and unknown cell types for the leiden clusters in non-epithelial group
IMM_STR_LABELS = {}
IMM_STR_LABELS['BOB_combined'] = ['unknown', 'stromal', 'stromal', 'stromal', 'immune',
                                  'immune', 'stromal', 'immune', 'immune', 'stromal',
                                  'stromal', 'immune', 'immune', 'stromal', 'immune',
                                  'unknown', 'unknown', 'stromal', 'immune', 'unknown',
                                  'unknown', 'stromal', 'stromal']

IMM_STR_LABELS['22-010A_combined'] = ['immune', 'stromal', 'stromal', 'immune', 'immune',
                                      'unknown', 'stromal', 'immune', 'immune', 'unknown',
                                      'immune', 'stromal', 'unknown', 'stromal', 'immune',
                                      'immune', 'unknown', 'stromal', 'unknown', 'stromal',
                                      'stromal', 'stromal', 'unknown', 'immune', 'unknown']

IMM_STR_LABELS['22-010C_combined'] = ['immune', 'immune', 'immune', 'stromal', 'unknown',
                                      'stromal', 'immune', 'stromal', 'unknown', 'immune',
                                      'stromal', 'immune', 'stromal', 'stromal', 'unknown',
                                      'immune', 'stromal', 'stromal', 'unknown', 'stromal',
                                      'immune', 'stromal', 'immune', 'stromal', 'stromal',
                                      'unknown']

IMM_STR_LABELS['23-009_combined'] = ['immune', 'unknown', 'immune', 'immune', 'immune',
                                     'stromal', 'immune', 'immune', 'unknown', 'stromal',
                                     'stromal', 'stromal', 'stromal', 'immune', 'immune',
                                     'unknown', 'stromal', 'immune', 'unknown', 'stromal',
                                     'stromal', 'stromal', 'unknown', 'immune', 'stromal',
                                     'stromal', 'immune', 'unknown', 'immune', 'unknown']

IMM_STR_LABELS['BOB_XP12446'] = ['unknown', 'immune', 'unknown', 'immune', 'immune',
                                 'stromal', 'unknown', 'stromal', 'stromal', 'stromal',
                                 'immune', 'stromal', 'immune', 'stromal', 'stromal',
                                 'immune', 'immune', 'unknown', 'immune', 'stromal',
                                 'unknown', 'immune']

IMM_STR_LABELS['22-010A_XP13083'] = ['unknown', 'immune', 'immune', 'unknown', 'immune',
                                     'stromal', 'stromal', 'immune', 'immune', 'unknown',
                                     'immune', 'stromal', 'unknown', 'unknown', 'stromal',
                                     'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                     'stromal', 'stromal', 'stromal', 'immune', 'stromal']

IMM_STR_LABELS['22-010C_XP13087'] = ['immune', 'unknown', 'immune', 'unknown', 'immune',
                                     'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                     'stromal', 'immune', 'immune', 'stromal', 'stromal',
                                     'stromal', 'stromal', 'stromal', 'unknown', 'stromal',
                                     'immune', 'unknown', 'unknown']

IMM_STR_LABELS['23-009_XP13086'] = ['immune', 'immune', 'immune', 'immune', 'stromal',
                                    'immune', 'immune', 'stromal', 'stromal', 'stromal',
                                    'stromal', 'stromal', 'stromal', 'immune', 'unknown',
                                    'unknown', 'stromal', 'stromal', 'stromal', 'immune',
                                    'immune', 'immune', 'stromal', 'stromal', 'immune']

# Task: layer 2 of cell type annotation
# epithelial
e_labels = ['luminal', 'luminal', 'luminal', 'luminal', 'undefined_epithelial',
            'luminal', 'undefined_epithelial', 'luminal', 'myoepithelial', 'luminal',
            'myoepithelial', 'luminal', 'myoepithelial', 'luminal', 'undefined_epithelial',
            'myoepithelial', 'luminal', 'myoepithelial', 'luminal', 'myoepithelial',
            'luminal', 'luminal', 'luminal', 'myoepithelial', 'luminal',
            'myoepithelial', 'myoepithelial', 'luminal', 'myoepithelial', 'luminal',
            'undefined_epithelial', 'luminal', 'luminal', 'luminal', 'luminal',
            'luminal', 'luminal', 'luminal', 'luminal', 'undefined_epithelial']

# immune and stromal together due to the number of markers


"""
Task: Cell type annotation for 3 TMAs only (22-010A, 22-010C, 23-009)
"""
layer2_epi = ['H3K27me3+', 'Myoepithelial', 'Cluster2', 'Undefined Epithelial', 'CK8/18+Luminal',
              'Ki67+', 'CK8/18+Luminal', 'Basal', 'CK5+CK7+Ki67+Basaluminal', 'CK7+Luminal',
              'CK8/18+Ki67+Luminal', 'Myoepithelial', 'B-Catenin', 'CK7+Luminal', 'Basal',
              'Ki67+', 'CK8/18+Ki67+Luminal', 'Basal', 'Undefined Epithelial', 'Ki67+Myoepithelial',
              'CK7+CK8/18+Luminal', 'Cluster21', 'CK8/18+Luminal', 'CK7+CK8/18+Luminal', 'H3K27me3+Luminal']

layer1_tme = ['Stromal', 'Immune', 'Stromal', 'Stromal', 'Immune',
              'Immune', 'Stromal', 'Stromal', 'Immune', 'Stromal',
              'Immune', 'Stromal', 'Immune', 'Stromal', 'Immune',
              'Stromal', 'Immune', 'Stromal', 'Stromal', 'Stromal']

layer2_tme = []

"""
Task: Cell type annotation for 3 TMAS (cohort_specific, TNBC)
"""
tme_layer1 = ['Immune', 'Immune', 'Immune', 'Stromal', 'Immune',
              'Undefined', 'Immune', 'Immune', 'Immune', 'Stromal',
              'Stromal', 'Stromal', 'Stromal', 'Stromal', 'Immune']

tme_layer2 = ['Blood Endothelial', 'Fibroblast', 'T Cell', 'T Cell', 'Macrophage',
              'B Cell', 'Macrophage', 'Smooth Muscle Cell', 'Smooth Muscle Cell', 'TME Cluster 9',
              'Blood Endothelial', 'Vimentin+', 'Vimentin+', 'Macrophage', 'Macrophage',
              'T Cell', 'Vimentin+', 'Macrophage', 'Fibroblast']

epi_layer2 = ['Epithelial CK_low', 'Epithelial CK5+', 'Epithelial H3K27me3+', 'Epithelial CK5+', 'Ki67+',
              'Epithelial CK8-18+', 'Epithelial CK7+', 'Myoepithelial', 'Myoepithelial', 'Epithelial CK5+CK7+',
              'Epithelial CK8-18+', 'Epithelial CK8-18+', 'Myoepithelial', 'Epithelial H3K27me3+',
              'Epithelial CK8-18+',
              'Epithelial CK_low', 'Myoepithelial H3K27me3+', 'Epithelial CK8-18+', 'Ki67+', 'Epithelial CK5+CK7+',
              'Epithelial CK7+', 'Epithelial CK5+CK8-18+', 'Epithelial CK5+', 'Epithelial CK5+CK7+',
              'Epithelial CK7+',
              'Epithelial CK7+', 'Epithelial CK_low', 'Epithelial CK8-18+', 'Epithelial H3K27me3+']

"""
Task: Cell type annotation for 5 TMAs (BOB, 22-010A, 22-010C, 23-005 (7), 23-009)
"""
tme_layer1 = ["Stromal", 'Immune', 'Immune', 'Stromal', 'Stromal',
              'Stromal', 'Stromal', 'Immune', 'Stromal', 'Immune',
              'Immune', 'Immune', 'Immune', 'Stromal', 'Immune',
              'Immune', 'Stromal', 'Stromal', 'Immune']

layer2_epi = ['CK7+CK8-18+', 'SMA+', 'CK8-18+', 'SMA+', 'CK8-18+',
              'CK5+CK7+', 'SMA+', 'CK_low', 'CK5+', 'CK_low',
              'CK8-18+', 'SMA+', 'CK7+CK8-18+', 'CK7+CK8-18+', 'CK_low',
              'CK5+', 'CK_low', 'CK_low', 'CK5+CK7+SMA+', 'CK_low',
              'CK_low', 'CK7+CK8-18+SMA+', 'CK5+CK7+CK8-18+', 'CK5+', 'CK5+CK7+SMA+',
              'CK5+CK7+', 'CK7+CK8-18+SMA+', 'CK7+', 'CK5+', 'CK_low',
              'CK_low', 'CK7+CK8-18+', 'CK5+', 'CK7+', 'CK5+CK7+CK8-18+SMA+',
              'CK5+SMA+', 'CK7+CK8-18+', 'CK5+CK7+', 'CK_low', 'CK7+CK8-18+SM+',
              'CK5+', 'CK7+', 'CK5+', 'CK_low', 'CK8-18+',
              'CK_low']
layer2_epi = ['Epithelial '+x for x in layer2_epi]

tme_layer2 = ['Blood Endothelial', 'T Cell', 'T Cell', 'Smooth Muscle Cell', 'Smooth Muscle Cell',
              'Fibroblast', 'Blood Endothelial', 'Macrophage', 'Fibroblast', 'Macrophage',
              'B Cell', 'Macrophage', 'T Cell', 'Vimentin+', 'B Cell',
              'B Cell', 'Fibroblast', 'vWF+', 'B Cell']


