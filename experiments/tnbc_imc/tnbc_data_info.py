"""
@author: Tina Hsu
Storing all the information of TNBC IMC dataset.
"""
# EXPERIMENT_LIST includes all the experiments
EXPERIMENT_LIST = ["XP12446", "XP13085",
                   "XP12813", "XP13083",
                   "XP13087", "XP13084",
                   "XP12814", "XP13086",
                   "XP12274r1", "XP12274r23", "XP12274r34"]
# "XP13328" and "XP13364" are still in progress (only few images)


# EXPERIMENT_MARKERS stores the passed markers of each experiment
EXPERIMENT_MARKERS = {}
EXPERIMENT_MARKERS["XP12446"] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP13085'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                 'erbB2',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP12813'] = ['CK5', 'CK7', 'CK8-18', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP13083'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP13087'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                 'ER-D8H8', 'ER-SP1', 'PR-SP2', 'erbB2',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP13084'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP12814'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'EZH2', 'Ki67', 'EGFR',
                                 'GATA3', 'p53', 'ER-D8H8', 'ER-SP1', 'erbB2',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin']
EXPERIMENT_MARKERS['XP13086'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                 'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                 'ER-D8H8', 'ER-SP1', 'PR-SP2',
                                 'CD3e', 'CD20', 'CD45', 'CD68',
                                 'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP12274r1'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                   'B-Catenin', 'H3K27me3', 'Ki67', 'p53', 'PR-SP2',
                                   'erbB2',
                                   'CD3e', 'CD20', 'CD45', 'CD68',
                                   'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP12274r23'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                   'B-Catenin', 'H3K27me3', 'Ki67', 'GATA3', 'p53',
                                   'ER-D8H8', 'ER-SP1', 'PR-SP2', 'erbB2',
                                   'CD3e', 'CD20', 'CD45', 'CD68',
                                   'CD31', 'SMA', 'Vimentin', 'vWF']
EXPERIMENT_MARKERS['XP12274r34'] = ['CK5', 'CK7', 'CK8-18', 'CK19', 'panCK',
                                   'B-Catenin', 'H3K27me3', 'Ki67', 'p53', 'ER-D8H8',
                                   'ER-SP1', 'PR-SP2', 'erbB2',
                                   'CD3e', 'CD20', 'CD45', 'CD68',
                                   'CD31', 'SMA', 'Vimentin', 'vWF']




