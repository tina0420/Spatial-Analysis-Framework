"""
@author: Tina Hsu
Comparing the experiments from the same TMA, ESI identification and distribution
"""
import os
from pathlib import Path
from experiments.tnbc_imc.tnbc_imc_setup_xp import *
from anndata import read_h5ad
from src.annotate import update_labels


def update_manual_labels(adata, old_cluster_key, new_cluster_key, new_labels, out_dir, name):
    """
    Update manual annotation to the immune, stromal and unknown cell types
    :param adata: Anndata
    :param old_cluster_key: indicate the column with the reference for mapping new labels, type: str
    :param new_cluster_key: indicate the column we need to update with the manual annotations, type: str
    :param new_labels: new labels for update, type: np.Array
    :param out_dir: output directory, type: Path
    :param name: extra name for the output file, type: str
    :return:
    """
    temp_adata = adata.copy()
    label_name_dict = dict(zip(range(len(new_labels)), new_labels))

    update_labels(temp_adata, new_cluster_key, [label_name_dict[int(i)] for i in adata.obs[old_cluster_key]])

    temp_adata.write(out_dir / f"{name}_non-epithelial.h5ad")


def update_all_labels(experiment, cluster_key, out_dir):
    """
    Update all the annotation together into the merged one
    :param experiment: experiment name, type: str
    :param cluster_key: indicate the column in obs, type: str
    :param out_dir: output directory, type: Path
    :return:
    """
    temp_adata = read_h5ad(out_dir / f"{experiment}_merge.h5ad")
    temp_epi = read_h5ad(out_dir / f"{experiment}_epithelial.h5ad")
    temp_tme = read_h5ad(out_dir / f"{experiment}_non-epithelial.h5ad")

    new_labels = np.zeros(temp_adata.shape[0], dtype=object)
    epi_labels = np.array(['epithelial'] * temp_epi.shape[0])
    tme_labels = temp_tme.obs[cluster_key].to_numpy()

    new_labels[np.where(temp_adata.obs[cluster_key] == 'epithelial')[0]] = epi_labels
    new_labels[np.where(temp_adata.obs[cluster_key] != 'epithelial')[0]] = tme_labels

    update_labels(temp_adata, cluster_key, new_labels)

    temp_adata.write(out_dir / f"{experiment}_merge.h5ad")


if __name__ == '__main__':
    # parameter setting
    TYPE = 'tnbc_imc'
    TMA = '22-010C'

    EXPERIMENT_1 = TMA_XP[TMA][0]
    EXPERIMENT_2 = TMA_XP[TMA][1]

    in_dir = Path(f"analysis/{TYPE}/{TMA}_test")
    out_dir = Path(f"analysis/{TYPE}/{TMA}_test")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # manual annotate the immune, stromal and unknown cell types
    IMM_STR_LABELS = {}

    # 22-010C
    # IMM_STR_LABELS[TMA] = ['stromal', 'stromal', 'unknown', 'unknown', 'unknown',
    #                        'unknown', 'stromal', 'unknown', 'unknown', 'unknown',
    #                        'unknown', 'immune', 'immune', 'immune', 'unknown',
    #                        'unknown', 'immune', 'unknown', 'immune', 'stromal',
    #                        'unknown', 'immune', 'unknown']
    #
    # IMM_STR_LABELS[EXPERIMENT_1] = ['immune', 'unknown', 'stromal', 'stromal', 'unknown',
    #                                 'immune', 'stromal', 'unknown', 'unknown', 'unknown',
    #                                 'unknown', 'unknown', 'unknown', 'immune', 'stromal',
    #                                 'stromal', 'immune', 'unknown', 'unknown', 'unknown',
    #                                 'unknown', 'unknown', 'immune', 'unknown']
    #
    # IMM_STR_LABELS[EXPERIMENT_2] = ['unknown', 'immune', 'unknown', 'unknown', 'stromal',
    #                                 'stromal', 'unknown', 'unknown', 'unknown', 'unknown',
    #                                 'stromal', 'unknown', 'immune', 'immune', 'immune',
    #                                 'unknown', 'unknown', 'stromal', 'stromal', 'immune',
    #                                 'immune']

    # update annotation
    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = IMM_STR_LABELS[TMA]
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    adata_1 = read_h5ad(out_dir / f"{EXPERIMENT_1}_non-epithelial.h5ad")
    labels = IMM_STR_LABELS[EXPERIMENT_1]
    update_manual_labels(adata_1, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, EXPERIMENT_1)

    adata_2 = read_h5ad(out_dir / f"{EXPERIMENT_2}_non-epithelial.h5ad")
    labels = IMM_STR_LABELS[EXPERIMENT_2]
    update_manual_labels(adata_2, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, EXPERIMENT_2)

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)
    update_all_labels(EXPERIMENT_1, CLUSTER_KEY, out_dir)
    update_all_labels(EXPERIMENT_2, CLUSTER_KEY, out_dir)

    # CHECK
    adata = read_h5ad(out_dir / f"{TMA}_merge.h5ad")






    imm_str_labels = {}
    imm_str_labels[TMA] = ['stromal', 'stromal', 'unknown', 'unknown', 'unknown',
                           'immune', 'stromal', 'stromal', 'unknown', 'unknown',
                           'unknown', 'immune', 'immune', 'immune', 'immune',
                           'unknown', 'immune', 'unknown', 'immune', 'stromal',
                           'unknown', 'immune', 'unknown']

    imm_str_labels[EXPERIMENT_1] = ['immune', 'unknown', 'stromal', 'stromal', 'unknown',
                                    'immune', 'stromal', 'unknown', 'unknown', 'unknown',
                                    'unknown', 'unknown', 'unknown', 'immune', 'stromal',
                                    'stromal', 'immune', 'unknown', 'unknown', 'unknown',
                                    'unknown', 'unknown', 'immune', 'unknown']

    imm_str_labels[EXPERIMENT_2] = ['unknown', 'immune', 'unknown', 'unknown', 'stromal',
                                    'stromal', 'unknown', 'unknown', 'unknown', 'unknown',
                                    'stromal', 'unknown', 'immune', 'immune', 'immune',
                                    'unknown', 'unknown', 'stromal', 'stromal', 'immune',
                                    'immune']

    # update the labels for each non-epithelial.h5ad
    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA]
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    adata_1 = read_h5ad(out_dir / f"{TMA_XP[TMA][0]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][0]]
    update_manual_labels(adata_1, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][0])

    adata_2 = read_h5ad(out_dir / f"{TMA_XP[TMA][1]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][1]]
    update_manual_labels(adata_2, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][1])

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][0], CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][1], CLUSTER_KEY, out_dir)

    # update the labels
    imm_str_labels = {}
    # 22-010A
    imm_str_labels[TMA] = ['unknown', 'stromal', 'immune', 'immune', 'stromal',
                           'immune', 'immune', 'immune', 'stromal', 'stromal',
                           'stromal', 'immune', 'immune', 'immune', 'stromal',
                           'stromal', 'stromal', 'stromal', 'immune', 'immune',
                           'unknown']

    # XP12813
    imm_str_labels[TMA_XP[TMA][0]] = ['immune', 'stromal', 'stromal', 'immune', 'immune',
                                      'immune', 'unknown', 'immune', 'stromal', 'unknown',
                                      'stromal', 'stromal', 'unknown', 'unknown', 'immune',
                                      'stromal', 'immune', 'immune', 'stromal', 'stromal',
                                      'unknown', 'stromal', 'immune', 'unknown']

    # XP13083
    imm_str_labels[TMA_XP[TMA][1]] = ['unknown', 'stromal', 'immune', 'unknown', 'immune',
                                      'stromal', 'immune', 'immune', 'immune', 'immune',
                                      'stromal', 'immune', 'stromal', 'unknown', 'unknown',
                                      'unknown', 'stromal', 'unknown', 'unknown', 'stromal',
                                      'stromal', 'stromal', 'immune', 'immune', 'unknown']

    # update the labels for each non-epithelial.h5ad
    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA]
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    adata_1 = read_h5ad(out_dir / f"{TMA_XP[TMA][0]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][0]]
    update_manual_labels(adata_1, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][0])

    adata_2 = read_h5ad(out_dir / f"{TMA_XP[TMA][1]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][1]]
    update_manual_labels(adata_2, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][1])

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][0], CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][1], CLUSTER_KEY, out_dir)

    # update the labels
    # 23-009
    TMA = '23-009'
    imm_str_labels['23-009'] = ['stromal', 'immune', 'immune', 'stromal', 'stromal',
                                'immune', 'immune', 'immune', 'stromal', 'immune',
                                'stromal', 'unknown', 'unknown', 'stromal', 'stromal',
                                'immune', 'unknown', 'immune', 'immune', 'immune',
                                'unknown']

    imm_str_labels[TMA_XP['23-009'][0]] = ['immune', 'stromal', 'stromal', 'unknown', 'immune',
                                           'stromal', 'immune', 'unknown', 'unknown', 'unknown',
                                           'unknown', 'stromal', 'immune', 'unknown', 'immune',
                                           'stromal', 'unknown', 'stromal', 'immune', 'unknown',
                                           'immune', 'unknown', 'stromal', 'stromal', 'unknown',
                                           'immune', 'unknown']

    imm_str_labels[TMA_XP['23-009'][1]] = ['stromal', 'immune', 'unknown', 'stromal', 'immune',
                                           'unknown', 'stromal', 'immune', 'immune', 'immune',
                                           'immune', 'stromal', 'unknown', 'stromal', 'immune',
                                           'unknown', 'immune', 'unknown', 'stromal', 'unknown',
                                           'immune', 'unknown', 'immune', 'stromal', 'stromal',
                                           'immune']

    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA]
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    adata_1 = read_h5ad(out_dir / f"{TMA_XP[TMA][0]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][0]]
    update_manual_labels(adata_1, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][0])

    adata_2 = read_h5ad(out_dir / f"{TMA_XP[TMA][1]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][1]]
    update_manual_labels(adata_2, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][1])

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][0], CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][1], CLUSTER_KEY, out_dir)

    # update the labels
    # BOB
    TMA = 'BOB'
    imm_str_labels['BOB'] = ['stromal', 'immune', 'stromal', 'stromal', 'stromal',
                             'immune', 'immune', 'immune', 'immune', 'stromal',
                             'stromal', 'unknown', 'stromal', 'immune', 'immune',
                             'immune', 'stromal', 'stromal']

    imm_str_labels[TMA_XP['BOB'][0]] = ['unknown', 'unknown', 'unknown', 'immune', 'unknown',
                                        'stromal', 'unknown', 'unknown', 'immune', 'stromal',
                                        'unknown', 'unknown', 'unknown', 'stromal', 'unknown',
                                        'stromal', 'unknown', 'immune']

    imm_str_labels[TMA_XP['BOB'][1]] = ['unknown', 'immune', 'unknown', 'unknown', 'immune',
                                        'stromal', 'unknown', 'stromal', 'unknown', 'unknown',
                                        'immune', 'stromal', 'unknown', 'unknown', 'stromal',
                                        'stromal', 'unknown', 'unknown', 'immune', 'unknown']

    OLD_CLUSTER_KEY = 'leiden'
    NEW_CLUSTER_KEY = 'esi'

    adata = read_h5ad(out_dir / f"{TMA}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA]
    update_manual_labels(adata, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA)

    adata_1 = read_h5ad(out_dir / f"{TMA_XP[TMA][0]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][0]]
    update_manual_labels(adata_1, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][0])

    adata_2 = read_h5ad(out_dir / f"{TMA_XP[TMA][1]}_non-epithelial.h5ad")
    labels = imm_str_labels[TMA_XP[TMA][1]]
    update_manual_labels(adata_2, OLD_CLUSTER_KEY, NEW_CLUSTER_KEY, labels, out_dir, TMA_XP[TMA][1])

    # all together in merge.h5ad
    CLUSTER_KEY = 'esi'
    update_all_labels(TMA, CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][0], CLUSTER_KEY, out_dir)
    update_all_labels(TMA_XP[TMA][1], CLUSTER_KEY, out_dir)

    # check
    adata = read_h5ad(out_dir / f"{TMA}_merge.h5ad")
