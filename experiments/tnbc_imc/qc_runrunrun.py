"""
@author: Tina Hsu
Marker quality control for spatial data. (main, call functions)
"""
from tnbc_imc_marker_quality_control import run_qc


if __name__ == '__main__':
    """
    Give parameters and call QC pipeline (main function)
    """

    # data setting parameters
    TYPE = "tnbc_imc_fluidigm"
    EXPERIMENT = "22-010B"

    # boolean setting for each step
    EXP_MATCH = True
    EXP_PLOT = True
    LOC_MATCH = False
    LOC_PLOT = True
    INTERSECTION_PLOT = True

    run_qc(disease_type=TYPE, experiment=EXPERIMENT,
           exp_match=EXP_MATCH, exp_plot=EXP_PLOT,
           loc_match=LOC_MATCH, loc_plot=LOC_PLOT,
           intersection_plot=INTERSECTION_PLOT)


