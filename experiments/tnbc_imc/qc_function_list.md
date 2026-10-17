Marker quality control for spatial datasets.
---
_____________________________________________

Move the plotting function to src.visualize (marker_corr_heatmap_plot in marker_quality_control.py)


### marker_quality_control.py
- `run_qc(disease_type, experiment, exp_match=True, exp_plot=True, loc_match=False, loc_plot=True, intersection_plot=True)`
- `get_exp_corr(adata, out_dir, name, match_info=True, plot_corr=True)`
- `get_loc_corr(adata, out_dir, name, match_info=False, plot_corr=True)`
- `get_marker_correlation(out_dir, name, plot_corr=True)`
- `plot_marker_correlations_heatmap(out_dir, corr_df, figsize, name=None)`

### qc_runrunrun.py
- `run_qc(disease_type, experiment, exp_match=True, exp_plot=True, loc_match=False, loc_plot=True, intersection_plot=True)`



