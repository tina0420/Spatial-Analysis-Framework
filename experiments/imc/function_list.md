# Spatial Analysis framework

## List of functions in Module

### src.segmentation
- `run_mesmer(nuclear_channel, cytoplasm_channels, in_dir, out_dir, deepcell_token,
  mes_image_mpp, mes_interior_threshold=0.2, mes_maxima_threshold=0.075,
  kernel_size=3, norm_method="minmax", resize=False,
  plot_mask=False, plot_overlay=False, output_pred_df=False)`


### marker_quality_control.py
- `run_qc(disease_type, experiment, exp_match=True, exp_plot=True, loc_match=False, loc_plot=True, intersection_plot=True)`
- `get_exp_corr(adata, out_dir, name, match_info=True, plot_corr=True)`
- `get_loc_corr(adata, out_dir, name, match_info=False, plot_corr=True)`
- `get_marker_correlation(out_dir, name, plot_corr=True)`
- `plot_marker_correlations_heatmap(out_dir, corr_df, figsize, name=None)`

### clustering.py
- `run_clustering(adata, cluster_key, marker_list, method, resolution=1.0)`
- `compare_clustering_resolutions(adata, resolutions, method)`

### eda.py
- ``
- ``




### visualization
- `plot_marker_correlations_heatmap(out_dir, corr_df, figsize=(7, 7), name=None)`
- `plot_marker_nuclei_staining(marker_list, nuclei_name, in_dir, out_dir)`
- `plot_marker_nuclear(x, out_dir, name)`
- `plot_zscore_cluster_mean_heatmap(adata, column_names, cluster_key, out_dir, vmax=None, vmin=None, figsize=(7, 7),
                                     name=None)`
- `plot_compare_clustering_line(adata, method, out_dir, figsize=(5, 7), name=None)`
- `plot_spatial_cluster_distribution(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=1,
                                      color=COLORS, name=None, dpi=150)`
- `plot_cluster_cell_count(adata, cluster_key, out_dir, figsize=(3.5, 2), name=None, dpi=150)`
- `plot_cell_area_count(adata, out_dir, figsize=(3.5, 2), name=None, dpi=150)`
- 


### analysis_setup.py
This file includes the customized settings for the experiment such as colour dictionaries for different groups.

