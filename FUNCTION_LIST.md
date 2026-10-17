# Sakura
A package for integrated spatial analysis for single-cell digital pathology.

## List of 64 Functions by Module

### src.create
- `simulate_spatial_ROI(r_rois, i_cells, j_genes, k_clusters, pixel_size, mixture_type,
                         init_pos=(0, 0), seed=None)`

### src.extract
- `extract_mcd_for_imc(in_dir, out_dir)`
- `process_qupath(path_dir)`
- `process_cellprofiler(path_dir, set_count=None)`


### src.segment
- `run_mesmer(nuclear_channel, cytoplasm_channels, in_dir, out_dir, deepcell_token,
               mes_image_mpp, mes_interior_threshold=0.2, mes_maxima_threshold=0.075,
               kernel_size=3, norm_method="minmax", resize=False,
               plot_mask=False, plot_overlay=False, output_pred_df=False)`


### src.preprocess
- `quantify_expression(path_dir, is_sparse=False)`
- `normalize(adata, method="arcsinh", arcsinh_cofactor=0.8, clip_percentile=99.5)`
- `read_neighbors(adata, rel_path="rel.csv", threshold=30)`
- `read_to_anndata_wrapper(path_dir, is_sparse=False)`


### src.annotate
- `update_labels(adata, cluster_key, labels)`
- `leiden_subset(adata, columns, n_neighbors=15, n_pcs=None)`
- `louvain_subset(adata, columns, n_neighbors=15, n_pcs=None)`


### src.annotate_evaluation
- `davies_bouldin(exp_mtx, labels)`
- `get_agreement_indices(labels_i, labels_j)`
- `plot_cluster_agreement_panel(adata, cluster_key, agree_idxs, out_dir, dotsize=1)`
- `plot_cluster_agreement_summary(adata, agree_idxs, disagree_idxs, out_dir, figsize=(3.5, 2))`
- `plot_cluster_disagreement_matrix(labels_i, labels_j, out_dir, figsize=(7, 7), annot=False, annot_size=8)`


### src.spatial_stats
- `get_labeled_connected_components(adata, cluster_key, label)`
- `get_mean_expression(adata, cell_idxs)`
- `get_centroid_cell_idx(adata, cell_idxs)`
- `get_peri_centroid_cell_idxs(adata, centroid_idx, radius)`
- `get_distance(cell_u, cell_v)`
- `get_cell_distribution(adata, cell_idxs, cluster_key)`
- `get_mean_distance_from_centroid(adata, centroid_idx, cell_idxs, cluster_key)`
- `get_global_mean_distance(adata)`
- `get_interacting_pairs(adata, cluster_key, type_i, type_j)`
- `get_filtered_interacting_pairs(adata, cluster_key, type_i, type_j, col_idx_i, col_idx_j, thres_i, thres_j)`
- `get_spatial_cluster_cell_idxs(adata, cluster_key, list_of_spatial_cluster_id)`
- `get_densest_point(adata)`
- `get_densest_cell_idx(adata)`
- `get_hot_spots_graph(adata, col_name)`
- `get_hot_spots_matrix(col_spot_mtx, window_size=3, weight_mtx=None)`
- `get_ligand_receptor_accessibility(adata, cluster_key, type_i, type_j, col_idx_i, col_idx_j, thres_i, thres_j,
                                      radius=150, decay_sigma=10)`
- `get_interacting_neighborhood_cell_idxs(adata, cell_idxs)`


### src.squidpy_stats
- `run_squidpy_statistics(adata, cluster_key)`
- `get_squidpy_data(adata, cluster_key)`
- `get_interaction_entropy(adata, cluster_key)`
- `get_spatially_variable_genes(adata, cluster_key)`


### src.visualize
- `plot_cell_area(adata, out_dir, figsize=(3.5, 2), name=None)`
- `plot_cell_expression_violin(adata, out_dir, figsize=(7, 3), name=None)`
- `plot_cell_expression_violin_split(adata, out_dir, set_count=8, figsize=(7, 3), name=None)`
- `plot_cell_expression_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_zscore_cell_expression_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_spatial_expression_panel(adata, out_dir, name=None)`
- `plot_spatial_correlation(adata, out_dir, threshold=0, figsize=(15, 12), plot_coeff=False, fontsize=15, name=None)`
- `plot_umap(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=1, name=None)`
- `plot_umap_panel(adata, cluster_key, out_dir, dotsize=0.1, name=None)`
- `plot_cluster_mean_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_zscore_cluster_mean_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_cluster_heatmap(adata, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_zscore_cluster_heatmap(adata, column_names, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_cluster_cell_count(adata, cluster_key, out_dir, figsize=(3.5, 2), name=None)`
- `plot_spatial_cluster_distribution(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=1, name=None)`
- `plot_spatial_cluster_panel(adata, cluster_key, out_dir, figsize=(7, 7), dotsize=0.1, name=None)`
- `plot_edge_count(adata, cluster_key, out_dir, figsize=(7, 7), rel_zero_indexing=True, annot=False, annot_size=8,
                    name=None)`
- `plot_cluster_cooccurrence(adata, cluster_key, label, out_dir, name=None)`
- `plot_cluster_interaction_matrix(adata, cluster_key, out_dir, figsize=(7, 7), name=None)`
- `plot_cluster_ripley_l(adata, cluster_key, out_dir, name=None)`
- `plot_neighbor_graph(adata, out_dir, figsize=(7, 7), dotsize=1, linewidth=0.2, name=None)`
- `plot_observation_distribution(adata, obs_key, out_dir, figsize=(7, 7), cell_idxs=None,
                                  color_non_cell_idxs=False, cmap="coolwarm", dotsize=1, name=None)`


### src.enrichment
- `get_adjacency_enrichment_matrix(adata, cluster_key)`
- `adjacency_enrichment_score(adata, cluster_key, type_i, type_j)`
- `get_neg_binomial_regression_params(cell_count_arr)`
- `plot_neg_binomial_regression(r, p, out_dir, figsize=(5, 5))`