"""
@author: Tina Hsu
Functions for basic spatial statistics.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde
from scipy.spatial import ConvexHull, distance_matrix
from scipy.sparse import issparse
from shapely.geometry import Point, Polygon


def get_labeled_connected_components(adata, cluster_key, label):
    """
    Get connected components with label of interest.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param label: label of interest, type: int
    :return: connected components with label of interest, type: list
    """
    neighbors = adata.uns["relation"]

    # Get indices of neighbors with label
    label_idxs = np.where(adata.obs[cluster_key] == label)[0]
    neighbors_with_label = neighbors[np.where(pd.Series(neighbors[:, 0]).isin(label_idxs) &
                                              pd.Series(neighbors[:, 1]).isin(label_idxs))[0]]

    # Get number of components
    adj_list = __neighbors_to_adjacency_list(neighbors=neighbors_with_label)
    conn_comps = __get_connected_components(adjacency_list=adj_list)

    return conn_comps


def __neighbors_to_adjacency_list(neighbors):
    """
    Transform neighbors to adjacency list.
    :param neighbors: array of neighbor relations in AnnData, type: np.array
    :return: adjacency list, type: dict
    """
    adj_list = {}

    for cell_i, cell_j in neighbors:
        if cell_i not in adj_list:
            adj_list[cell_i] = []

        if cell_j not in adj_list:
            adj_list[cell_j] = []

        adj_list[cell_i].append(cell_j)
        adj_list[cell_j].append(cell_i)

    return adj_list


def __get_connected_components(adjacency_list):
    """
    Get connected components from adjacency list.
    :param adjacency_list: adjacency list, type: dict
    :return: connected components, type: list[list]
    """
    visited = {cell: False for cell in adjacency_list}
    components = [__depth_first_search(cell, adjacency_list, visited) for cell in adjacency_list if not visited[cell]]

    return components


def __depth_first_search(cell, adjacency_list, visited):
    """
    Depth first search.
    :param cell: index of cell of interest, type: int
    :param adjacency_list: adjacency list, type: dict
    :param visited: visited list, type: dict
    :return: single component, type: list
    """
    component = []
    stack = [cell]

    while stack:
        curr = stack.pop()
        if not visited[curr]:
            visited[curr] = True
            component.append(curr)

            for neighbor in adjacency_list[curr]:
                if not visited[neighbor]:
                    stack.append(neighbor)

    return component


def get_mean_expression(adata, cell_idxs):
    """
    Get mean expression across cells of interest.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :return: mean expression, type: np.array
    """
    if issparse(adata.X):
        exp_arr = adata.X[cell_idxs]
        mean_exp = np.mean(exp_arr.toarray(), axis=0)
    else:
        exp_arr = adata.X[cell_idxs]
        mean_exp = np.mean(exp_arr, axis=0)

    return mean_exp


def get_centroid_cell_idx(adata, cell_idxs):
    """
    Get index of centroid across cells of interest.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :return: centroid index, type: int
    """
    location_cells = adata.obsm["spatial"][cell_idxs]
    centroid = np.mean(location_cells, axis=0)
    centroid_idx = cell_idxs[np.argmin(np.linalg.norm(location_cells - centroid, axis=1))]

    return centroid_idx


def get_peri_centroid_cell_idxs(adata, centroid_idx, radius):
    """
    Get indices of all cells surrounding centroid in a defined radius.
    :param adata: AnnData
    :param centroid_idx: centroid index, type: int
    :param radius: radius of interest, type: float
    :return: list of indices, type: list
    """
    location = adata.obsm["spatial"]
    centroid = location[centroid_idx]

    # Get indices of neighbors within square to limit selection
    min_x, max_x = centroid[0] - radius, centroid[0] + radius
    min_y, max_y = centroid[1] - radius, centroid[1] + radius
    idxs_within_square = np.where((location[:, 0] <= max_x) & (location[:, 0] >= min_x) &
                                  (location[:, 1] >= min_y) & (location[:, 1] <= max_y))[0]

    # Get indices of neighbors within radius
    idxs_within_radius = [idx for idx in idxs_within_square if get_distance(centroid, location[idx]) <= radius]

    return idxs_within_radius


def get_distance(cell_u, cell_v):
    """
    Distance between two cells.
    :param cell_u: first cell, type: tuple/list
    :param cell_v: second cell, type: tuple/list
    :return: distance, type: float
    """
    return ((cell_u[0] - cell_v[0]) ** 2 + (cell_u[1] - cell_v[1]) ** 2) ** (1 / 2)


def get_cell_distribution(adata, cell_idxs, cluster_key):
    """
    Get cell distribution across cells of interest.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :return: cell distribution, type: pd.Series
    """
    labels_ser = adata.obs[cluster_key].iloc[cell_idxs]
    value_counts_ser = labels_ser.value_counts().sort_index()

    return value_counts_ser


def get_mean_distance_from_centroid(adata, centroid_idx, cell_idxs, cluster_key):
    """
    Get mean distance to cell types across cells of interest.
    :param adata: AnnData
    :param centroid_idx: centroid index, type: int
    :param cell_idxs: cell indices of interest, type: np.array
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :return: mean distance to cell types, type: pd.Series
    """
    labels_ser = adata.obs[cluster_key].iloc[cell_idxs]
    centroid = adata.obsm["spatial"][centroid_idx]
    location_cells = adata.obsm["spatial"][cell_idxs]

    # Get mean distance per label
    mean_dist_arr = []
    for lbl in pd.unique(labels_ser):
        loc_with_label = location_cells[np.where(labels_ser == lbl)[0]]
        dists = np.array([get_distance(centroid, coord) for coord in loc_with_label])
        mean_dist_arr.append(np.average(dists))
    mean_dist_ser = pd.Series(mean_dist_arr, index=pd.unique(labels_ser)).sort_index()

    return mean_dist_ser


def get_global_mean_distance(adata):
    """
    Get mean distance between all cells.
    :param adata: AnnData
    :return: mean distance, type: float
    """
    location = adata.obsm["spatial"]
    neighbors = adata.uns["relation"]

    # Get mean distance across all neighbors
    dists = np.array([get_distance(location[cell_i], location[cell_j]) for cell_i, cell_j in neighbors])
    mean_dist = np.sum(dists) / len(neighbors)

    if mean_dist == float('inf'):
        raise Exception("Radius is too small")

    return mean_dist


def get_interacting_pairs(adata, cluster_key, type_i, type_j):
    """
    Get all pairs of cells of interest that are interacting.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param type_i: first cell type of interest
    :param type_j: second cell type of interest
    :return: pairs of interacting cells, type: np.array
    """
    neighbors = adata.uns["relation"]
    idx_label_map = dict(zip(range(len(adata.obs[cluster_key])), adata.obs[cluster_key]))
    mapped_arr = np.array([[idx_label_map[u], idx_label_map[v]] for u, v in neighbors])

    # Filter by cell type
    pairs = neighbors[np.where(((mapped_arr[:, 0] == type_i) | (mapped_arr[:, 1] == type_i)) &
                               ((mapped_arr[:, 0] == type_j) | (mapped_arr[:, 1] == type_j)))[0]]

    return pairs


def get_filtered_interacting_pairs(adata, cluster_key, type_i, type_j, col_idx_i, col_idx_j, thres_i, thres_j):
    """
    Get all pairs of cells of interest that are interacting satisfying expression threshold.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param type_i: first cell type of interest
    :param type_j: second cell type of interest
    :param col_idx_i: first index of column of interest
    :param col_idx_j: second index of column of interest
    :param thres_i: threshold of first index of column of interest
    :param thres_j: threshold of second index of column of interest
    :return: pairs of cells of interest that are interacting satisfying expression threshold, type: np.array
    """
    pairs = get_interacting_pairs(adata, cluster_key, type_i, type_j)

    # Get cells, expression, labels from pairs
    selected_cells = np.array(list(set(pairs.flatten())))
    selected_labels = adata.obs[cluster_key].iloc[selected_cells]

    if issparse(adata.X):
        selected_exp = adata.X[selected_cells].toarray()
    else:
        selected_exp = adata.X[selected_cells]

    # Filter by column threshold
    filtered_type_i = np.where((selected_labels == type_i) &
                               (selected_exp[:, col_idx_i].flatten() > thres_i))[0]
    filtered_type_j = np.where((selected_labels == type_j) &
                               (selected_exp[:, col_idx_j].flatten() > thres_j))[0]

    # Get cells that pass filtering
    filtered_cells = selected_cells[list(set(list(filtered_type_i) + list(filtered_type_j)))]

    # Get pairs by indices
    filtered_pairs = pairs[np.where(pd.Series(pairs[:, 0]).isin(filtered_cells) &
                                    pd.Series(pairs[:, 1]).isin(filtered_cells))[0]]

    return filtered_pairs


def get_spatial_cluster_cell_idxs(adata, cluster_key, list_of_spatial_cluster_id):
    """
    Get list of cell indices in a spatial cluster.
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param list_of_spatial_cluster_id: list of follicle idxs from dbscan to process, type: list
    :return: list of cell indices in a spatial cluster, type: list
    """
    follicle_cell_idxs = []
    for lst in list_of_spatial_cluster_id:
        # Get polygon defined by convex hull
        convex_hull_idxs = __get_convex_hull_idxs(adata, np.where(adata.obs[cluster_key].isin(lst))[0])
        polygon = Polygon(adata.obsm["spatial"][convex_hull_idxs])

        # Get indices in the polygon
        idxs_in_polygon = [i for i in range(len(adata)) if polygon.contains(Point(adata.obsm["spatial"][i]))]

        follicle_cell_idxs.append(idxs_in_polygon)

    return follicle_cell_idxs


def __get_convex_hull_idxs(adata, cell_idxs):
    """
    Get convex hull indices.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :return: convex hull indices, type: np.array
    """
    location_cells = adata.obsm["spatial"][cell_idxs]

    # Get convex hull based on cell indices
    ch = ConvexHull(location_cells)
    coord_idx = cell_idxs[ch.vertices]

    return coord_idx


def get_densest_point(adata):
    """
    Calculate the densest point
    :param adata: AnnData
    :return: x, y coordinates of the densest point, type: float, float
    """
    # Extract the x and y values
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    # Perform KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # Evaluate KDE on a grid
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = np.linspace(y.min(), y.max(), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(x_grid.shape)

    # Find the coordinates of the densest point
    max_density_idx = np.argmax(z)
    densest_x, densest_y = grid_coords[:, max_density_idx]

    return densest_x, densest_y


def get_densest_cell_idx(adata):
    """
    Get the cell index nearest to the densest point
    :param adata: AnnData
    :return: cell index, type: int
    """
    # Calculate the densest point
    dense_x, dense_y = get_densest_point(adata)

    # Calculate all distances from the densest point
    distances = np.linalg.norm(adata.obsm["spatial"] - np.array([dense_x, dense_y]), axis=1)

    # Find the densest cell index
    densest_idx = np.argmin(distances)

    return densest_idx


def get_hot_spots_graph(adata, col_name):
    """
    Detects hot spots for a graph.
    :param adata: AnnData
    :param col_name: column name of interest in observations, type: str
    :return: hot spot array, type: np.array
    """
    # Z-score critical value for a 95% confidence level
    z_critical = norm.ppf(0.975)

    # Detect hot spots by comparing Gi* to z at 95% confidence
    gi_star_arr = __calculate_gi_star_graph(adata, col_name)

    hot_spot_arr = np.array([1 if i >= z_critical else 0 for i in gi_star_arr])

    return hot_spot_arr


def __calculate_gi_star_graph(adata, col_name):
    """
    Calculating Gi*
    :param adata: AnnData
    :param col_name: column name of interest in observations, type: str
    :return: Gi*, type: np.array
    """
    # Column of interest
    data = adata.obs[col_name]

    # Create adjacency list from neighbors
    adj_list = __neighbors_to_adjacency_list(neighbors=adata.uns['relation'])

    # Calculate parameters
    n = len(adata)
    global_mean = np.mean(data)
    global_std = np.sqrt(np.sum(data ** 2) / n - global_mean ** 2)

    # Iteratively calculate Gi* over each cell in the original matrix
    gi_star_arr = np.zeros(len(adata))

    for i in range(len(adata)):
        # Base case: no neighbors
        if i not in adj_list.keys():
            continue

        # Get neighbors
        neighbors = adj_list[i]
        neighbors.append(i)
        window_sum = np.sum(np.array([data[j] for j in neighbors]))
        weight_arr = len(neighbors)

        gi_star_arr[i] = (window_sum -
                          global_mean * weight_arr) / \
                         (global_std *
                          np.sqrt((n * weight_arr - weight_arr ** 2) / (n - 1)))

    return gi_star_arr


def get_hot_spots_matrix(col_spot_mtx, window_size=3, weight_mtx=None):
    """
    Detects hot spots for a matrix (spot-based technologies).
    :param col_spot_mtx: column matrix of interest in observations, type: np.array
    :param window_size: window size parameter, type: int
    :param weight_mtx: weight matrix parameter, should match window size, type: np.array
    :return: hot spot matrix, type: np.array
    """
    # Z-score critical value for a 95% confidence level
    z_critical = norm.ppf(0.975)

    # Detect hot spots by comparing Gi* to z at 95% confidence
    gi_star_mtx = __calculate_gi_star_matrix(col_spot_mtx, window_size=window_size, weight_mtx=weight_mtx)

    hot_spot_mtx = np.zeros(col_spot_mtx.shape)

    for i in range(col_spot_mtx.shape[0]):
        for j in range(col_spot_mtx.shape[1]):
            if gi_star_mtx[i, j] >= z_critical:
                hot_spot_mtx[i, j] = 1
            else:
                hot_spot_mtx[i, j] = 0

    return hot_spot_mtx


def __calculate_gi_star_matrix(col_spot_mtx, window_size=3, weight_mtx=None):
    """
    Calculating Gi*
    :param col_spot_mtx: column matrix of interest in observations, type: np.array
    :param window_size: window size parameter, type: int
    :param weight_mtx: weight matrix parameter, should match window size, type: np.array
    :return: Gi*, type: np.array
    """
    # Pad data to handle edges
    pad_data = np.pad(col_spot_mtx, pad_width=window_size // 2, mode='constant', constant_values=0)

    # Default to unit weight matrix
    if weight_mtx is None:
        weight_mtx = np.ones((window_size, window_size))

    # Calculate parameters
    n = col_spot_mtx.size
    global_mean = np.mean(col_spot_mtx)
    global_std = np.sqrt(np.sum(col_spot_mtx ** 2) / n - global_mean ** 2)

    # Iteratively calculate Gi* over each cell in the original matrix
    gi_star_mtx = np.zeros(col_spot_mtx.shape)

    for i in range(col_spot_mtx.shape[0]):
        for j in range(col_spot_mtx.shape[1]):
            window = pad_data[i:i + window_size, j:j + window_size]
            window_values = window[weight_mtx > 0]
            gi_star_mtx[i, j] = (np.sum(window_values) -
                                 global_mean * np.sum(weight_mtx)) / \
                                (global_std *
                                 np.sqrt((n * np.sum(weight_mtx ** 2) - np.sum(weight_mtx) ** 2) / (n - 1)))

    return gi_star_mtx


def get_ligand_receptor_accessibility(adata, cluster_key, type_i, type_j, col_idx_i, col_idx_j, thres_i, thres_j,
                                      radius=150, decay_sigma=10):
    """
    Calculating ligand-receptor accessibility based on the Two-Step Floating Catchment Area (2SFCA) method
    :param adata: AnnData
    :param cluster_key: indicate which cluster key to use in obs, type: str
    :param type_i: first cell type of interest
    :param type_j: second cell type of interest
    :param col_idx_i: first index of column of interest
    :param col_idx_j: second index of column of interest
    :param thres_i: threshold of first index of column of interest
    :param thres_j: threshold of second index of column of interest
    :param radius: cell distance for catchment, type: float
    :param decay_sigma: variance parameter for Gaussian decay, type: float
    :return: pairs of cells of interest that are interacting satisfying expression threshold, type: np.array
    """
    # Get expression and labels
    selected_labels = adata.obs[cluster_key]

    if issparse(adata.X):
        selected_exp = adata.X.toarray()
    else:
        selected_exp = adata.X

    # Filter by column threshold
    filtered_type_i = np.where((selected_labels == type_i) &
                               (selected_exp[:, col_idx_i].flatten() > thres_i))[0]
    filtered_type_j = np.where((selected_labels == type_j) &
                               (selected_exp[:, col_idx_j].flatten() > thres_j))[0]

    # Get expression and location of two groups of cells
    receptor_exp = selected_exp[filtered_type_i, col_idx_i]
    receptor_loc = adata.obsm["spatial"][filtered_type_i]
    ligand_exp = selected_exp[filtered_type_j, col_idx_j]
    ligand_loc = adata.obsm["spatial"][filtered_type_j]

    # Calculate modified 2SFCA
    twosfca = __calculate_modified_twosfca(receptor_exp, receptor_loc, ligand_exp, ligand_loc,
                                           radius, decay_sigma)

    # Return accessibility for each ligand
    ligand_accessibility = np.zeros(len(selected_labels))
    ligand_accessibility[filtered_type_j] = twosfca

    return ligand_accessibility


def __calculate_modified_twosfca(receptor_exp, receptor_loc, ligand_exp, ligand_loc, radius=150, decay_sigma=10):
    """
    Calculating modified ligand-receptor Two-Step Floating Catchment Area (2SFCA)
    :param receptor_exp: receptor expression of selected cell type, type: np.array
    :param receptor_loc: receptor location of selected cell type, type: np.array
    :param ligand_exp: ligand expression of selected cell type, type: np.array
    :param ligand_loc: ligand location of selected cell type, type: np.array
    :param radius: cell distance for catchment, type: float
    :param decay_sigma: variance parameter for Gaussian decay, type: float
    :return: accessibility scores for each receptor cell, np.array
    """
    dist_matrix = distance_matrix(receptor_loc, ligand_loc)
    gaussian_decay = lambda d: np.exp(- (d ** 2) / (2 * decay_sigma ** 2))

    # Step 1: Immune Ligand to Malignant B Receptor Ratios
    weighted_receptor_pop = np.zeros(len(receptor_exp))

    for i in range(len(receptor_exp)):
        within_radius = dist_matrix[i, :] <= radius
        weights = gaussian_decay(dist_matrix[i, within_radius])
        weighted_receptor_pop[i] = np.sum(ligand_exp[within_radius] * weights)

    ratio = receptor_exp / weighted_receptor_pop

    # Step 2: Ligand Accessibility Score
    accessibility_scores = np.zeros(len(ligand_exp))

    for i in range(len(ligand_exp)):
        within_radius = dist_matrix[:, i] <= radius
        weights = gaussian_decay(dist_matrix[within_radius, i])
        accessibility_scores[i] = np.sum(ratio[within_radius] * weights)

    return accessibility_scores


def get_interacting_neighborhood_cell_idxs(adata, cell_idxs):
    """
    Get indices of all cells surrounding target indices.
    :param adata: AnnData
    :param cell_idxs: cell indices of interest, type: np.array
    :return:
    """
    # Create adjacency list from neighbors
    adj_list = __neighbors_to_adjacency_list(neighbors=adata.uns['relation'])

    # Exclude target indices and return cells that neighbor with target indices
    neighborhood = []

    for i in cell_idxs:
        if i in adj_list:
            neighborhood.extend(adj_list[i])

    neighborhood = list(set(neighborhood) - set(cell_idxs))

    return neighborhood
