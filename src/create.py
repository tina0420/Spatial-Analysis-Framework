"""
@author: Tina Hsu
Functions for simulating spatial datasets.
"""
import numpy as np
from scipy import stats


def simulate_spatial_ROI(r_rois, i_cells, j_genes, k_clusters, pixel_size, mixture_type,
                         init_pos=(0, 0), seed=None):
    """
    Algorithm to obtain attributes of a spatial dataset:
    - Sample mean expression for each cluster
    - Sample interaction type for each cluster
        where bias means cells of the same type are more likely to be around each other
        where uni means cells of any type have a random chance of likelihood to be around each other
    - Sample cell size for each cluster
    - For each cell:
        - Sample cell type
        - Sample cell expression
        - Sample cell coordinates
        - Sample cell area

    :param r_rois: number of region of interests or images, type: int
    :param i_cells: number of cells, type: int
    :param j_genes: number of genes, type: int
    :param k_clusters: number of clusters, type: int
    :param pixel_size: average pixel size, type: float
    :param mixture_type: one of ['uni', 'bias'] for type of beta, type: str
    :param init_pos: initialized position of the graph, default: (0, 0), type: tuple(int, int)
    :param seed: random seed, default: None, type: int
    :return: [1] cell type labels, [2] cell areas, [3] expression, [4] spatial coordinates, [5] ROI IDs, type: dict
    """
    # Initialization
    if seed:
        np.random.seed(seed)

    # Initialize mean expression, cluster interactions, cell radii
    mu = __sample_cluster_mean_expression(k_clusters, j_genes, mu_mean=0, mu_stdev=1)
    beta = __sample_cluster_interaction(k_clusters, mixture_type)
    cell_radii = __sample_cluster_cell_radii(k_clusters, pixel_size)

    # Iterate over rois
    rois = {"label": [],
            "area": [],
            "X": [],
            "spatial": [],
            "image_id": []}

    for r in range(r_rois):
        prev_label = 0
        pos = init_pos
        label = np.zeros(i_cells, dtype=int)
        area = np.zeros(i_cells)
        X = np.zeros((i_cells, j_genes))
        spatial = np.zeros((i_cells, 2))

        # Algorithm
        for i in range(i_cells):
            cell_i = __sample_label(prev_label=prev_label, beta=beta)
            X[i] = __sample_expression(mean_exp=mu[cell_i])
            spatial[i] = __sample_coordinates(prev_pos=pos, pixel_size=pixel_size)
            label[i] = cell_i
            area[i] = __sample_area(cell_radius=cell_radii[cell_i])
            prev_label = cell_i

        # Append all images
        rois["label"].append(label)
        rois["area"].append(area)
        rois["X"].append(X)
        rois["spatial"].append(spatial)
        rois["image_id"] = rois["image_id"] + [f"roi_{r}"] * i_cells

    rois["label"] = np.hstack(rois["label"])
    rois["area"] = np.hstack(rois["area"])
    rois["X"] = np.vstack(rois["X"])
    rois["spatial"] = np.vstack(rois["spatial"])

    return rois


def __sample_cluster_mean_expression(k_clusters, j_genes, mu_mean, mu_stdev):
    """
    mu_{k,j} ~ N(mu_mean, mu_stdev)
    :param k_clusters: number of clusters, type: int
    :param j_genes: number of genes, type: int
    :param mu_mean: mean of expression, type: float
    :param mu_stdev: standard deviation of expression, type: float
    :param seed: random seed, default: None, type: int
    :return: mean expression matrix (k * j), type: np.array
    """
    return stats.norm.rvs(loc=mu_mean, scale=mu_stdev,
                          size=k_clusters * j_genes).reshape(k_clusters, j_genes)


def __sample_cluster_interaction(k_clusters, mixture_type):
    """
    beta_{k} ~ Beta(1, 1)
    :param k_clusters: number of clusters, type: int
    :param mixture_type: one of ['uni', 'bias'] for type of interaction, type: str
    :return: interaction matrix (k * k), type: np.array
    """
    autonomous_interaction = stats.beta.rvs(a=1, b=1, size=k_clusters)
    non_autonomous_interaction = 1 - autonomous_interaction
    beta = np.vstack((autonomous_interaction, non_autonomous_interaction)).T

    # If bias, swap the strong interaction term to autonomous interactions
    if mixture_type == "bias":
        for i in range(len(beta)):
            if beta[i][0] < beta[i][1]:
                beta[i] = [beta[i][1], beta[i][0]]

    # Interaction matrix normalized by row
    beta_mtx = np.zeros((k_clusters, k_clusters))
    for i in range(k_clusters):
        for j in range(k_clusters):
            if i != j:
                beta_mtx[i, j] = beta[i][1] * beta[j][1]
            else:
                beta_mtx[i, j] = beta[i][0] * beta[j][0]
        beta_mtx[i] = beta_mtx[i] / np.sum(beta_mtx[i])

    return beta_mtx


def __sample_cluster_cell_radii(k_clusters, pixel_size):
    """
    Sample cell area per cluster from a Gaussian distribution.
    :param k_clusters: number of clusters, type: int
    :param pixel_size: size of each pixel, type: float
    :param seed: random seed, default: None, type: int
    :return: cell area per cluster, type: np.array
    """
    return stats.norm.rvs(loc=pixel_size * 2, scale=1, size=k_clusters)


def __sample_label(prev_label, beta):
    """
    Sample label for interaction distribution.
    :param prev_label: previous label sampled, type: int
    :param beta: interaction matrix, type: np.array
    :return: label sampled from categorical, type: int
    """
    return np.random.choice(a=len(beta), p=beta[prev_label])


def __sample_area(cell_radius):
    """
    Sample cell area of the current cell.
    :param cell_radius: cell radius for the current cell type, type: float
    :return:
    """
    cell_r = __sample_radius(mean_radius=cell_radius)

    return np.pi * (cell_r ** 2)


def __sample_radius(mean_radius):
    """
    Sample radius from a Gaussian distribution.
    :param mean_radius: mean radius of current cell type, type: float
    :return: radius sampled from Gaussian, type: float
    """
    return stats.norm.rvs(loc=mean_radius, scale=0.1)


def __sample_coordinates(prev_pos, pixel_size):
    """
    Sample coordinates of the current cell.
    :param prev_pos: coordinates of the previous cell, type: tuple(int, int)
    :param pixel_size: size of each pixel, type: float
    :return: coordinates of the current cell, type: tuple(int, int)
    """
    cell_dist = __sample_distance(pixel_size=pixel_size)
    cell_angle = __sample_angle()

    return cell_dist * np.cos(cell_angle) + prev_pos[0], cell_dist * np.sin(cell_angle) + prev_pos[1]


def __sample_distance(pixel_size):
    """
    Sample radius from a Gaussian distribution.
    :param pixel_size: size of each pixel, type: float
    :return: distance sampled from Gaussian, type: float
    """
    return stats.norm.rvs(loc=pixel_size * 3, scale=0.5)


def __sample_angle():
    """
    Sample angle from a uniform distribution.
    :return: angle sampled from uniform, type: float
    """
    return stats.uniform.rvs(loc=0, scale=2*np.pi)


def __sample_expression(mean_exp):
    """
    Sample expression from Gaussian distribution centered at mu_{k}.
    :param mean_exp: mean expression of current cell type, type: numpy array
    :return: expression sampled from Gaussian, type: numpy array
    """
    return stats.norm.rvs(loc=mean_exp, scale=0.1)
