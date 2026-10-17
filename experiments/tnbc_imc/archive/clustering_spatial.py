"""
@author: Tina Hsu
Clustering method by using spatial data.
"""
import os
from pathlib import Path
from anndata import read_h5ad
import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from src.annotate import run_leiden
from scipy.spatial.distance import cdist
import networkx as nx
import leidenalg as la
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualize import *
import igraph as ig

from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
"""
Two ideas trying now:
1. network, GNN, transform the data into weighted undirected network
    +fn: constructing network with feature values of each cell as a node and wights calculation
    +fn: definition of neighbors in spatial, binary, neighbor or not neighbor
    +fn: definition of neighbors in spatial, continues value, distance between nodes
    +fn: visualization of the distribution of the cells (src.visualize) and network (igraph)

2. variational autoencoders (VAE) with protein expression and spatial separately first and then merge the latent space
   +fn: VAE model with setting of encoder, reparameterize, decoder, forward
   +fn: loss function calculation
   +fn: visualization
   
   -step1: traditional VAE with only expression data
   -step2: combine expression and KNN
   -step3: combine expression and spatial
   
3. graph neural network (GNN) individual in network format or combined with autoencoder

--to-do--
1. whether or not adding extra fully networks in encoder and decoder in VAE, will it overfit?
2. is there any difference between TME and all cells?
3. 
"""

"""
--notes on 25/02/14 1:1 meeting

1. need more epochs in training (used 15 and 30 before)
2. can separate and show the components in loss function (e.g. MSE and KL showing separately)
3. can try different loss function, maybe ...entropy (not binary one)
4. check latent features in latent space, if they are really crucial? can plot with each of them
5. can try to add the distance value as another feature or y in the training (idea)

"""


def plot_nodes(loc_arr):
    plt.figure(figsize=(7, 7))
    plt.scatter(x=loc_arr[:, 0], y=loc_arr[:, 1], s=10)
    plt.xlabel("X", fontsize=15)
    plt.ylabel("Y", fontsize=15)
    plt.show()


def calculate_weight(loc1, loc2):
    similarity = np.exp(np.sum((loc2 - loc1) ** 2))

    return similarity


def neighbor_binary(adata):
    pass


def neighbor_value(adata):
    neighbors = adata.uns['neighbors']

    return calculate_weight()


def get_neighbor(adata, method):
    if method == "binary":
        neighbor_mtx = neighbor_binary()
    elif method == "weights":
        neighbor_mtx = neighbor_value()

    return neighbor_mtx


class VAE(nn.Module):
    def __init__(self, input, latent):
        """
        Create objects/layers for VAE
        :param input: dimension of the input metric, type: int
        :param latent: dimension of the latent space, type: int
        """
        super(VAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(input, 64)
        self.fc12 = nn.Linear(64, 32)
        self.fc21 = nn.Linear(32, latent)
        self.fc22 = nn.Linear(32, latent)

        # decoder
        self.fc3 = nn.Linear(latent, 32)
        self.fc32 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, input)  # output size

    def encode(self, x):
        """
        Encoding the input data into mean and log-variance of latent space
        :param x: input metric in (m x n), type: np.Array
        :return: mean and log-variance of latent space both in (m x latent_dim), type: [np.Array], [np.Array]
        """
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc12(h1))

        mean = self.fc21(h2)
        log_var = self.fc22(h2)

        return mean, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick
        :param mu: mean of latent space in (m x latent_dim), type: np.Array
        :param logvar: log-variance of latent space in (m x latent_dim), type:np.Array
        :return: reparameterization trick result for latent space in (m x latent_dim), type: np.Array
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   # sample epsilon

        return mu + eps * std

    def decode(self, z):
        """
        Reconstruct the data
        :param z: latent space output for decoder, type:np.array
        :return: decoded result , type: np.array
        """
        h3 = torch.relu(self.fc3(z))
        h4 = torch.relu(self.fc32(h3))

        return self.fc4(h4)

    def forward(self, x):
        """
        Work in latent space
        :param x: input metric in mxn, type: np.Array
        :return: reconstructed data and mean and log-variance of latent space
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """
    Calculate the loss function
    :param recon_x: reconstruction data, type: np.array
    :param x: original inout data, type: np.array
    :param mu: mean of latent space, type: np.array
    :param logvar: log-variance of latent space, type: np.array
    :return: loss value, type: float
    """
    # mean square error
    mse = nn.MSELoss(reduction='sum')(recon_x, x)
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse, kld, (mse + kld)


def get_latent_num(input_mtx):
    """
    Get latent dimension based on pca result
    :param input_mtx: input metric in (m x n), type: np.array
    :return: latent number of latent space, type: int
    """
    pca = PCA()
    pca.fit(input_mtx)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # latent number
    latent_num = np.argmax(cumulative_variance >= 0.95) + 1

    return latent_num


if __name__ == '__main__':
    # path and file
    EXPERIMENT = "XP12446"
    TYPE = "tnbc_imc"
    print(f"Processing {EXPERIMENT}")

    in_dir = Path(f"data/{TYPE}/{EXPERIMENT}")
    out_dir = Path(f"output/{TYPE}/{EXPERIMENT}/methods")

    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    adata_all = read_h5ad(in_dir / f"{EXPERIMENT}.h5ad")

    # only expression clustering
    print("doing leiden")
    run_leiden(adata_all, columns=adata_all.var_names, n_neighbors=150, n_pcs=None, name="leiden_exp")

    print(set(adata_all.obs["image_id"]))
    # clustering with expression and expression


    # focus on one image
    img = "ROI10_ROI_010.ome.tiff"
    adata = adata_all[np.where(adata_all.obs.image_id == img)[0], :]
    run_leiden(adata, columns=adata.var_names, n_neighbors=150, n_pcs=None, name="leiden_exp")
    plot_spatial_cluster_distribution(adata, "leiden_exp", out_dir, figsize=(7, 7), dotsize=1, name="leiden_exp")

    # separate epithelial and tme
    adata_epi = adata[np.where(adata.obs.gmm_4ck == 0)[0], :]
    adata_tme = adata[np.where(adata.obs.gmm_4ck == 1)[0], :]
    run_leiden(adata_epi, columns=adata_epi.var_names, n_neighbors=150, n_pcs=None, name="leiden")
    plot_spatial_cluster_distribution(adata_epi, "leiden", out_dir, figsize=(7, 7), dotsize=1, name="epi_leiden")
    run_leiden(adata_tme, columns=adata_tme.var_names, n_neighbors=150, n_pcs=None, name="leiden")
    plot_spatial_cluster_distribution(adata_tme, "leiden", out_dir, figsize=(7, 7), dotsize=1, name="tme_leiden")

    # Way 2: VAE

    exp_mtx = adata.X

    # binary first, cells are neighbor or not neighbor
    neighbor_binary_mtx = get_neighbor(adata=adata, method="binary")

    # similarity
    neighbor_mtx = 0

    # device = torch.device("")
    # vae_model = VAE().to(device)

    input_dim = exp_mtx.shape[1]
    latent_dim = get_latent_num(exp_mtx)
    vae_model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

    # Convert the data into PyTorch tensors
    X_tensor = torch.tensor(exp_mtx, dtype=torch.float32)

    # Create a DataLoader for batching
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    error_record = []
    # mse_list = []
    # kl_list = []
    epochs = 200
    for epoch in range(epochs):
        vae_model.train()
        train_loss = 0
        print("----------------------------------------------------")
        print("epoch: ", epoch)
        for batch_idx, (data,) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(data)
            mse, kl, loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print("lossssssssss function")
            # mse_list.append(mse.item())
            # kl_list.append(kl.item())

        error_record.append(round(train_loss / len(dataloader.dataset), 4))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(dataloader.dataset):.4f}")

    plt.figure(figsize=(7, 7))
    plt.title("Total Error")
    plt.plot(error_record)
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", bbox_inches='tight')
    plt.clf()

    # plt.figure(figsize=(7, 7))
    # plt.title("Mean Square Error of training")
    # plt.plot(mse_list)
    # plt.tight_layout()
    # plt.savefig(out_dir / f"training_vae_mse_{epoch}.png", bbox_inches='tight')
    # plt.clf()
    #
    # plt.figure(figsize=(7, 7))
    # plt.title("KL Divergence of training")
    # plt.plot(kl_list)
    # plt.tight_layout()
    # plt.savefig(out_dir / f"training_vae_kl_{epoch}.png", bbox_inches='tight')
    # plt.clf()

    # generate new data from the latent space
    vae_model.eval()
    with torch.no_grad():
        # randomly sample z from N(0, 1)
        z = torch.randn(exp_mtx.shape[0], latent_dim)
        generated_data = vae_model.decode(z)

    # test the difference between original clustering and VAE before clustering
    # 1. original_exp_leiden
    # 2. vae_exp_leiden
    # 3. vae_spatial_leiden

    # 1. use original expression data to do leiden clustering
    original_exp_leiden = AnnData(adata.X, var=pd.DataFrame([], index=adata.var_names))
    sc.tl.pca(original_exp_leiden, svd_solver='arpack')
    sc.pp.neighbors(original_exp_leiden, n_neighbors=15)
    sc.tl.leiden(original_exp_leiden)  # 15 clusters
    plot_zscore_cluster_heatmap(original_exp_leiden, original_exp_leiden.var_names,
                                "leiden", out_dir, figsize=(7, 7), name="original_exp_leiden")

    plot_zscore_cluster_mean_heatmap(original_exp_leiden, original_exp_leiden.var_names, "leiden", out_dir, figsize=(7, 7), name="original_exp_leiden")

    # 2. use the constructed data from VAE to do leiden clustering
    generated_data = generated_data.cpu().numpy()
    vae_exp_leiden = AnnData(generated_data, var=pd.DataFrame([], index=adata.var_names))
    sc.tl.pca(vae_exp_leiden, svd_solver='arpack')
    sc.pp.neighbors(vae_exp_leiden, n_neighbors=15)
    sc.tl.leiden(vae_exp_leiden)  # 18 clusters, new: 14 clusters
    plot_zscore_cluster_heatmap(vae_exp_leiden, vae_exp_leiden.var_names,
                                "leiden", out_dir, figsize=(7, 7), name="vae_exp_leiden")

    plot_zscore_cluster_mean_heatmap(vae_exp_leiden, vae_exp_leiden.var_names, "leiden", out_dir,
                                     figsize=(7, 7), name="vae_exp_leiden_all_clusters")

    # 3. plus

    



    # check the cohort data
    organize_metadata = pd.read_table(out_dir / "TNBC ctDNA cohort summary.xlsx")



