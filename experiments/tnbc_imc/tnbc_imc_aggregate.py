"""
@author: Tina Hsu
Data/samples group and aggregation compare
"""
import os
from pathlib import Path
from anndata import read_h5ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from datetime import datetime

if __name__ == "__main__":
    # Set disease type and which experiment to process
    TYPE = 'tnbc_imc'
    PURPOSE = 'cohort_specific'
    XP = 'cohort_tnbc'

    in_dir = Path(f"analysis/{TYPE}/{PURPOSE}")
    out_dir = Path(f"analysis/{TYPE}/{PURPOSE}/aggregate")
    if os.path.exists(out_dir) == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Read in adata for tils analysis
    adata = read_h5ad(in_dir / f"{XP}_finalize_binary_relapse.h5ad")

    # Create a df with image_key, cell_counts, relapse (descending with cell count number)
    cell_numbers_df = pd.DataFrame({"image_key": adata.obs.image_key.value_counts().index,
                                    "cell_counts": adata.obs.image_key.value_counts().values,
                                    "relapse": [adata.obs.iloc[np.where(adata.obs.image_key == i)[0], :].relapse[0] for
                                                i in adata.obs.image_key.value_counts().index]})

    cell_numbers_df.sort_values(by="cell_counts", ascending=True, inplace=True)

    # Directly use Kmeans to get 5 groups
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(np.array(cell_numbers_df.cell_counts).reshape(-1, 1))

    # Step 3: Get the cluster labels
    labels = kmeans.labels_
    print("Cluster Labels:", labels)

    cell_numbers_df["group"] = labels

    # reorder the index
    cell_numbers_df.reset_index(inplace=True, drop=True)
    # Step 4: Visualize the clusters (optional)
    plt.figure(figsize=(12, 9))
    color_dict = {0: "tab:blue", 1: "tab:pink", 2: "tab:orange", 3: "tab:green", 4: "tab:purple"}
    scatter = sns.scatterplot(x=cell_numbers_df.image_key, y=cell_numbers_df.cell_counts,
                              hue=cell_numbers_df.group, palette=color_dict)
    scatter = plt.scatter(cell_numbers_df.image_key, cell_numbers_df.cell_counts, c=cell_numbers_df.group,
                          cmap="", s=25)
    plt.title("Classification of image group", fontsize=18, fontweight='bold')

    # plt.xlabel("Image", fontsize=15)
    plt.ylabel("Cell counts", fontsize=15)
    plt.xticks(fontsize=13, rotation=90)

    for i in np.unique(cell_numbers_df.group):
        plt.scatter([], [], color=scatter.cmap(scatter.norm(i)), label=f"Group {i}")

    plt.legend(title="Groups")

    plt.tight_layout()
    plt.show()
    plt.clf()

    # counts_diff = np.diff(cell_numbers)
    # counts_diff_svd = counts_diff / np.mean(counts_diff)

    # # Plotting
    # plt.figure(figsize=(12, 9))
    # plt.plot(cell_numbers, marker='o')
    # # plt.plot(counts_diff, marker='o')
    # # plt.plot(counts_diff/np.mean(counts_diff), marker='o')
    # plt.title("Cell counts of the images", fontsize=15, fontweight='bold')
    # # plt.title("Difference of cell counts in the images", fontsize=15, fontweight='bold')
    # # plt.title("Difference of cell counts in the images(value-mean)", fontsize=15, fontweight='bold')
    # plt.show()
    # plt.clf()

