"""
@author: Tina Hsu
Testing
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from tqdm import tqdm
from scipy.stats import combine_pvalues, tmean
from statsmodels.stats.multitest import multipletests
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc
from experiments.tnbc_imc.tnbc_imc_setup_analysis_e import *


# logistic regression
def run_logistic_regression(df, features):
    X = df[features].values
    y = df['Recurrence status'].values

    # L2 Regularization prevents overfitting
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')

    y_pred_probs = np.zeros(len(y))

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit and predict
        model.fit(X_train_scaled, y_train)

        # Store relapse probability
        y_pred_probs[test_index] = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    auroc = roc_auc_score(y, y_pred_probs)
    auprc = average_precision_score(y, y_pred_probs)

    print(f"Model Performance (LOOCV)")
    print(f"AUROC: {auroc:.3f}")
    print(f"AUPRC: {auprc:.3f}\n")

    # Extract Feature Importance (Odds Ratios)
    X_scaled_full = StandardScaler().fit_transform(X)

    # Add a constant (intercept) - statsmodels requires this explicitly
    X_full_with_const = sm.add_constant(X_scaled_full)

    # Fit logistic regression for inference
    logit_model = sm.Logit(y, X_full_with_const)
    result = logit_model.fit(disp=False)  # disp=False hides the optimization text output
    # result = logit_model.fit(method='bfgs', maxiter=500, disp=False)  # If the above doesn't work, use this as a fix

    inference_df = pd.DataFrame({
        'Features': ['Intercept'] + features,
        'Odds Ratio': np.exp(result.params),
        '95% CI Lower': np.exp(result.conf_int()[:, 0]),
        '95% CI Upper': np.exp(result.conf_int()[:, 1]),
        'p-val': result.pvalues
    })

    # Removing the Intercept
    inference_df = inference_df[inference_df['Features'] != 'Intercept'].reset_index(drop=True)

    return inference_df


def plot_odds_ratio(inference_df, model_type, out_dir, figsize=(9, 4), name=None, dpi=150):
    """
    Plot
    """
    # Set global plot style for publication-quality figures
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # # Figure: ROC and PRC
    # fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    #
    # # AUROC
    # fpr, tpr, _ = roc_curve(y, y_pred_probs)
    # roc_auc = auc(fpr, tpr)
    #
    # ax1.plot(fpr, tpr, color='#d62728', lw=2, label=f'AUC = {roc_auc:.2f}')
    # ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Random chance line
    # ax1.set_xlim([0.0, 1.0])
    # ax1.set_ylim([0.0, 1.05])
    # ax1.set_xlabel('False Positive Rate')
    # ax1.set_ylabel('True Positive Rate')
    # ax1.set_title('Receiver Operating Characteristic (ROC)')
    # ax1.legend(loc="lower right")
    # ax1.grid(alpha=0.3)
    #
    # plt.tight_layout()
    # if "/" in name:
    #     temp_name = name.replace("/", "_")
    #     plt.savefig(out_dir / f"{temp_name}_ROC.png", dpi=300)
    # else:
    #     plt.savefig(out_dir / f"{name}_ROC.png", dpi=300)
    #
    # plt.clf()

    # Figure: Odds ratio
    plot_df = inference_df.sort_values(by='Odds Ratio', ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(plot_df))
    ors = plot_df['Odds Ratio']
    lowers = plot_df['95% CI Lower']
    uppers = plot_df['95% CI Upper']
    pvals = plot_df['p-val']
    feature_names = plot_df['Features']

    print(features)
    print(ors)
    print(pvals)
    print("_______________________")

    for o, p in zip(ors, pvals):
        if p <= 0.05:
            print(features)
            print(p)
            print("_______")

    # Define colors based on significance
    colors = []
    for o, p in zip(ors, pvals):
        if p >= 0.05:
            colors.append('#808080')  # Non-significant (background noise)
        elif o > 1:
            colors.append('#d62728')  # Significant Risk Factor (Red)
        else:
            colors.append('#1f77b4')  # Significant Protective Factor (Blue)

    fig, ax = plt.subplots(figsize=figsize)

    # 95% Confidence Intervals as error bars
    xerr_lower = ors - lowers
    xerr_upper = uppers - ors
    # ax.errorbar(ors, y_pos, xerr=[xerr_lower, xerr_upper],
    #             fmt='none', ecolor="gray", capsize=6, elinewidth=2.5, zorder=2)

    ax.errorbar(ors, y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='none', ecolor="gray", capsize=6, elinewidth=3.5, zorder=2)

    # Odds Ratios
    ax.scatter(ors, y_pos, color=colors, s=110, zorder=2)

    # No effect line
    ax.axvline(x=1.0, color='black', linestyle='--', lw=1.5, zorder=0)

    ax.set_yticks(y_pos)
    # ax.set_yticklabels(feature_names, fontsize=15, rotation=90, va='center')
    ax.set_yticklabels(feature_names, fontsize=14, va='center')
    ax.set_xscale('log')
    ax.set_xlabel('Odds Ratio (Log Scale)', fontsize=13)
    if model_type == "univariate":
        ax.set_title('Univariate Predictor of Recurrence', weight="bold", fontsize=13)
    elif model_type == "multivariate":
        ax.set_title('Multivariate Predictor of Recurrence', weight="bold", fontsize=13)

    # if len(features) > 1:
    #     ax.set_title('Multivariate Predictor of Recurrence', weight="bold", fontsize=13)
    # elif len(features) == 1:
    #     ax.set_title('Univariate Predictor of Recurrence', weight="bold", fontsize=13)

    for i in range(len(y_pos)):
        p_str = "p < 0.01" if pvals[i] < 0.01 else f"p = {pvals[i]:.2f}"
        anno_text = f"{p_str}"
        ax.text(1.02, y_pos[i], anno_text, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=13, color=colors[i])

    plt.xticks(fontsize=10)
    plt.tight_layout()
    if "/" in name:
        temp_name = name.replace("/", "_")
        plt.savefig(out_dir / f"{temp_name}_OR.png", dpi=dpi)
    else:
        plt.savefig(out_dir / f"{name}_OR.png", dpi=dpi)
    plt.clf()


# correlation between features
def plot_feature_correlation(df, features, method='spearman', figsize=(13, 12)):
    """
    Calculates correlation between features and plots a lower-triangle heatmap.
    """
    # 1. Calculate the correlation matrix
    corr_matrix = df[features].corr(method=method)

    # 2. Create a mask to hide the upper triangle (redundant information)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 3. Initialize the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # 4. Generate a custom diverging colormap (classic Blue-White-Red)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 5. Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": 0.5, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
                          "label": "Spearman Correlation Coefficient"},
                annot=True,  # Shows the actual numbers
                fmt=".2f",  # Limit to 2 decimal places
                annot_kws={"size": 16},
                ax=ax)

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title(f'Feature Correlation Matrix', fontsize=25, pad=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_correlation.png", dpi=200)

    return corr_matrix


# my_features = ['Niche_1', 'Niche_4', 'IIS', 'Epi_Stroma_Dist']
# corr_df = plot_feature_correlation(df, my_features)

if __name__ == "__main__":
    """
    Input
    
    # Toy example dataframe
    np.random.seed(42)
    df = pd.DataFrame({'CID': [f'P{i:02d}' for i in range(1, 50)],
                       'Recurrence status': np.array([1] * 18 + [0] * 31),
                       'Niche_4_Abundance': np.concatenate([np.random.normal(loc=0.35, scale=0.20, size=18),
                                                            np.random.normal(loc=0.15, scale=0.20, size=31)]),
                       'IIS': np.concatenate([np.random.normal(loc=0.20, scale=0.10, size=18),
                                              np.random.normal(loc=0.55, scale=0.15, size=31)]),
                       'Niche_7_Abundance': np.random.normal(loc=0.10, scale=0.05, size=49)})
    df['Niche_4_Abundance'] = df['Niche_4_Abundance'].clip(lower=0)
    df['IIS'] = df['IIS'].clip(lower=0)
    df['Niche_7_Abundance'] = df['Niche_7_Abundance'].clip(lower=0)
    """
    # Merge all features together: SMA+Stromal cell type prop, Epithelial_Predominant cell niches prop, IIS score,
    #                              median distance b/w tumour and T cells, H3K27me3 mean expression,
    #                              Ki67+ expression (in specific cell types?)

    df_IIS = pd.read_csv(IN_DIR / "IMC_immune_hot_cold.csv")
    df = df_IIS[["CID", "Recurrence status", "Affected epithelial prop"]]
    df_distance = pd.read_csv(IN_DIR / "tumour-tme_distances.csv")
    df_distance = df_distance[df_distance["cell_type"] == "T-cell"].reset_index(drop=True)[["CID", "median_dist"]]
    df = pd.merge(df, df_distance, on="CID", how="left")

    # Ge the props
    # CLUSTER_KEY = "layer_2"
    CLUSTER_KEY = "cn_cluster_name"
    counts = FULL_ADATA.obs.pivot_table(
        index="CID",
        columns=CLUSTER_KEY,
        aggfunc='size',
        fill_value=0
    )
    prop_df = counts.div(counts.sum(axis=1), axis=0)
    # to_add = pd.DataFrame({"CID": list(prop_df.index), "SMA+ Stromal prop": prop_df["SMA+ Stromal"].values})
    to_add = pd.DataFrame(
        {"CID": list(prop_df.index), "Epithelial_Predominant prop": prop_df["Epithelial_Predominant"].values})
    df = pd.merge(df, to_add, on="CID", how="left")

    df.columns = ['CID', 'Recurrence status', 'Affected epithelial prop', 'Median_dist_tumour_T-cell',
                  'SMA+ Stromal prop', 'Epithelial_Predominant prop']

    # H3K27me3/Ki67 mean/median expression
    MARKER = "H3K27me3"
    UNIT_KEY = 'CID'
    # CLUSTER_KEY = 'layer_2'
    CLUSTER_KEY = 'cn_cluster_name'

    temp_df = pd.DataFrame({
        UNIT_KEY: FULL_ADATA.obs[UNIT_KEY],
        CLUSTER_KEY: FULL_ADATA.obs[CLUSTER_KEY],
        f'{MARKER} median expression': FULL_ADATA.to_df()[MARKER]
    })
    grouped_df = temp_df.groupby([UNIT_KEY, CLUSTER_KEY])[f'{MARKER} median expression'].median().reset_index()

    pivot_df = grouped_df.pivot(index=UNIT_KEY, columns=CLUSTER_KEY, values=f'{MARKER} median expression').reset_index()

    new_column_names = {}
    for col in pivot_df.columns:
        if col != UNIT_KEY:
            # Formats the column as "T-cell Ki67 median expression"
            new_column_names[col] = f"{col} {MARKER} median expression"

    final_patient_df = pivot_df.rename(columns=new_column_names)

    final_patient_df[list(new_column_names.values())] = final_patient_df[list(new_column_names.values())].fillna(0)

    df = pd.merge(df, final_patient_df, on="CID", how="left")

    # drop the functional marker features which didn't show significant results
    to_drop = ["CK7+ Luminal Ki67 median expression", "CK7+ Luminal Ki67 mean expression",
               "Stromal_Predominant Ki67 mean expression", "Stromal_Predominant Ki67 median expression"]
    df.drop(columns=to_drop, inplace=True)

    df.columns = ['CID', 'Recurrence status', 'Immune Impact Score', 'Median distance between tumour and T-cell',
                  'SMA+ Stromal proportion', 'Epithelial_Predominant proportion', 'CK5+ Basal H3K27me3 median expression',
                  'CK7+ Luminal H3K27me3 median expression', 'CK8/18+ Luminal H3K27me3 median expression',
                  'CK+ Epithelial H3K27me3 median expression', 'T-cell H3K27me3 median expression',
                  'B-cell H3K27me3 median expression', 'Macrophage H3K27me3 median expression',
                  'Endothelial H3K27me3 median expression', 'SMA+ Stromal H3K27me3 median expression',
                  'Stromal H3K27me3 median expression', 'CK5+ Basal Ki67 median expression',
                  'CK8/18+ Luminal Ki67 median expression', 'CK+ Epithelial Ki67 median expression',
                  'T-cell Ki67 median expression', 'B-cell Ki67 median expression',
                  'Macrophage Ki67 median expression', 'Endothelial Ki67 median expression',
                  'SMA+ Stromal Ki67 median expression', 'Stromal Ki67 median expression',
                  'CK5+ Basal Ki67 mean expression', 'CK8/18+ Luminal Ki67 mean expression',
                  'CK+ Epithelial Ki67 mean expression', 'T-cell Ki67 mean expression',
                  'B-cell Ki67 mean expression', 'Macrophage Ki67 mean expression',
                  'Endothelial Ki67 mean expression', 'SMA+ Stromal Ki67 mean expression',
                  'Stromal Ki67 mean expression', 'CK5+ Basal H3K27me3 mean expression',
                  'CK7+ Luminal H3K27me3 mean expression', 'CK8/18+ Luminal H3K27me3 mean expression',
                  'CK+ Epithelial H3K27me3 mean expression',
                  'T-cell H3K27me3 mean expression', 'B-cell H3K27me3 mean expression',
                  'Macrophage H3K27me3 mean expression', 'Endothelial H3K27me3 mean expression',
                  'SMA+ Stromal H3K27me3 mean expression',
                  'Stromal H3K27me3 mean expression', 'B_Patch H3K27me3 mean expression',
                  'Endothelial_Predominant H3K27me3 mean expression',
                  'Epithelial_Predominant H3K27me3 mean expression',
                  'Stromal_Predominant H3K27me3 mean expression', 'T_Infiltration_Basal H3K27me3 mean expression',
                  'T_Infiltration_Luminal H3K27me3 mean expression', 'T_Mp_Aggregation H3K27me3 mean expression',
                  'T_Resident_Stromal H3K27me3 mean expression',
                  'B_Patch Ki67 mean expression',
                  'Endothelial_Predominant Ki67 mean expression',
                  'Epithelial_Predominant Ki67 mean expression',
                  'T_Infiltration_Basal Ki67 mean expression',
                  'T_Infiltration_Luminal Ki67 mean expression',
                  'T_Mp_Aggregation Ki67 mean expression',
                  'T_Resident_Stromal Ki67 mean expression',
                  'B_Patch Ki67 median expression',
                  'Endothelial_Predominant Ki67 median expression',
                  'Epithelial_Predominant Ki67 median expression',
                  'T_Infiltration_Basal Ki67 median expression',
                  'T_Infiltration_Luminal Ki67 median expression',
                  'T_Mp_Aggregation Ki67 median expression',
                  'T_Resident_Stromal Ki67 median expression',
                  'B_Patch H3K27me3 median expression',
                  'Endothelial_Predominant H3K27me3 median expression',
                  'Epithelial_Predominant H3K27me3 median expression',
                  'Stromal_Predominant H3K27me3 median expression',
                  'T_Infiltration_Basal H3K27me3 median expression',
                  'T_Infiltration_Luminal H3K27me3 median expression',
                  'T_Mp_Aggregation H3K27me3 median expression',
                  'T_Resident_Stromal H3K27me3 median expression']

    df.to_csv(IN_DIR / "all_spatial_features_for_logreg.csv", index=False)

    """
    Logistic regression
    """

    # Do univariate first, then pick the top two OR, then run multivariate
    OUT_DIR = OUT_DIR / "multivariate test"
    df = pd.read_csv(IN_DIR / "all_spatial_features_for_logreg.csv")

    features = ['Immune Impact Score']
    features = ['T-cell H3K27me3 mean expression']
    features = ['Immune Impact Score', 'Endothelial_Predominant H3K27me3 mean expression',
                'Epithelial_Predominant proportion', 'T_Mp_Aggregation H3K27me3 mean expression',
                'T-cell H3K27me3 mean expression']
    features = ['Immune Impact Score', 'T-cell H3K27me3 mean expression']

    # Go through every two sig features
    OUT_DIR = OUT_DIR / "Features/Significant p"
    for f in features:
        run_logistic_regression(df, [f], OUT_DIR, f"6A_{f}")
    for f in list(df.columns)[2:]:
        run_logistic_regression(df, [f], OUT_DIR, f"6A_{f}")

    for i in range(len(features)):
        print(i)
        if i == 0:
            log_df = run_logistic_regression(df, [features[i]])
        else:
            log_df_2 = run_logistic_regression(df, [features[i]])
            log_df = pd.concat([log_df, log_df_2], axis=0)

    log_df = run_logistic_regression(df, features)
    plot_odds_ratio(log_df, "multivariate", (9, 4), OUT_DIR, "Multivariate_plot_together")

    # Plot all univariate OR together
    # plot_odds_ratio(log_df, OUT_DIR, "Univariate_plot_together")
    plot_odds_ratio(log_df, "univariate", (12, 5), OUT_DIR, "Univariate_plot_together")

    sig_features = ['Endothelial_Predominant H3K27me3 mean expression', 'T-cell H3K27me3 mean expression',
                    'Epithelial_Predominant proportion', 'Immune Impact Score',
                    'T_Mp_Aggregation H3K27me3 mean expression', 'T_Mp_Aggregation H3K27me3 median expression']

    # Run correlation between single significant feature
    my_features = list(df.columns)[2:]
    my_features = ['Affected epithelial prop', 'Endothelial_Predominant H3K27me3 mean expression',
                   'Epithelial_Predominant prop', 'T_Mp_Aggregation H3K27me3 mean expression',
                   'T-cell H3K27me3 mean expression']
    corr_df = plot_feature_correlation(df, features)

    # plot
    fig, ax = plt.subplots(figsize=(7, 7))

    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 3. Draw the heatmap
    sns.heatmap(
        corr_df,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={"shrink": 0.5, "label": "Spearman Correlation Coefficient"},
        annot=True,
        annot_kws={"size": 5},
        ax=ax
    )

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "test_sig_feature_corr.png", dpi=200)
    plt.clf()
    plt.show()
