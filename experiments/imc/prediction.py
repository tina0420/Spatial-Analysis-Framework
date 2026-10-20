"""
@author: Tina Hsu
Risk factors predicting outcomes
"""
from visualization import plot_feature_correlation, plot_roc_curve
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def features_correlation(df, features, method='spearman', plot_heatmap=None, out_dir=None, figsize=(13, 12), dpi=150):
    """
    Calculate feature correlation.
    :param df: table with feature information, type: pd.DataFrame
    :param features: feature names, type: list[str]
    :param method: method used for feature correlation, type: str
    :param plot_heatmap: plot a heatmap for visualization or not, type: bool
    :param out_dir: output directory, type: Path
    :param figsize: figure size, type: tuple
    :param dpi: dpi setting, type: int
    :return: correction matrix, type: pd.DataFrame
    """
    # Calculate correlation
    corr_matrix = df[features].corr(method=method)

    # Generate plot
    if plot_heatmap:
        plot_feature_correlation(corr_matrix, out_dir, figsize=figsize, dpi=dpi)

    return corr_matrix


def run_logistic_regression(df, features, plot_roc=False, out_dir=None, figsize=(6, 5), dpi=150):
    """
    Run logistic regression model and return odds ratio matrix to see the feature importance.
    :param df: table fed into the model, type: pd.DataFrame
    :param features: feature names, type: list[str]
    :param plot_roc: plot roc curve or not, type: bool
    :param out_dir: output directory, type: Path
    :param figsize: figure size, type: tuple
    :param dpi: dpi setting, type: int
    :return: Odds Ratio matrix, type: pd.DataFrame
    """
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

    # Evaluate the prediction results by AUROC
    auroc = roc_auc_score(y, y_pred_probs)

    print(f"Model Performance")
    print(f"AUROC: {auroc:.3f}")

    if plot_roc:
        plot_roc_curve(y, y_pred_probs, out_dir, figsize=figsize, name=str(features), dpi=dpi)

    # Extract Feature Importance by using Odds Ratios
    X_scaled_full = StandardScaler().fit_transform(X)

    # Add a intercept
    X_full_with_const = sm.add_constant(X_scaled_full)

    # Fit logistic regression for inference
    logit_model = sm.Logit(y, X_full_with_const)
    result = logit_model.fit(disp=False)

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

