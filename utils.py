"""
Created on Oct 1, 2023.
utils.py
visualizations and explainability

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import scipy.stats as stats
import seaborn as sns
import pdb
import matplotlib.pylab as plt
import pandas as pd
from sklearn.metrics import roc_curve
import shap



def compute_mean_and_ci(data, confidence=0.95):
    """
    """
    mean_val = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean_val, (mean_val - ci, mean_val + ci)


def ROC_curves():
    # Loading the provided CSV files
    ds_predictions = pd.read_csv('/PATH/DS_predictions.csv')
    ada_predictions = pd.read_csv('/PATH/ADA_predictions.csv')
    ds_bootstrap = pd.read_csv('/PATH/DS_bootstrap.csv')
    ada_bootstrap = pd.read_csv('/PATH/ADA_bootstrap.csv')

    # Compute mean and 95% CI for Validatory and ChatGPT ADA methods using bootstrapped data
    ds_mean_cardio, ds_ci_cardio = compute_mean_and_ci(ds_bootstrap['AUC'])
    ada_mean_cardio, ada_ci_cardio = compute_mean_and_ci(ada_bootstrap['AUC'])

    # Rounding the values to 3 decimal places
    ds_mean_cardio_rounded = round(ds_mean_cardio, 3)
    ds_ci_cardio_rounded = (round(ds_ci_cardio[0], 3), round(ds_ci_cardio[1], 3))
    ada_mean_cardio_rounded = round(ada_mean_cardio, 3)
    ada_ci_cardio_rounded = (round(ada_ci_cardio[0], 3), round(ada_ci_cardio[1], 3))

    # Calculating the ROC curves
    ds_fpr_cardio, ds_tpr_cardio, _ = roc_curve(ds_predictions['ground_truth'],
                                                ds_predictions['probability'])
    ada_fpr_cardio, ada_tpr_cardio, _ = roc_curve(ada_predictions['ground_truth'],
                                                  ada_predictions['probability'])

    # Generating the ROC curve plot
    plt.figure(figsize=(10, 8))

    # Plotting the ROC curve for Validatory method with dotted line, red color, and thicker line width
    plt.plot(ds_fpr_cardio, ds_tpr_cardio, color='red', lw=3, linestyle=':',
             label=f'Validatory (AUROC = {ds_mean_cardio_rounded} [95% CI: {ds_ci_cardio_rounded[0]}, {ds_ci_cardio_rounded[1]}])')

    # Plotting the ROC curve for ChatGPT ADA method with solid line, blue color, and thicker line width
    plt.plot(ada_fpr_cardio, ada_tpr_cardio, color='blue', lw=3,
             label=f'ChatGPT ADA (AUROC = {ada_mean_cardio_rounded} [95% CI: {ada_ci_cardio_rounded[0]}, {ada_ci_cardio_rounded[1]}])')

    # Plotting the diagonal line with a dashed style and original thickness
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

    # Setting the plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title('Hereditary Hearing Loss [Otolaryngology]', fontsize=24, loc='left', pad=20)
    plt.legend(loc="lower right", fontsize=18)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(False)
    plt.tight_layout()





def SHAP_analysis_GPT(X_train, clf):
    shap.initjs()

    # Compute SHAP values using TreeExplainer (optimized for tree-based models)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)

    # Calculate the mean absolute SHAP values for each feature
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(shap_values[1]).mean(axis=0)
    })

    # Sort features by importance
    sorted_feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    # Top 10 features
    top_10_features = sorted_feature_importance.head(10)

    # Calculate the mean absolute SHAP values for each feature
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(shap_values[1]).mean(axis=0)
    })

    # Sort features by importance
    sorted_feature_importance = feature_importance.sort_values(by='importance', ascending=False)

    # Extract SHAP values for the top 10 features
    top_feature_names = top_10_features['feature'].values
    top_shap_values = np.array([shap_values[1][X_train.columns.get_loc(col)] for col in top_feature_names]).T

    # Save the underlying data for the SHAP Box Plot to a CSV file
    shap_box_data = pd.DataFrame(top_shap_values, columns=top_feature_names)
    shap_box_data.to_csv("/PATH/shap_box_data.csv", index=False)

    # Correctly extract SHAP values for the top 10 features
    top_feature_indices = [X_train.columns.get_loc(col) for col in top_feature_names]
    top_shap_values = shap_values[1][:, top_feature_indices]

    # Save the underlying data for the SHAP Box Plot to a CSV file
    shap_box_data = pd.DataFrame(top_shap_values, columns=top_feature_names)
    shap_box_data.to_csv("/PATH/shap_box_data.csv", index=False)

    # Top 10 features
    top_10_features = sorted_feature_importance.head(10)

    # Extract the SHAP values for the top 10 features
    top_shap_values = shap_values[0][:, top_feature_indices]

    # Save the underlying data for the SHAP Box Plot to a CSV file
    shap_box_data = pd.DataFrame(top_shap_values, columns=top_feature_names)
    shap_box_data.to_csv("/PATH/shap_box_data.csv", index=False)

    # Inspect the structure of shap_values
    shap_values_shape = np.array(shap_values).shape

    # Extract the SHAP values for the top 10 features
    top_shap_values = shap_values[:, top_feature_indices]

    # Create the SHAP Box Plot
    shap.summary_plot(top_shap_values, X_train[top_feature_names], plot_type="box")

    # Save the underlying data for the SHAP Box Plot to a CSV file
    shap_box_data = pd.DataFrame(top_shap_values, columns=top_feature_names)
    shap_box_data.to_csv("/PATH/shap_box_data.csv", index=False)

    shap_box_data.head()






if __name__ == '__main__':
    ROC_curves()
    SHAP_analysis_GPT()

