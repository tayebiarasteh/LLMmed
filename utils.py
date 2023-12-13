"""
Created on Oct 1, 2023.
utils.py
visualizations

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
