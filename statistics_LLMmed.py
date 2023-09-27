"""
Created on Sep 26, 2023.
statistics_LLMmed.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import pdb
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import ranksums
from scipy.stats import norm
from mne.stats import fdr_correction



def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)



def bootstrap_metric_statistics_and_samples(y_true, y_pred, metric_func, n_bootstrap=1000):
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[sample_indices]
        y_pred_sample = y_pred[sample_indices]
        metric_sample = metric_func(y_true_sample, y_pred_sample)
        values.append(metric_sample)
    mean_metric = np.mean(values)
    std_metric = np.std(values)
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    return mean_metric, std_metric, (ci_lower, ci_upper), values


def bootstrap_p_value(y_true, y_pred1, y_pred2, metric_func, n_bootstrap=1000):
    n = len(y_true)
    diff_original = metric_func(y_true, y_pred1) - metric_func(y_true, y_pred2)
    count = 0
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[sample_indices]
        y_pred1_sample = y_pred1[sample_indices]
        y_pred2_sample = y_pred2[sample_indices]
        diff_sample = metric_func(y_true_sample, y_pred1_sample) - metric_func(y_true_sample, y_pred2_sample)
        if diff_original > 0 and diff_sample >= diff_original:
            count += 1
        elif diff_original < 0 and diff_sample <= diff_original:
            count += 1
    p_value = count / n_bootstrap
    return p_value


def save_bootstrapped_samples_to_csv(samples_dict, csv_filename):
    df = pd.DataFrame(samples_dict)
    df.to_csv(csv_filename, index=False)


def print_results_updated_v3(results_dict, filename="output.txt"):
    with open(filename, "w") as file:
        # Print and write the main header
        print("Statistical Analysis Results:\n")
        file.write("Statistical Analysis Results:\n\n")

        for metric, values in results_dict.items():
            # Print and write metric name
            print(f"Metric: {metric}")
            file.write(f"Metric: {metric}\n")

            for method, stats in values.items():
                if method != "p_value":
                    # Print and write method name
                    print(f"  {method}:")
                    file.write(f"  {method}:\n")

                    # Print and write metric statistics
                    print(f"    - Mean: {stats['Mean']:.3f}")
                    file.write(f"    - Mean: {stats['Mean']:.3f}\n")

                    print(f"    - Std: {stats['Std']:.3f}")
                    file.write(f"    - Std: {stats['Std']:.3f}\n")

                    print(f"    - 95% CI: ({stats['95% CI'][0]:.3f}, {stats['95% CI'][1]:.3f})")
                    file.write(f"    - 95% CI: ({stats['95% CI'][0]:.3f}, {stats['95% CI'][1]:.3f})\n")

            if values['p_value'] is not None:
                # Print and write p-value
                print(f"  p-value: {values['p_value']:.3f}\n")
                file.write(f"  p-value: {values['p_value']:.3f}\n\n")


def main_analysis_with_saving(csv_file1, csv_file2, output_file1, output_file2, n_bootstrap=1000):
    results1 = pd.read_csv(csv_file1)
    results2 = pd.read_csv(csv_file2)
    y_true1 = results1["ground_truth"]
    y_pred1 = (results1["probability"] > 0.5).astype(int)
    y_pred_proba1 = results1["probability"]
    y_true2 = results2["ground_truth"]
    y_pred2 = (results2["probability"] > 0.5).astype(int)
    y_pred_proba2 = results2["probability"]


    # # ############################## Youden's ##############################
    # # # threshold finding for metrics calculation (Youden's theorem)
    # optimal_threshold = np.zeros(y_true1.shape[0])
    # for idx in range(y_true1.shape[0]):
    #     fpr, tpr, thresholds = roc_curve(y_true1, y_pred_proba1)
    #     # optimal_idx = np.argmax(tpr - fpr)
    #     optimal_idx = np.argmax(tpr + (1 - fpr))
    #     optimal_threshold[idx] = thresholds[optimal_idx]
    # y_true1 = (y_pred_proba1 > optimal_threshold).astype(np.int32)
    #
    # optimal_threshold = np.zeros(y_true2.shape[0])
    # for idx in range(y_true2.shape[0]):
    #     fpr, tpr, thresholds = roc_curve(y_true2, y_pred_proba2)
    #     # optimal_idx = np.argmax(tpr - fpr)
    #     optimal_idx = np.argmax(tpr + (1 - fpr))
    #     optimal_threshold[idx] = thresholds[optimal_idx]
    # y_true2 = (y_pred_proba2 > optimal_threshold).astype(np.int32)
    # # ############################## Youden's ##############################



    metrics = [roc_auc_score, accuracy_score, f1_score, sensitivity, specificity]
    metric_names = ["AUC", "Accuracy", "F1 Score", "Sensitivity", "Specificity"]

    # Collect samples for CSV saving
    samples1 = {}
    samples2 = {}

    results = {}
    for metric, metric_name in zip(metrics, metric_names):
        mean_metric1, std_metric1, ci1, values1 = bootstrap_metric_statistics_and_samples(y_true1,
                                                                                          y_pred1 if metric != roc_auc_score else y_pred_proba1,
                                                                                          metric, n_bootstrap)
        mean_metric2, std_metric2, ci2, values2 = bootstrap_metric_statistics_and_samples(y_true2,
                                                                                          y_pred2 if metric != roc_auc_score else y_pred_proba2,
                                                                                          metric, n_bootstrap)
        # Add samples to the dictionary
        samples1[metric_name] = values1
        samples2[metric_name] = values2

        p_value = bootstrap_p_value(y_true1, y_pred1 if metric != roc_auc_score else y_pred_proba1,
                                    y_pred2 if metric != roc_auc_score else y_pred_proba2, metric, n_bootstrap)

        # Apply Benjamini-Hochberg FDR correction to p-values
        reject_fdr, p_value = fdr_correction(p_value, alpha=0.05, method='indep')

        results[metric_name] = {
            "Data Scientist": {
                "Mean": mean_metric1,
                "Std": std_metric1,
                "95% CI": ci1
            },
            "Using ChatGPT ADA": {
                "Mean": mean_metric2,
                "Std": std_metric2,
                "95% CI": ci2
            },
            "p_value": p_value
        }

    # Save the samples to CSV files
    save_bootstrapped_samples_to_csv(samples1, output_file1)
    save_bootstrapped_samples_to_csv(samples2, output_file2)

    # Print the results
    print_results_updated_v3(results)






if __name__ == '__main__':


    main_analysis_with_saving(csv_file1, csv_file2, output_file1, output_file2)
