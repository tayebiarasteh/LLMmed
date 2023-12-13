"""
shap_LLMmed.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import warnings
warnings.filterwarnings('ignore')



class cohort1():
    def __int__(self):
        pass


    def shap_plot(self):
        shap_values_df = pd.read_csv('final_shap_1.csv')
        avg_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)
        top_features = avg_abs_shap_values.index.tolist()

        min_avg_shap = avg_abs_shap_values.min()
        max_avg_shap = avg_abs_shap_values.max()
        norm_dataset = plt.Normalize(vmin=min_avg_shap, vmax=max_avg_shap)
        sm_dataset = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_dataset)
        palette_dict_custom = {feature: sm_dataset.to_rgba(value) for feature, value in avg_abs_shap_values.items()}

        means_dataset = shap_values_df[top_features].mean()

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.set_style("white")
        sns.boxplot(data=shap_values_df[top_features], orient='h', palette=palette_dict_custom,
                    fliersize=5, flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                    ax=ax)
        ax.scatter(means_dataset, range(len(top_features)), marker='x', color='black', s=500, zorder=5,
                   label='Mean')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=3)

        x_min_specified = -0.08
        x_max_specified = 0.20
        specified_xticks = [-3, -2, -1, 0, 1, 2, 3, 4]

        ax.set_xlim(x_min_specified, x_max_specified)
        ax.set_xticks(specified_xticks)
        ax.set_xticklabels([f"{int(value)}" for value in specified_xticks],
                           fontsize=30)
        ax.set_yticklabels(top_features, fontsize=30)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=35, labelpad=20)
        ax.set_ylabel("")
        ax.set_title("Metastatic Disease [Endocrinologic Oncology]", fontsize=40, ha='left', x=0, y=1.02)

        cax_new = fig.add_axes([0.93, 0.125, 0.02, 0.75])
        cbar = plt.colorbar(sm_dataset, cax=cax_new, orientation='vertical')
        cbar.set_ticks(np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5))
        cbar.set_ticklabels(
            [f"{value:.2f}" for value in np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5)])
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Mean absolute SHAP value', rotation=270, fontsize=35, labelpad=40)

        plt.show()




class cohort2():
    def __int__(self):
        pass


    def shap_plot(self):
        shap_values_df = pd.read_csv('final_shap_2.csv')
        avg_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)
        top_features = avg_abs_shap_values.index.tolist()

        min_avg_shap = avg_abs_shap_values.min()
        max_avg_shap = avg_abs_shap_values.max()
        norm_dataset = plt.Normalize(vmin=min_avg_shap, vmax=max_avg_shap)
        sm_dataset = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_dataset)
        palette_dict_custom = {feature: sm_dataset.to_rgba(value) for feature, value in avg_abs_shap_values.items()}

        means_dataset = shap_values_df[top_features].mean()

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.set_style("white")
        sns.boxplot(data=shap_values_df[top_features], orient='h', palette=palette_dict_custom,
                    fliersize=5, flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                    ax=ax)
        ax.scatter(means_dataset, range(len(top_features)), marker='x', color='black', s=500, zorder=5,
                   label='Mean')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=3)

        x_min_specified = -0.08
        x_max_specified = 0.20
        specified_xticks = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        ax.set_xlim(x_min_specified, x_max_specified)
        ax.set_xticks(specified_xticks)
        ax.set_xticklabels([f"{int(value)}" for value in specified_xticks],
                           fontsize=30)
        ax.set_yticklabels(top_features, fontsize=30)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=35, labelpad=20)
        ax.set_ylabel("")
        ax.set_title("Oesophageal Cancer [Gastrointestinal Oncology]", fontsize=40, ha='left', x=0, y=1.02)

        cax_new = fig.add_axes([0.93, 0.125, 0.02, 0.75])
        cbar = plt.colorbar(sm_dataset, cax=cax_new, orientation='vertical')
        cbar.set_ticks(np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5))
        cbar.set_ticklabels(
            [f"{value:.2f}" for value in np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5)])
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Mean absolute SHAP value', rotation=270, fontsize=35, labelpad=40)

        plt.show()




class cohort3():
    def __int__(self):
        pass


    def shap_plot(self):
        shap_values_df = pd.read_csv('final_shap_3.csv')
        avg_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)
        top_features = avg_abs_shap_values.index.tolist()

        min_avg_shap = avg_abs_shap_values.min()
        max_avg_shap = avg_abs_shap_values.max()
        norm_dataset = plt.Normalize(vmin=min_avg_shap, vmax=max_avg_shap)
        sm_dataset = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_dataset)
        palette_dict_custom = {feature: sm_dataset.to_rgba(value) for feature, value in avg_abs_shap_values.items()}

        means_dataset = shap_values_df[top_features].mean()

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.set_style("white")
        sns.boxplot(data=shap_values_df[top_features], orient='h', palette=palette_dict_custom,
                    fliersize=5, flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                    ax=ax)
        ax.scatter(means_dataset, range(len(top_features)), marker='x', color='black', s=500, zorder=5,
                   label='Mean')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=3)

        x_min = -0.5
        x_max = 0.4
        specified_xticks = np.linspace(x_min, x_max, num=10)

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(specified_xticks)
        ax.set_xticklabels([f"{value:.1f}" for value in specified_xticks], fontsize=30)
        ax.set_yticklabels(top_features, fontsize=30)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=35, labelpad=20)
        ax.set_ylabel("")
        ax.set_title("Hereditary Hearing Loss [Otolaryngology]", fontsize=40, ha='left', x=0, y=1.02)

        cax = fig.add_axes([0.93, 0.125, 0.02, 0.75])
        cbar = plt.colorbar(sm_dataset, cax=cax, orientation='vertical')
        cbar.set_ticks(
            np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5))
        cbar.set_ticklabels([f"{value:.2f}" for value in
                                          np.linspace(avg_abs_shap_values.min(),
                                                      avg_abs_shap_values.max(), 5)])
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Mean absolute SHAP value', rotation=270, fontsize=35, labelpad=40)

        plt.show()



class cohort4():
    def __int__(self):
        pass



    def shap_plot(self):
        shap_values_df = pd.read_csv('final_shap_4.csv')
        avg_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)
        top_features = avg_abs_shap_values.index.tolist()

        min_avg_shap = avg_abs_shap_values.min()
        max_avg_shap = avg_abs_shap_values.max()
        norm_dataset = plt.Normalize(vmin=min_avg_shap, vmax=max_avg_shap)
        sm_dataset = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm_dataset)
        palette_dict_custom = {feature: sm_dataset.to_rgba(value) for feature, value in avg_abs_shap_values.items()}

        means_dataset = shap_values_df[top_features].mean()

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.set_style("white")
        sns.boxplot(data=shap_values_df[top_features], orient='h', palette=palette_dict_custom,
                    fliersize=5, flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
                    ax=ax)
        ax.scatter(means_dataset, range(len(top_features)), marker='x', color='black', s=500, zorder=5,
                   label='Mean')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=3)

        x_min = -0.1
        x_max = 0.2
        specified_xticks = [-0.1, 0, 0.1, 0.2]

        ax.set_xlim(x_min, x_max)
        ax.set_xticks(specified_xticks)
        ax.set_xticklabels([f"{value:.1f}" for value in specified_xticks], fontsize=30)
        ax.set_yticklabels(top_features, fontsize=30)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=35, labelpad=20)
        ax.set_ylabel("")
        ax.set_title("Cardiac Amyloidosis [Cardiology]", fontsize=40, ha='left', x=0, y=1.02)

        cax = fig.add_axes([0.93, 0.125, 0.02, 0.75])
        cbar = plt.colorbar(sm_dataset, cax=cax, orientation='vertical')
        cbar.set_ticks(
            np.linspace(avg_abs_shap_values.min(), avg_abs_shap_values.max(), 5))
        cbar.set_ticklabels([f"{value:.2f}" for value in
                                          np.linspace(avg_abs_shap_values.min(),
                                                      avg_abs_shap_values.max(), 5)])
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('Mean absolute SHAP value', rotation=270, fontsize=35, labelpad=40)

        plt.show()
