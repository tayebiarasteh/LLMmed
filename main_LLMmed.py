"""
Created on Sep 26, 2023.
main_LLMmed.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import os
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import ranksums
from scipy.stats import norm
from sklearn.impute import KNNImputer
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import warnings
warnings.filterwarnings('ignore')



class cohort1():
    def __int__(self):
        pass


    def main_train_GPT(self):
        """Uses Gradient boosting machine
        """

        # Load the data
        data = pd.read_excel("/home/soroosh/Documents/datasets/LLMmed/1/DataZonodo_v2_original.xlsx")

        # Display the first few rows of the data
        data.head()

        # Set the first row as the column headers
        data.columns = data.iloc[0]
        data = data.drop(0)

        # Split the data into training and test sets
        train_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "IT"]
        test_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "EV"]

        # Convert the relevant columns to their appropriate data types
        cols_to_convert = [
            "Age at diagnosis of first tumor {s)",
            "Plasma NMN pg/ml",
            "Plasma MN pg/ml",
            "Plasma MTY  pg/ml",
            "Spherical volume of  primary  tumor{s)"
        ]

        for col in cols_to_convert:
            train_data[col] = train_data[col].astype(float)
            test_data[col] = test_data[col].astype(float)

        # Check for missing values in training and test data
        missing_train = train_data.isnull().sum()
        missing_test = test_data.isnull().sum()

        # Impute missing values with median for both training and test data
        for col in cols_to_convert:
            median_train = train_data[col].median()

            # Impute missing values in train and test set with median from training set
            train_data[col].fillna(median_train, inplace=True)
            test_data[col].fillna(median_train, inplace=True)

        # Check if there are any more missing values
        missing_train = train_data.isnull().sum().max()
        missing_test = test_data.isnull().sum().max()

        # print(missing_train, missing_test)

        # Identify categorical columns for encoding
        categorical_cols = [
            "Sex (M/F)",
            "Previous history of PPGLs?  (YES/NO)",
            "Adrenal/Extra-adrenal location of primary tumor",
            "Presence of SDHB",
            "Tumor category of primary tumor(S;B, M)"
        ]

        # Initialize the label encoder
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            label_encoders[col] = le


        print('training started ...\n')
        # Split the data into features (X) and target (y) sets
        X_train = train_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_train = train_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        X_test = test_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_test = test_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Train the Gradient Boosting classifier
        clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        print("Chosen hyperparameters:", clf.get_params())

        # Predict on the test set
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # saving the results in a CSV

        # Create a dataframe with Patient ID, Predicted Labels, and Probabilities
        result_df = test_data[["Patient ID"]].copy()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        # Save the predictions to a CSV file
        result_file_path = "/home/soroosh/Documents/datasets/LLMmed/1/ADA_predictions.csv"
        result_df.to_csv(result_file_path, index=False)


        # Calculate evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)



    def main_train_DS(self):
        """Uses Adaboost ensemble tree classifier with grid search optimization
        """
        # Load the data
        data = pd.read_excel("/home/soroosh/Documents/datasets/LLMmed/1/DataZonodo_v2_original.xlsx")

        # Display the first few rows of the data
        data.head()

        # Set the first row as the column headers
        data.columns = data.iloc[0]
        data = data.drop(0)

        # Split the data into training and test sets
        train_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "IT"]
        test_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "EV"]

        print(train_data.shape, test_data.shape)

        # columns with number data types
        cols_to_convert = [
            "Age at diagnosis of first tumor {s)",
            "Plasma NMN pg/ml",
            "Plasma MN pg/ml",
            "Plasma MTY  pg/ml",
            "Spherical volume of  primary  tumor{s)"
        ]

        # Impute missing values with median for both training and test data
        for col in cols_to_convert:
            median_train = train_data[col].median()

            # Impute missing values in train and test set with median from training set
            train_data[col].fillna(median_train, inplace=True)
            test_data[col].fillna(median_train, inplace=True)

        # Identify categorical columns for encoding
        categorical_cols = [
            "Sex (M/F)",
            "Previous history of PPGLs?  (YES/NO)",
            "Adrenal/Extra-adrenal location of primary tumor",
            "Presence of SDHB",
            "Tumor category of primary tumor(S;B, M)"
        ]

        # Initialize the label encoder
        label_encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            label_encoders[col] = le

        print('training started ...\n')
        # Split the data into features (X) and target (y) sets
        X_train = train_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_train = train_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        X_test = test_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_test = test_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        # Define the base model
        base_clf = DecisionTreeClassifier()

        # Define AdaBoost with the base model
        ada_clf = AdaBoostClassifier(base_estimator=base_clf)

        # Define hyperparameters to tune
        param_grid = {
            'base_estimator__max_depth': [1, 2, 3, 4],  # max depth of the decision tree
            'n_estimators': [10, 50, 100, 200],  # number of trees
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]  # learning rate
        }

        # Use GridSearchCV
        grid_search = GridSearchCV(ada_clf, param_grid, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_ada_clf = grid_search.best_estimator_

        # Evaluate the best model on test data
        y_pred = best_ada_clf.predict(X_test)
        y_pred_proba = best_ada_clf.predict_proba(X_test)[:, 1]


        # saving the results in a CSV
        result_df = test_data[["Patient ID"]].copy()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        result_file_path = "/home/soroosh/Documents/datasets/LLMmed/1/DS_predictions.csv"
        result_df.to_csv(result_file_path, index=False)


        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print("Chosen hyperparameters:", ada_clf.get_params())
        print("Best parameters:", grid_search.best_params_)

        print(auc, accuracy, f1, sensitivity, specificity)




class cohort2():
    def __int__(self):
        pass


    def main_train_DS(self):
        """Uses light Gradient boosting
        """

        # Load the train and test datasets
        train = pd.read_csv("/home/soroosh/Documents/datasets/LLMmed/2/train.csv")
        test = pd.read_csv("/home/soroosh/Documents/datasets/LLMmed/2/test.csv")

        # Check for missing values in the train and test datasets
        missing_train = train.isnull().sum().sum()
        missing_test = test.isnull().sum().sum()

        print(missing_train, missing_test)

        # Define the target variable
        target = 'GroundTruth_bi'

        # Split the data into features and target variable
        X_train = train.drop([target], axis=1)
        y_train = train[target]

        X_test = test.drop([target], axis=1)
        y_test = test[target]

        # Initialize the label encoder
        le = LabelEncoder()
        X_train['Ethnic'] = le.fit_transform(X_train['Ethnic'])
        # Replace unobserved categories with a placeholder
        X_test['Ethnic'] = [le.transform([val])[0] if val in le.classes_ else 0 for val in X_test['Ethnic']]

        # Remove non-numeric features
        X_train = X_train.drop(['No.', 'CheckID'], axis=1)
        X_test = X_test.drop(['No.', 'CheckID', 'DuodenalOther'], axis=1)

        # Identify non-numeric columns in the training set
        non_numeric_columns_train = X_train.select_dtypes(include=['object']).columns.tolist()

        # Identify non-numeric columns in the test set
        non_numeric_columns_test = X_test.select_dtypes(include=['object']).columns.tolist()

        # Identify additional features in the test set that are not in the training set
        additional_features_test = set(X_test.columns) - set(X_train.columns)

        # Drop these additional features from the test set
        X_test = X_test.drop(additional_features_test, axis=1)

        # Check again for missing values
        missing_train_imputed = X_train.isnull().sum().sum()
        missing_test_imputed = X_test.isnull().sum().sum()

        print(missing_train_imputed, missing_test_imputed)
        X_train_imputed = X_train
        X_test_imputed = X_test


        print('training started ...\n')
        # # Initialize the Gradient Boosting Classifier
        # gb = GradientBoostingClassifier(random_state=0)

        # Initialize the LGBMClassifier
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, objective='binary', random_state=42)
        pdb.set_trace()

        # Fit the model
        clf.fit(X_train_imputed, y_train)

        # Predict using the best model
        y_pred_proba = clf.predict_proba(X_test_imputed)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(np.int32)  # Default threshold

        # saving the results in a CSV
        # Create a dataframe with Probabilities
        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        # Save the predictions to a CSV file
        result_file_path = "/home/soroosh/Documents/datasets/LLMmed/2/DS_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        # # threshold finding for metrics calculation (Youden's theorem)
        # # Calculate the ROC curve
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
        # # Calculate the Youden's Index for each threshold
        # J = tpr - fpr
        # # Identify the optimal threshold
        # optimal_threshold = thresholds[np.argmax(J)]
        # # Use the optimal threshold to classify predicted probabilities into predicted classes
        # y_pred = (y_pred_proba > optimal_threshold).astype(np.int32)

        # Calculate evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)



class cohort3():
    def __int__(self):
        pass

    def main_train_DS(self):
        """Uses SVM
        """

        discovery_set_path = "/home/soroosh/Documents/datasets/LLMmed/3/Discovery Set.xlsx"
        discovery_set = pd.read_excel(discovery_set_path, header=1)

        # Load the validation set
        validation_set_path = "/home/soroosh/Documents/datasets/LLMmed/3/Validation Set_original.xlsx"
        validation_set = pd.read_excel(validation_set_path, header=1)  # Skipping the first row to use the second row as the header

        # Separate features and target for discovery and validation sets
        X_discovery = discovery_set.drop(columns=['Diagnoses'])
        y_discovery = discovery_set['Diagnoses']
        X_validation = validation_set.drop(columns=['Diagnoses'])
        y_validation = validation_set['Diagnoses']

        # Check for any non-numeric values in the discovery set
        non_numeric_discovery = {col: X_discovery[col].apply(lambda x: isinstance(x, str)).sum() for col in
                                 X_discovery.columns}
        non_numeric_discovery = {k: v for k, v in non_numeric_discovery.items() if v > 0}

        # Check for any non-numeric values in the validation set
        non_numeric_validation = {col: X_validation[col].apply(lambda x: isinstance(x, str)).sum() for col in
                                  X_validation.columns}
        non_numeric_validation = {k: v for k, v in non_numeric_validation.items() if v > 0}

        # print(non_numeric_discovery, non_numeric_validation)

        # Convert the problematic column to numeric, handling any non-numeric values
        X_discovery['c.235delC'] = pd.to_numeric(X_discovery['c.235delC'], errors='coerce')

        # Fill any NaN values with the median of the column
        X_discovery['c.235delC'].fillna(X_discovery['c.235delC'].median(), inplace=True)

        # Scale the features
        scaler = MinMaxScaler()
        X_discovery = scaler.fit_transform(X_discovery)
        X_validation = scaler.transform(X_validation)  # Use the same scaler to transform validation set

        print('training started ...\n')

        # # Train a RandomForest model
        # model = RandomForestClassifier(random_state=42)
        # model.fit(X_discovery, y_discovery)

        # # Predict using the best model
        # y_pred = model.predict(X_validation)
        # y_pred_proba = model.predict_proba(X_validation)[:, 1]
        # y_test = y_validation

        # Define the SVM classifier and parameters for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
            # 'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            # 'degree': [2, 3, 4],  # only used when kernel='poly'
            # 'coef0': [0, 1]  # useful for 'poly' and 'sigmoid'
        }
        svc = SVC(probability=True)

        # Define 10-fold stratified cross-validation
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
        #  'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
        #  'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples',
        #  'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss',
        #  'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance',
        #  'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error',
        #  'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision',
        #  'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall',
        #  'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo',
        #  'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']

        clf = GridSearchCV(svc, param_grid, cv=stratified_kfold, scoring='accuracy')
        clf.fit(X_discovery, y_discovery)

        # Use the optimized SVM model to predict
        y_pred = clf.predict(X_validation)
        y_pred_proba = clf.predict_proba(X_validation)[:, 1]
        y_test = y_validation

        # saving the results in a CSV
        # Create a dataframe with Probabilities
        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        # Save the predictions to a CSV file
        result_file_path = "/home/soroosh/Documents/datasets/LLMmed/3/DS_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        # Calculate evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)




class cohort4():
    def __int__(self):
        pass

    def main_train_DS(self):
        """Uses RF
        """

        # Loading the dataset
        train_path = "/home/soroosh/Documents/datasets/LLMmed/4/training_data.csv"
        train_data = pd.read_csv(train_path)
        test_path = "/home/soroosh/Documents/datasets/LLMmed/4/test_data.csv"
        test_data = pd.read_csv(test_path)

        # Checking the distribution of the positive and negative classes in the "cohort_flag" column
        class_distribution = train_data['cohort_flag'].value_counts()
        print(class_distribution)

        # Separating features and target variable
        X_train = train_data.drop(columns=['patient_id', 'cohort_flag', 'cohort_type'])
        y_train = train_data['cohort_flag']

        # Separating features and target variable
        X_test = test_data.drop(columns=['patient_id', 'cohort_flag', 'cohort_type'])
        y_test = test_data['cohort_flag']

        # Checking the distribution of the positive and negative classes in the training and test sets
        train_class_distribution = y_train.value_counts()
        test_class_distribution = y_test.value_counts()

        print(train_class_distribution, test_class_distribution)

        # Checking for missing values
        missing_values = X_train.isnull().sum().sum()

        # Scaling the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(missing_values)


        print('training started ...\n')

        # # Train a RandomForest model
        # model = RandomForestClassifier(random_state=42)
        # model.fit(X_train_scaled, y_train)
        #
        # # Predict using the best model
        # y_pred = model.predict(X_test_scaled)
        # y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Checking for missing values
        missing_values = X_train.isnull().sum().sum()
        print(missing_values)

        # Scaling the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print('training started ...\n')

        # Setting hyperparameters for grid search
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 3, 5, 7],
            'min_samples_leaf': [2, 3, 4, 5],
            'bootstrap': [True, False]
        }

        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, verbose=1, n_jobs=-1, scoring='accuracy', cv=5)
        clf.fit(X_train_scaled, y_train)

        print("Best hyperparameters found: ", clf.best_params_)

        # Predict using the best model
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

        # saving the results in a CSV
        # Create a dataframe with Probabilities
        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        # Save the predictions to a CSV file
        result_file_path = "/home/soroosh/Documents/datasets/LLMmed/4/DS_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        # Calculate evaluation metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)









if __name__ == '__main__':
    # cohort = cohort1()
    cohort = cohort2()
    # cohort = cohort3()
    # cohort = cohort4()

    cohort.main_train_DS()
    # cohort.main_train_GPT()
