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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from scipy.stats import ranksums
from scipy.stats import norm
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import warnings
warnings.filterwarnings('ignore')



class cohort1():
    def __int__(self):
        pass

    def main_train_GPT_ADA(self):
        """Codes written by ChatGPT ADA
        Uses Gradient boosting machine
        """

        # Load the data
        data = pd.read_excel("/mnt/data/DataZonodo_v2_unlabeled.xlsx")

        # Display the first few rows of the dataset to inspect its structure
        print(data.head())

        # Set the first row as the column header
        data.columns = data.iloc[0]
        data = data.drop(0)

        # Convert columns to appropriate data types
        data = data.convert_dtypes()

        # Split data into training (IT) and test (EV) sets
        train_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "IT"]
        test_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "EV"]

        # Display the first few rows of the training dataset
        print(train_data.head())

        # Check for missing values in the training dataset
        missing_values = train_data.isnull().sum()

        print(missing_values)

        # Impute missing values with median for the numerical columns
        for column in ["Age at diagnosis of first tumor {s)", "Plasma NMN pg/ml", "Plasma MN pg/ml",
                       "Plasma MTY  pg/ml", "Spherical volume of  primary  tumor{s)"]:
            median_val = train_data[column].median()
            train_data[column].fillna(median_val, inplace=True)

        # Standardize numerical features
        numerical_features = ["Age at diagnosis of first tumor {s)", "Plasma NMN pg/ml", "Plasma MN pg/ml",
                              "Plasma MTY  pg/ml", "Spherical volume of  primary  tumor{s)"]
        scaler = StandardScaler()
        train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])

        # Encode categorical features
        label_encoders = {}
        categorical_features = ["Sex (M/F)", "Previous history of PPGLs?  (YES/NO)",
                                "Adrenal/Extra-adrenal location of primary tumor", "Presence of SDHB",
                                "Tumor category of primary tumor(S;B, M)", "Metastatic YES/NO"]
        for feature in categorical_features:
            le = LabelEncoder()
            train_data[feature] = le.fit_transform(train_data[feature])
            label_encoders[feature] = le

        # Display the preprocessed training data
        print(train_data.head())

        # Separate features and target variable
        X_train = train_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_train = train_data["Metastatic YES/NO"]

        # Train a Gradient Boosting Classifier
        gb_classifier = GradientBoostingClassifier()
        gb_classifier.fit(X_train, y_train)

        # Model is trained, now we'll preprocess the test set
        # Impute missing values with median for the numerical columns in test data
        for column in ["Age at diagnosis of first tumor {s)", "Plasma NMN pg/ml", "Plasma MN pg/ml",
                       "Plasma MTY  pg/ml", "Spherical volume of  primary  tumor{s)"]:
            median_val = test_data[column].median()
            test_data[column].fillna(median_val, inplace=True)

        # Standardize numerical features in test data
        test_data[numerical_features] = scaler.transform(test_data[numerical_features])

        # Encode categorical features in test data
        for feature in categorical_features:
            if feature != "Metastatic YES/NO":  # Exclude the target column
                test_data[feature] = label_encoders[feature].transform(test_data[feature])

        # Display the preprocessed test data
        print(test_data.head())

        # Extract features from the test data
        X_test = test_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])

        # Predict the probability of metastatic disease for test data
        probs = gb_classifier.predict_proba(X_test)[:, 1]  # Probabilities of the positive class (YES)

        # Predict the metastatic disease (YES/NO) for test data
        predictions = gb_classifier.predict(X_test)

        # Map the encoded predictions back to their original labels (YES/NO)
        predictions_labels = label_encoders["Metastatic YES/NO"].inverse_transform(predictions)

        # Create a dataframe with Patient ID, Predicted Labels, and Probabilities
        result_df = test_data[["Patient ID"]].copy()
        result_df["Predicted Metastatic"] = predictions_labels
        result_df["Probability"] = probs

        # Save the predictions to a CSV file
        result_file_path = "/mnt/data/predicted_metastatic_results.csv"
        result_df.to_csv(result_file_path, index=False)

        print(result_df.head())


    def main_train_Validatory(self):
        """Uses Adaboost ensemble tree classifier with grid search optimization
        """
        data = pd.read_excel("/PATH/DataZonodo_v2_original.xlsx")

        data.columns = data.iloc[0]
        data = data.drop(0)

        train_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "IT"]
        test_data = data[data["Internal Testing (IT)/External Validatio (EV)"] == "EV"]

        cols_to_convert = [
            "Age at diagnosis of first tumor {s)",
            "Plasma NMN pg/ml",
            "Plasma MN pg/ml",
            "Plasma MTY  pg/ml",
            "Spherical volume of  primary  tumor{s)" ]

        for col in cols_to_convert:
            median_train = train_data[col].median()
            train_data[col].fillna(median_train, inplace=True)
            test_data[col].fillna(median_train, inplace=True)

        categorical_cols = [
            "Sex (M/F)",
            "Previous history of PPGLs?  (YES/NO)",
            "Adrenal/Extra-adrenal location of primary tumor",
            "Presence of SDHB",
            "Tumor category of primary tumor(S;B, M)"]

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            label_encoders[col] = le

        print('training started ...\n')

        X_train = train_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_train = train_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        X_test = test_data.drop(
            columns=["Patient ID", "Internal Testing (IT)/External Validatio (EV)", "Metastatic YES/NO"])
        y_test = test_data["Metastatic YES/NO"].map({"YES": 1, "NO": 0})

        base_clf = DecisionTreeClassifier()
        ada_clf = AdaBoostClassifier(base_estimator=base_clf)

        param_grid = {
            'base_estimator__max_depth': [1, 2, 3, 4],
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]}

        grid_search = GridSearchCV(ada_clf, param_grid, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_ada_clf = grid_search.best_estimator_
        y_pred = best_ada_clf.predict(X_test)
        y_pred_proba = best_ada_clf.predict_proba(X_test)[:, 1]

        result_df = test_data[["Patient ID"]].copy()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test
        result_file_path = "/PATH/Validatory_predictions.csv"
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

    def optimize_threshold(self, probs, y_true):
        """Codes written by ChatGPT ADA
        Optimize the threshold for binary classification based on the training data."""
        best_threshold = 0
        best_sum = 0
        best_accuracy = 0

        # Test thresholds between 0 and 1 in increments of 0.01
        for threshold in [i * 0.01 for i in range(100)]:
            predictions = (probs[:, 1] > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Update best threshold if the sum of sensitivity and specificity is improved
            if sensitivity + specificity > best_sum:
                best_sum = sensitivity + specificity
                best_threshold = threshold
                best_accuracy = accuracy

        return best_threshold, best_accuracy


    def main_train_GPT_ADA(self):
        """Codes written by ChatGPT ADA
        Uses Gradient boosting
        """

        # Load the training dataset
        train_data = pd.read_csv('/mnt/data/train.csv')

        # Display the first few rows of the dataset and its basic information
        train_data_info = train_data.info()
        train_data_head = train_data.head()

        print(train_data_info, train_data_head)

        # Check the distribution of the target variable
        target_distribution = train_data['GroundTruth_bi'].value_counts(normalize=True)

        # Check for missing values in the dataset
        missing_values = train_data.isnull().sum().sort_values(ascending=False)

        print(target_distribution, missing_values[missing_values > 0])

        # Load the feature explanations file
        feature_explanations = pd.read_excel('/mnt/data/Feature explanations.xlsx')

        # Display the first few rows of the explanations
        print(feature_explanations.head())

        # Load the test dataset
        test_unlabeled = pd.read_csv('/mnt/data/test_unlabeled.csv')

        # Display the first few rows of the test dataset and its basic information
        test_unlabeled_info = test_unlabeled.info()
        test_unlabeled_head = test_unlabeled.head()

        print(test_unlabeled_info, test_unlabeled_head)

        # Identify columns that are not in the training dataset but are in the test dataset
        extra_cols = set(test_unlabeled.columns) - set(train_data.columns)

        # Drop these extra columns from the test dataset
        test_unlabeled = test_unlabeled.drop(columns=extra_cols)

        # Ensure that the test dataset now has the same columns as the training dataset (minus the target variable)
        aligned = set(train_data.columns) - set(test_unlabeled.columns) == {'GroundTruth_bi'}

        print(aligned)

        # Separate features and target variable from the training dataset
        X_train = train_data.drop(columns=['GroundTruth_bi'])
        y_train = train_data['GroundTruth_bi']

        # Scale the features
        scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # test_unlabeled_scaled = scaler.transform(test_unlabeled)
        #
        # Train the Gradient Boosting Classifier
        clf = GradientBoostingClassifier()
        # clf.fit(X_train_scaled, y_train)
        #
        # # Training accuracy
        # training_accuracy = clf.score(X_train_scaled, y_train)
        # print(training_accuracy)

        # Identify non-numeric columns in the training data
        non_numeric_cols = X_train.select_dtypes(exclude=['float64', 'int64']).columns

        # Display the unique values of these non-numeric columns
        unique_values = {col: X_train[col].unique() for col in non_numeric_cols}
        print(unique_values)

        # Drop the 'No.' column and encode the 'Ethnic' column using one-hot encoding
        X_train = pd.get_dummies(X_train, columns=['Ethnic'], drop_first=True)
        test_unlabeled = pd.get_dummies(test_unlabeled, columns=['Ethnic'], drop_first=True)

        # Ensure both datasets have the same columns after one-hot encoding
        missing_cols = set(X_train.columns) - set(test_unlabeled.columns)
        for col in missing_cols:
            test_unlabeled[col] = 0
        test_unlabeled = test_unlabeled[X_train.columns]

        # # Scale the features
        # X_train_scaled = scaler.fit_transform(X_train)
        # test_unlabeled_scaled = scaler.transform(test_unlabeled)

        # # Train the Gradient Boosting Classifier again
        # clf.fit(X_train_scaled, y_train)
        #
        # # Training accuracy
        # training_accuracy = clf.score(X_train_scaled, y_train)
        # print(training_accuracy)

        # Drop the 'No.' column from both training and test datasets
        X_train = X_train.drop(columns=['No.'])
        test_unlabeled = test_unlabeled.drop(columns=['No.'])

        # # Scale the features again
        # X_train_scaled = scaler.fit_transform(X_train)
        # test_unlabeled_scaled = scaler.transform(test_unlabeled)

        # # Train the Gradient Boosting Classifier again
        # clf.fit(X_train_scaled, y_train)
        #
        # # Training accuracy
        # training_accuracy = clf.score(X_train_scaled, y_train)
        # print(training_accuracy)

        # Identify non-numeric columns in the training data after previous processing
        remaining_non_numeric_cols = X_train.select_dtypes(exclude=['float64', 'int64']).columns

        # Display the unique values of these non-numeric columns
        remaining_unique_values = {col: X_train[col].unique() for col in remaining_non_numeric_cols}
        print(remaining_unique_values)

        # Identify columns with object data type in the training dataset
        object_cols_train = X_train.select_dtypes(include=['object']).columns

        # Identify columns with object data type in the test dataset
        object_cols_test = test_unlabeled.select_dtypes(include=['object']).columns

        print(object_cols_train, object_cols_test)

        # Convert the 'CheckID' column in the test dataset to numeric type
        test_unlabeled['CheckID'] = pd.to_numeric(test_unlabeled['CheckID'], errors='coerce')

        # Scale the features again
        X_train_scaled = scaler.fit_transform(X_train)
        test_unlabeled_scaled = scaler.transform(test_unlabeled)

        # # Train the Gradient Boosting Classifier again
        # clf.fit(X_train_scaled, y_train)
        #
        # # Training accuracy
        # training_accuracy = clf.score(X_train_scaled, y_train)
        # print(training_accuracy)

        # Get predicted probabilities from the classifier on the training data
        predicted_probs_train = clf.predict_proba(X_train_scaled)

        # Optimize the threshold
        best_threshold, best_accuracy_train = self.optimize_threshold(predicted_probs_train, y_train)
        print(best_threshold, best_accuracy_train)

        # Fill NaN values in the test dataset with the median of the respective columns
        test_unlabeled_filled = test_unlabeled.fillna(test_unlabeled.median())

        # Scale the filled test dataset
        test_unlabeled_scaled_filled = scaler.transform(test_unlabeled_filled)

        # Predict the probabilities on the filled test dataset using the trained classifier
        predicted_probs_test = clf.predict_proba(test_unlabeled_scaled_filled)

        # Make predictions based on the optimized threshold
        predicted_labels_test = (predicted_probs_test[:, 1] > best_threshold).astype(int)

        # Prepare the results for export to a CSV file
        results = pd.DataFrame({
            'PatientID': test_unlabeled_filled.index,
            'PredictedLabel': predicted_labels_test,
            'probability': predicted_probs_test[:, 1]})

        # Save the results to a CSV file
        results_file_path = "/mnt/data/predicted_results.csv"
        results.to_csv(results_file_path, index=False)


    def main_train_Validatory(self):
        """Uses light Gradient boosting
        """
        train = pd.read_csv("/PATH/train.csv")
        test = pd.read_csv("/PATH/test.csv")

        target = 'GroundTruth_bi'

        X_train = train.drop([target], axis=1)
        y_train = train[target]
        X_test = test.drop([target], axis=1)
        y_test = test[target]

        le = LabelEncoder()
        X_train['Ethnic'] = le.fit_transform(X_train['Ethnic'])
        X_test['Ethnic'] = [le.transform([val])[0] if val in le.classes_ else 0 for val in X_test['Ethnic']]

        X_train = X_train.drop(['No.', 'CheckID'], axis=1)
        X_test = X_test.drop(['No.', 'CheckID', 'DuodenalOther'], axis=1)

        additional_features_test = set(X_test.columns) - set(X_train.columns)
        X_test = X_test.drop(additional_features_test, axis=1)
        X_train_imputed = X_train
        X_test_imputed = X_test

        print('training started ...\n')

        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, objective='binary', class_weight='balanced', random_state=42)
        print(clf.get_params())
        clf.fit(X_train_imputed, y_train)
        y_pred_proba = clf.predict_proba(X_test_imputed)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(np.int32)  # Default threshold

        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        result_file_path = "/PATH/Validatory_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)





class cohort3():
    def __int__(self):
        pass

    def main_train_GPT_ADA(self):
        """Codes written by ChatGPT ADA
        Uses RF
        """
        # Load the "Discovery Set" data
        discovery_set = pd.read_excel("/mnt/data/Discovery Set.xlsx")

        # Display the first few rows of the dataset
        print(discovery_set.head())

        # Load the "Validation Set_unlabeled" data
        validation_set_unlabeled = pd.read_excel("/mnt/data/Validation Set_unlabeled.xlsx")

        # Display the first few rows of the dataset
        print(validation_set_unlabeled.head())

        # Set column headers for both datasets and drop the first row
        discovery_set.columns = discovery_set.iloc[0]
        discovery_set = discovery_set.drop(0)

        validation_set_unlabeled.columns = validation_set_unlabeled.iloc[0]
        validation_set_unlabeled = validation_set_unlabeled.drop(0)

        # Convert the data to numeric for model training
        discovery_set = discovery_set.apply(pd.to_numeric)
        validation_set_unlabeled = validation_set_unlabeled.apply(pd.to_numeric)

        # Display the cleaned discovery set
        print(discovery_set.head())

        # Replace whitespace or non-numeric values with 0
        discovery_set = discovery_set.replace(' ', 0)

        # Convert the data to numeric again
        discovery_set = discovery_set.apply(pd.to_numeric)

        # Display the cleaned discovery set
        print(discovery_set.head())

        # Separate features and target variable
        X_train = discovery_set.drop(columns=["Diagnoses"])
        y_train = discovery_set["Diagnoses"]

        # Initialize the Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Validate the model's performance using cross-validation
        cross_val_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")

        # Train the classifier on the entire training data
        clf.fit(X_train, y_train)

        cross_val_scores.mean()

        # Make predictions on the "Validation Set_unlabeled"
        predicted_labels = clf.predict(validation_set_unlabeled)
        predicted_probs = clf.predict_proba(validation_set_unlabeled)[:, 1]  # Probability of class 1

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            "Predicted Label": predicted_labels,
            "probability": predicted_probs})

        # Save the results to a CSV file
        results_file_path = "/mnt/data/predicted_results.csv"
        results_df.to_csv(results_file_path, index=False)


    def main_train_Validatory(self):
        """Uses SVM
        """

        discovery_set_path = "/PATH/Discovery Set.xlsx"
        discovery_set = pd.read_excel(discovery_set_path, header=1)
        validation_set_path = "/PATH/Validation Set_original.xlsx"
        validation_set = pd.read_excel(validation_set_path, header=1)

        X_discovery = discovery_set.drop(columns=['Diagnoses'])
        y_discovery = discovery_set['Diagnoses']
        X_validation = validation_set.drop(columns=['Diagnoses'])
        y_validation = validation_set['Diagnoses']

        X_discovery['c.235delC'] = pd.to_numeric(X_discovery['c.235delC'], errors='coerce')
        X_discovery['c.235delC'].fillna(X_discovery['c.235delC'].median(), inplace=True)

        scaler = MinMaxScaler()
        X_discovery = scaler.fit_transform(X_discovery)
        X_validation = scaler.transform(X_validation)

        print('training started ...\n')

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear']
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

        y_pred = clf.predict(X_validation)
        y_pred_proba = clf.predict_proba(X_validation)[:, 1]
        y_test = y_validation

        print(clf.get_params())
        print(clf.best_estimator_)
        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        result_file_path = "/PATH/Validatory_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)





class cohort4():
    def __int__(self):
        pass

    def main_train_GPT_ADA(self):
        """Codes written by ChatGPT ADA
        Uses RF
        """
        # Load the training data
        training_data = pd.read_csv('/mnt/data/training_data.csv')

        # Display the first few rows of the training data
        print(training_data.head())

        # Load the test data
        test_data_unlabeled = pd.read_csv('/mnt/data/test_data_unlabeled.csv')

        # Display the first few rows of the test data
        print(test_data_unlabeled.head())

        # Separate features and target in the training data
        X_train = training_data.drop(columns=['patient_id', 'cohort_type', 'cohort_flag'])
        y_train = training_data['cohort_flag']

        # Features for the test data
        X_test = test_data_unlabeled.drop(columns=['patient_id'])

        print(X_train.shape, y_train.shape, X_test.shape)

        # Train a Random Forest classifier
        rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_train)

        # Predict probabilities for the test data
        test_probabilities_rf = rf_clf.predict_proba(X_test)[:, 1]

        print(test_probabilities_rf[:5])  # Display the first 5 predicted probabilities for inspection

        # Threshold the probabilities to make binary predictions
        test_predictions = (test_probabilities_rf >= 0.5).astype(int)

        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            'patient_id': test_data_unlabeled['patient_id'],
            'predicted_label': test_predictions,
            'probability': test_probabilities_rf})

        # Save the results to a CSV file
        output_filepath = '/mnt/data/predicted_results.csv'
        results_df.to_csv(output_filepath, index=False)



    def main_train_Validatory(self):
        """Uses RF
        """

        # Loading the dataset
        train_path = "/PATH/training_data.csv"
        train_data = pd.read_csv(train_path)
        test_path = "/PATH/test_data.csv"
        test_data = pd.read_csv(test_path)

        X_train = train_data.drop(columns=['patient_id', 'cohort_flag', 'cohort_type'])
        y_train = train_data['cohort_flag']
        X_test = test_data.drop(columns=['patient_id', 'cohort_flag', 'cohort_type'])
        y_test = test_data['cohort_flag']

        print('training started ...\n')

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
            'bootstrap': [True, False]}

        clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, verbose=1, n_jobs=-1, scoring='accuracy', cv=5)
        clf.fit(X_train_scaled, y_train)

        print("Best hyperparameters found: ", clf.best_params_)
        print(clf.get_params())
        print(clf.best_estimator_)

        # Predict using the best model
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        result_file_path = "/PATH/Validatory_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, f1, sensitivity, specificity)






if __name__ == '__main__':
    # cohort = cohort1()
    # cohort = cohort2()
    # cohort = cohort3()
    cohort = cohort4()

    cohort.main_train_Validatory()
    # cohort.main_train_GPT_ADA()
