import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import seaborn as sns
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Load data
def load_preprocessed_data(train_load_path, test_load_path):
    df_train = pd.read_csv(train_load_path)
    df_test = pd.read_csv(test_load_path)
    return df_train, df_test


# Save model
def save_model(model, models_folder, model_name):
    model_path = os.path.join(models_folder, f"{model_name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


# Load model
def load_model(models_folder, model_name):
    model_path = os.path.join(models_folder, f"{model_name.replace(' ', '_').lower()}.pkl")
    return joblib.load(model_path)


# Apply scaling conditionally
def conditional_scaling(X_train, X_val, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled
    return X_train, X_val


# Preprocess 
def preprocess_dataframe(df, vectorizer=None, fit_vectorizer=False, scale=True):
    stop_words = set(stopwords.words('english')) | set(stopwords.words('german'))
    stop_words_list = list(stop_words)

    y = df['label']
    df['body'] = df['body'].fillna("")

    X_text = df['body']
    X_numerical = df.drop(columns=['label', 'body'], errors='ignore')

    if fit_vectorizer:
        vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_features=5000)
        X_tfidf = vectorizer.fit_transform(X_text)

    else:
        if vectorizer is None:
            raise ValueError("A fitted vectorizer must be provided if fit_vectorizer is False.")
        X_tfidf = vectorizer.transform(X_text)
    
    _, X_numerical_proc = conditional_scaling(X_numerical, X_numerical, scale=scale)
    X_combined = np.hstack([X_tfidf.toarray(), X_numerical_proc])

    if fit_vectorizer:
        return X_combined, y, vectorizer
    else:
        return X_combined, y
    

# Train and save models
def train_and_save_model(model, param_grid, X_train, y_train, model_name, models_folder, scale_features=False):
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    save_model(best_model, models_folder, model_name)

    print(f"Best {model_name} params:", grid_search.best_params_)
    return best_model


# Save outpust
def save_classification_report(report_df, output_folder, model_name):
    report_path = os.path.join(output_folder, f"{model_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_df.to_string())

def save_confusion_matrix(conf_matrix, output_folder, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Phishing"], yticklabels=["Legit", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    cm_path = os.path.join(output_folder, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

def save_feature_distributions(X_val, X_test, output_folder):
    for feature in X_val.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(X_val[feature], kde=True, color='blue', label='Validation', stat="density", bins=30, alpha=0.6)
        sns.histplot(X_test[feature], kde=True, color='red', label='Test', stat="density", bins=30, alpha=0.6)
        plt.title(f"Feature Distribution - {feature}")
        plt.xlabel(feature)
        plt.legend()
        feature_path = os.path.join(output_folder, f"feature_{feature}.png")
        plt.savefig(feature_path)
        plt.close()


# Evaluate
def evaluate_model(model, X_test_combined, y_test, model_name, output_folder):
    y_pred = model.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    conf_matrix = confusion_matrix(y_test, y_pred)
    save_classification_report(report_df, output_folder, model_name)
    save_confusion_matrix(conf_matrix, output_folder, model_name)
    return y_pred, acc, report_df, conf_matrix


# Load all models
def load_all_models(models_folder, model_names):
    models = [(name, load_model(models_folder, name)) for name in model_names]
    return models


# Ensemble model
def create_and_train_ensemble(models_folder, model_names):
    models = {name: load_model(models_folder, name) for name in model_names}

    voting_clf = VotingClassifier(
        estimators=[
            ('naive_bayes', models['naive_bayes']),
            ('decision_tree', models['decision_tree']),
            ('random_forest', models['random_forest'])
        ],
        voting='hard'
    )

    return voting_clf


# Feature Distribution
def plot_feature_distributions(val_df, test_df, feature):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(val_df[feature].dropna(), kde=True, color='blue', label='Validation', stat="density", bins=30, alpha=0.6)
    sns.histplot(test_df[feature].dropna(), kde=True, color='red', label='Test', stat="density", bins=30, alpha=0.6)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.legend()

    plt.subplot(1, 2, 2)
    data = [val_df[feature].dropna(), test_df[feature].dropna()]
    plt.boxplot(data, labels=['Validation', 'Test'])
    plt.title(f'Boxplot of {feature}')
    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()


# TFID Stats
def tfidf_statistics(tfidf_matrix):
    tfidf_dense = tfidf_matrix.toarray()
    avg_tfidf = tfidf_dense.mean(axis=1)
    nonzero_counts = (tfidf_dense > 0).sum(axis=1)
    return avg_tfidf, nonzero_counts


# Analyse Distribution
def analyze_feature_distributions(X_val_numerical, X_test_numerical, X_val_tfidf, X_test_tfidf):
    features_to_compare = list(X_val_numerical.columns)

    for feature in features_to_compare:
        plot_feature_distributions(X_val_numerical, X_test_numerical, feature)

    val_avg, val_nonzero = tfidf_statistics(X_val_tfidf)
    test_avg, test_nonzero = tfidf_statistics(X_test_tfidf)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(val_avg, kde=True, color='blue', label='Validation', stat="density", bins=30, alpha=0.6)
    sns.histplot(test_avg, kde=True, color='red', label='Test', stat="density", bins=30, alpha=0.6)
    plt.title("Average TF-IDF per Document")
    plt.xlabel("Average TF-IDF")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(val_nonzero, kde=True, color='blue', label='Validation', stat="density", bins=30, alpha=0.6)
    sns.histplot(test_nonzero, kde=True, color='red', label='Test', stat="density", bins=30, alpha=0.6)
    plt.title("Nonzero TF-IDF Count per Document")
    plt.xlabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

    for feature in features_to_compare:
        ks_test(X_val_numerical, X_test_numerical, feature)

    stat, p = ks_2samp(val_avg, test_avg)
    print(f"KS test for Average TF-IDF:\nStatistic: {stat:.4f}, p-value: {p:.4f}")

    stat, p = ks_2samp(val_nonzero, test_nonzero)
    print(f"KS test for Nonzero TF-IDF Count:\nStatistic: {stat:.4f}, p-value: {p:.4f}")


# KS Test
def ks_test(val_data, test_data, feature):
    statistic, p_value = ks_2samp(val_data[feature].dropna(), test_data[feature].dropna())
    print(f"KS test for {feature}:\nStatistic: {statistic:.4f}, p-value: {p_value:.4f}\n")
