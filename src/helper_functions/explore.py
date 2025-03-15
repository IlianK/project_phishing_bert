import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from langdetect import detect
from wordcloud import WordCloud
from helper_functions.assemble_collect import detect_language


##############################################
# Data Loading 
##############################################

def read_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {file_path}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def show_dataset_info(df, file_path):
    dataset_name = os.path.basename(file_path)
    shape = df.shape
    class_distribution = df['label'].value_counts()
    
    print("\nDataset Information:")
    print("---------------------")
    print(f"Dataset Name   : {dataset_name}")
    print(f"Shape          : {shape[0]} rows, {shape[1]} columns")
    print(f"Columns        : {', '.join(df.columns)}")
    print("\nClass Distribution:")
    print("--------------------")
    print(class_distribution)
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', data=df)
    ax.bar_label(ax.containers[0])
    plt.title(f"Class Distribution in {dataset_name}")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()


def show_nans(df):
    missing_data = df.isnull().sum().to_frame('Missing Values')
    print("\nMissing Values:")
    print("----------------")
    print(missing_data)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()


##############################################
# Text Length Analysis
##############################################

def compute_text_lengths(df):
    df = df.copy()
    df['subject_length'] = df['subject'].apply(lambda x: len(str(x).split()))
    df['body_length'] = df['body'].apply(lambda x: len(str(x).split()))
    return df


def show_text_length_boxplots(df, ignore_nans=False):
    temp_df = df.copy()
    if ignore_nans:
        initial_row_count = len(temp_df)
        temp_df = temp_df[temp_df['subject'].notna() & temp_df['body'].notna()]
        temp_df = temp_df[temp_df['subject'].str.strip() != '']
        temp_df = temp_df[temp_df['body'].str.strip() != '']
        final_row_count = len(temp_df)
        print(f"Row count ignoring NaNs: {final_row_count}/{initial_row_count} ({final_row_count/initial_row_count*100:.2f}%)")
    
    temp_df = compute_text_lengths(temp_df)
    
    print("\nText Lengths (Subject & Body):")
    print("-------------------------------")
    print(f"Max subject length: {temp_df['subject_length'].max()}")
    print(f"Min subject length: {temp_df['subject_length'].min()}")
    print(f"Avg subject length: {temp_df['subject_length'].mean():.2f}")
    print(f"Max body length: {temp_df['body_length'].max()}")
    print(f"Min body length: {temp_df['body_length'].min()}")
    print(f"Avg body length: {temp_df['body_length'].mean():.2f}")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=temp_df['subject_length'])
    plt.title("Subject Length Distribution")
    plt.xlabel("Subject Length")
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=temp_df['body_length'])
    plt.title("Body Length Distribution")
    plt.xlabel("Body Length")
    plt.show()


def show_text_length_bins(df):
    bins = [0, 50, 100, 200, 300, 400, 512, 1024, 2048]
    bin_labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-512', '513-1024', '>1024']
    
    df = df.copy()
    df['body_length'] = df['body'].apply(lambda x: len(str(x).split()))
    df['body_length_range'] = pd.cut(df['body_length'], bins=bins, labels=bin_labels, right=False)
    
    plt.figure(figsize=(10, 6))
    body_length_distribution = df['body_length_range'].value_counts().sort_index()
    sns.barplot(x=body_length_distribution.index, y=body_length_distribution.values, 
                hue=body_length_distribution.index, palette='RdYlBu', dodge=False, legend=False)
    plt.title("Body Text Length Ranges")
    plt.xlabel("Length Range")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()


def show_text_length_boxplots_multi(dfs, dataset_names):
    num_datasets = len(dfs)
    cols = 4
    rows = (num_datasets // cols) + (num_datasets % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()
    
    for i, (df, name) in enumerate(zip(dfs, dataset_names)):
        if 'subject' not in df.columns or 'body' not in df.columns:
            continue
        df = compute_text_lengths(df)
        sns.boxplot(x=df['body_length'], ax=axes[i])
        axes[i].set_title(f"{name} - Body Length")
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


def show_text_length_bins_multi(dfs, dataset_names):
    num_datasets = len(dfs)
    cols = 4
    rows = (num_datasets // cols) + (num_datasets % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()
    
    bins = [0, 50, 100, 200, 300, 400, 512, 1024, 2048]
    bin_labels = ['0-50', '51-100', '101-200', '201-300', '301-400', '401-512', '513-1024', '>1024']
    
    for i, (df, name) in enumerate(zip(dfs, dataset_names)):
        if 'body' not in df.columns:
            continue
        df['body_length'] = df['body'].apply(lambda x: len(str(x).split()))
        df['body_length_range'] = pd.cut(df['body_length'], bins=bins, labels=bin_labels, right=False)
        dist = df['body_length_range'].value_counts().sort_index()
        sns.barplot(x=dist.index, y=dist.values, hue=dist.index, palette='RdYlBu', dodge=False, legend=False, ax=axes[i])
        axes[i].set_title(f"{name} - Body Length Ranges")
        axes[i].set_xlabel("Length Range")
        axes[i].set_ylabel("Frequency")
        axes[i].set_xticks(range(len(bin_labels)))
        axes[i].set_xticklabels(bin_labels, rotation=45)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


##############################################
# Wordcloud and Language Analysis Functions
##############################################

def generate_wordclouds(df, label=None):
    if label is not None:
        df = df[df['label'] == label]
    
    text_body = " ".join(str(x) for x in df['body'].dropna() if str(x).strip() != "")
    text_subject = " ".join(str(x) for x in df['subject'].dropna() if str(x).strip() != "")
    
    print("\nWordclouds:")
    print("------------------------------------------")
    
    body_wc = WordCloud(width=800, height=400, background_color='white').generate(text_body)
    print(f"Wordcloud for Body (Label {label}):")
    print(f"Total words in body: {len(text_body.split())}")
    plt.figure(figsize=(10, 6))
    plt.imshow(body_wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Wordcloud for Body (Label {label})")
    plt.show()
    
    subject_wc = WordCloud(width=800, height=400, background_color='white').generate(text_subject)
    print(f"Wordcloud for Subject (Label {label}):")
    print(f"Total words in subject: {len(text_subject.split())}")
    plt.figure(figsize=(10, 6))
    plt.imshow(subject_wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Wordcloud for Subject (Label {label})")
    plt.show()


def analyze_language_distribution(df):
    def inner_detect(text):
        try:
            return detect(str(text))
        except Exception:
            return "unknown"
    
    df['body_language'] = df['body'].apply(inner_detect)
    df['subject_language'] = df['subject'].apply(inner_detect)
    
    body_counts = df['body_language'].value_counts().head(10)
    subject_counts = df['subject_language'].value_counts().head(10)
    
    distribution = pd.DataFrame({
        'Body Language': body_counts.astype(int),
        'Subject Language': subject_counts.astype(int)
    }).fillna(0)
    
    print("\nLanguage Distribution (Subject and Body):")
    print("-------------------------------------------")
    print(distribution)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=body_counts.index, y=body_counts.values, hue=body_counts.index, palette='viridis', dodge=False, legend=False)
    plt.title("Language Distribution in Body")
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=subject_counts.index, y=subject_counts.values, hue=subject_counts.index, palette='viridis', dodge=False, legend=False)
    plt.title("Language Distribution in Subject")
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


##############################################
# Sender and Recipient 
##############################################

def count_mails_with_urls(df):
    url_pattern = r'http[s]?://(?:[a-zA-Z0-9$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df['contains_url'] = df['body'].apply(lambda text: bool(re.search(url_pattern, str(text))))
    num_with = df['contains_url'].sum()
    
    print("\nMails with URL:")
    print("---------------------------------")
    url_summary = pd.DataFrame({
        'Label': ["Mails with URLs", "Mails without URLs"],
        'Count': [num_with, len(df) - num_with]
    })
    print(url_summary)
    
    plt.figure(figsize=(6, 6))
    sns.barplot(x="Label", y="Count", data=url_summary, palette="Set2", dodge=False, legend=False)
    plt.title("Mails with and without URLs")
    plt.ylabel("Count")
    plt.show()


def show_most_common_sender_recipient(df):
    if 'sender' not in df.columns or 'receiver' not in df.columns:
        print("Columns 'sender' or 'receiver' are missing in the dataframe.")
        return
    
    most_common_sender = df['sender'].mode()[0]
    most_common_receiver = df['receiver'].mode()[0]
    
    print("\nMost Common Sender and Receiver:")
    print("---------------------------------")
    print(f"Most common sender: {most_common_sender}")
    print(f"Most common receiver: {most_common_receiver}")
    
    sender_counts = df['sender'].value_counts().head(10)
    receiver_counts = df['receiver'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sender_counts.index, y=sender_counts.values, color='skyblue')
    plt.title("Top 10 Most Common Senders")
    plt.xlabel("Sender")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=receiver_counts.index, y=receiver_counts.values, color='lightgreen')
    plt.title("Top 10 Most Common Receivers")
    plt.xlabel("Receiver")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()


##############################################
# Dataset Summarize
##############################################

def summarize_datasets(dfs, dataset_names):
    summary = []
    for df, name in zip(dfs, dataset_names):
        if df is not None:
            shape = df.shape
            df['subject'] = df['subject'].apply(lambda x: np.nan if str(x).strip() == '' else x)
            df['body'] = df['body'].apply(lambda x: np.nan if str(x).strip() == '' else x)
            
            missing_values = df[['subject', 'body']].isnull().sum().to_dict()
            class_distribution = df['label'].value_counts().to_dict()
            
            df = compute_text_lengths(df)
            df['text_length'] = df['subject_length'] + df['body_length']
            avg_text_length = df['text_length'].mean()
            
            df['combined_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
            df['language'] = df['combined_text'].apply(detect_language)
            top_languages = dict(Counter(df['language']).most_common(3))
            
            summary.append({
                "Dataset": name,
                "Shape": shape,
                "NaNs": missing_values,
                "Class Distribution": class_distribution,
                "Avg. Text Length": avg_text_length,
                "Top Languages": top_languages
            })
    
    return pd.DataFrame(summary)
