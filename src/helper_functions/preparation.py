import pandas as pd
import textstat
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


################################################################
# COMMON
################################################################

# Read dataset
def read_dataset(file_path):
    return pd.read_csv(file_path)


# Process missing values
def process_text_columns(df):
    df['subject'] = df['subject'].fillna('[NO_SUBJECT]').astype(str)
    df['body'] = df['body'].fillna('[NO_BODY]').astype(str)
    return df


# Clean text 
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    substitutions = [
        (r'https?://\S+|www\.\S+', '[URL]'),                                    # Replace URLs
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),     # Replace emails
        (r'-{2,}', ' '), (r'!{2,}', '!'), (r'\?{2,}', '?'),                     # Remove repeated punctuation
        (r'[_+*]{2,}', ' '), (r'[=+]{3,}', ' '), (r'[~]{3,}', ' '),
        (r'[#]{3,}', ' '), (r'[<]{3,}', ' '), (r'[>]{3,}', ' ')
    ]
    for pattern, repl in substitutions:
        text = re.sub(pattern, repl, text)
    return text.strip()


# Combine subject and body 
def combine_text_fields(df):
    df['subject'] = df['subject'].apply(clean_text)
    df['body'] = df['body'].apply(clean_text)
    df['text'] = df['subject'] + " [SEP] " + df['body']
    return df


# Full preprocessing
def prepare_bert_data(df):
    df = process_text_columns(df)
    df = combine_text_fields(df)
    return df[['text', 'label']]


################################################################
# META
################################################################

def correlation_analysis(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    correlated_features = {col: upper[col][upper[col] > threshold].index.tolist() for col in to_drop}
    remaining_features = [col for col in df.columns if col not in to_drop]

    print("\n**Feature Correlation Report**")
    print(f"- Correlation Threshold: {threshold}")
    print(f"- Features Removed: {len(to_drop)}")
    print(f"- Remaining Features: {len(remaining_features)}")

    for key, value in correlated_features.items():
        print(f"- {key} is highly correlated with: {value}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    return remaining_features, correlated_features


def feature_correlation(df, feature_columns):
    remaining_features, correlated_report = correlation_analysis(df[feature_columns])
    return remaining_features


def extract_meta_features(df):
    df['subject_length'] = df['subject'].apply(len)
    df['body_length'] = df['body'].apply(len)
    df['digit_count'] = df['body'].apply(lambda x: len(re.findall(r'\d', x)))
    df['url_count'] = df['body'].apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    df['uppercase_word_count'] = df['body'].apply(lambda x: len([word for word in x.split() if word.isupper()]))
    df['exclamation_mark_count'] = df['body'].apply(lambda x: x.count('!'))
    df['question_mark_count'] = df['body'].apply(lambda x: x.count('?'))
    df['html_tag_count'] = df['body'].apply(lambda x: len(re.findall(r'<[^>]+>', x)))
    df['repeated_char_count'] = df['body'].apply(lambda x: len(re.findall(r'(.)\1{2,}', x)))
    df['email_address_count'] = df['body'].apply(lambda x: len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', x)))
    df['readability_score'] = df['body'].apply(lambda x: textstat.flesch_reading_ease(x))
    df['special_char_count'] = df['body'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    df['word_count'] = df['body'].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['body'].apply(lambda x: sum(len(word) for word in x.split()) / max(len(x.split()), 1))
    return df


def process_text_columns(df):
    df['subject'] = df['subject'].fillna('').astype(str)
    df['body'] = df['body'].fillna('').astype(str)
    return df


def preprocess_and_save_meta_data(train_file, test_file, train_save_path, test_save_path, features_to_remove=None):
    feature_columns = [
        'subject_length', 'body_length', 'special_char_count', 'digit_count', 'url_count',
        'uppercase_word_count', 'exclamation_mark_count', 'question_mark_count',
        'word_count', 'avg_word_length', 'html_tag_count', 'repeated_char_count',
        'email_address_count', 'readability_score'
    ]

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train = process_text_columns(df_train)
    df_test = process_text_columns(df_test)

    df_train = extract_meta_features(df_train)
    df_test = extract_meta_features(df_test)

    print("\n NaN counts in Train:")
    print(df_train.isna().sum())

    print("\nNaN counts in Test:")
    print(df_test.isna().sum())

    if features_to_remove:
        # Remove specified features
        df_train = df_train.drop(columns=[col for col in features_to_remove if col in df_train.columns])
        df_test = df_test.drop(columns=[col for col in features_to_remove if col in df_test.columns])
        remaining_features = [col for col in feature_columns if col not in features_to_remove]
    else:
        # Correlation-based feature removal
        remaining_features = feature_correlation(df_train, feature_columns)

    df_train = df_train[remaining_features + ['label', 'body']]
    df_test = df_test[remaining_features + ['label', 'body']]

    df_train.to_csv(train_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)


################################################################
# BERT
################################################################

def calculate_clean_reduction(original, processed):
    original_length = len(original)
    processed_length = len(processed)
    if original_length == 0:
        return 0
    reduction_percentage = (original_length - processed_length) / original_length * 100
    return reduction_percentage


def sample_preprocessed_bert_data(input_file, n=1):
    pd.set_option('display.max_colwidth', None)
    
    df = read_dataset(input_file)
    df_sample = df.sample(n).copy()
    
    # Original
    df_sample['full_text_original'] = df_sample['subject'].astype(str) + " " + df_sample['body'].astype(str)
    df_sample['word_count_original'] = df_sample['full_text_original'].apply(lambda x: len(x.split()))
    df_sample['char_count_original'] = df_sample['full_text_original'].apply(lambda x: len(x))
    
    print("Original Data:")
    display(df_sample[['subject', 'body', 'label', 'word_count_original', 'char_count_original']])
    
    # Processed
    df_processed = prepare_bert_data(df_sample).copy()
    df_processed['word_count_processed'] = df_processed['text'].apply(lambda x: len(x.split()))
    df_processed['char_count_processed'] = df_processed['text'].apply(lambda x: len(x))
    
    # Reduction in percentage
    df_processed['char_reduction_%'] = df_sample.apply(
        lambda row: calculate_clean_reduction(
            row['full_text_original'], 
            prepare_bert_data(pd.DataFrame([row]))['text'].iloc[0]),
        axis=1
    )
    
    print("Processed Data:")
    display(df_processed[['text', 'label', 'word_count_processed', 'char_count_processed', 'char_reduction_%']])


def prepare_and_save_bert_data(input_file, output_file):
    # Load data
    df = read_dataset(input_file)
    
    # Original
    df['full_text_original'] = df['subject'].astype(str) + " " + df['body'].astype(str)
    df['char_count_original'] = df['full_text_original'].apply(lambda x: len(x))
    
    # Process 
    df_processed = prepare_bert_data(df).copy()
    df_processed['char_count_processed'] = df_processed['text'].apply(lambda x: len(x))
    
    # Reduction
    total_orig_chars = df['char_count_original'].sum()
    total_proc_chars = df_processed['char_count_processed'].sum()
    overall_reduction = (total_orig_chars - total_proc_chars) / total_orig_chars * 100 if total_orig_chars > 0 else 0
    
    # Avg. reduction
    df['reduction_%'] = df.apply(
        lambda row: calculate_clean_reduction(
            row['full_text_original'], 
            prepare_bert_data(pd.DataFrame([row]))['text'].iloc[0]
        ),
        axis=1
    )
    avg_reduction = df['reduction_%'].mean()
    
    print(f"Char Count (Original): {total_orig_chars}")
    print(f"Char Count (Processed): {total_proc_chars}")
    print(f"Char Reduction (Overall): {overall_reduction:.2f}%")
    print(f"Char Reduction (Avg. per row): {avg_reduction:.2f}%")
    
    # Save 
    df_processed.to_csv(output_file, index=False)
    print(f"Processed saved to {output_file}")


################################################################

