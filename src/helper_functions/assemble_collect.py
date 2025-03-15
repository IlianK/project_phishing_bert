##############################################
# Imports common to all functions
##############################################
import os
import sys
import re
import csv
import socket
import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from functools import lru_cache
from deep_translator import GoogleTranslator
from tqdm.notebook import tqdm
from langdetect import detect, DetectorFactory
from langdetect import lang_detect_exception
from bs4 import BeautifulSoup
import imaplib
import email
from email import policy
from email.header import decode_header
from email.parser import BytesParser
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

DetectorFactory.seed = 42

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper_functions.path_resolver import DynamicPathResolver



##############################################
#  data_assembly.ipynb
##############################################
translator = GoogleTranslator(source="en", target="de")
tqdm.pandas()


def detect_language(text):
    try:
        return detect(str(text))
    except Exception:
        return "unknown"


def translate_to_de(text):
    try:
        return translator.translate(text)
    except Exception:
        return text


def build_balanced_set(base_file, out_file, total_size, language="en"):
    per_class = total_size // 2
    df = pd.read_csv(base_file)
    df = df[df["label"].isin([0, 1]) & (df["language"] == language)]
    legit_samp = df[df["label"] == 0].sample(n=per_class, random_state=42)
    phish_samp = df[df["label"] == 1].sample(n=per_class, random_state=42)
    balanced = pd.concat([legit_samp, phish_samp], ignore_index=True)
    balanced.fillna({"subject": "", "body": ""}, inplace=True)
    balanced.to_csv(out_file, index=False)


def translate_entire_dataset(eng_file, out_file):
    df = pd.read_csv(eng_file)
    print(f"Translating dataset: {eng_file}")
    df['subject'] = [translate_to_de(text) for text in tqdm(df['subject'], desc="Translating subjects")]
    df['body'] = [translate_to_de(text) for text in tqdm(df['body'], desc="Translating bodies")]
    df['language'] = "de"
    df.to_csv(out_file, index=False)
    print(f"Translated dataset saved: {out_file}")


def sample_from_existing_datasets(eng_file, de_file, out_file, english_size, german_size):
    for file, desc in zip([eng_file, de_file], ["English", "German"]):
        tqdm.write(f"Processing {desc} file: {file}")
    df_eng = pd.read_csv(eng_file)
    df_de = pd.read_csv(de_file)
    eng_phishing = df_eng[df_eng["label"] == 1]
    eng_legit = df_eng[df_eng["label"] == 0]
    de_phishing = df_de[df_de["label"] == 1]
    de_legit = df_de[df_de["label"] == 0]
    eng_phishing_sample = eng_phishing.iloc[:english_size // 2]  
    eng_legit_sample = eng_legit.iloc[:english_size // 2]
    de_phishing_sample = de_phishing.iloc[-german_size // 2:]  
    de_legit_sample = de_legit.iloc[-german_size // 2:]
    combined = pd.concat([eng_phishing_sample, eng_legit_sample, de_phishing_sample, de_legit_sample], ignore_index=True)
    combined.to_csv(out_file, index=False)
    print(f"Built multilingual dataset: {out_file}")
    return combined


def verify(df, name):
    print(f"\n{name}, Rows: {len(df)}")
    print("----------------------------------------")
    class_counts = df["label"].value_counts().to_dict()
    lang_counts = df["language"].value_counts().to_dict()
    grouped = df.groupby(["label", "language"]).size().to_dict()
    print(f"Class Distribution: {class_counts}")
    print(f"Language Distribution: {lang_counts}")
    print(f"Detailed (Class, Language) Distribution: {grouped}")
    print("----------------------------------------")


def add_lang_and_create_base(file_paths, out_file):
    dfs = []
    for path in tqdm(file_paths, desc="Preprocessing and combining files", unit="file"):
        df = pd.read_csv(path)
        df["language"] = df["subject"].progress_apply(detect_language)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_file, index=False)
    print(f"Combined preprocessed file saved: {out_file}")
    return out_file


def balance_and_split_dataset(base_file, train_file, test_file, test_size=0.2, random_state=42):
    df = pd.read_csv(base_file)
    df = df[df["label"].isin([0, 1])]
    class_counts = df["label"].value_counts()
    print("Original dataset distribution:\n", class_counts)
    min_class_size = min(class_counts)
    legit_sample = df[df["label"] == 0]
    phish_sample = df[df["label"] == 1].sample(n=min_class_size, random_state=random_state)
    df_balanced = pd.concat([legit_sample, phish_sample], ignore_index=True)
    print("Balanced dataset distribution:\n", df_balanced["label"].value_counts())
    train_df, test_df = train_test_split(
        df_balanced, 
        test_size=test_size, 
        stratify=df_balanced["label"], 
        random_state=random_state
    )
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Train set saved: {train_file}, Size: {len(train_df)}")
    print(f"Test set saved: {test_file}, Size: {len(test_df)}")


def check_duplicates(csv_path1, csv_path2):
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    combined_df = pd.concat([df1, df2])
    duplicates = combined_df.duplicated(keep=False)
    num_duplicates = duplicates.sum()

    print(f"Number of duplicate rows across both CSVs: {num_duplicates}")
    return num_duplicates


def sample_non_overlapping(train_curated_paths, used_train_file, output_file, sample_size=4000):
    df_list = [pd.read_csv(path) for path in train_curated_paths]
    full_data = pd.concat(df_list, ignore_index=True)
    used_data = pd.read_csv(used_train_file)

    full_data = full_data.drop_duplicates(subset=["subject", "body"])
    used_data = used_data.drop_duplicates(subset=["subject", "body"])

    full_data["language"] = full_data["body"].apply(detect_language)
    full_data = full_data[full_data["language"] == "en"]

    merged = full_data.merge(
        used_data,
        on=["subject", "body"],
        how="left",
        indicator=True,
        suffixes=("", "_drop")
    )

    new_data = (
        merged
        .query("_merge == 'left_only'")
        .drop(columns=["_merge"] + [col for col in merged.columns if col.endswith("_drop")])
    )

    per_class = sample_size // 2
    legit_samples = new_data[new_data["label"] == 0].sample(n=per_class, random_state=42)
    phish_samples = new_data[new_data["label"] == 1].sample(n=per_class, random_state=42)
    balanced_sample = pd.concat([legit_samples, phish_samples], ignore_index=True)
    balanced_sample.to_csv(output_file, index=False)

    class_dist = balanced_sample["label"].value_counts()
    print("\nClass Distribution:\n", class_dist)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_dist.index, y=class_dist.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()

    return balanced_sample


##############################################
# Functions from data_collect.ipynb
##############################################

def extract_email_address(sender):
    email_match = re.search(r'<(.+?)>', sender)
    return email_match.group(1) if email_match else sender


def decode_mime_header(header_value):
    if header_value:
        decoded_parts = decode_header(header_value)
        decoded_text = []
        for part, encoding in decoded_parts:
            try:
                if isinstance(part, bytes):
                    decoded_text.append(part.decode(encoding or "utf-8", errors="replace"))
                else:
                    decoded_text.append(part)
            except LookupError:
                decoded_text.append(part.decode("utf-8", errors="replace"))
        return " ".join(decoded_text)
    return "Unknown"


def fetch_emails(mail, folder, label):
    mail.select(folder)
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    email_data = []
    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                print(f"Raw Subject: {msg['Subject']}")
                print(f"Decoded Subject: {decode_mime_header(msg['Subject'])}")
                subject = decode_mime_header(msg["Subject"]) or "No Subject"
                sender = decode_mime_header(msg["From"])
                sender = extract_email_address(sender)
                date = msg["Date"]
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = part.get("Content-Disposition", "")
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                email_data.append([label, date, sender, subject, body])
    return email_data


def save_to_csv(emails_data, path):
    file_exists = os.path.isfile(path)
    df = pd.DataFrame(emails_data, columns=["label", "date", "sender", "subject", "body"])
    df.to_csv(
        path,
        mode='a',
        header=not file_exists,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
    )
    return df


def clean_html_and_save(df, save_path, filter):
    def clean_html(text):
        return BeautifulSoup(str(text), "html.parser").get_text(separator="\n", strip=True)
    if filter:
        df["body"] = df["body"].apply(clean_html)
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"Cleaned dataset saved as {save_path}")


def label_known_legit(df, legit_senders):
    df.loc[df["sender"].isin(legit_senders), "label"] = 0
    legit_count = df["label"].value_counts().get(0, 0)
    print(f"Auto-labeled {legit_count} emails as LEGIT.")
    return df


def label_known_phish(df, spam_senders):
    df.loc[df["sender"].isin(spam_senders), "label"] = 1
    spam_count = df["label"].value_counts().get(1, 0)
    print(f"Auto-labeled {spam_count} emails as SPAM.")
    return df


def label_by_domain(df, legit_domains, spam_domains):
    df.loc[df["sender"].str.endswith(tuple(legit_domains)), "label"] = 0
    legit_count = df["label"].value_counts().get(0, 0)
    print(f"Auto-labeled {legit_count} emails as LEGIT based on domain endings.")
    df.loc[df["sender"].str.endswith(tuple(spam_domains)), "label"] = 1
    spam_count = df["label"].value_counts().get(1, 0)
    print(f"Auto-labeled {spam_count} emails as SPAM based on domain endings.")
    return df


def auto_label_emails(df, legit_senders, spam_senders, legit_domains, spam_domains, save_path):
    if "label" not in df.columns:
        df["label"] = -1
    df = label_known_legit(df, legit_senders)
    df = label_known_phish(df, spam_senders)
    df = label_by_domain(df, legit_domains, spam_domains)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Auto-labeling complete. Labeled dataset saved at: {save_path}")
    return df


def display_next_top_senders(df, start_index=0, batch_size=100):
    pd.set_option("display.max_colwidth", None)
    sorted_senders_df = df["sender"].value_counts().reset_index()
    sorted_senders_df.columns = ["sender", "count"]
    next_senders_df = sorted_senders_df.iloc[start_index:start_index + batch_size]

    print(f"Total Unique Senders: {len(sorted_senders_df)}")
    print(f"Showing senders {start_index + 1} to {start_index + batch_size}")
    print(next_senders_df["sender"].to_string(index=False))


def label_senders(df, start_index=0, batch_size=1, 
                  exclude_domain="kleinanzeigen.de",
                  known_spam=None,
                  known_legit=None,
                  legit_domains=None,
                  spam_domains=None):
    if known_spam is None:
        known_spam = set()
    if known_legit is None:
        known_legit = set()
    if legit_domains is None:
        legit_domains = set()
    if spam_domains is None:
        spam_domains = set()

    pd.set_option("display.max_colwidth", None)
    sorted_senders_df = df["sender"].value_counts().reset_index()
    sorted_senders_df.columns = ["sender", "Count"]
    sorted_senders_df["domain"] = sorted_senders_df["sender"].str.split("@").str[-1]

    filtered_senders_df = sorted_senders_df[
        (~sorted_senders_df["sender"].str.endswith(exclude_domain)) &  
        (~sorted_senders_df["sender"].isin(known_spam)) &              
        (~sorted_senders_df["sender"].isin(known_legit)) &             
        (~sorted_senders_df["domain"].isin(spam_domains)) &            
        (~sorted_senders_df["domain"].isin(legit_domains))             
    ]

    while start_index < len(filtered_senders_df):
        sender_row = filtered_senders_df.iloc[start_index]
        sender = sender_row["sender"]
        print(f"sender: {sender} ({sender_row['Count']} emails)")
        user_input = input("Label as Phish (1) or Legit (0): ").strip()
        if user_input == "1":
            df.loc[df["sender"] == sender, "label"] = "phish"
            known_spam.add(sender)
            print(f"Labeled {sender} as Phish.")

        elif user_input == "0":
            df.loc[df["sender"] == sender, "label"] = "legit"
            known_legit.add(sender)
            print(f"Labeled {sender} as Legit.")

        else:
            print("Invalid input, try again.")
            continue

        start_index += 1
        print(f"Processed {start_index}/{len(filtered_senders_df)} senders.\n")

    print("All senders processed.")


def concat_and_save(csv_path1, csv_path2, output_dir, output_filename="combined_own.csv"):
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    if 'label' not in combined_df.columns:
        raise ValueError("Column 'label' not found in CSV files.")
    class_distribution = combined_df['label'].value_counts()
    print("\nClass Distribution:")
    print(class_distribution)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined dataset saved at: {output_path}")
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_distribution.index, y=class_distribution.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.show()
    return combined_df


def extract_email_data(email_file):
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1']
    for enc in encodings:
        try:
            with open(email_file, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            sender = msg.get("From", None)
            subject = msg.get("Subject", None)
            to = msg.get("To", None)
            date = msg.get("Date", None)
            email_text = ""
            if msg.is_multipart():
                for part in msg.iter_parts():
                    if part.get_content_type() == 'text/plain':
                        email_text = part.get_payload(decode=True).decode(enc, errors='replace')
                        break
            else:
                if msg.get_content_type() == 'text/plain':
                    email_text = msg.get_payload(decode=True).decode(enc, errors='replace')
            email_text = email_text.replace('\r\n', '\n').replace('\r', '\n')
            return sender, subject, to, date, email_text.strip() if email_text.strip() else None
        except Exception as e:
            print(f"Failed to read {email_file} with encoding {enc}: {e}")
    return None, None, None, None, None


def process_spam_folder(spam_folder_path, output_csv_path):
    email_data = []
    if not os.path.exists(spam_folder_path):
        print(f"Folder not found: {spam_folder_path}")
        return
    for root, dirs, files in os.walk(spam_folder_path):
        for file in files:
            email_file_path = os.path.join(root, file)
            sender, subject, to, date, body = extract_email_data(email_file_path)
            email_data.append([sender if sender else "", 
                               subject if subject else "", 
                               to if to else "", 
                               date if date else "", 
                               body if body else ""])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sender', 'subject', 'receiver', 'date', 'body'])
        writer.writerows(email_data)
    print(f"Data saved in: {output_csv_path}")


def label_spam_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['label'] = 1
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Labeled CSV saved to: {csv_file}")
