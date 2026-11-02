"""UCI HAR Dataset ingestion, extraction, and preprocessing module."""

import os
import zipfile
import urllib.request
import numpy as np
import yaml

def get_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def download_and_extract_uci_har(data_dir="data"):
    target_dir = os.path.join(data_dir, "uci_har")
    extracted_marker = os.path.join(target_dir, "UCI HAR Dataset")
    if os.path.exists(extracted_marker):
        return extracted_marker

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "uci_har.zip")
    
    urls = [
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "https://raw.githubusercontent.com/jgb/UCI-HAR-Dataset/master/UCI%20HAR%20Dataset.zip"
    ]
    
    downloaded = False
    for url in urls:
        try:
            print(f"Downloading UCI HAR from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            
    if not downloaded:
        raise RuntimeError("Failed to download UCI HAR Dataset from all sources.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    # If the zip extracted an inner zip (UCI official repo has nested zip)
    nested_zip = os.path.join(target_dir, "UCI HAR Dataset.zip")
    if os.path.exists(nested_zip):
        print(f"Extracting nested zip {nested_zip}...")
        with zipfile.ZipFile(nested_zip, 'r') as nested_ref:
            nested_ref.extractall(target_dir)

    return extracted_marker

def load_inertial_signals(dataset_dir, split="train"):
    """
    Loads 6 raw inertial channels:
    total_acc_x, total_acc_y, total_acc_z, body_gyro_x, body_gyro_y, body_gyro_z.
    Each file has shape (N_windows, 128).
    Returns (N_windows, 128, 6).
    """
    signals_dir = os.path.join(dataset_dir, split, "Inertial Signals")
    channel_files = [
        f"total_acc_x_{split}.txt",
        f"total_acc_y_{split}.txt",
        f"total_acc_z_{split}.txt",
        f"body_gyro_x_{split}.txt",
        f"body_gyro_y_{split}.txt",
        f"body_gyro_z_{split}.txt",
    ]
    
    channels = []
    for f in channel_files:
        f_path = os.path.join(signals_dir, f)
        data = np.loadtxt(f_path)
        channels.append(data)
        
    # Stack along channels axis: (N, 128, 6)
    stacked = np.stack(channels, axis=-1)
    
    # Load labels: (N,) 1-indexed in original dataset, convert to 0-indexed
    y_path = os.path.join(dataset_dir, split, f"y_{split}.txt")
    labels = np.loadtxt(y_path, dtype=int) - 1
    
    # Load subject IDs: (N,)
    sub_path = os.path.join(dataset_dir, split, f"subject_{split}.txt")
    subjects = np.loadtxt(sub_path, dtype=int)
    
    return stacked, labels, subjects

def load_har_splits(data_dir="data"):
    dataset_dir = download_and_extract_uci_har(data_dir)
    config = get_config()
    val_subjects = set(config["har"]["val_subjects"])
    
    train_x, train_y, train_sub = load_inertial_signals(dataset_dir, split="train")
    test_x, test_y, test_sub = load_inertial_signals(dataset_dir, split="test")
    
    val_mask = np.isin(train_sub, list(val_subjects))
    train_mask = ~val_mask
    
    X_val, y_val, sub_val = train_x[val_mask], train_y[val_mask], train_sub[val_mask]
    X_train, y_train, sub_train = train_x[train_mask], train_y[train_mask], train_sub[train_mask]
    X_test, y_test, sub_test = test_x, test_y, test_sub
    
    return {
        "train": (X_train, y_train, sub_train),
        "val": (X_val, y_val, sub_val),
        "test": (X_test, y_test, sub_test),
        "classes": config["har"]["classes"]
    }

if __name__ == "__main__":
    splits = load_har_splits()
    print("HAR Train:", splits["train"][0].shape, "labels:", np.bincount(splits["train"][1]))
    print("HAR Val:", splits["val"][0].shape, "labels:", np.bincount(splits["val"][1]))
    print("HAR Test:", splits["test"][0].shape, "labels:", np.bincount(splits["test"][1]))
