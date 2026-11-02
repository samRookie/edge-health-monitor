"""Main training pipeline for HAR 1D-CNN and KWS DS-CNN.
Produces baseline models, overfit/underfit comparisons, and classical/unsupervised baselines.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import random
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

from src.models import (
    build_har_model, build_har_underfit_model, build_har_overfit_model,
    build_kws_model
)
from src.data_har import load_har_splits
from src.data_kws import load_kws_splits
from src.evaluate import evaluate_classifier

# Styling settings for matplotlib
PAGE_BG = '#ECEAE4'
PANEL_BG = '#F7F6F2'
BORDER_COLOR = '#CBC8BF'
TEXT_PRIMARY = '#1E2328'
TEXT_SECONDARY = '#58626B'
ACCENT_COLOR = '#1F6F78'
OCHRE_COLOR = '#A06F18'

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_and_save_curves(history, model_name, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Save CSV
    csv_path = os.path.join(results_dir, f"{model_name}_training_curves.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
        for ep in epochs:
            idx = ep - 1
            f.write(f"{ep},{history.history['loss'][idx]:.6f},{history.history['accuracy'][idx]:.6f},"
                    f"{history.history['val_loss'][idx]:.6f},{history.history['val_accuracy'][idx]:.6f}\n")
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=PAGE_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=BORDER_COLOR, linewidth=1.0)
        
    # Accuracy plot
    ax1.plot(epochs, history.history['accuracy'], color=ACCENT_COLOR, linewidth=1.5, label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], color=OCHRE_COLOR, linewidth=1.5, label='Val Accuracy')
    ax1.set_title(f"{model_name.upper()} Accuracy", fontsize=11, color=TEXT_PRIMARY)
    ax1.set_xlabel("Epoch", fontsize=10, color=TEXT_SECONDARY)
    ax1.set_ylabel("Accuracy", fontsize=10, color=TEXT_SECONDARY)
    ax1.legend(loc='lower right', framealpha=1.0, facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    
    # Loss plot
    ax2.plot(epochs, history.history['loss'], color=ACCENT_COLOR, linewidth=1.5, label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], color=OCHRE_COLOR, linewidth=1.5, label='Val Loss')
    ax2.set_title(f"{model_name.upper()} Loss", fontsize=11, color=TEXT_PRIMARY)
    ax2.set_xlabel("Epoch", fontsize=10, color=TEXT_SECONDARY)
    ax2.set_ylabel("Loss", fontsize=10, color=TEXT_SECONDARY)
    ax2.legend(loc='upper right', framealpha=1.0, facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    
    plt.tight_layout()
    png_path = os.path.join(results_dir, f"{model_name}_training_curves.png")
    fig.savefig(png_path, dpi=150, facecolor=PAGE_BG)
    plt.close(fig)
    print(f"Saved curves to {png_path} and CSV to {csv_path}")

def train_har(config, quick=False):
    print("\n=== Training HAR 1D-CNN Baseline ===")
    har_splits = load_har_splits()
    X_train, y_train, _ = har_splits["train"]
    X_val, y_val, _ = har_splits["val"]
    X_test, y_test, _ = har_splits["test"]
    classes = har_splits["classes"]
    
    epochs = config["training"]["quick_epochs"] if quick else config["training"]["har_epochs"]
    batch_size = config["training"]["har_batch_size"]
    
    model = build_har_model(input_shape=(128, 6), num_classes=len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config["training"]["patience"], restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    plot_and_save_curves(history, "har_baseline")
    evaluate_classifier(model, X_test, y_test, classes, "har_baseline")
    
    os.makedirs("models", exist_ok=True)
    model.save("models/har_baseline.keras")
    print("Saved models/har_baseline.keras")
    return model, har_splits

def train_kws(config, quick=False):
    print("\n=== Training KWS DS-CNN Baseline ===")
    kws_splits = load_kws_splits(quick=quick)
    X_train, y_train = kws_splits["train"]
    X_val, y_val = kws_splits["val"]
    X_test, y_test = kws_splits["test"]
    classes = kws_splits["classes"]
    
    # Expand dims for channel axis: (N, 98, 40, 1)
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)
    
    epochs = config["training"]["quick_epochs"] if quick else config["training"]["kws_epochs"]
    batch_size = config["training"]["kws_batch_size"]
    
    model = build_kws_model(input_shape=(98, 40, 1), num_classes=len(classes))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config["training"]["patience"], restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    plot_and_save_curves(history, "kws_baseline")
    evaluate_classifier(model, X_test, y_test, classes, "kws_baseline")
    
    os.makedirs("models", exist_ok=True)
    model.save("models/kws_baseline.keras")
    print("Saved models/kws_baseline.keras")
    return model, kws_splits

def run_overfit_underfit_experiment(har_splits, quick=False):
    print("\n=== Running Overfit / Underfit Demonstration ===")
    X_train, y_train, _ = har_splits["train"]
    X_val, y_val, _ = har_splits["val"]
    
    epochs = 6 if quick else 12
    
    # 1. Underfit Model
    under_model = build_har_underfit_model(input_shape=(128, 6), num_classes=6)
    under_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    under_hist = under_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=128, verbose=0)
    
    # 2. Overfit Model
    over_model = build_har_overfit_model(input_shape=(128, 6), num_classes=6)
    over_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    over_hist = over_model.fit(X_train[:1000], y_train[:1000], validation_data=(X_val, y_val), epochs=epochs, batch_size=32, verbose=0)
    
    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=PAGE_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=BORDER_COLOR)
        
    ep_range = range(1, epochs + 1)
    ax1.plot(ep_range, under_hist.history['accuracy'], color=ACCENT_COLOR, label='Train Acc')
    ax1.plot(ep_range, under_hist.history['val_accuracy'], color=OCHRE_COLOR, linestyle='--', label='Val Acc')
    ax1.set_title("Underfitting Demo (High Bias, Low Capacity)", fontsize=10, color=TEXT_PRIMARY)
    ax1.set_xlabel("Epoch", fontsize=9, color=TEXT_SECONDARY)
    ax1.set_ylabel("Accuracy", fontsize=9, color=TEXT_SECONDARY)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    
    ax2.plot(ep_range, over_hist.history['accuracy'], color=ACCENT_COLOR, label='Train Acc')
    ax2.plot(ep_range, over_hist.history['val_accuracy'], color=OCHRE_COLOR, linestyle='--', label='Val Acc')
    ax2.set_title("Overfitting Demo (High Variance, No Reg)", fontsize=10, color=TEXT_PRIMARY)
    ax2.set_xlabel("Epoch", fontsize=9, color=TEXT_SECONDARY)
    ax2.set_ylabel("Accuracy", fontsize=9, color=TEXT_SECONDARY)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    
    plt.tight_layout()
    fig.savefig("results/overfit_underfit_curves.png", dpi=150, facecolor=PAGE_BG)
    plt.close(fig)
    
    res = {
        "underfit_params": int(under_model.count_params()),
        "underfit_final_train_acc": float(under_hist.history['accuracy'][-1]),
        "underfit_final_val_acc": float(under_hist.history['val_accuracy'][-1]),
        "overfit_params": int(over_model.count_params()),
        "overfit_final_train_acc": float(over_hist.history['accuracy'][-1]),
        "overfit_final_val_acc": float(over_hist.history['val_accuracy'][-1])
    }
    with open("results/overfit_underfit.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results/overfit_underfit_curves.png and results/overfit_underfit.json")

def run_classical_and_unsupervised_comparison(har_splits):
    print("\n=== Running Classical ML & Unsupervised Comparison ===")
    X_train, y_train, _ = har_splits["train"]
    X_test, y_test, _ = har_splits["test"]
    
    # Feature extraction: mean, std, min, max, energy for each of 6 channels -> 30 features
    def extract_summary_features(X):
        means = np.mean(X, axis=1)
        stds = np.std(X, axis=1)
        mins = np.min(X, axis=1)
        maxs = np.max(X, axis=1)
        energy = np.mean(X ** 2, axis=1)
        return np.concatenate([means, stds, mins, maxs, energy], axis=-1)
        
    feat_train = extract_summary_features(X_train)
    feat_test = extract_summary_features(X_test)
    
    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(feat_train, y_train)
    rf_pred = rf.predict(feat_test)
    rf_acc = float(accuracy_score(y_test, rf_pred))
    print(f"Random Forest Test Accuracy: {rf_acc*100:.2f}%")
    
    # 2. Unsupervised K-Means Clustering (k=6)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(feat_test)
    
    ari = float(adjusted_rand_score(y_test, cluster_labels))
    nmi = float(normalized_mutual_info_score(y_test, cluster_labels))
    print(f"K-Means Adjusted Rand Index: {ari:.4f}, NMI: {nmi:.4f}")
    
    comp_results = {
        "random_forest": {
            "test_accuracy": rf_acc,
            "feature_dim": feat_train.shape[1],
            "n_estimators": 50
        },
        "kmeans_unsupervised": {
            "n_clusters": 6,
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi
        }
    }
    with open("results/classical_comparison.json", "w") as f:
        json.dump(comp_results, f, indent=2)
    print("Saved results/classical_comparison.json")

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    set_seeds(config["project"]["random_seed"])
    
    quick = "--quick" in sys.argv
    
    har_model, har_splits = train_har(config, quick=quick)
    run_overfit_underfit_experiment(har_splits, quick=quick)
    run_classical_and_unsupervised_comparison(har_splits)
    
    train_kws(config, quick=quick)
    print("\n=== Phase 2 Training Completed Successfully ===")

if __name__ == "__main__":
    main()
