"""Evaluation utilities, metric computation, confusion matrix generation, and CSV/JSON export."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Matplotlib styling for Edge Health Monitor calm aesthetic
plt.rcParams['font.sans-serif'] = 'Segoe UI, Arial, sans-serif'
plt.rcParams['axes.edgecolor'] = '#CBC8BF'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#CBC8BF'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.6

ACCENT_COLOR = '#1F6F78'
PANEL_BG = '#F7F6F2'
PAGE_BG = '#ECEAE4'
TEXT_PRIMARY = '#1E2328'
TEXT_SECONDARY = '#58626B'

def evaluate_classifier(model, X_test, y_test, class_names, model_name, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # Run test set predictions
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=-1)
    
    # Overall test accuracy
    test_acc = float(np.mean(y_pred == y_test))
    
    # Classification report (per-class precision, recall, f1, support)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save metrics JSON
    metrics = {
        "model_name": model_name,
        "test_accuracy": test_acc,
        "total_test_samples": int(len(y_test)),
        "class_names": class_names,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1_score": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"])
            }
            for cls in class_names
        },
        "macro_avg_f1": float(report["macro avg"]["f1-score"]),
        "weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": cm.tolist()
    }
    
    json_path = os.path.join(results_dir, f"{model_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {model_name} metrics to {json_path}")
    
    # Save Confusion Matrix CSV
    cm_csv_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.csv")
    with open(cm_csv_path, "w") as f:
        f.write("True_Class," + ",".join([f"Pred_{c}" for c in class_names]) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i]}," + ",".join(map(str, row)) + "\n")
            
    # Plot Confusion Matrix with 5 discrete single-hue steps of accent
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PAGE_BG)
    ax.set_facecolor(PANEL_BG)
    
    # Normalize for coloring
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    # 5 discrete steps
    from matplotlib.colors import ListedColormap
    cm_cmap = ListedColormap(['#F7F6F2', '#D2E4E6', '#A5C9CC', '#639FA6', '#1F6F78'])
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cm_cmap, vmin=0.0, vmax=1.0)
    
    # Print integer counts in each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            # Use light text for dark cells (high percentage)
            text_color = '#ECEAE4' if cm_norm[i, j] > 0.6 else TEXT_PRIMARY
            ax.text(j, i, f"{count}", ha="center", va="center", color=text_color, fontsize=10, fontfamily='Consolas')
            
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9, color=TEXT_PRIMARY)
    ax.set_yticklabels(class_names, fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Predicted Class", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("True Class", fontsize=10, color=TEXT_SECONDARY)
    ax.set_title(f"{model_name.upper()} Confusion Matrix (Test Accuracy: {test_acc*100:.2f}%)", fontsize=11, color=TEXT_PRIMARY, pad=12)
    
    plt.tight_layout()
    cm_png_path = os.path.join(results_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(cm_png_path, dpi=150, facecolor=PAGE_BG)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {cm_png_path}")
    
    return metrics
