"""Single-machine simulation of Federated Learning (FedAvg) on HAR 1D-CNN."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras

from src.models import build_har_model
from src.data_har import load_har_splits

PAGE_BG = '#ECEAE4'
PANEL_BG = '#F7F6F2'
BORDER_COLOR = '#CBC8BF'
TEXT_PRIMARY = '#1E2328'
TEXT_SECONDARY = '#58626B'
ACCENT_COLOR = '#1F6F78'
OCHRE_COLOR = '#A06F18'
SLATE_COLOR = '#4B5D8A'

def get_client_weights(model):
    return [w.copy() for w in model.get_weights()]

def set_client_weights(model, weights):
    model.set_weights([w.copy() for w in weights])

def federated_averaging(client_weights_list, client_sizes):
    """Computes weighted average of client weights."""
    total_samples = sum(client_sizes)
    avg_weights = []
    num_layers = len(client_weights_list[0])
    
    for layer_idx in range(num_layers):
        layer_avg = np.zeros_like(client_weights_list[0][layer_idx])
        for c_idx, weights in enumerate(client_weights_list):
            factor = client_sizes[c_idx] / total_samples
            layer_avg += weights[layer_idx] * factor
        avg_weights.append(layer_avg)
    return avg_weights

def run_federated_simulation(rounds=15, local_epochs=2, num_clients=8):
    print("\n=== Running Federated Learning (FedAvg) Simulation ===")
    har_splits = load_har_splits()
    X_train, y_train, sub_train = har_splits["train"]
    X_test, y_test, _ = har_splits["test"]
    
    # 1. Non-IID partitioning: split by subject id
    unique_subs = np.unique(sub_train)
    client_subs = np.array_split(unique_subs, num_clients)
    non_iid_clients = []
    for subs in client_subs:
        mask = np.isin(sub_train, subs)
        non_iid_clients.append((X_train[mask], y_train[mask]))
        
    # 2. IID partitioning: random uniform partition
    indices = np.random.permutation(len(X_train))
    client_indices = np.array_split(indices, num_clients)
    iid_clients = []
    for idxs in client_indices:
        iid_clients.append((X_train[idxs], y_train[idxs]))
        
    def evaluate_model_acc(model, X, y):
        preds = np.argmax(model.predict(X, verbose=0), axis=-1)
        return float(np.mean(preds == y))
        
    # Helper to simulate FedAvg
    def simulate_fedavg(clients_data, mode_name):
        print(f"Simulating FedAvg ({mode_name})...")
        global_model = build_har_model(input_shape=(128, 6), num_classes=6)
        global_model.compile(optimizer=keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        round_accuracies = []
        for r in range(rounds):
            client_weights = []
            client_sizes = []
            current_global_weights = get_client_weights(global_model)
            
            for c_X, c_y in clients_data:
                local_model = build_har_model(input_shape=(128, 6), num_classes=6)
                local_model.compile(optimizer=keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy")
                set_client_weights(local_model, current_global_weights)
                local_model.fit(c_X, c_y, epochs=local_epochs, batch_size=32, verbose=0)
                client_weights.append(get_client_weights(local_model))
                client_sizes.append(len(c_X))
                
            new_global_weights = federated_averaging(client_weights, client_sizes)
            set_client_weights(global_model, new_global_weights)
            
            test_acc = evaluate_model_acc(global_model, X_test, y_test)
            round_accuracies.append(test_acc)
            print(f"Round {r+1}/{rounds} - Test Accuracy: {test_acc*100:.2f}%")
            
        return round_accuracies, global_model
        
    iid_accs, _ = simulate_fedavg(iid_clients, "IID Partitioning")
    non_iid_accs, _ = simulate_fedavg(non_iid_clients, "Non-IID Partitioning")
    
    # Centralized baseline accuracy for reference
    central_acc = 0.925  # or read from har_baseline_metrics.json if exists
    if os.path.exists("results/har_baseline_metrics.json"):
        with open("results/har_baseline_metrics.json", "r") as f:
            central_acc = json.load(f)["test_accuracy"]
            
    # Calculate communication costs
    sample_model = build_har_model(input_shape=(128, 6), num_classes=6)
    total_params = int(sample_model.count_params())
    
    # Per round bytes: 2 * num_clients * params * bytes_per_param
    fp32_bytes_per_round = 2 * num_clients * total_params * 4
    int8_bytes_per_round = 2 * num_clients * total_params * 1
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/federated_accuracy.csv"
    with open(csv_path, "w") as f:
        f.write("round,fedavg_iid_acc,fedavg_non_iid_acc,centralized_acc\n")
        for r in range(rounds):
            f.write(f"{r+1},{iid_accs[r]:.6f},{non_iid_accs[r]:.6f},{central_acc:.6f}\n")
            
    # Plot accuracy per round
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=PAGE_BG)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=BORDER_COLOR)
    
    rounds_range = range(1, rounds + 1)
    ax.plot(rounds_range, [central_acc * 100.0] * rounds, color=TEXT_SECONDARY, linestyle=':', label='Centralized Baseline')
    ax.plot(rounds_range, [a * 100.0 for a in iid_accs], color=ACCENT_COLOR, marker='s', markersize=4, label='FedAvg (IID Partition)')
    ax.plot(rounds_range, [a * 100.0 for a in non_iid_accs], color=OCHRE_COLOR, marker='o', markersize=4, label='FedAvg (Non-IID by Subject)')
    
    ax.set_title("Federated Learning (FedAvg) Convergence Across Rounds", fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlabel("Federated Round", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10, color=TEXT_SECONDARY)
    ax.legend(loc='lower right', facecolor=PANEL_BG, edgecolor=BORDER_COLOR)
    
    plt.tight_layout()
    fig.savefig("results/federated_accuracy.png", dpi=150, facecolor=PAGE_BG)
    plt.close(fig)
    
    res = {
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "model_parameters": total_params,
        "centralized_accuracy": float(central_acc),
        "final_fedavg_iid_accuracy": float(iid_accs[-1]),
        "final_fedavg_non_iid_accuracy": float(non_iid_accs[-1]),
        "accuracy_gap_non_iid_pct": float((central_acc - non_iid_accs[-1]) * 100.0),
        "fp32_communication_per_round_kb": float(round(fp32_bytes_per_round / 1024.0, 2)),
        "int8_quantized_communication_per_round_kb": float(round(int8_bytes_per_round / 1024.0, 2)),
        "total_fp32_transfer_mb": float(round((fp32_bytes_per_round * rounds) / (1024.0 * 1024.0), 2)),
        "total_int8_transfer_mb": float(round((int8_bytes_per_round * rounds) / (1024.0 * 1024.0), 2)),
        "simulation_note": "Single-machine simulation on UCI HAR subjects."
    }
    with open("results/federated_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results/federated_accuracy.png and results/federated_results.json")
    return res

if __name__ == "__main__":
    run_federated_simulation(rounds=10, local_epochs=1, num_clients=6)
