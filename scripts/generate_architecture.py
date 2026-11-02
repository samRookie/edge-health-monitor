"""Generates system architecture diagram docs/architecture.png using matplotlib."""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram():
    os.makedirs("docs", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#ECEAE4')
    ax.set_facecolor('#ECEAE4')
    ax.axis('off')
    
    # Palette tokens
    BG_PANEL = '#F7F6F2'
    BORDER_COLOR = '#CBC8BF'
    ACCENT = '#1F6F78'
    TEXT_PRI = '#1E2328'
    TEXT_SEC = '#58626B'
    STATUS_OK = '#3E7A4D'
    STATUS_WARN = '#A06F18'
    
    # Title
    ax.text(0.5, 0.96, "Edge Health Monitor: System Architecture & Dataflow",
            ha='center', va='center', fontsize=15, weight='bold', color=TEXT_PRI)
    ax.text(0.5, 0.925, "TinyML Dual-Inference Wearable Pipeline mapped across Units 1 to 5",
            ha='center', va='center', fontsize=10, color=TEXT_SEC)
            
    def draw_box(x, y, w, h, title, subtitle, tag=None, tag_color=STATUS_OK):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.2, edgecolor=BORDER_COLOR,
                                 facecolor=BG_PANEL, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=10, weight='bold', color=TEXT_PRI, zorder=3)
        ax.text(x + w/2, y + h*0.35, subtitle, ha='center', va='center', fontsize=8, color=TEXT_SEC, zorder=3)
        if tag:
            tbox = patches.Rectangle((x + w - 0.08, y + h - 0.03), 0.075, 0.025,
                                     linewidth=0.8, edgecolor=BORDER_COLOR, facecolor='#ECEAE4', zorder=4)
            ax.add_patch(tbox)
            ax.text(x + w - 0.042, y + h - 0.018, tag, ha='center', va='center', fontsize=7, color=tag_color, weight='bold', zorder=5)

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.5, mutation_scale=12),
                    zorder=1)
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.02, label, ha='center', va='bottom', fontsize=8, color=TEXT_SEC, zorder=3)

    # 1. Ingestion Layer
    ax.text(0.12, 0.85, "1. INGESTION LAYER", fontsize=9, weight='bold', color=ACCENT)
    draw_box(0.02, 0.68, 0.20, 0.12, "Android Phone Node", "Chrome: IMU (50Hz) + Mic", "Live Source", STATUS_OK)
    draw_box(0.02, 0.50, 0.20, 0.12, "Simulated Replay", "demo_replay.py (offline)", "Fallback", STATUS_WARN)
    
    # 2. Preprocessing Layer
    ax.text(0.35, 0.85, "2. PREPROCESSING", fontsize=9, weight='bold', color=ACCENT)
    draw_box(0.26, 0.68, 0.18, 0.12, "IMU Windowing", "128 samples @ 50 Hz (2.56s)\n6 axes normalized (g, rad/s)", "Unit 4")
    draw_box(0.26, 0.50, 0.18, 0.12, "MFCC Extraction", "16 kHz audio -> 40 mel bands\n98x40 spectrogram", "Unit 4")

    # 3. TinyML Inference Engine
    ax.text(0.57, 0.85, "3. TINYML RUNTIME", fontsize=9, weight='bold', color=ACCENT)
    draw_box(0.48, 0.68, 0.20, 0.12, "HAR 1D-CNN (INT8)", "6 classes (24k params)\nFlash: 32KB, Arena: 18KB", "Unit 2, 3")
    draw_box(0.48, 0.50, 0.20, 0.12, "KWS DS-CNN (INT8)", "8 classes (20k params)\nFlash: 68KB, Arena: 34KB", "Unit 2, 3")

    # 4. Alert & Decision Logic
    ax.text(0.80, 0.85, "4. ALERT STATE ENGINE", fontsize=9, weight='bold', color=ACCENT)
    draw_box(0.72, 0.68, 0.24, 0.12, "Fall Rule (Option 1)", "Spike > 2.5g + Stillness\nConfirmed by LAYING posture", "Unit 4, 5")
    draw_box(0.72, 0.50, 0.24, 0.12, "Keyword Rule (Option 2)", "Keyword 'stop' spotted\n+ abnormal posture/inactivity", "Unit 4, 5")
    
    # 5. Output Layer
    draw_box(0.72, 0.32, 0.24, 0.11, "Live Dashboard", "Studio aesthetic, 14 pages\nAlert banner & Events table", "Unit 1-5")

    # Connect feedforward lines
    draw_arrow(0.22, 0.74, 0.26, 0.74)
    draw_arrow(0.22, 0.56, 0.26, 0.56)
    draw_arrow(0.44, 0.74, 0.48, 0.74)
    draw_arrow(0.44, 0.56, 0.48, 0.56)
    draw_arrow(0.68, 0.74, 0.72, 0.74)
    draw_arrow(0.68, 0.56, 0.72, 0.56)
    draw_arrow(0.84, 0.68, 0.84, 0.62)
    draw_arrow(0.84, 0.50, 0.84, 0.43)

    # Offline Pipeline (Bottom Row)
    rect_offline = patches.Rectangle((0.02, 0.05), 0.94, 0.20, linewidth=1.0, edgecolor=BORDER_COLOR,
                                     facecolor='#F0EDE6', linestyle='--', zorder=1)
    ax.add_patch(rect_offline)
    ax.text(0.04, 0.22, "OFFLINE RESEARCH & TRAINING PIPELINE (Units 1, 2, 5)", fontsize=9, weight='bold', color=TEXT_PRI)
    
    draw_box(0.04, 0.08, 0.19, 0.11, "Supervised & Classical", "1D-CNN, RF, K-Means\nOverfit/Underfit demo", "Unit 2")
    draw_box(0.26, 0.08, 0.20, 0.11, "Compression Lifecycle", "Pruning (50%, 80%)\nFP16, INT8, Distillation", "Unit 2, 3")
    draw_box(0.49, 0.08, 0.22, 0.11, "Federated Learning", "FedAvg (IID vs Non-IID)\nCommunication deltas", "Unit 5")
    draw_box(0.74, 0.08, 0.20, 0.11, "Adversarial & Noise", "FGSM attack & defense\nSNR degradation analysis", "Unit 5")

    draw_arrow(0.23, 0.135, 0.26, 0.135)
    draw_arrow(0.46, 0.135, 0.49, 0.135)
    draw_arrow(0.71, 0.135, 0.74, 0.135)
    
    plt.tight_layout()
    png_path = os.path.join("docs", "architecture.png")
    fig.savefig(png_path, dpi=200, facecolor='#ECEAE4')
    plt.close(fig)
    print(f"Generated architecture diagram at {png_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
