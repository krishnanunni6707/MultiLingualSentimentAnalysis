import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def find_default_metrics_csv() -> str:
    """Find training_metrics.csv across standard workspace relative paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "../checkpoints/training_metrics.csv",
        script_dir / "../../checkpoints/training_metrics.csv",
        Path.cwd() / "backend/checkpoints/training_metrics.csv",
        Path.cwd() / "checkpoints/training_metrics.csv",
        Path.cwd() / "training_metrics.csv",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return str(script_dir / "../checkpoints/training_metrics.csv")


def find_default_dataset_csv() -> Optional[str]:
    """Find default Hindi emotion dataset CSV across standard relative paths."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "EmoInHindi/LREC_EmoInHindi_reduced.csv",
        script_dir / "EmoInHindi/LREC_EmoInHindi.csv",
        Path.cwd() / "backend/data/EmoInHindi/LREC_EmoInHindi_reduced.csv",
        Path.cwd() / "backend/data/EmoInHindi/LREC_EmoInHindi.csv",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def plot_training_metrics(csv_path: str, output_path: str, show_plot: bool = True):
    """Generate comprehensive 4-panel training and validation metrics dashboard."""
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metrics log file not found at: {csv_path}\n"
            f"Please run model training first or verify the path."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Metrics log file {csv_path} is empty.")

    print(f"[+] Loaded {len(df)} epoch records from: {csv_path}")

    # Set visual styling
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Identify best epoch (by is_best flag or max val_emotion_f1_macro)
    if 'is_best' in df.columns and df['is_best'].max() > 0:
        best_epoch_idx = df[df['is_best'] == 1]['val_emotion_f1_macro'].idxmax()
    elif 'val_emotion_f1_macro' in df.columns:
        best_epoch_idx = df['val_emotion_f1_macro'].idxmax()
    else:
        best_epoch_idx = df.index[-1]

    best_epoch = int(df.iloc[best_epoch_idx]['epoch'])
    best_macro_f1 = df.iloc[best_epoch_idx].get('val_emotion_f1_macro', 0.0)

    fig.suptitle(
        f'Multi-Task Sentiment Model: Training & Validation Performance Overview\n'
        f'Total Epochs: {len(df)} | Best Epoch: #{best_epoch} (Val Macro F1: {best_macro_f1:.4f})',
        fontsize=15, fontweight='bold', y=0.98
    )

    epochs = df['epoch']

    # ---------------------------------------------------------
    # Subplot 1: Total & Sub-task Losses
    # ---------------------------------------------------------
    ax1 = axes[0, 0]
    val_loss_col = 'val_total_loss' if 'val_total_loss' in df.columns else 'total_loss'
    
    ax1.plot(epochs, df['train_total_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2.5, marker='o')
    if val_loss_col in df.columns:
        ax1.plot(epochs, df[val_loss_col], label='Val Total Loss', color='#d62728', linewidth=2.5, marker='s', linestyle='--')
    
    # Sub-task breakdown lines if available
    if 'train_emotion_loss' in df.columns and 'emotion_loss' in df.columns:
        ax1.plot(epochs, df['train_emotion_loss'], label='Train Emotion Loss', color='#2ca02c', linewidth=1.5, linestyle=':')
        ax1.plot(epochs, df['emotion_loss'], label='Val Emotion Loss', color='#8c564b', linewidth=1.5, linestyle=':')
    
    ax1.set_title('Loss Trajectory (Train vs. Validation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss Value', fontweight='bold')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.axvline(x=best_epoch, color='#e377c2', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Subplot 2: Validation Emotion Classification Metrics
    # ---------------------------------------------------------
    ax2 = axes[0, 1]
    if 'val_emotion_f1_micro' in df.columns:
        ax2.plot(epochs, df['val_emotion_f1_micro'], label='F1 Micro', color='#2ca02c', linewidth=2.5, marker='^')
    if 'val_emotion_f1_macro' in df.columns:
        ax2.plot(epochs, df['val_emotion_f1_macro'], label='F1 Macro', color='#ff7f0e', linewidth=2.5, marker='v')
    if 'val_emotion_subset_accuracy' in df.columns:
        ax2.plot(epochs, df['val_emotion_subset_accuracy'], label='Subset Accuracy', color='#9467bd', linewidth=2.5, marker='d')
    if 'val_emotion_precision_micro' in df.columns:
        ax2.plot(epochs, df['val_emotion_precision_micro'], label='Precision Micro', color='#17becf', linewidth=1.8, linestyle='-.')
    if 'val_emotion_recall_micro' in df.columns:
        ax2.plot(epochs, df['val_emotion_recall_micro'], label='Recall Micro', color='#bcbd22', linewidth=1.8, linestyle='-.')

    ax2.axvline(x=best_epoch, color='#e377c2', linestyle='--', alpha=0.7)
    ax2.set_title('Validation Emotion Classification Performance', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Score (0.0 - 1.0)', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(loc='lower right', fontsize=9, frameon=True)
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Subplot 3: Polarity & Intensity Performance (Twin Y-Axis)
    # ---------------------------------------------------------
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    plots = []
    if 'val_polarity_pearson' in df.columns:
        p1 = ax3.plot(epochs, df['val_polarity_pearson'], label='Polarity Pearson r', color='#1f77b4', linewidth=2.5, marker='o')
        plots.extend(p1)
    if 'val_intensity_pearson' in df.columns:
        p2 = ax3.plot(epochs, df['val_intensity_pearson'], label='Intensity Pearson r', color='#e377c2', linewidth=2.5, marker='s')
        plots.extend(p2)
    
    if 'val_polarity_mae' in df.columns:
        p3 = ax3_twin.plot(epochs, df['val_polarity_mae'], label='Polarity MAE', color='#1f77b4', linestyle=':', linewidth=2, marker='.')
        plots.extend(p3)
    if 'val_intensity_mae' in df.columns:
        p4 = ax3_twin.plot(epochs, df['val_intensity_mae'], label='Intensity MAE', color='#e377c2', linestyle=':', linewidth=2, marker='.')
        plots.extend(p4)

    ax3.set_title('Polarity & Intensity Validation Metrics', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Pearson Correlation (r)', fontweight='bold', color='#1f77b4')
    ax3_twin.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold', color='#8c564b')
    ax3.set_ylim(0, 1.05)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.axvline(x=best_epoch, color='#e377c2', linestyle='--', alpha=0.7)

    labels = [p.get_label() for p in plots]
    ax3.legend(plots, labels, loc='center right', fontsize=9, frameon=True)
    ax3.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Subplot 4: Best Epoch Per-Emotion F1 Scores
    # ---------------------------------------------------------
    ax4 = axes[1, 1]
    emotion_cols = [
        c for c in df.columns 
        if c.startswith('val_emotion_f1_') and c not in ['val_emotion_f1_micro', 'val_emotion_f1_macro']
    ]

    if emotion_cols:
        best_emotions = df.iloc[best_epoch_idx][emotion_cols]
        emotion_names = [c.replace('val_emotion_f1_', '').capitalize() for c in emotion_cols]

        sorted_pairs = sorted(zip(emotion_names, best_emotions.values), key=lambda x: float(x[1]))
        sorted_names = [p[0] for p in sorted_pairs]
        sorted_values = [float(p[1]) for p in sorted_pairs]

        colors = [
            '#d62728' if v < 0.4 else ('#ff7f0e' if v < 0.7 else '#2ca02c') 
            for v in sorted_values
        ]

        bars = ax4.barh(sorted_names, sorted_values, color=colors, height=0.65)
        ax4.set_title(f'Per-Emotion F1 Scores (Best Epoch #{best_epoch})', fontsize=12, fontweight='bold')
        ax4.set_xlabel('F1 Score', fontweight='bold')
        
        # Ensure right padding so bar numbers up to 1.00 don't clip off screen
        max_val = max(sorted_values) if sorted_values else 1.0
        ax4.set_xlim(0, max(1.15, max_val + 0.12))
        ax4.grid(True, alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax4.text(
                width + 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f'{width:.2f}',
                va='center',
                ha='left',
                fontsize=8.5,
                fontweight='bold'
            )
    else:
        ax4.text(0.5, 0.5, 'No per-emotion F1 columns found', ha='center', va='center')

    # Adjust layout & Save / Show
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[+] Dashboard saved successfully → {output_file.resolve()}")

    if show_plot and os.environ.get('DISPLAY'):
        try:
            plt.show()
        except Exception as e:
            print(f"[i] GUI display skipped ({e}). Image saved to file.")
    plt.close(fig)


def plot_dataset_statistics(data_csv_path: str, output_path: str, show_plot: bool = True):
    """Generate exploratory analysis dashboard for raw Hindi dataset CSV."""
    data_path = Path(data_csv_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {data_path}")

    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"[+] Loaded {len(df)} dataset rows from: {data_path}")

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Hindi Sentiment Dataset Analysis Overview\nFile: {data_path.name} ({len(df)} total utterances)', fontsize=15, fontweight='bold', y=0.98)

    # 1. Emotion Frequency Breakdown
    emotion_counts = {}
    label_counts_per_sample = []

    for emo_str in df['emotions'].fillna('neutral'):
        labels = [l.strip().lower() for l in str(emo_str).split(',') if l.strip()]
        label_counts_per_sample.append(len(labels))
        for l in labels:
            emotion_counts[l] = emotion_counts.get(l, 0) + 1

    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    emo_names = [e[0].capitalize() for e in sorted_emotions[:15]]
    emo_vals = [e[1] for e in sorted_emotions[:15]]

    ax1 = axes[0, 0]
    bars1 = ax1.barh(emo_names[::-1], emo_vals[::-1], color=sns.color_palette("mako", len(emo_names)))
    ax1.set_title('Top Emotion Label Frequencies', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Count', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + max(emo_vals)*0.01, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center', fontsize=8)

    # 2. Multi-Label Complexity Distribution
    ax2 = axes[0, 1]
    multi_counts = pd.Series(label_counts_per_sample).value_counts().sort_index()
    ax2.bar(multi_counts.index.astype(str), multi_counts.values, color='#9467bd', width=0.5)
    ax2.set_title('Multi-Label Complexity (Emotions per Utterance)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Co-occurring Emotions', fontweight='bold')
    ax2.set_ylabel('Sample Count', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for i, val in enumerate(multi_counts.values):
        ax2.text(i, val + max(multi_counts.values)*0.01, f'{val}', ha='center', fontsize=9, fontweight='bold')

    # 3. Emotion Intensity Value Distribution
    ax3 = axes[1, 0]
    intensities = []
    for int_str in df['emoIntensity'].fillna('1'):
        for val in str(int_str).split(','):
            try:
                intensities.append(float(val.strip()))
            except ValueError:
                pass

    if intensities:
        sns.histplot(intensities, ax=ax3, bins=np.arange(0.5, 4.5, 0.5), color='#ff7f0e', discrete=True)
        ax3.set_title('Emotion Intensity Distribution (1.0 to 3.0 scale)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Intensity Level', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No valid intensity values', ha='center', va='center')

    # 4. Author Role Distribution
    ax4 = axes[1, 1]
    if 'authorRole' in df.columns:
        role_counts = df['authorRole'].value_counts()
        ax4.pie(role_counts.values, labels=[r.capitalize() for r in role_counts.index], autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        ax4.set_title('Speaker Author Role Distribution', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No authorRole column present', ha='center', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[+] Dataset visualization saved successfully → {output_file.resolve()}")

    if show_plot and os.environ.get('DISPLAY'):
        try:
            plt.show()
        except Exception as e:
            print(f"[i] GUI display skipped ({e}). Image saved to file.")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Sentiment Analysis Data & Metrics Visualizer")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["metrics", "dataset"],
        default="metrics",
        help="Visualization mode: 'metrics' for training performance, 'dataset' for raw CSV statistics"
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=find_default_metrics_csv(),
        help="Path to training_metrics.csv file"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=find_default_dataset_csv(),
        help="Path to raw dataset CSV file (for 'dataset' mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path for saved dashboard plot"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not invoke interactive GUI plt.show()"
    )
    args = parser.parse_args()

    show_plot = not args.no_show

    if args.mode == "metrics":
        out_path = args.output or "training_metrics_dashboard.png"
        plot_training_metrics(args.metrics_csv, out_path, show_plot=show_plot)
    elif args.mode == "dataset":
        data_csv = args.data_csv or find_default_dataset_csv()
        if not data_csv:
            print("[-] Error: No dataset CSV specified or found.")
            sys.exit(1)
        out_path = args.output or "dataset_statistics_dashboard.png"
        plot_dataset_statistics(data_csv, out_path, show_plot=show_plot)


if __name__ == "__main__":
    main()