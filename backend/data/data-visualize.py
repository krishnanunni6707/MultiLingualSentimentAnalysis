import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load training data
df = pd.read_csv('../checkpoints/training_metrics.csv')

# 2. Set visual style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Model Training & Validation Performance Overview', fontsize=16, fontweight='bold', y=0.98)

# ---------------------------------------------------------
# Subplot 1: Total & Task Losses
# ---------------------------------------------------------
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train_total_loss'], label='Train Total Loss', color='#1f77b4', linewidth=2.5, marker='o')
ax1.plot(df['epoch'], df['total_loss'], label='Val Total Loss', color='#d62728', linewidth=2.5, marker='s', linestyle='--')
ax1.set_title('Loss Trajectory (Train vs. Validation)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_xticks(df['epoch'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---------------------------------------------------------
# Subplot 2: Validation Emotion Classification Metrics
# ---------------------------------------------------------
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['val_emotion_f1_micro'], label='F1 Micro', color='#2ca02c', linewidth=2.5, marker='^')
ax2.plot(df['epoch'], df['val_emotion_f1_macro'], label='F1 Macro', color='#ff7f0e', linewidth=2.5, marker='v')
ax2.plot(df['epoch'], df['val_emotion_subset_accuracy'], label='Subset Accuracy', color='#9467bd', linewidth=2.5, marker='d')
ax2.set_title('Validation Emotion Classification Performance', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.set_xticks(df['epoch'])
ax2.legend()
ax2.grid(True, alpha=0.3)

# ---------------------------------------------------------
# Subplot 3: Polarity & Intensity Performance
# ---------------------------------------------------------
ax3 = axes[1, 0]
ax3.plot(df['epoch'], df['val_polarity_pearson'], label='Polarity Pearson r', color='#1f77b4', linewidth=2.5, marker='o')
ax3.plot(df['epoch'], df['val_intensity_pearson'], label='Intensity Pearson r', color='#e377c2', linewidth=2.5, marker='s')
ax3.plot(df['epoch'], df['val_polarity_mae'], label='Polarity MAE', color='#1f77b4', linestyle=':', marker='.')
ax3.plot(df['epoch'], df['val_intensity_mae'], label='Intensity MAE', color='#e377c2', linestyle=':', marker='.')
ax3.set_title('Polarity & Intensity Validation Metrics', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score / Error')
ax3.set_xticks(df['epoch'])
ax3.legend()
ax3.grid(True, alpha=0.3)

# ---------------------------------------------------------
# Subplot 4: Final Epoch Per-Emotion F1 Scores
# ---------------------------------------------------------
ax4 = axes[1, 1]
emotion_cols = [c for c in df.columns if c.startswith('val_emotion_f1_') and c not in ['val_emotion_f1_micro', 'val_emotion_f1_macro']]
latest_emotions = df.iloc[-1][emotion_cols]
emotion_names = [c.replace('val_emotion_f1_', '').capitalize() for c in emotion_cols]

# Sort by F1 score
sorted_indices = latest_emotions.values.argsort()
sorted_names = [emotion_names[i] for i in sorted_indices]
sorted_values = [latest_emotions.values[i] for i in sorted_indices]

colors = sns.color_palette("viridis", len(sorted_names))
bars = ax4.barh(sorted_names, sorted_values, color=colors, height=0.7)
ax4.set_title('Per-Emotion F1 Scores (Final Epoch)', fontsize=12, fontweight='bold')
ax4.set_xlabel('F1 Score')
ax4.set_xlim(0, 1.0)
ax4.grid(True, alpha=0.3)

# Add numeric value labels on bars
for bar in bars:
    width = bar.get_width()
    ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
             va='center', ha='left', fontsize=8, fontweight='semibold')

# Adjust layout & Save / Display
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('training_metrics_dashboard.png', dpi=300)
plt.show()