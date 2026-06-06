"""Step 9: Analyze extracted features and generate diagnostic plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('figures', exist_ok=True)

df = pd.read_csv('data/feature_dataset.csv')
class_names = ['CLEAN', 'NOISY', 'DRIFTED', 'ATTENUATED', 'SPIKED']
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
features = [c for c in df.columns if c != 'label']

print(f"Dataset shape: {df.shape}")
print(f"Features: {features}")

# ── Figure 1: Feature Distributions ───────────────────
print("\nPlotting feature distributions...")
fig, axes = plt.subplots(3, 5, figsize=(24, 14))
axes = axes.flatten()

for i, feat in enumerate(features):
    for cls in range(5):
        subset = df[df['label'] == cls][feat]
        axes[i].hist(subset, bins=40, alpha=0.55,
                     label=class_names[cls], color=colors[cls])
    axes[i].set_title(feat, fontweight='bold', fontsize=10)
    axes[i].legend(fontsize=6)
    axes[i].grid(True, alpha=0.3)

# Hide unused subplot
for j in range(len(features), len(axes)):
    axes[j].axis('off')

plt.suptitle('Feature Distributions by Signal Class',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved figures/feature_distributions.png")

# ── Figure 2: Correlation Matrix ───────────────────────
print("Plotting correlation matrix...")
plt.figure(figsize=(13, 10))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', 
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png', dpi=150)
plt.show()
print("Saved figures/correlation_matrix.png")

# ── Figure 3: PCA Plot ─────────────────────────────────
print("Plotting PCA separability...")
X = df[features].values
y = df['label'].values

scaler  = StandardScaler()
X_scaled= scaler.fit_transform(X)

pca     = PCA(n_components=2)
X_pca   = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
for cls in range(5):
    mask = y == cls
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[cls], label=class_names[cls],
                alpha=0.4, s=12)

plt.xlabel(
    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance explained)',
    fontsize=11)
plt.ylabel(
    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance explained)',
    fontsize=11)
plt.title('PCA — Feature Space Separability by Fault Class',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11, markerscale=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pca_plot.png', dpi=150)
plt.show()
print("Saved figures/pca_plot.png")
print("\nIf the 5 clusters are clearly separated in the PCA plot,")
print("your feature set is doing its job correctly.")
