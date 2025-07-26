import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# ---- Step 1: Choose dataset ---- #
dataset_choice = 'digits'  # Change to 'iris' for Iris dataset

if dataset_choice == 'digits':
    data = load_digits()
    title_prefix = "Digits"
    cmap = 'tab10'
elif dataset_choice == 'iris':
    data = load_iris()
    title_prefix = "Iris"
    cmap = 'viridis'

X = data.data
y = data.target
labels = data.target_names if dataset_choice == 'iris' else np.unique(y)

# ---- Step 2: Dimensionality Reduction ---- #
# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# t-SNE
perplexity = 30
learning_rate = 200
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
X_tsne = tsne.fit_transform(X)

# ---- Step 3: Plot PCA vs t-SNE ---- #
fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

# PCA plot
p1 = axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, s=60, edgecolor='k', alpha=0.85)
axs[0].set_title(f"{title_prefix} PCA (Linear)")
axs[0].set_xlabel("Principal Component 1")
axs[0].set_ylabel("Principal Component 2")
axs[0].grid(True)

# t-SNE plot
p2 = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, s=60, edgecolor='k', alpha=0.85)
axs[1].set_title(f"{title_prefix} t-SNE (Nonlinear)\nPerplexity={perplexity}, LR={learning_rate}")
axs[1].set_xlabel("t-SNE Dimension 1")
axs[1].set_ylabel("t-SNE Dimension 2")
axs[1].grid(True)

# Single shared colorbar under the plots
cbar = fig.colorbar(p2, ax=axs.ravel().tolist(), orientation='horizontal', pad=0.1, aspect=50)
cbar.set_label('Target Label')

# Main title
fig.suptitle(f"{title_prefix} Dataset: PCA vs t-SNE", fontsize=16)
plt.show()
