import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load Data (update your path)
base_path = "C:/Users/skyline/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/"

X = pd.read_csv(base_path + "X_train.txt", delim_whitespace=True, header=None)
y = pd.read_csv(base_path + "y_train.txt", header=None).squeeze()

# Step 2: Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: KMeans clustering on full normalized data
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
labels_k = kmeans.fit_predict(X_scaled)

# Step 4: Reduce dimensions for DBSCAN
pca_10 = PCA(n_components=10)
X_reduced = pca_10.fit_transform(X_scaled)

# Step 5: DBSCAN clustering on reduced data
dbscan = DBSCAN(eps=3, min_samples=10)
labels_d = dbscan.fit_predict(X_reduced)

# Step 6: 2D PCA for visualization of both
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)

# Step 7: Plot side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_k, cmap='tab10', s=10)
axs[0].set_title("KMeans Clustering (PCA 2D)")
axs[0].set_xlabel("PC 1")
axs[0].set_ylabel("PC 2")

axs[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_d, cmap='tab10', s=10)
axs[1].set_title("DBSCAN Clustering (PCA reduced data)")
axs[1].set_xlabel("PC 1")
axs[1].set_ylabel("PC 2")

plt.tight_layout()
plt.show()

# Optional: Print DBSCAN cluster counts
num_clusters = len(set(labels_d)) - (1 if -1 in labels_d else 0)
num_noise = list(labels_d).count(-1)
print(f"DBSCAN clusters (excluding noise): {num_clusters}")
print(f"Noise points: {num_noise}")
