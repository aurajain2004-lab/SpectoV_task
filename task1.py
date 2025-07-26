import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN

# 1. Generate datasets

# Nested crescent moons (2 clusters)
X_moons, _ = make_moons(n_samples=300, noise=0.07, random_state=42)

# One concentric circle dataset (2 rings)
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

# 2. Shift circles right and up to separate from moons
X_circles += np.array([4.5, 6])

# 3. Combine moons and circles into one dataset
X_combined = np.vstack((X_moons, X_circles))

# 4. Apply KMeans (expecting 4 clusters roughly: 2 moons + 2 circles)
kmeans = KMeans(n_clusters=4, n_init='auto', random_state=42)
kmeans_labels = kmeans.fit_predict(X_combined)

# 5. Apply DBSCAN with tuned eps=0.18 to keep outer circle as one cluster
dbscan = DBSCAN(eps=0.18, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_combined)

# 6. Plot side-by-side

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# KMeans plot
axs[0].scatter(
    X_combined[:, 0], X_combined[:, 1], c=kmeans_labels, cmap='tab10', s=80, edgecolor='k', alpha=0.9
)
axs[0].scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    marker='X', c='red', s=250, label='Centroids', edgecolor='k'
)
axs[0].set_title("K-Means Clustering")
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")

# DBSCAN plot
axs[1].scatter(
    X_combined[:, 0], X_combined[:, 1], c=dbscan_labels, cmap='tab10', s=80, edgecolor='k', alpha=0.9
)
axs[1].set_title("DBSCAN Clustering")
axs[1].grid(True)
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

# 7. Print cluster info
print("K-Means Clustering:")
print(f"  Number of clusters: {len(set(kmeans_labels))}")

print("\nDBSCAN Clustering:")
print(f"  Number of clusters (excluding noise): {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
print(f"  Number of noise points: {list(dbscan_labels).count(-1)}")
