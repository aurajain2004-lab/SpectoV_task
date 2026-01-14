# Unsupervised Clustering and Image Segmentation for AR/VR

This repository contains implementations of unsupervised learning tasks focused on clustering and image segmentation, designed to build foundational skills useful in AR/VR object recognition and sensor data interpretation.

---

## Project Overview

The project is divided into six main tasks:

### ✅ Task 1: Understand Clustering Algorithms (K-Means and DBSCAN)

- **Aim:** Learn and implement K-Means and DBSCAN clustering algorithms on synthetic 2D datasets.
- **Approach:**  
  - Generate datasets using `make_blobs`, `make_moons`, and `make_circles` from scikit-learn.  
  - Apply K-Means clustering with a chosen number of clusters.  
  - Apply DBSCAN, tweaking `eps` and `min_samples` parameters.  
  - Visualize and compare cluster results highlighting strengths and weaknesses of each method.

---

### ✅ Task 2: Image Segmentation with K-Means

- **Aim:** Segment images into color regions using K-Means clustering.
- **Approach:**  
  - Load images using OpenCV or PIL.  
  - Reshape pixels to 2D array of RGB values.  
  - Cluster pixels into `k` color groups and reconstruct segmented images.  
  - Experiment with different `k` values and color spaces (grayscale, LAB).

---

### ✅ Task 3: Dimensionality Reduction with PCA and t-SNE

- **Aim:** Visualize high-dimensional data in 2D using PCA and t-SNE.
- **Approach:**  
  - Use built-in datasets (digits, Iris) from scikit-learn.  
  - Apply PCA and t-SNE separately.  
  - Visualize clusters in 2D scatter plots.  
  - Experiment with t-SNE parameters like perplexity and learning rate.

---

### ✅ Task 4: Clustering Real-World Sensor Data

- **Aim:** Apply clustering on real-world sensor data to detect user activity patterns.
- **Approach:**  
  - Download clean sensor datasets (e.g., UCI HAR Dataset).  
  - Preprocess: normalize, remove missing values.  
  - Cluster using K-Means and DBSCAN.  
  - Visualize clusters with PCA/t-SNE to interpret behaviors.  
  - Experiment with different feature combinations (acceleration vs. gyroscope).

---

### ✅ Task 5: Overlay Clustering Output on Image

- **Aim:** Simulate AR-style visualization by overlaying clustering results on images.
- **Approach:**  
  - Detect contours/boundaries in clustered image using OpenCV.  
  - Draw bounding boxes or polygons around clusters.  
  - Overlay these on the original image for clear visual feedback.  
  - (Optional) Apply the process frame-by-frame on video streams to simulate real-time AR.

---

### ✅ Task 6: Implement a Real GitHub Repo

- **Aim:** Clone and run a real-world unsupervised image segmentation repo.
- **Approach:**  
  - Fork or clone a lightweight GitHub repo performing K-Means-based image segmentation.  
  - Run the segmentation script on sample or personal images.  
  - Understand code structure: model definition, preprocessing, visualization.  
  - Experiment with parameters like cluster count and image resolution.

---

## How to Use This Repo

1. Clone the repository.
2. Follow each task folder or notebook for step-by-step implementation.
3. Modify parameters and inputs to experiment and visualize results.
4. Refer to the included resources for deeper understanding.


## Author

Aura Jain

---

