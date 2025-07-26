import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load image using OpenCV (BGR format)
img_bgr = cv2.imread('C:/Users/skyline/OneDrive/Desktop/istockphoto-133305215-612x612.jpg')  # Replace with your image path
if img_bgr is None:
    raise ValueError("Image not found or path incorrect")

# 2. Convert to RGB for display and processing
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. Reshape to a 2D array of pixels (num_pixels x 3)
pixel_values = img_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)  # Convert to float

# 4. Choose number of clusters (k)
k = 8

# 5. Apply KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_values)
centers = kmeans.cluster_centers_

# 6. Replace each pixel color with its cluster center
segmented_img = centers[labels].reshape(img_rgb.shape).astype(np.uint8)

# 7. Plot original and segmented images side-by-side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.axis('off')
plt.imshow(img_rgb)

plt.subplot(1, 2, 2)
plt.title(f'Segmented Image with k={k}')
plt.axis('off')
plt.imshow(segmented_img)
plt.show()
