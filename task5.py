import cv2
import numpy as np
from sklearn.cluster import KMeans

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance
    resized = cv2.resize(frame, (320, 240))
    
    # Blur to smooth image
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Prepare data for clustering
    pixels = blurred.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(pixels)
    clustered_pixels = kmeans.cluster_centers_[labels].astype(np.uint8)
    clustered_img = clustered_pixels.reshape((240, 320, 3))

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw polygon boundaries on original resized image
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(resized, [approx], -1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("AR-style Segmentation", resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()