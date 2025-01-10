import cv2
import numpy as np

# Load the image
image = cv2.imread('example_02.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
# Apply thresholding to create a binary image
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Use morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Find contours
contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
min_area = 100  # Adjust as needed
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Count the objects
object_count = len(filtered_contours)

# Draw contours on the original image
result = image.copy()
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

# Display results
print(f"Number of objects detected: {object_count}")
cv2.imshow('opening', opening)
cv2.imshow('Processed Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
