import cv2
import numpy as np

image = cv2.imread('example_03.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's thresholding (adapt based on object/background brightness)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours based on area
min_area = 100 
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

object_count = len(filtered_contours)

result = image.copy()
cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

print(f"Number of objects detected: {object_count}")
cv2.imshow('Objects Detected', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
