import cv2 as cv
import numpy as np

# Load images
meowth = cv.imread("meowth.jpg")
jigglyPuff = cv.imread("jigglePuff.jpg")

gray_image_meowth = cv.cvtColor(meowth, cv.COLOR_BGR2GRAY)
gray_image_jigglyPuff = cv.cvtColor(jigglyPuff, cv.COLOR_BGR2GRAY)

height_meowth, width_meowth = gray_image_meowth.shape[:2]
height_jiggly, width_jiggly = gray_image_jigglyPuff.shape[:2]

gray_image_jigglyPuff_resized = cv.resize(
    gray_image_jigglyPuff, (width_meowth, height_jiggly)
)

circle_image= gray_image_jigglyPuff_resized.copy()
width, height = circle_image.shape[:2]
circle_radius = 40
circle_center = (width_meowth - circle_radius, height - circle_radius)
cv.circle(circle_image, circle_center, circle_radius, 0, -1)

combined_image_left = np.vstack((gray_image_meowth, circle_image))
combined_image_right = np.vstack((gray_image_meowth, gray_image_jigglyPuff_resized))

ret, binary_combined_image_right = cv.threshold(combined_image_right, 127, 255, cv.THRESH_BINARY)

combined_image = np.hstack((combined_image_left, binary_combined_image_right))

cv.imshow("Combined Image", combined_image)
cv.waitKey(0)
