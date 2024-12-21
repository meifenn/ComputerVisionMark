import cv2 as cv 

color_offset=0
def nothing(x):
    pass

title_window="Monsters"
cv.namedWindow(title_window)
cv.createTrackbar("color_offset","Monsters",0,255,nothing)

img=cv.imread("jigglypuff.jpg")
cv.imshow("Pokemon",img)

while True:
    img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)

    color_offset=cv.getTrackbarPos('color_offset',title_window)
    for row in range(0,img_hsv.shape[0]):
        for column in range(0,img_hsv.shape[1]):
            img_hsv[row][column][0] = img_hsv[row][column][0] + color_offset

    img_final=cv.cvtColor(img_hsv,cv.COLOR_HSV2BGR)

    cv.imshow(title_window,img_final)

    if cv.waitKey(1) & 0xFF ==27:
        break
cv.destroyAllWindows()