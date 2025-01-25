import cv2 as cv
import numpy as np
import numpy.typing as npt
import argparse

def order_points(pts: npt.NDArray) -> npt.NDArray:
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def four_point_transform(image: npt.NDArray, pts: npt.NDArray) -> npt.NDArray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct destination points
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)
    # Apply perspective transform
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def main() -> None:
    parser=argparse.ArgumentParser(description='image')
    parser.add_argument("-i",'--image',required=True,help="Path to the iamge to be scanned")
    args=parser.parse_args()

    image=cv.imread(args.image)
    if image is None:
        raise ValueError("The Image is missing")
    
    ratio = 1000/image.shape[0]
    # orig = image.copy()
    imageResize = cv.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv.INTER_CUBIC)

    gray = cv.cvtColor(imageResize, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 75, 200)

    contours,_=cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv.contourArea,reverse=True)[:5]

    screen_cnt=None
    for c in contours:
        peri=cv.arcLength(c,True)
        approx=cv.approxPolyDP(c,0.02*peri,True)

        if len(approx)==4:
            screen_cnt=approx
            break

    warp=four_point_transform(imageResize, screen_cnt.reshape(4,2))
    warp_gray= cv.cvtColor(warp,cv.COLOR_BGR2GRAY)
    warp_final= cv.adaptiveThreshold(warp_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,10)
    cv.imshow('warp',warp_final)

    cv.drawContours(imageResize, [screen_cnt], -1, (0, 255, 0), 5)
    cv.imshow('original',imageResize)
    cv.imshow('gray',gray)
    cv.imshow('edged',edged)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()