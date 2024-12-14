import cv2 as cv
print(cv.__version__)

image = cv.imread("meowth.jpg")

grey_image= cv.cvtColor(image,cv.COLOR_BGR2GRAY)

def draw_line(image,start_x,start_y,end_x,end_y):
    x=start_x
    for y in range (start_y,end_y):
            image[y][x]=0

def draw_rectangle(image,start_x,start_y,end_x,end_y):
    for y in range (start_y,end_y):
        for x in range(start_x,end_x):
            image[y][x]=0

def draw_line2(image,start_x,start_y,end_x,end_y):
    for y,x in zip(range(start_y,end_y),range(start_x,end_x)):
            image[y][x]=0

def threasholding(image,threashold_val):
    binary_image=image.copy()
    height,width=image.shape
    for y in range(0,height):
        for x in range(0,width):
                if binary_image[y][x] <= threashold_val:
                    binary_image[y][x]=0
                else:
                    binary_image[y][x]=255
    return binary_image
                    
# binary_image=threasholding(grey_image,150)
ret,binary_image=cv.threshold(grey_image,127,255,cv.THRESH_BINARY)

# draw_rectangle(grey_image,50,20,200,300)

print(image.shape)
print(grey_image.shape)

cv.imshow("pokemon",binary_image)
cv.waitKey(0)