import cv2
import numpy as np
import math
from BoundaryDetection import getSegMask

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def checkCircular(con,area,perimeter):

    circularity = 4*math.pi*(area/(perimeter*perimeter))
    return circularity


def checkRectangle(con,area):

    rect = cv2.minAreaRect(con)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area_rect = box[1][0]*box[1][1]
    
    return area/area_rect,box



def detectShapes(thresh,img_color):

    orig = img_color.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(thresh,contours,-1,(255,255,0),3)
    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        # area = cv2.contourArea(con)
        # if 10< area < 1000000:
        contours_area.append(con)
        
    
    contours_allowed = []

# check if contour is of circular shape
    for con in contours_area:
        temp_img = orig
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        cv2.drawContours(img_color,contours_allowed,-1,(255,255,0),3)
        cv2.imshow("ss",img_color)
        cv2.waitKey(0)

        if perimeter == 0:
            break
        circularity = checkCircular(con,area,perimeter)
      
        # print circularity
        # if 0.6 < circularity < 1.2:
        #     contours_allowed.append(con)
        #     continue
        
        # approx = cv2.approxPolyDP(con,0.01*perimeter,True)
        # print (len(approx))
        # if len(approx) == 4:
        #     contours_allowed.append(con)

    # cv2.drawContours(img_color,contours_allowed,-1,(255,255,0),3)
    # cv2.imshow("ss",img_color)
    # cv2.waitKey(0)



img = cv2.imread("/home/jit2307/Downloads/test3.jpeg")
# img = image_resize(img,height=1080)
img_orig = img.copy()

kernel = np.ones((7,7),np.float32)/25
# img = cv2.GaussianBlur(img,(3,3),0)
# img = cv2.GaussianBlur(img,(15,15),0)
# img = cv2.medianBlur(img,3)

img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img = img[:,:,0]
img = cv2.bitwise_not(img)
# cv2.imshow("ss",img)
# cv2.waitKey(0)
# ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
img = cv2.bitwise_not(img)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("ss",img)
cv2.waitKey(0)
kernel = np.ones((3,3),np.float32)/9

# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
img = cv2.Canny(img,100,200)
cv2.imshow("ss",img)
cv2.waitKey(0)

# kernel = kernel = np.ones((3,3),np.float32)/9
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# mask = getSegMask(img_orig)
# print(img.shape)
# img = cv2.bitwise_and(img, img, mask = mask)
detectShapes(img,img_orig) 


# cv2.imshow("sss",img)
# cv2.waitKey(0)



