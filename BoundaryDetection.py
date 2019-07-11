import cv2
import numpy as np
import math

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



def detectBoundary(thresh,img_color):

    new_img = np.zeros_like(thresh)
    orig = img_color.copy()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(orig,contours,0,(156,34,255),2)
 

    contours_hull = list(map(cv2.convexHull,contours))
    contour_area = list(map(cv2.contourArea,contours_hull))
    max_area_cont = np.argmax(contour_area)
    # print (max_area_cont)
    # print(contour_area)
    
    for ii,con in enumerate(contours) :
        if (ii == max_area_cont):
            continue
        # print (hierarchy[count])
        cv2.fillPoly(thresh, pts =[contours_hull[ii]], color=(255,255,255))
        

    return thresh

def getSegMask(img):

    
    img_orig = img.copy()

    kernel = np.ones((5,5),np.float32)/25
    img = cv2.GaussianBlur(img,(31,31),0)
    img = cv2.GaussianBlur(img,(15,15),0)
 


    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img[:,:,0]
    img = cv2.bitwise_not(img)


    ret,img=cv2.threshold(img,170,255,cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)

    # img = cv2.Canny(img,100,200)
    # cv2.imshow("mass",img)
    # cv2.waitKey(0)

    return detectBoundary(img,img_orig) 




# img = cv2.imread("/home/jbmai/Downloads/IMG_20181228_093937.jpg")
# img = image_resize(img,height=1080)
# mask = getSegMask(img)
# print (mask.shape)
# print(img.shape)
# masked_img = cv2.bitwise_and(img,img,mask = mask)

# cv2.imshow("ss",masked_img)
# cv2.waitKey(0)


