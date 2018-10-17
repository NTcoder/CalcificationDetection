import cv2
import numpy as np

img = cv2.imread('kmeans.jpg',0)
kernel = np.ones((5,5),np.uint8)
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#erosion = cv2.erode(img,kernel,iterations = 10)
cv2.imshow('kmeans',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()