import cv2
import numpy as np
img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.png")
diff = cv2.absdiff(img1, img2)
cv2.imshow('diff',diff)
cv2.waitKey(5000)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
cv2.imshow('mask',mask)
cv2.waitKey(5000)
th = 1
imask =  mask>th
canvas = np.zeros_like(img2, np.uint8)
canvas[imask] = img2[imask]
cv2.imshow('canvas',canvas)
cv2.waitKey(5000)
cv2.imwrite("result.png", canvas)