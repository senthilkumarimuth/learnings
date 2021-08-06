# importing openCV library
import cv2

im1 = cv2.imread(r'input_data\2058 commented.jpg')
im2 = cv2.imread(r'input_data\2058 Revised.jpg')

im1_g = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_g = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(im1_g, None)
kp2,des2 = orb.detectAndCompute(im2_g, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
n_bf_match = bf.match(des1,des2)
n_bf_match = sorted(n_bf_match, key=lambda x: x.distance)

output_image = cv2.drawMatches(im1_g, kp1, im2_g, kp2, n_bf_match[:100], None, flags=2)
cv2.imwrite('Output image.png', output_image)

#Check better detector i.e BEBLID now available which is better than 14% compared with ORB