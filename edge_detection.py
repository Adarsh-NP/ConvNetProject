import numpy as np
import cv2

src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks12\\campus.jpg"

og_image = cv2.imread(src_path, cv2.IMREAD_COLOR)

#we transform the original image to grayscale

gray_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)

#creating the laplacian kernel
kernel = np.array([[0,1,0], [1, -4 , 1], [0,1,0]])
edge = cv2.filter2D(gray_image, -1, kernel)

#instead of manually doing things, we can use builtin edge detection tool of openCV
# edge = cv2.Laplacian(gray_image, -1)

#sharpening the image
sharpen_kernel = np.array([[0,-1,0], [-1, 7, -1], [0, -1, 0]])
sharp_image = cv2.filter2D(og_image, -1, sharpen_kernel)

cv2.imshow('Original_Image', og_image)
cv2.imshow('edge detection', edge)
cv2.imshow('Sharpened Image', sharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



