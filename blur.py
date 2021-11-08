import numpy as np
import cv2
import os

src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks12\\campus.jpg"

og_image = cv2.imread(src_path, cv2.IMREAD_COLOR)

#as we move from 0 to 255 we go from darker to brighter shade
print(og_image.shape)  #shows the shape of capture vector as row x col
print(og_image)        #shows the pixel intensity in grayscale (0-255)

#divide by 25 to normalize the values because we had 25 items
kernel = np.ones((5, 5))/25

#-1 means the blur capture depth is same as og capture depth
blur_image = cv2.filter2D(og_image, -1, kernel)

#gaussian blur is used to reduce the noise in the input

cv2.imshow('Original_Image', og_image)

cv2.imshow('Blurred_Image', blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



