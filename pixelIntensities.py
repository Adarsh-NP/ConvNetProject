import cv2
import numpy as np
#opening the capture
src_path = r"C:\\Users\\asus\Desktop\\CNNProject\\Resources\\Weeks12\\camus.jpg"

image = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

#as we move from 0 to 255 we go from darker to brighter shade
print(image.shape)  #shows the shape of capture vector as row x col
print(image)        #shows the pixel intensity in grayscale (0-255)

#opening the capture window
cv2.imshow('Computer Vision', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


