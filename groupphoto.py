import numpy as np
import cv2
import matplotlib.pyplot as plt

src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks56\\haarcascade_frontalface_alt2.xml"
cascade_classifier = cv2.CascadeClassifier(src_path)
photo = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks56\\groupphoto.jpg"
image = cv2.imread(photo)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap="gray")

detected_face = cascade_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=[30,30])

print(detected_face)

for (x, y, width, height) in detected_face:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 5)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()