import  cv2

src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks56\\haarcascade_frontalface_alt2.xml"
cascade_classifier = cv2.CascadeClassifier(src_path)

#0 means we are going to use the default camera
video_capture = cv2.VideoCapture(0)

#index-> width of capturing window,  width
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    ret, img = video_capture.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=[30,30])

    for (x, y, width, height) in detected_faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 5)

    cv2.imshow('Real-time face detection', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
