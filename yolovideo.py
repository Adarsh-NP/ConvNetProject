import cv2
import numpy as np
import matplotlib.pyplot as plt

src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks78\\"
configfile = src_path + "yolov3.cfg"
weights = src_path + "yolov3.weights"


# threshold
THRESHOLD = 0.5
SUPRESSION_THRESHOLD = 0.5
YOLO_IMG_SIZE = 320

# 0: person, 2:car, 5:bus
classes = ['car', 'person', 'bus']


def find_objects(model_outputs):
    box_locations = []
    class_ids = []
    confidence_val = []

    for output in model_outputs:
        for prediction in output:
            class_probs = prediction[5:]
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2] * YOLO_IMG_SIZE), int(prediction[3] * YOLO_IMG_SIZE)
                x, y = int(prediction[0] * YOLO_IMG_SIZE - w / 2), int(prediction[1] * YOLO_IMG_SIZE - h / 2)
                box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_val.append(float(confidence))

    box_indices_to_keep = cv2.dnn.NMSBoxes(box_locations, confidence_val, THRESHOLD, SUPRESSION_THRESHOLD)
    return box_indices_to_keep, box_locations, class_ids, confidence_val


def show_detected_image(img, box_id, boxes, classid, confval, wratio, hratio):
    for index in box_id:
        bounding_box = boxes[index[0]]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])

        # transform the location and coordinates
        x = int(x * wratio)
        y = int(y * hratio)
        w = int(w * wratio)
        h = int(h * hratio)

        if classid[index[0]] == 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), ( 255, 0,0), 2)
            classwithconf = 'CAR' + str(int(confval[index[0]] * 100)) + '%'
            # 0.5 is scale factor
            cv2.putText(img, classwithconf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0, 0), 1)

        if classid[index[0]] == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), ( 255,0, 0), 2)
            classwithconf = 'PERSON' + str(int(confval[index[0]] * 100)) + '%'
            # 0.5 is scale factor
            cv2.putText(img, classwithconf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,0, 0), 1)

capture = cv2.VideoCapture(src_path + "yolo_test.mp4")
nnetwork = cv2.dnn.readNetFromDarknet(configfile, weights)

# define if we want to use GPU
nnetwork.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
nnetwork.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    frame_grabbed, frame = capture.read()
    if not frame_grabbed:
        break
    # transform the capture to blob (binary large objects)
    orwidth, orheight = frame.shape[1], frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMG_SIZE, YOLO_IMG_SIZE), True,
                                 crop=False)  # tuple is the size of op img, True is for conversion of RGB to BGR
    nnetwork.setInput(blob)

    # Yolo network has 3 output layers with indices starting from 1
    layerNames = nnetwork.getLayerNames()
    outputlayers = nnetwork.getUnconnectedOutLayers()
    outputnames = [layerNames[index[0] - 1] for index in outputlayers]

    outputs = nnetwork.forward(outputnames)

    # there are 300 bounding boxes and 85 is the prediction vector, 80 class probabilities
    print(outputs[0].shape)
    # print(layer)

    predicted_objects, box_loc, class_id, conf_val = find_objects(outputs)
    show_detected_image(frame, predicted_objects, box_loc, class_id, conf_val, orwidth / YOLO_IMG_SIZE,
                        orheight / YOLO_IMG_SIZE)

    cv2.imshow('YOLO_ALGO', frame)
    cv2.waitKey(5)

capture.release()
cv2.destroyAllWindows()
