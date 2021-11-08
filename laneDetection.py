import cv2
import numpy
import numpy as np

#video is several frames one by one
src_path = r"C:\\Users\\asus\\Desktop\\CNNProject\\Resources\\Weeks34\\lane_detection_video.mp4"

video = cv2.VideoCapture(src_path)

def drawLines(image, lines):
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are x and y coordinates for start and end
    for line in lines:
        for x1, x2, y1, y2 in line:
            cv2.line(lines_image, (x1, x2), (y1, y2), (0, 255, 0), thickness=2)

    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines

def getImageofInterest(image, regionofInterest):
    mask = numpy.zeros_like(image)
    cv2.fillPoly(mask, regionofInterest, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def getDetectedLane(image):
    (height, width) = (image.shape[0], image.shape[1])

    #turn the capture in gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #edge det kernel
    canny_image = cv2.Canny(gray_image, 100, 120)
    regionOfInterest = [(0, height), (width/2, height*0.6), (width, height)]
    croppedImage = getImageofInterest(canny_image, np.array([regionOfInterest], np.int32))

    lines = cv2.HoughLinesP(croppedImage, rho=2, theta=np.pi/180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100 )
    imagewithLines = drawLines(image, lines)
    return imagewithLines

while video.isOpened():
    is_grabbed, frame = video.read()
    #end of the while loop
    if not is_grabbed:
        break
    #else
    frame = getDetectedLane(frame)

    cv2.imshow('Lane detection video', frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()




