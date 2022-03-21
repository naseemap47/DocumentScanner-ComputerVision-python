import cv2
from my_utils import preProcessing, getContour

###############################
width_img = 640
height_img = 480
width_frame = 640
height_frame = 480
###############################

cap = cv2.VideoCapture(0)
cap.set(3, width_frame)
cap.set(4, height_frame)

while True:
    success, img = cap.read()
    cv2.resize(img, (width_img, height_img))
    canny_img = preProcessing(img)
    getContour(canny_img)
    cv2.imshow("Image", canny_img)
    cv2.waitKey(1)