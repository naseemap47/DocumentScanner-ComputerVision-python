import cv2
from my_utils import preProcessing

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
    output_img = preProcessing(img)
    cv2.imshow("Image", output_img)
    cv2.waitKey(1)