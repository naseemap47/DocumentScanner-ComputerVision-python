import cv2
from my_utils import preProcessing, getContour, getWarp, reOrder

###############################
width_img = 640
height_img = 480
width_frame = 640
height_frame = 480
###############################

# doc_photo = cv2.imread('document sample.jpeg')

cap = cv2.VideoCapture(0)
cap.set(3, width_frame)
cap.set(4, height_frame)

while True:
    success, img = cap.read()
    cv2.resize(img, (width_img, height_img))
    canny_img = preProcessing(img)
    large_approx = getContour(canny_img, img)
    # print(large_approx)
    if len(large_approx) != 0:
        # print(len(large_approx))
        ordered_points = reOrder(large_approx)
        doc_img = getWarp(img, ordered_points, width_img, height_img)
    else:
        doc_img = img
    cv2.imshow("Image", img)
    cv2.imshow("Canny Image", canny_img)
    cv2.imshow("Output Image", doc_img)
    cv2.waitKey(1)