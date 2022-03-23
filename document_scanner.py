import cv2
from my_utils import preProcessing, getContour, getWarp

###############################
width_img = 640
height_img = 480
width_frame = 640
height_frame = 480
###############################

doc_photo = cv2.imread('document sample.jpeg')


# cap = cv2.VideoCapture(0)
# cap.set(3, width_frame)
# cap.set(4, height_frame)

# while True:
    # success, img = cap.read()
cv2.resize(doc_photo, (width_img, height_img))
canny_img = preProcessing(doc_photo)
large_approx = getContour(canny_img, doc_photo)
print(large_approx)
doc_img = getWarp(doc_photo, large_approx, width_img, height_img)
    # if len(large_approx) != 0:
    #     print(len(large_approx))
    # cv2.imshow("Image", img)
cv2.imshow("Canny Image", canny_img)
cv2.imshow("Output Image", doc_img)
cv2.waitKey(0)