import cv2

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
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny_img = cv2.Canny(blur_img, 50, 50)
    dial_img = cv2.dilate(canny_img, (5, 5), iterations=2)
    erode_img = cv2.erode(dial_img, (5, 5), iterations=1)

    cv2.imshow("Image", img)
    cv2.imshow("Gray Image", gray_img)
    cv2.imshow("Blur Image", blur_img)
    cv2.imshow("Canny Image", canny_img)
    cv2.imshow("Dilate Image", dial_img)
    cv2.imshow("Erode Image", erode_img)
    cv2.waitKey(1)