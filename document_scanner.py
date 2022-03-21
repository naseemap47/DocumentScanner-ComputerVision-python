import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny_img = cv2.Canny(blur_img, 50, 50)

    cv2.imshow("Image", img)
    cv2.imshow("Gray Image", gray_img)
    cv2.imshow("Blur Image", blur_img)
    cv2.imshow("Canny Image", canny_img)
    cv2.waitKey(1)