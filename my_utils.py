import cv2
import numpy as np


def preProcessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny_img = cv2.Canny(blur_img, 50, 50)

    # Canny image is thin, if there is any shadow like anything it will NOT detect properly
    # So we are using dilate and erode function to solve this problem
    dial_img = cv2.dilate(canny_img, (5, 5), iterations=2)
    erode_img = cv2.erode(dial_img, (5, 5), iterations=1)
    return erode_img


def getContour(img, draw_img):
    max_area = 0
    biggest_approx = np.array([])
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            cv2.drawContours(draw_img, cnt, -1, (0, 255, 0), 5)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            # print(len(approx))
            if area > max_area and len(approx) == 4:
                biggest_approx = approx
                max_area = area
    return biggest_approx