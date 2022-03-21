import cv2

def preProcessing(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    canny_img = cv2.Canny(blur_img, 50, 50)

    # Canny image is thin, if there is any shadow like anything it will NOT detect properly
    # So we are using dilate and erode function to solve this problem
    dial_img = cv2.dilate(canny_img, (5, 5), iterations=2)
    erode_img = cv2.erode(dial_img, (5, 5), iterations=1)
    return erode_img