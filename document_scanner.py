import cv2
import os

import my_utils
from my_utils import preProcessing, getContour, getWarp, reOrder

###############################
# Switches
image_scanner = True
camera_scanner = False

# Size Controllers
width_img = 640
height_img = 380
width_frame = 640
height_frame = 480
###############################

# initializing Trackbars
my_utils.initializingTrackbars(initialVal=100)

# Image Scanner
if image_scanner:
    doc_img = cv2.imread('Document Images/sample1.jpeg')
    cv2.resize(doc_img, (width_img, height_img))
    doc_img_copy = doc_img.copy()
    cv2.resize(doc_img_copy, (width_img, height_img))
    threshold1, threshold2 = my_utils.valTrackbar()
    canny_img = preProcessing(doc_img, threshold1, threshold2)
    large_approx = getContour(canny_img, doc_img)
    # print(large_approx)
    if len(large_approx) != 0:
        # print(len(large_approx))
        ordered_points = reOrder(large_approx)
        output_img = getWarp(doc_img_copy, ordered_points, width_img, height_img)
    else:
        output_img = doc_img
        cv2.putText(
            output_img, str('Failed to Scan Document'),
            (width_img//2 - 50, height_img//2 - 30),
            cv2.FONT_HERSHEY_PLAIN, 1.5,
            (0, 0, 255), 2
        )
    cv2.imshow("Image", doc_img)
    cv2.imshow("Canny Image", canny_img)
    cv2.imshow("Output Image", output_img)

    # To Save Scanned Document
    # Press 's' key to Save
    if cv2.waitKey(0) & 0xFF == ord('s'):
        number_docs = os.listdir('Scanned Documents')
        number_docs = len(number_docs)
        cv2.imwrite('Scanned Documents/Document' + str(number_docs + 1) +'.jpeg', output_img)
        cv2.putText(output_img, 'Scan Saved', (width_img // 2 - 230, height_img // 2 + 20),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
        cv2.waitKey(300)

# Web-cam or Camera Scanner
if camera_scanner:
    cap = cv2.VideoCapture(0)
    cap.set(3, width_frame)
    cap.set(4, height_frame)

    while True:
        success, img = cap.read()
        cv2.resize(img, (width_img, height_img))
        threshold1, threshold2 = my_utils.valTrackbar()
        canny_img = preProcessing(img, threshold1, threshold2)
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

        # To Save Scanned Document
        # Press 's' key to Save
        if cv2.waitKey(1) & 0xFF == ord('s'):
            number_docs = os.listdir('Scanned Documents')
            number_docs = len(number_docs)
            cv2.imwrite('Scanned Documents/Document' + str(number_docs + 1) + '.jpeg', doc_img)
            cv2.putText(doc_img, 'Scan Saved', (width_img // 2 - 230, height_img // 2 + 20),
                        cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)
            cv2.waitKey(300)