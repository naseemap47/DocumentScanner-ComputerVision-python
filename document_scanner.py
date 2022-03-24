import cv2
from my_utils import preProcessing, getContour, getWarp, reOrder

###############################
# Switches
image_scanner = True
camera_scanner = False

# Size Controllers
width_img = 640
height_img = 480
width_frame = 640
height_frame = 480
###############################

# Image Scanner
if image_scanner:
    doc_img = cv2.imread('Document Images/sample1.jpeg')
    cv2.resize(doc_img, (width_img, height_img))
    doc_img_copy = doc_img.copy()
    cv2.resize(doc_img_copy, (width_img, height_img))
    canny_img = preProcessing(doc_img)
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
    cv2.waitKey(0)


# Web-cam or Camera Scanner
if camera_scanner:
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