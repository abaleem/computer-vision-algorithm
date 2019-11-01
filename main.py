import cv2
from EdgeDetector import canny_detector
from HoughTransform import hough_lines


if __name__ == '__main__':

    # reading test image
    image = cv2.imread("./Images/Input/test2.bmp")

    # generating edge image using canny detector
    edges = canny_detector(image)

    # detected lines using hough line transform
    lined_image = hough_lines(edges, image)

    # saving image
    cv2.imwrite("./Images/Hough/test2_hough.png", lined_image)