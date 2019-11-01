import cv2
import numpy as np

def _hough_matrix(edge_image):
    """
    Computers the hough matrix in p and theta for an edge image.
    :param edge_image: image with edges only. output from canny filter
    :return: returns the accumulator matrix.
    """
    # input image dimensions
    input_height = edge_image.shape[0]
    input_width = edge_image.shape[1]

    # theta from 0 to 180 with steps of one unit
    theta = np.arange(0, 360, 1)

    # maximum value p is the square root of sum of squared dimensions
    p_length =  int(round(np.sqrt(input_height ** 2 + input_width ** 2)))
    theta_length = theta.shape[0]

    # accumulator matrix
    accumulator = np.zeros((p_length, theta_length), dtype=np.int16)

    # pixels and their coordinates where there is an edge (as detected by the canny edge detector)
    edge_pixels = np.where(edge_image == 255)
    edge_coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    # iterates over each non-zero point for every theta and populates the accumulator matrix
    for cord in edge_coordinates:
        for t in theta:
            p = int(round(cord[1] * np.cos(np.radians(t)) + cord[0] * np.sin(np.radians(t))))
            accumulator[p, t] = accumulator[p, t] + 1

    return accumulator


def _local_maxima(array):
    """
    Zeros any element in the array whose any immediate neighbour is bigger than the element itself.
    :param array: 2d array
    :return: locally maximized 2d array
    """
    image_height = array.shape[0]
    image_width = array.shape[1]
    output_image = np.zeros((image_height, image_width), dtype=np.int16)

    # iterates over all pixels except the ones on the edges.
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            # all neighbouring pixels
            top_left_pixel = array[i - 1, j - 1]
            top_pixel = array[i - 1, j]
            top_right_pixel = array[i - 1, j + 1]
            left_pixel = array[i, j - 1]
            right_pixel = array[i, j + 1]
            bottom_left_pixel = array[i + 1, j - 1]
            bottom__pixel = array[i + 1, j]
            bottom_right_pixel = array[i + 1, j + 1]

            # retaining value if value of pixel higher than neighbours otherwise setting to zero
            if (array[i, j] >= top_left_pixel) and (array[i, j] >= top_pixel) and (array[i, j] >= top_right_pixel) and \
                    (array[i, j] >= left_pixel) and (array[i, j] >= right_pixel) and (array[i, j] >= bottom_left_pixel) and \
                    (array[i, j] >= bottom__pixel) and (array[i, j] >= bottom_right_pixel):
                output_image[i, j] = array[i, j]
            else:
                output_image[i, j] = 0

    return output_image


def _thresholding(array, ratio):
    """
    Returns a list of points whose value is greater than the ratio to maximum
    :param array: any 2d array
    :param ratio: the ratio to max, above which we want to keep values
    :return: list of points that are above the given ratio
    """
    # max value of the array
    max_value = np.max(array)

    # threshold value using max value and ratio provided
    threshold = ratio * max_value
    print("threshold = {}".format(threshold))

    # points above threshold
    lines = np.where(array > threshold)

    # making list of coordinates for the points found
    coordinates = list(zip(lines[0], lines[1]))

    return coordinates


def _add_lines(image, coordinates):
    """
    Add lines to the image using coordiantes in parameter space (p and theta)
    :param image: any image
    :param coordinates: list of points(tuple) where each tuple represents (p, theta)
    :return: image with points drawn as line in image space
    """
    # converts points back into x and y form and draws them. formula found on the internet.
    for point in coordinates:
        a = np.cos(np.radians(point[1]))
        b = np.sin(np.radians(point[1]))
        x0 = a * point[0]
        y0 = b * point[0]
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return image


def hough_lines(edges, image, ratio=0.7):
    """
    Identifies lines on an image using edge image.
    :param edges: image whoses edges have been found. Using Canny or any other edge detector
    :param ratio: ratio for thresholding
    :return: image on which lines have been drawn and accumulator matrix
    """
    # creates the accumulator hough matrix
    accumulator = _hough_matrix(edges)

    # removes points that are not locally maximun
    accumulator_local_max = _local_maxima(accumulator)

    # removes points below the threshold
    coordinates = _thresholding(accumulator_local_max, ratio)

    # draw lines on the image
    lined_image = _add_lines(image, coordinates)

    return lined_image