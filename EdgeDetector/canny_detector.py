import numpy as np
from Filters import gaussian_filter
from Operations import conv

def _gaussian_smoothing(image, kernel_size, sigma):
    return conv(image, gaussian_filter(kernel_size, sigma))


# computes the magnitude and angle using image gradients
def _image_gradient(image):
    # defining sobel Filters
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # computes the gradient by convolving image with sobel Filters
    grad_x = conv(image, sobel_x)
    grad_y = conv(image, sobel_y)

    # computes the magnitude by taking the square root of sum of squared derivates
    mag = np.hypot(grad_x, grad_y)

    # normalized the derivates for the image range
    mag = mag / np.max(mag) * 255

    # converts the magnitude in float to int
    # mag = mag.astype(np.int16)

    # computes the tan inverse using x and y gradients
    theta = np.arctan2(grad_y, grad_x)

    return mag, theta


# applies non max supression to find locally prominent points
def _non_max_suppression(mag, theta_radian):
    image_height = mag.shape[0]
    image_width = mag.shape[1]
    output_image = np.zeros((image_height, image_width), dtype=np.int16)

    # converting to degree for ease of calculation
    theta_degree = theta_radian * 180. / np.pi

    # taking the mod would work on angles paralle to planes but not on diagonal angles.
    # adding 180 to make sure the direction remains the same.
    theta_degree[theta_degree < 0] += 180

    # iterates over all pixels except the ones on the edges.
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            # all possible groups the can detemine the next and the previous pixels
            if (0 <= theta_degree[i, j] < 22.5):
                next_pixel = mag[i, j + 1]
                previous_pixel = mag[i, j - 1]
            elif (22.5 <= theta_degree[i, j] < 67.5):
                next_pixel = mag[i + 1, j + 1]
                previous_pixel = mag[i - 1, j - 1]
            elif (67.5 <= theta_degree[i, j] < 112.5):
                next_pixel = mag[i + 1, j]
                previous_pixel = mag[i - 1, j]
            elif (112.5 <= theta_degree[i, j] < 157.5):
                next_pixel = mag[i - 1, j + 1]
                previous_pixel = mag[i + 1, j - 1]
            elif (157.5 <= theta_degree[i, j] <= 180):
                next_pixel = mag[i, j + 1]
                previous_pixel = mag[i, j - 1]
            else:
                print("theta degree exceeds defined ranges")

            # retaining value if mag highest otherwise setting to zero
            if (mag[i, j] >= previous_pixel) and (mag[i, j] >= next_pixel):
                output_image[i, j] = mag[i, j]
            else:
                output_image[i, j] = 0

    return output_image


# convert weak points to strong points if the neighbour with other strong points hence creating links
def _edge_linking(image, high_threshold_ratio):
    image_height = image.shape[0]
    image_width = image.shape[1]

    # defines the high and low threshold. high using ratio provided and low is half of high threshold
    high_threshold = np.max(image) * high_threshold_ratio
    low_threshold = high_threshold / 2

    # array where the output of threshold is stored.
    threshold_output = np.zeros((image_height, image_width), dtype=np.int16)

    # checks every pixel and assigns them a value of strong, weak and low depending on threshold
    for i in range(0, image_height):
        for j in range(0, image_width):
            if (image[i, j] >= high_threshold):
                threshold_output[i, j] = 255
            elif ((image[i, j] < high_threshold) & (image[i, j] >= low_threshold)):
                threshold_output[i, j] = 50
            else:
                threshold_output[i, j] = 0

    # checks the sides of weak pixels to see if there is a strong pixel hence connecting the strong pixels
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            # checking for weak pixels only
            if (threshold_output[i, j] == 50):
                # checks on all sides to see if there is any strong pixel
                if ((threshold_output[i, j - 1] == 255) or (threshold_output[i, j + 1] == 255)):
                    threshold_output[i, j] = 255
                elif ((threshold_output[i - 1, j] == 255) or (threshold_output[i + 1, j] == 255)):
                    threshold_output[i, j] = 255
                elif ((threshold_output[i + 1, j - 1] == 255) or (threshold_output[i - 1, j + 1] == 255)):
                    threshold_output[i, j] = 255
                elif ((threshold_output[i - 1, j - 1] == 255) or (threshold_output[i + 1, j + 1] == 255)):
                    threshold_output[i, j] = 255
                else:
                    threshold_output[i, j] = 0

    return threshold_output


def canny_detector(input_image, kernel_size=5, sigma=1, threshold=0.2):

    # Applies gaussian smoothing on the image. The numerical parameters are kernel size and sigma resectively.
    smoothed_output = _gaussian_smoothing(input_image, kernel_size, sigma)
    #cv2.imwrite("./smoothed_output.png", smoothed_output)

    # Computes the magnitude and directon using sobel fitlers
    mag, theta = _image_gradient(smoothed_output[:, :, 0])
    #cv2.imwrite("./mag_output.png", mag)

    # Applies the non max supression technqiue to improve edges
    supression_output = _non_max_suppression(mag, theta)
    #cv2.imwrite("./supression_output.png", supression_output)

    # Uses edge linking technique make edges more promiment
    edge_linking_output = _edge_linking(supression_output, threshold)
    #cv2.imwrite("./linked_output.png", edge_linking_output)

    return edge_linking_output
