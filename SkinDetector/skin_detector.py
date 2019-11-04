import os
import cv2
import numpy as np


def skin_detector_train(imageFolder="./Images/SkinTraining"):
    # creating a 2d histogram. got maximum dimensions from cv2 change color space documentation.
    histogram = np.zeros((180,256))

    for imageName in os.listdir(imageFolder):           
        # reads image as BGR
        image = cv2.imread(os.path.join(imageFolder,imageName))
        # coverts image to HSV
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # for each pixel finds the hue and saturation and increments that value in histogram by 1.
        if image is not None:    
            for h in range(imageHSV.shape[0]):
                for w in range(imageHSV.shape[1]):                
                    hue = imageHSV[h,w,0]
                    saturation = imageHSV[h,w,1]
                    histogram[hue, saturation] += 1

    # normalizes the array using a maximum value in the array
    histogram = histogram / np.max(histogram)

    return histogram



def skin_detector_test(testImageFilepath):
    # reads the test image in BGR
    testImage = cv2.imread(testImageFilepath)
    # converts the test image to HSV color space
    testImageHSV = cv2.cvtColor(testImage, cv2.COLOR_BGR2HSV)

    # threshold which works the best for the given training data. found by trail and error.
    threshold = 0.001

    # deepcopies the input image to output image for we dont alter the input image.
    outputImage = np.copy(testImage)

    # for each pixel checks if it is greater the threshold. if not changes that pixel to black.
    for h in range(testImageHSV.shape[0]):
        for w in range(testImageHSV.shape[1]):
            testHue = testImageHSV[h,w,0]
            testSaturation = testImageHSV[h,w,1]
            
            if(histogram[testHue, testSaturation] < threshold):
                outputImage[h,w,:] = 0

    return outputImage