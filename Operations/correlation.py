import numpy as np

def corr(image, kernel, padding=True):
    # checks to see if kernel is valid. where valid means its nxn matrix and n is odd
    if (kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 == 0):
        print("Invalid kernel")
        return

    #  image height and width
    image_height = image.shape[0]
    image_width = image.shape[1]

    # checks if the image  is coloured or b/w. changes shapes of b/w from 2d to 3d where channel is 1
    try:
        image_channel = image.shape[2]
    except IndexError:
        image_channel = 1
        image = image.reshape((image_height, image_width, image_channel))

    # checks to see if padding is true. if padding to false the size of output image would decrease depending upon kernel.
    if (padding == True):
        image_output = np.zeros(image.shape, dtype=np.int16)
        padding = int((kernel.shape[0] - 1) / 2)
        image_padded = np.zeros((image_height + 2 * padding, image_width + 2 * padding, image_channel), dtype=np.int16)
        image_padded[padding:image_height + padding, padding:image_width + padding, :] = image
        image = image_padded
    else:
        image_output = np.zeros((image_height - 2 * padding, image_width - 2 * padding, image_channel), dtype=np.int16)

    # performs the sum of multiplication of image unit and the kernel
    for channel in range(image_output.shape[2]):
        for height in range(image_output.shape[0]):
            for width in range(image_output.shape[1]):
                image_unit = image[height:height + padding * 2 + 1, width:width + padding * 2 + 1, channel]
                image_output[height, width, channel] = round(np.sum(np.multiply(image_unit, kernel)))

    return image_output

