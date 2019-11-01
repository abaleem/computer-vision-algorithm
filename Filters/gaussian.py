import numpy as np

def gaussian_filter(size, sigma):
    # creates the x and y values for the filter
    linspace_size = int(size / 2)
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))

    # creates the gaussian filter using formula in book
    fltr = np.zeros((size, size), dtype=np.float32)
    fltr = (1 / (2 * (22 / 7) * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    # renormalizing because of computing accuracy
    fltr /= np.sum(fltr)

    return fltr