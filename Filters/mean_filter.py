import numpy as np

def mean_filter(size):
    # creates a mean filter
    fltr = np.ones((size, size), dtype=np.float32)
    fltr = np.divide(fltr, size ** 2)
    return fltr