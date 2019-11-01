import numpy as np

def sharpening_filter(size, value):
    # error handling for shapening filter
    if (size % 2 == 0):
        print("Invalid Filter Size")
        return

    # creates the first filter that has 2(value) in center and zero elsewhere
    fltr1 = np.zeros((size, size), dtype=np.float32)
    fltr1[int(size / 2), int(size / 2)] = value

    # creates the mean filter
    fltr2 = np.ones((size, size), dtype=np.float32)
    fltr2 = np.divide(fltr2, size ** 2)

    # creates the shapening filter by subtracting mean from the first one
    fltr = fltr1 - fltr2

    return fltr
