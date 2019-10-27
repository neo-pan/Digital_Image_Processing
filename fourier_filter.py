import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from cv2 import cv2
from numba import njit
from pprint import pprint

def low_pass_filter(image, width=0.1):
    rows, cols = image.shape
    c_row = rows // 2
    c_col = cols // 2
    filter_width = int(width * min(rows, cols))
    low_pass_filter = np.zeros_like(image, dtype=np.int)
    low_pass_filter[
        c_row - filter_width : c_row + filter_width,
        c_col - filter_width : c_col + filter_width,
    ] = 1
    img_fft = fft.fft2(image)
    img_fft_shift = fft.fftshift(img_fft)
    img_fft = img_fft_shift * low_pass_filter
    img_fft_ishift = fft.ifftshift(img_fft)
    image = np.abs(fft.ifft2(img_fft_ishift))

    return low_pass_filter, image

if __name__ == "__main__":
    image = cv2.imread("imgs/camera.tiff", cv2.IMREAD_UNCHANGED)
    print(image.shape)
    plt.figure("Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    _filter, result = low_pass_filter(image)
    plt.figure("Low Pass")
    plt.imshow(result, cmap="gray")
    plt.axis("off")
    plt.figure("Low Pass Filter")
    plt.imshow(_filter, cmap="gray")
    plt.axis("off")

    plt.show()
