import argparse

import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, njit

HISTOGRAM_ALPHA = 0.5
DISCRETE_VALUE_NUM = 256
COLORS = ["red", "green", "blue"]


@njit(cache=True)
def get_hist(image):
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    num_channels = image.shape[2]
    hist = np.zeros((num_channels, DISCRETE_VALUE_NUM), dtype=np.int64)
    for i in range(height):
        for j in range(width):
            for c in range(num_channels):
                hist[c][image[i, j, c]] += 1
    return hist


@njit(cache=True)
<<<<<<< HEAD
def _one_dim_hist_normalize(hist):
=======
def _one_dim_normalize_hist(hist):
>>>>>>> 490cb344929d6037d90f6165626636ac05f22ba9
    assert hist.ndim == 1
    hist_norm = np.divide(hist, hist.sum())
    return hist_norm


@njit(cache=True)
def _one_dim_hist_equlize(hist):
    assert hist.ndim == 1
    assert hist.shape[0] == DISCRETE_VALUE_NUM
<<<<<<< HEAD
    norm_hist = _one_dim_hist_normalize(hist.astype(np.uint8))
=======
    norm_hist = _one_dim_normalize_hist(hist.astype(np.uint8))
>>>>>>> 490cb344929d6037d90f6165626636ac05f22ba9
    eq_histmap = np.zeros_like(hist, dtype=np.uint8)
    _sum = 0
    for i in range(DISCRETE_VALUE_NUM):
        _sum += norm_hist[i]
        eq_histmap[i] = int((DISCRETE_VALUE_NUM - 1) * _sum + 0.5)

    return eq_histmap


@njit(cache=True)
def _one_dim_hist_specify(source_hist, target_hist):
    assert source_hist.ndim == target_hist.ndim == 1
    eq_source_histmap = _one_dim_hist_equlize(source_hist)
    eq_target_histmap = _one_dim_hist_equlize(target_hist)
    sp_histmap = np.zeros_like(source_hist, dtype=np.int16)
    small_nearest = 0
    for i in range(DISCRETE_VALUE_NUM):
        eq_i = eq_source_histmap[i]
        reverse_eq_i = np.where(eq_target_histmap == eq_i, 1, 0)
        num_nozero = len(np.nonzero(reverse_eq_i)[0])
        if int(num_nozero) != 0:
            small_nearest = sp_histmap[i] = int(
                np.sum(np.multiply(eq_target_histmap, reverse_eq_i)) / num_nozero
            )
        else:
            sp_histmap[i] = small_nearest
    return sp_histmap


@njit(cache=True)
def image_match_histmap(image, histmap):
    assert histmap.ndim == 2
    assert image.ndim == 3
    assert image.shape[2] == histmap.shape[0]
    height = image.shape[0]
    width = image.shape[1]
    num_channels = image.shape[2]
    result = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            for c in range(num_channels):
                result[i, j, c] = histmap[c][image[i, j, c]]
    return result


@njit(cache=True)
def hist_equlize(hist):
    assert hist.ndim == 2
    num_channels = hist.shape[0]
    eq_histmap = np.zeros_like(hist, dtype=np.uint8)
    for i in range(num_channels):
        eq_histmap[i] = _one_dim_hist_equlize(hist[i]).astype(np.uint8)
    return eq_histmap


@njit(cache=True)
def hist_specify(source_hist, target_hist):
    assert source_hist.ndim == target_hist.ndim == 2
    assert source_hist.shape[0] == target_hist.shape[0]
    num_channels = source_hist.shape[0]
    sp_histmap = np.zeros_like(source_hist, dtype=np.uint8)
    for i in range(num_channels):
        sp_histmap[i] = _one_dim_hist_specify(source_hist[i], target_hist[i]).astype(
            np.uint8
        )
    return sp_histmap


def image_hist_equlize(image):
    hist_map = hist_equlize(get_hist(image))
    result = image_match_histmap(image, hist_map)
    return result


def image_hist_specifiy(source_image, target_image):
    hist_map = hist_specify(get_hist(source_image), get_hist(target_image))
    result = image_match_histmap(source_image, hist_map)
    return result


def plot_hist(hist, title):
    assert hist.ndim == 2
    assert hist.shape[1] == DISCRETE_VALUE_NUM
    num_channels = hist.shape[0]
    x_coord = np.arange(0, DISCRETE_VALUE_NUM)
    params = {"alpha": HISTOGRAM_ALPHA, "width": 1}
    plt.figure(title)
    for i in range(num_channels):
        plt.bar(
            x_coord,
            hist[i],
            **params,
            color=COLORS[i],
            label="Channel: {}".format(i + 1)
        )
    plt.legend()


def plot_image_hist(image, title):
    assert image.ndim == 3
    num_channels = image.shape[2]
    params = {"alpha": HISTOGRAM_ALPHA, "bins": DISCRETE_VALUE_NUM}
    plt.figure(title)
    for i in range(num_channels):
        plt.hist(
            image[:, :, i].flatten(),
            **params,
            color=COLORS[i],
            label="Channel: {}".format(i + 1)
        )
    plt.legend()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=str)
    parser.add_argument("target_path", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_image = imgplt.imread(args.source_path)
    target_image = imgplt.imread(args.target_path)

    plt.figure("Source Image")
    plt.imshow(source_image)
    plt.axis("off")
    plot_image_hist(source_image, "Source Histogram")
    plt.figure("Target Image")
    plt.imshow(target_image)
    plt.axis("off")
    plot_image_hist(target_image, "Target Histogram")

    result_image = image_hist_specifiy(source_image, target_image)

    plt.figure("Result Image")
    plt.imshow(result_image)
    plt.axis("off")
    plot_hist(get_hist(result_image), "Result Histogram")

    plt.show()


if __name__ == "__main__":
    main()
