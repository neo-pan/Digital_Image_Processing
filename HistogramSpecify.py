import argparse

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from numba import jit, njit

HISTOGRAM_ALPHA = 0.5
DISCRETE_VALUE_NUM = 256
COLORS = ["red", "green", "blue", "black"]


@njit(cache=True)
def get_hist(image):
    """获取图像灰度值出现频数的统计数据
    Args:
        image:  numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                数组中数据代表相应位置上的灰度值
    Returns:
        hist:   numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                数组中数据代表相应灰度值在图像中出现的频数
    """
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
def _one_dim_hist_normalize(hist):
    """对单个通道的直方图数据进行归一化
    Args: 
        hist:       numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在图像中出现的频数
    Returns:
        hist_norm:  numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在图像中出现的频率
    """
    assert hist.ndim == 1
    hist_norm = np.divide(hist, hist.sum())

    return hist_norm


@njit(cache=True)
def _one_dim_hist_prob_sum(hist):
    """对单个通道的直方图数据进行累计求和
    Args: 
        hist:       numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在图像中出现的频数
    Returns:
        hist_sum:  numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在图像中出现的累计频率
    """
    assert hist.ndim == 1
    hist_norm = np.divide(hist, hist.sum())
    hist_sum = np.cumsum(hist_norm)

    return hist_sum


@njit(cache=True)
def _one_dim_hist_equlize(hist):
    """对单个通道的直方图数据进行均衡化
    Args: 
        hist:       numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在图像中出现的频数
    Returns:
        hist_map:  numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在均衡化后对应的灰度值
    """
    assert hist.ndim == 1
    assert hist.shape[0] == DISCRETE_VALUE_NUM
    hist_sum = _one_dim_hist_prob_sum(hist)
    eq_histmap = ((DISCRETE_VALUE_NUM - 1) * hist_sum + 0.5).astype(np.uint8)

    return eq_histmap


@njit(cache=True)
def _one_dim_hist_specify(source_hist, target_hist):
    """对单个通道的直方图数据进行规定化
    Args: 
        source_hist:numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在源图像中出现的频数
        target_hist:numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在目标图像中出现的频数
    Returns:
        hist_map:   numpy数组, shape为(256, )-(8位离散灰度值总数, )，
                    数组中数据代表相应灰度值在规定化后对应的灰度值
    """
    assert source_hist.ndim == target_hist.ndim == 1
    source_hist_sum = _one_dim_hist_prob_sum(source_hist)
    target_hist_sum = _one_dim_hist_prob_sum(target_hist)
    sp_histmap = np.zeros_like(source_hist, dtype=np.uint8)
    # 计算源图像与目标图像任意灰度值累计概率的差值, hist_sum_map.shape=(256, 256)
    hist_sum_map = np.abs(
        np.subtract(source_hist_sum.reshape(-1, 1), target_hist_sum.reshape(-1, 1).T)
    )
    # 从目标图像灰度值中找到累计概率差值最小的, 作为映射目标
    for i in range(DISCRETE_VALUE_NUM):
        sp_histmap[i] = np.argmin(hist_sum_map[i])

    return sp_histmap


@njit(cache=True)
def image_match_histmap(image, histmap):
    """使图像根据特定的灰度值映射进行匹配
    Args:
        image:      numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                    数组中数据代表相应位置上的灰度值
        hist_map:   numpy数组, shape为(C, 256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在映射关系中对应的灰度值
    Returns：
        result:     numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                    数组中数据代表变换后新图像相应位置上的灰度值
    """
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
    """对多通道的直方图数据进行规定化
    Args: 
        hist:       numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在图像中出现的频数
    Returns:
        hist_map:  numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在均衡化后对应的灰度值
    """
    assert hist.ndim == 2
    num_channels = hist.shape[0]
    eq_histmap = np.zeros_like(hist, dtype=np.uint8)
    for i in range(num_channels):
        eq_histmap[i] = _one_dim_hist_equlize(hist[i])

    return eq_histmap


@njit(cache=True)
def hist_specify(source_hist, target_hist):
    """对多通道的直方图数据进行规定化
    Args: 
        source_hist:numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在源图像中出现的频数
        target_hist:numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在目标图像中出现的频数
    Returns:
        hist_map:  numpy数组, shape为(C,256)-(通道数, 8位离散灰度值总数)，
                    数组中数据代表相应灰度值在均衡化后对应的灰度值
    """
    assert source_hist.ndim == target_hist.ndim == 2
    assert source_hist.shape[0] == target_hist.shape[0]
    num_channels = source_hist.shape[0]
    sp_histmap = np.zeros_like(source_hist, dtype=np.uint8)
    for i in range(num_channels):
        sp_histmap[i] = _one_dim_hist_specify(source_hist[i], target_hist[i])

    return sp_histmap


def image_hist_equlize(image):
    """对图像进行直方图均衡化
    Args:
        image:      numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                    数组中数据代表相应位置上的灰度值
    Returns：
        result:     numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                    数组中数据代表均衡化后新图像相应位置上的灰度值
    """
    hist_map = hist_equlize(get_hist(image))
    result = image_match_histmap(image, hist_map)

    return result


def image_hist_specifiy(source_image, target_image):
    """对图像进行直方图规定化
    Args:
        source_image:   numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                        数组中数据代表源图像相应位置上的灰度值
        target_image:   numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                        数组中数据代表目标图像相应位置上的灰度值
    Returns：
        result:         numpy数组, shape为(H,W,C)-(长, 宽, 通道数), 
                        数组中数据代表规定化后新图像相应位置上的灰度值
    """
    hist_map = hist_specify(get_hist(source_image), get_hist(target_image))
    result = image_match_histmap(source_image, hist_map)

    return result


def plot_hist(hist, title):
    """根据统计信息绘制直方图
    """
    assert hist.ndim == 2
    assert hist.shape[1] == DISCRETE_VALUE_NUM
    num_channels = hist.shape[0]
    x_coord = np.arange(0, DISCRETE_VALUE_NUM)
    params = {"alpha": HISTOGRAM_ALPHA, "width": 1}
    plt.figure(title)
    for i in range(num_channels):
        plt.bar(
            x_coord,
            _one_dim_hist_normalize(hist[i]),
            **params,
            color=COLORS[i],
            label="Channel: {}".format(i + 1)
        )
    plt.legend()


def image_plot_hist(image, title):
    """根据图像原始信息绘制直方图
    """
    assert image.ndim == 3
    num_channels = image.shape[2]
    params = {"alpha": HISTOGRAM_ALPHA, "bins": DISCRETE_VALUE_NUM, "density": True}
    plt.figure(title)
    for i in range(num_channels):
        plt.hist(
            image[:, :, i].flatten(),
            **params,
            color=COLORS[i],
            label="Channel: {}".format(i + 1)
        )
    plt.legend()


def plot_image_and_hist(image, title):
    """绘制图像及其直方图
    """
    plt.figure("{} Image".format(title))
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")
    plot_hist(get_hist(image), "{} Histogram".format(title))


def parse_args():
    parser = argparse.ArgumentParser(description="将给定图像进行直方图均衡化或正规化")
    parser.add_argument(
        "task",
        type=str,
        choices=["equalize", "specify"],
        help="指定处理任务, 直方图均衡化: equalize, 或直方图规定化: specify",
    )
    parser.add_argument("-s", "--source_path", type=str, default=None, help="源图像地址")
    parser.add_argument("-t", "--target_path", type=str, default=None, help="目标图像地址")

    args = parser.parse_args()
    return args


def image_pre_process(image):
    if image.ndim == 2:
        # 单通道图像数组增加一个维度以满足后续处理函数要求
        image = image[:, :, np.newaxis]
    assert image.ndim == 3
    # 将 BGR 顺序改为 RGB, 便于 matplotlib 绘图
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return image


def main():
    args = parse_args()
    if args.task == "equalize":
        assert args.source_path is not None
        source_image = image_pre_process(
            cv2.imread(args.source_path, cv2.IMREAD_UNCHANGED)
        )
        plot_image_and_hist(source_image, "Source")
        result_image = image_hist_equlize(source_image)

    elif args.task == "specify":
        assert args.source_path is not None and args.target_path is not None
        source_image = image_pre_process(
            cv2.imread(args.source_path, cv2.IMREAD_UNCHANGED)
        )
        target_image = image_pre_process(
            cv2.imread(args.target_path, cv2.IMREAD_UNCHANGED)
        )
        assert source_image.shape[-1] == target_image.shape[-1], "源图像与目标图像通道数需要相同！"
        if source_image.shape[-1] != 1:
            print("多通道图像将分别按各个通道处理!")
        plot_image_and_hist(source_image, "Source")
        plot_image_and_hist(target_image, "Target")
        result_image = image_hist_specifiy(source_image, target_image)

    plot_image_and_hist(result_image, "{} Result".format(args.task.capitalize()))
    plt.show()


if __name__ == "__main__":
    main()
