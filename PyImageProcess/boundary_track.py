import argparse

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
from pprint import pprint
from cv2 import cv2
from numba import njit
from numba.typed import List

from region_label import generate_color_map, region_label

# 默认的前景像素值
FOREGROUND_VALUE = 255
# 顺时针8-邻域追踪顺序
# -------------
# | 5 | 6 | 7 |
# -------------
# | 4 | x | 0 |
# -------------
# | 3 | 2 | 1 |
# -------------
EIGHT_TRACK_DIRS = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
# 顺时针4-邻域追踪顺序
# -------------
# |   | 1 |   |
# -------------
# | 2 | x | 0 |
# -------------
# |   | 3 |   |
# -------------
FOUR_TRACK_DIRS = ((0, 1), (-1, 0), (0, -1), (1, 0))


@njit(cache=True)
def boundary_track(image, n=8):
    """对二值图像进行区域外边界跟踪
    """
    # 确保为二值化图像
    # assert image.dtype == np.uint8  #! numba 优化后无法正常判断
    assert image.ndim == 2
    count = np.bincount(image.ravel())
    assert count[0] + count[-1] == image.size
    # 获取区域标记
    assert n in [4, 8]
    label_mask = region_label(image, n=n)
    num_regions = np.max(label_mask)
    # 用于标记该区域的边界是否已被跟踪
    region_tracked = np.zeros((num_regions,), dtype=np.int8)

    # numba.typed.List 需要提前指定其内部元素类型
    boundary = List()
    boundaries = List()
    boundary.append((-1, -1))
    boundaries.append(list(boundary))
    boundary.clear()
    boundaries.clear()

    height = image.shape[0]
    width = image.shape[1]

    # np.pad(image, 1, "constant", constant_values=0)

    for i in range(height):
        for j in range(width):
            label = label_mask[i, j]
            if label != 0 and region_tracked[label - 1] == 0:
                P0 = (i, j)
                P1, direct = track_next(image, boundary, point=P0, direct=0, n=n)
                P_n = P1
                P_n_pre = P_n
                while not (P_n == P1 and P_n_pre == P0):
                    P_n_pre = P_n
                    P_n, direct = track_next(image, boundary, P_n_pre, direct, n)

                region_tracked[label - 1] = 1
                boundaries.append(list(boundary.copy()))
                boundary.clear()
    # 返回时去除最初加入的 (-1, -1)
    return list(boundaries)


@njit(cache=True)
def track_next(image, boundary, point, direct, n):
    """从给定点开始, 在其邻域中跟踪下一个边界点
    """
    assert n in [4, 8]
    boundary.append((point[0], point[1]))
    if n == 4:
        direct = (direct + 3) % n
    else:
        if direct % 2 == 0:
            direct = (direct + (n - 1)) % n
        else:
            direct = (direct + (n - 2)) % n

    for _ in range(n):
        if n == 4:
            n_point = np.add(np.array(point), np.array(FOUR_TRACK_DIRS[direct % n]))
        else:
            n_point = np.add(np.array(point), np.array(EIGHT_TRACK_DIRS[direct % n]))
        if _is_within_rect(image.shape, n_point):
            if image[n_point[0], n_point[1]] == FOREGROUND_VALUE:
                return (n_point[0], n_point[1]), direct % n
        direct += 1
    # 若区域仅一个像素, 返回该点自身
    return (point[0], point[1]), 0


@njit(cache=True)
def _is_within_rect(shape, point):
    """判断给定点是否在图像内部
    """
    assert len(shape) == len(point) == 2
    if point[0] < 0 or point[0] >= shape[0]:
        return False
    if point[1] < 0 or point[1] >= shape[1]:
        return False
    return True


def plot_image_with_boundaries(image, boundaries, title=""):
    boundaries = np.vstack([np.asarray(boundary) for boundary in boundaries])
    
    boundary_mask = np.zeros_like(image)
    boundary_mask[boundaries[:, 0], boundaries[:, 1]] = 255
    # 将 boundary_mask 转化为4通道图像, 增加透明度通道便于叠加显示
    boundary_mask = np.dstack(
        (
            boundary_mask - boundary_mask + 255,    # R
            255 - boundary_mask,                    # G
            255 - boundary_mask,                    # B
            boundary_mask,                          # A
        )
    )
    plt.figure("{} Boundaries".format(title))
    plt.imshow(image, cmap="gray")
    plt.imshow(boundary_mask)
    plt.axis("off")


def image_preprocess(image, foreground_value=None):
    """将图像进行二值化处理
    """
    # 对输入的单通道矩阵逐像素进行阈值分割
    assert foreground_value in [None, 0, 255]
    count = np.bincount(image.ravel())
    if count[0] + count[-1] != image.size:
        ret, image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
        )
        print("threshold value {}".format(ret))

    if foreground_value is not None and foreground_value == 0:
        # 若前景为0, 做反色处理
        image = cv2.bitwise_not(image)

    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="""对给定的二值图像进行边界追踪(外边界),
        若给定图像并非二值图像, 则先根据全局阈值对其进行二值化处理
        注意: 二值图像中默认 0 为背景, 255 为前景
        """
    )
    parser.add_argument("image_path", type=str, help="待处理图像的地址")
    parser.add_argument(
        "-n", type=int, choices=[4, 8], default=8, help="考虑的边界邻接区域, 4-邻域或8-邻域, 默认为8"
    )
    parser.add_argument(
        "-fg",
        "--foreground_value",
        type=int,
        choices=[0, 255],
        default=255,
        help="前景像素值, 默认为255",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure("Source Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    image = image_preprocess(image, args.foreground_value)

    boundaries = boundary_track(image, n=args.n)
    pprint(boundaries, compact=True)
    
    plot_image_with_boundaries(image, boundaries, "Binary Image with")

    plt.show()


if __name__ == "__main__":
    main()
