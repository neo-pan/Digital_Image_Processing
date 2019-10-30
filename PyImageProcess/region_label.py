import argparse
import os

import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numba import njit

# 4-邻域
FOUR_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# 8-邻域
EIGHT_NEIGHBOURS = FOUR_NEIGHBOURS + ((1, 1), (1, -1), (-1, 1), (-1, -1))
# 前景像素值
FOREGROUND_VALUE = 255


@njit(cache=True)
def region_label(image, n=None):
    """对二值图像进行联通区域标记
    """
    region_id = 1
    # 确保为二值化图像
    # assert image.dtype == np.uint8  #! numba 优化后无法正常判断
    assert image.ndim == 2
    count = np.bincount(image.ravel())
    assert count[0] + count[-1] == image.size
    label_mask = np.zeros_like(image, dtype=np.int64)

    height = image.shape[0]
    width = image.shape[1]

    queue = list()

    for i in range(height):
        for j in range(width):
            if image[i, j] == FOREGROUND_VALUE and label_mask[i, j] == 0:
                queue.append((i, j))
                while queue:
                    point = queue.pop()
                    valid_neighbors = pixel_label(
                        image, label_mask, point, region_id, n
                    )
                    queue += valid_neighbors

                region_id += 1

    return label_mask


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


@njit(cache=True)
def get_neighbours_indecis(shape, point, n=None):
    """得到图中某一点的4-邻域或8-邻域点坐标
    """
    assert len(shape) == len(point) == 2
    if n is None:
        n = 4
    assert n in (4, 8)
    neighbours = list()
    if n == 8:
        for neighbour in EIGHT_NEIGHBOURS:
            n_point = np.add(np.array(point), np.array(neighbour))
            neighbours.append(list(n_point))
    else:
        for neighbour in FOUR_NEIGHBOURS:
            n_point = np.add(np.array(point), np.array(neighbour))
            neighbours.append(list(n_point))
    neighbours = [
        neighbour for neighbour in neighbours if _is_within_rect(shape, neighbour)
    ]

    return neighbours


@njit(cache=True)
def pixel_label(image, label_mask, point, region_id, n=None):
    """标记特定像素点, 并返回其尚未被标记的邻居前景点的坐标
    """
    assert image.ndim == 2
    assert len(point) == 2
    assert point[0] < image.shape[0] and point[1] < image.shape[1]

    label_mask[point[0], point[1]] = region_id

    neighbours = get_neighbours_indecis(image.shape, point, n)
    valid_neighbors = list()
    for neighbour in neighbours:
        if (
            image[neighbour[0], neighbour[1]] == FOREGROUND_VALUE
            and label_mask[neighbour[0], neighbour[1]] == 0
        ):
            valid_neighbors.append((neighbour[0], neighbour[1]))
    return valid_neighbors


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
    # 若前景像素为0, 将图像反色后再进行处理
    if foreground_value is not None and foreground_value == 0:
        image = cv2.bitwise_not(image)

    return image


def generate_color_map(MAX_CLASSES_NUM):
    """根据标签总数产生 color map, 用于绘图
    """
    color_map = cm.get_cmap("tab20")
    cmap = [color_map(i % 20)[:3] for i in range(MAX_CLASSES_NUM)]
    cmap = ["black"] + cmap
    cmap = matplotlib.colors.ListedColormap(cmap)
    return cmap


def parse_args():
    parser = argparse.ArgumentParser(
        description="""对给定的二值图像进行区域标记, 
        若给定图像并非二值图像, 则先根据全局阈值对其进行二值化处理
        注意: 二值图像中默认 0 为背景, 255 为前景
        """
    )
    parser.add_argument("image_path", type=str, help="待处理图像的地址")
    parser.add_argument(
        "-n", type=int, choices=[4, 8], default=4, help="考虑的邻接区域, 4-邻域或8-邻域, 默认为4"
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
    # 检验图像路径是否可用
    assert os.path.exists(args.image_path), "图像不存在"
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    image = image_preprocess(image, args.foreground_value)

    plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.title("Binary Image")

    label_mask = region_label(image, n=args.n)

    print("region numbers: {}".format(np.max(label_mask)))

    cmap = generate_color_map(np.max(label_mask))
    plt.subplot(122)
    plt.imshow(label_mask, cmap=cmap)
    plt.title("Region Label")
    plt.show()


if __name__ == "__main__":
    main()
