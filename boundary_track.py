import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from numba import njit
from numba.typed import List


# 前景像素值
FOREGROUND_VALUE = 255
# 顺时针追踪顺序
# -------------
# | 5 | 6 | 7 |
# -------------
# | 4 | x | 0 |
# -------------
# | 3 | 2 | 1 |
# -------------
TRACK_DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


@njit(cache=True)
def boundary_track(image):
    """对二值图像进行区域外边界跟踪, 要求区域内部无孔洞
    """
    # 确保为二值化图像
    # assert image.dtype == np.uint8  #! numba 优化后无法正常判断
    assert image.ndim == 2
    count = np.bincount(image.ravel())
    assert count[0] + count[-1] == image.size
    boundaries = List()
    # numba.typed.List 需要提前指定其内部元素类型
    boundaries.append((-1, -1))

    height = image.shape[0]
    width = image.shape[1]
    inside_boundary = False

    for i in range(height):
        for j in range(width):
            if (i, j) in boundaries:
                # 若经过边界点, inside_boundary 标记为 True
                inside_boundary = True
            if image[i, j] != FOREGROUND_VALUE:
                # 若经过背景点, inside_boundary 标记为 False
                inside_boundary = False
            if image[i, j] == FOREGROUND_VALUE and not inside_boundary:
                # 若开始追踪某条边界, inside_boundary 标记为 False
                inside_boundary = True
                P0 = (i, j)
                P1, direct = track_next(image, boundaries, point=P0, direct=0)
                P_n = P1
                P_n_pre = P_n
                while True:
                    P_n_pre = P_n
                    P_n, direct = track_next(image, boundaries, P_n_pre, direct)
                    if P_n == P1 and P_n_pre == P0:
                        # plot_image_with_boundaries(image, np.array(boundaries[1:]))
                        # plt.show()
                        break
            if j == width - 1:
                # 若进行换行, inside_boundary 标记为 False
                inside_boundary = False
    # 返回时去除最初加入的 (-1, -1)
    return boundaries[1:]


@njit(cache=True)
def track_next(image, boundaries, point, direct):
    """从给定点开始, 在其8-邻域中跟踪下一个边界点
    """
    boundaries.append((point[0], point[1]))
    if direct % 2 == 0:
        direct = (direct + 7) % 8
    else:
        direct = (direct + 6) % 8
    for _ in range(8):
        n_point = np.add(np.array(point), np.array(TRACK_DIRS[direct % 8]))
        if _is_within_rect(image.shape, n_point):
            if image[n_point[0], n_point[1]] == FOREGROUND_VALUE:
                return (n_point[0], n_point[1]), direct % 8
        direct += 1
    # 若区域仅一个像素, 返回该点自身 #! 应通过形态学处理消除仅有一个像素的点
    return (point[0], point[1]), 7


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
    boundary_mask = np.zeros_like(image)
    boundary_mask[boundaries[:, 0], boundaries[:, 1]] = 255
    # 将 boundary_mask 转化为4通道图像, 增加透明度通道便于叠加显示
    boundary_mask = np.dstack(
        (
            boundary_mask - boundary_mask + 255,  # R
            255 - boundary_mask,  # G
            255 - boundary_mask,  # B
            boundary_mask,  # A
        )
    )
    plt.figure("{} Boundaries".format(title))
    plt.imshow(image, cmap="gray")
    plt.imshow(boundary_mask)
    plt.axis("off")


def image_pre_process(image):
    """将图像进行二值化处理
    """
    # 对输入的单通道矩阵逐像素进行阈值分割
    count = np.bincount(image.ravel())
    if count[0] + count[-1] != image.size:
        ret, image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE
        )
        print("threshold value {}".format(ret))

    # 进行泛洪填充消除内部孔洞
    if FOREGROUND_VALUE == 0:
        # 若前景为0, 做反色处理
        image = cv2.bitwise_not(image)
    img_floodfill = image.copy()
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = image | img_floodfill_inv
    if FOREGROUND_VALUE == 0:
        img_out = cv2.bitwise_not(img_out)
    return img_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="""对给定的二值图像进行边界追踪(外边界),
        若给定图像并非二值图像, 则先根据全局阈值对其进行二值化处理
        预处理过程将去除图像区域内部孔洞
        注意: 二值图像中默认 0 为背景, 255 为前景
        """
    )
    parser.add_argument("image_path", type=str, help="待处理图像的地址")
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
    global FOREGROUND_VALUE
    args = parse_args()
    FOREGROUND_VALUE = args.foreground_value
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    plt.figure("Binary Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    processed_image = image_pre_process(image)
    boundaries = boundary_track(processed_image)
    boundaries = np.array(boundaries)

    print(boundaries)
    plot_image_with_boundaries(image, boundaries, "Source Image with")
    plot_image_with_boundaries(processed_image, boundaries, "Processed Image with")

    plt.show()


if __name__ == "__main__":
    main()
