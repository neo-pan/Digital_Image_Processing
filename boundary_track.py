import argparse

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from numba import njit

FOREGROUND_VALUE = 255
TRACK_DIRS = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))


@njit(cache=True)
def boundary_track(image):
    """对二值图像进行区域边界跟踪
    """
    # 确保为二值化图像
    # assert image.dtype == np.uint8  #! numba 优化后无法正常判断
    assert image.ndim == 2
    count = np.bincount(image.ravel())
    assert count[0] + count[-1] == image.size
    boundary_mask = np.zeros_like(image, dtype=np.uint8)

    height = image.shape[0]
    width = image.shape[1]
    inside_boundary = False

    for i in range(height):
        for j in range(width):
            if boundary_mask[i, j] == 1:
                inside_boundary = True
            if image[i, j] != FOREGROUND_VALUE:
                inside_boundary = False
            if image[i, j] == FOREGROUND_VALUE and not inside_boundary:
                inside_boundary = True
                P0 = (i, j)
                P1, direct = track_next(image, boundary_mask, point=P0, direct=7)
                P_n = P1
                while True:
                    P_n_pre = P_n
                    P_n, direct = track_next(image, boundary_mask, P_n_pre, direct)
                    if P_n == P1 and P_n_pre == P0:
                        break

    return boundary_mask


@njit(cache=True)
def track_next(image, boundary_mask, point, direct):
    """从给定点开始, 在其8-邻接点中跟踪下一个边界点
    """
    boundary_mask[point[0], point[1]] = 1
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


def image_pre_process(image):
    """将图像进行二值化处理
    """
    if image.ndim == 2:
        count = np.bincount(image.ravel())
        if count[0] + count[-1] == image.size:
            return image
    # 把输入图像灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 对输入的单通道矩阵逐像素进行阈值分割
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print("threshold value {}".format(ret))

    return binary


def parse_args():
    parser = argparse.ArgumentParser(
        description="""对给定的二值图像进行边界追踪(外边界),
        若给定图像并非二值图像, 则先根据全局阈值对其进行二值化处理
        注意: 二值图像中 0 应代表背景, 255 应代表前景
        """
    )
    parser.add_argument("image_path", type=str, help="待处理图像的地址")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    image = image_pre_process(cv2.imread(args.image_path))

    plt.figure("Binary Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    boundary = boundary_track(image)

    plt.figure("Boundaries")
    plt.imshow(boundary, alpha=1, cmap="binary")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
