import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from numba import njit
from numba import types
from numba.typed import Dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("rotate_angle", type=float)
    parser.add_argument(
        "interpolation_method", type=str, choices=["bilinear", "nearest"]
    )

    args = parser.parse_args()
    return args


@njit(cache=True)
def cross_product(vec_1, vec_2):
    assert len(vec_1) == len(vec_2) == 2
    return vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]


@njit(cache=True)
def bilinear_interpolation(image, f_x, f_y):
    height = image.shape[0]
    width = image.shape[1]
    x_origin, y_origin = _get_origin(width, height)

    f_x = f_x + x_origin
    f_y = f_y + y_origin
    if f_x < 0:
        f_x = 0
        x_0 = f_x
    elif f_x > height - 1:
        f_x = height - 1
        x_0 = f_x - 1
    else:
        x_0 = math.floor(f_x)
    x_0 = int(x_0)
    x_1 = x_0 + 1

    if f_y < 0:
        f_y = 0
        y_0 = f_y
    elif f_y > width - 1:
        f_y = width - 1
        y_0 = f_y - 1
    else:
        y_0 = math.floor(f_y)
    y_0 = int(y_0)
    y_1 = y_0 + 1

    value_y_1 = (f_x - x_0) * image[x_1, y_1, :] + (x_1 - f_x) * image[x_0, y_1, :]
    value_y_0 = (f_x - x_0) * image[x_1, y_0, :] + (x_1 - f_x) * image[x_0, y_0, :]
    value = (f_y - y_0) * value_y_1 + (y_1 - f_y) * value_y_0

    return value


@njit(cache=True)
def nearest_neighbor_interpolation(image, f_x, f_y):
    height = image.shape[0]
    width = image.shape[1]
    x_origin, y_origin = _get_origin(width, height)

    f_x = f_x + x_origin
    f_y = f_y + y_origin

    x = round(f_x)
    y = round(f_y)

    if x < 0:
        x = 0
    elif x >= height:
        x = height - 1
    if y < 0:
        y = 0
    elif y >= width:
        y = width - 1

    return image[x, y, :]


@njit(cache=True)
def _get_origin(width, height):
    return height // 2, width // 2


@njit(cache=True)
def _get_rotated_coord(x, y, theta):
    rX = x * math.cos(theta) + y * math.sin(theta)
    rY = -x * math.sin(theta) + y * math.cos(theta)
    return rX, rY


@njit(cache=True)
def _extend_rect_vertices(rect_vertices):
    assert len(rect_vertices) == 2
    assert len(rect_vertices[0]) == 2
    x_0, y_0 = rect_vertices[0]
    x_1, y_1 = rect_vertices[1]
    x_2, y_2 = -x_0, -y_0
    x_3, y_3 = -x_1, -y_1
    rect_vertices.append((x_2, y_2))
    rect_vertices.append((x_3, y_3))


@njit(cache=True)
def _is_within_rect(x_point, y_point, rect_vertices):
    assert len(rect_vertices) == 4
    assert len(rect_vertices[0]) == 2

    for i in range(len(rect_vertices)):
        x, y = rect_vertices[i]
        x_prime, y_prime = rect_vertices[(i + 1) % len(rect_vertices)]
        product = cross_product((x_point - x, y_point - y), (x_prime - x, y_prime - y))
        if product < 0:
            return False
    return True


@njit(cache=True)
def _is_within_simple_rect(x, y, height, width):
    return x >= 0 and x < height and y >= 0 and y < width


@njit(cache=True)
def imrotate(image, angle, method):
    methods = Dict.empty(key_type=types.string, value_type=types.functions)
    methods["bilinear"] = bilinear_interpolation
    methods["nearest"] = nearest_neighbor_interpolation

    angle = angle % 360
    height = image.shape[0]
    width = image.shape[1]

    x_origin, y_origin = _get_origin(width, height)  # 计算原点坐标

    # 角度转换为弧度
    rad_angle = angle / 180.0 * math.pi
    # 计算旋转后图片的大小
    new_height = math.ceil(
        width * abs(math.sin(rad_angle)) + height * abs(math.cos(rad_angle))
    )
    new_width = math.ceil(
        width * abs(math.cos(rad_angle)) + height * abs(math.sin(rad_angle))
    )
    # 计算新图片的中心点坐标
    new_x_origin, new_y_origin = _get_origin(new_width, new_height)
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # 逐个处理新图片中的像素点
    for x in range(new_height):
        for y in range(new_width):
            _x = x - new_x_origin
            _y = y - new_y_origin
            x_old, y_old = _get_rotated_coord(_x, _y, -rad_angle)
            if _is_within_simple_rect(
                x_old + x_origin, y_old + y_origin, height, width
            ):
                if method in methods:
                    new_image[x, y, :] = methods[method](image, x_old, y_old)

    return new_image


def main():
    args = parse_args()
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure("Original Image")
    plt.imshow(image)
    plt.axis("off")

    image = imrotate(image, args.rotate_angle, args.interpolation_method)

    plt.figure("Rotated Image")
    plt.imshow(image)
    plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    main()
