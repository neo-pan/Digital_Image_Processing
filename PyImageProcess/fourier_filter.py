import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft

from spatial_filter import get_average_kernel, get_gaussian_kernel, get_laplacian_kernel


def create_circular_mask(h, w, center=None, radius=None):
    """产生一个圆形的 mask
    """
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def fourier_filter(image, filter_type, width=None, kernel=None):
    """多维的傅里叶滤波
    Args:
        image:          多维图像数据
        filter_type:    滤波器类别, ["low_pass", "high_pass", "kernel"]
        width:          简单的频域低通 / 高通滤波器的宽度
        kernel:         空域滤波器的 kernel
    Returns:
        result_image:   滤波后图像
    """
    if image.ndim == 2:
        result_image = _one_dim_fourier_filter(image, filter_type, width, kernel)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        result_image = np.zeros_like(image)
        # 分别对各通道进行傅里叶滤波
        for i in range(num_channels):
            result_image[..., i] = _one_dim_fourier_filter(
                image[..., i], filter_type, width, kernel
            )

    return result_image


def _one_dim_fourier_filter(image, filter_type, width=None, kernel=None):
    """一维的傅里叶滤波
    Args:
        image:          一维图像数据
        filter_type:    滤波器类别, ["low_pass", "high_pass", "kernel"]
        width:          简单的频域低通 / 高通滤波器的宽度
        kernel:         空域滤波器的 kernel
    Returns:
        result_image:   滤波后图像
    """
    assert image.ndim == 2
    rows, cols = image.shape
    assert filter_type in ["low_pass", "high_pass", "kernel"]
    if filter_type == "kernel":
        assert kernel is not None
        assert kernel.ndim == 2
        # 计算将kernel置于中心所需补零的数量
        pad_rows = (rows - kernel.shape[0]) // 2
        pad_extra_row = (rows - kernel.shape[0]) % 2
        pad_cols = (cols - kernel.shape[1]) // 2
        pad_extra_col = (cols - kernel.shape[1]) % 2
        # 在kernel周围补零
        kernel_padded = np.pad(
            kernel,
            (
                (pad_rows, pad_rows + pad_extra_row),
                (pad_cols, pad_cols + pad_extra_col),
            ),
            "constant",
        )
        kernel_padded = fft.fftshift(kernel_padded)
        fourier_filter = fft.fft2(kernel_padded)
    else:
        assert width is not None
        simple_filter = create_circular_mask(rows, cols, radius=width)
        if filter_type == "low_pass":
            fourier_filter = simple_filter.astype(np.uint8)
        else:
            fourier_filter = (~simple_filter).astype(np.uint8)
        fourier_filter = fft.fftshift(fourier_filter)

    img_fft = fft.fft2(image)
    img_fft = img_fft * fourier_filter
    result_image = np.abs(fft.ifft2(img_fft))

    return result_image


def image_preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def parse_args():
    parser = argparse.ArgumentParser(
        description="""对给定的图像进行傅里叶滤波,
        图像读取时透明度维度将被忽略
        """
    )
    parser.add_argument("image_path", type=str, help="待处理图像的地址")
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        choices=["low_pass", "high_pass", "kernel"],
        required=True,
        help="""
        选择傅里叶滤波器类别,
        low_pass: 简单的频域低通滤波 (圆形mask),
        high_pass: 简单的频域高通滤波 (圆形mask), 
        kernel: 使用给定的空域kernel进行傅里叶滤波
        """,
    )
    parser.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=["average", "gaussian", "laplacian"],
        default=None,
        help="""
        选择空域kernel类别, 当--filter不为'kernel'时该项设置无效.
        average: 邻域平均模板, 形状为7x7,
        gaussian: 高斯模板, 形状为7x7, sigma为1,
        laplacian: 拉普拉斯模板, 形状为3x3,
        注意: kernel具体参数暂时无法由命令行输入, 修改需自行编码 
        """,
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=10,
        help="简单低通/高通滤波器的mask宽度, 当--filter为 'kernel'时该项设置无效",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # 检验图像路径是否可用
    assert os.path.exists(args.image_path), "图像不存在"
    image = cv2.imread(args.image_path)
    image = image_preprocess(image)

    plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap="gray")
    plt.title("Source Image")
    if args.filter in ["low_pass", "high_pass"]:
        result = fourier_filter(image, args.filter, width=args.width)
    elif args.filter == "kernel":
        assert args.kernel is not None
        if args.kernel == "average":
            kernel = get_average_kernel(shape=(7, 7))
        elif args.kernel == "gaussian":
            kernel = get_gaussian_kernel(shape=(7, 7), sigma=1)
        elif args.kernel == "laplacian":
            kernel = get_laplacian_kernel()
        result = fourier_filter(image, args.filter, kernel=kernel)

    plt.subplot(122)
    plt.imshow(result, cmap="gray")
    plt.title("Filtered Image")
 
    plt.show()
