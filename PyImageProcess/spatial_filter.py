import numpy as np


def get_gaussian_kernel(shape, sigma):
    """
    生成二维高斯模板
    fspecial('gaussian',[shape],[sigma])
    """
    assert type(shape) is tuple
    assert len(shape) == 2
    assert sigma is not None
    assert type(sigma) in [int, float]
    m, n = [(ss - 1) / 2 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def get_average_kernel(shape):
    """
    生成均值模板
    """
    assert type(shape) is tuple
    assert len(shape) == 2
    h = np.ones(shape) / np.prod(shape)
    return h


def get_laplacian_kernel():
    """
    生成拉普拉斯模板
    """
    h = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return h


def get_gaussian_kernel_gradients(shape, sigma):
    """
    计算高斯模板水平方向 (hx) 和竖直方向 (hy) 的导数
    """
    h = get_gaussian_kernel(shape, sigma)
    hy, hx = np.gradient(h)
    return hx, hy


def get_filter_kernel(name, *args, **kargs):
    """
    根据参数 name 的值生成相应的模板
    """
    assert name in ["average", "gaussian", "laplacian"]
    filter_kernels = {
        "average": get_average_kernel,
        "gaussian": get_gaussian_kernel,
        "laplacian": get_laplacian_kernel,
    }

    return filter_kernels[name](*args, **kargs)


def main():
    # 测试各函数功能
    shape1 = (3, 3)
    shape2 = (2, 3)
    h = get_gaussian_kernel(shape1, sigma=1)
    print("Gaussian Kernel, shape: {}".format(shape1))
    print(h)
    print("Gaussian Kernel Gradients, shape: {}".format(shape1))
    hx, hy = get_gaussian_kernel_gradients(shape1, sigma=1)
    print("HX:\n {}".format(hx))
    print("HY:\n {}".format(hy))
    h = get_gaussian_kernel(shape2, sigma=1)
    print("Gaussian Kernel, shape: {}".format(shape2))
    print(h)
    hx, hy = get_gaussian_kernel_gradients(shape2, sigma=1)
    print("Gaussian Kernel Gradients, shape: {}".format(shape2))
    print("HX:\n {}".format(hx))
    print("HY:\n {}".format(hy))
    h = get_average_kernel(shape1)
    print("Average Kernel, shape: {}".format(shape1))
    print(h)
    h = get_average_kernel(shape2)
    print("Average Kernel, shape: {}".format(shape2))
    print(h)
    h = get_laplacian_kernel()
    print("Laplacian Kernel")
    print(h)


if __name__ == "__main__":
    main()
