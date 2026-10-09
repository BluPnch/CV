"""
Нативная реализация усредняющего фильтра Гаусса на чистом Python.

Не используются ни OpenCV, ни NumPy: изображение представлено как список
строк (list[list[int]]) для полутонового изображения или список строк
пикселей-кортежей для цветного. Все вычисления — обычные циклы Python.

Реализованы два варианта свёртки:
  * gaussian_blur_2d        — «прямая» двумерная свёртка с ядром k x k,
                              сложность O(W * H * kx * ky);
  * gaussian_blur_separable — сепарабельная свёртка (сначала по строкам,
                              затем по столбцам одномерным ядром),
                              сложность O(W * H * (kx + ky)).
"""

import math


def sigma_from_ksize(ksize):
    """Сигма по размеру ядра — та же формула, что в OpenCV при sigma <= 0."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


# Для малых ядер при sigma <= 0 OpenCV берёт фиксированные биномиальные коэффициенты вместо формулы
SMALL_KERNELS = {
    1: [1.0],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}


def gaussian_kernel_1d(ksize, sigma):
    """Одномерное нормированное ядро Гаусса длины ksize (ksize — нечётное)."""
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("Размер ядра должен быть положительным нечётным числом")
    if sigma <= 0:
        if ksize in SMALL_KERNELS:
            return list(SMALL_KERNELS[ksize])
        sigma = sigma_from_ksize(ksize)
    center = (ksize - 1) / 2
    kernel = [math.exp(-((i - center) ** 2) / (2 * sigma * sigma)) for i in range(ksize)]
    total = sum(kernel)
    return [v / total for v in kernel]


def gaussian_kernel_2d(ksize_x, ksize_y, sigma_x, sigma_y):
    """Двумерное ядро как внешнее произведение двух одномерных ядер."""
    kx = gaussian_kernel_1d(ksize_x, sigma_x)
    ky = gaussian_kernel_1d(ksize_y, sigma_y)
    return [[wy * wx for wx in kx] for wy in ky]


def reflect_101(i, n):
    """Индекс с отражением относительно границы (BORDER_REFLECT_101)."""
    if n == 1:
        return 0
    period = 2 * n - 2
    i = abs(i) % period
    return period - i if i >= n else i


def _clip_round(v):
    v = int(v + 0.5)
    return 0 if v < 0 else (255 if v > 255 else v)


def _blur_channel_2d(channel, kernel):
    h, w = len(channel), len(channel[0])
    ky, kx = len(kernel), len(kernel[0])
    ry, rx = ky // 2, kx // 2
    col_idx = [[reflect_101(x + dx, w) for dx in range(-rx, rx + 1)] for x in range(w)]
    out = []
    for y in range(h):
        rows = [channel[reflect_101(y + dy, h)] for dy in range(-ry, ry + 1)]
        out_row = []
        for x in range(w):
            cols = col_idx[x]
            acc = 0.0
            for krow, src in zip(kernel, rows):
                for kv, cx in zip(krow, cols):
                    acc += kv * src[cx]
            out_row.append(_clip_round(acc))
        out.append(out_row)
    return out


def _blur_channel_separable(channel, kx, ky, round_result=True):
    h, w = len(channel), len(channel[0])
    rx, ry = len(kx) // 2, len(ky) // 2

    # Проход 1: горизонтальная свёртка (результат во float).
    col_idx = [[reflect_101(x + dx, w) for dx in range(-rx, rx + 1)] for x in range(w)]
    tmp = []
    for row in channel:
        tmp.append([sum(kv * row[cx] for kv, cx in zip(kx, cols)) for cols in col_idx])

    # Проход 2: вертикальная свёртка.
    out = []
    for y in range(h):
        rows = [tmp[reflect_101(y + dy, h)] for dy in range(-ry, ry + 1)]
        out_row = [0] * w
        for x in range(w):
            acc = 0.0
            for kv, src in zip(ky, rows):
                acc += kv * src[x]
            out_row[x] = _clip_round(acc) if round_result else acc
        out.append(out_row)
    return out


def _split_channels(image):
    """Цветное изображение (строки из кортежей) -> список каналов."""
    n = len(image[0][0])
    return [[[px[c] for px in row] for row in image] for c in range(n)]


def _merge_channels(channels):
    return [[tuple(px) for px in zip(*rows)] for rows in zip(*channels)]


def _is_color(image):
    return isinstance(image[0][0], (tuple, list))


def _normalize_params(ksize, sigma_x, sigma_y):
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    ksize_x, ksize_y = ksize
    if sigma_y is None or sigma_y <= 0:
        sigma_y = sigma_x
    return ksize_x, ksize_y, sigma_x, sigma_y


def gaussian_blur_2d(image, ksize, sigma_x=0.0, sigma_y=None):
    """Фильтр Гаусса прямой двумерной свёрткой.=

    image   — list[list[int]] (полутон) или list[list[tuple]] (цвет);
    ksize   — нечётное int или пара (ksize_x, ksize_y);
    sigma_x, sigma_y — параметры гауссианы (<= 0 — вычислить по размеру ядра).
    """
    ksize_x, ksize_y, sigma_x, sigma_y = _normalize_params(ksize, sigma_x, sigma_y)
    kernel = gaussian_kernel_2d(ksize_x, ksize_y, sigma_x, sigma_y)
    if _is_color(image):
        return _merge_channels([_blur_channel_2d(ch, kernel) for ch in _split_channels(image)])
    return _blur_channel_2d(image, kernel)


def gaussian_blur_separable(image, ksize, sigma_x=0.0, sigma_y=None, round_result=True):
    """Фильтр Гаусса двумя проходами одномерным ядром
    """
    ksize_x, ksize_y, sigma_x, sigma_y = _normalize_params(ksize, sigma_x, sigma_y)
    kx = gaussian_kernel_1d(ksize_x, sigma_x)
    ky = gaussian_kernel_1d(ksize_y, sigma_y)
    if _is_color(image):
        return _merge_channels([_blur_channel_separable(ch, kx, ky, round_result)
                                for ch in _split_channels(image)])
    return _blur_channel_separable(image, kx, ky, round_result)
