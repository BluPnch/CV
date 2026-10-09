"""Фильтр Гаусса средствами библиотеки OpenCV."""

import cv2
import numpy as np


def gaussian_blur_opencv(image, ksize, sigma_x=0.0, sigma_y=None):
    """Обёртка над cv2.GaussianBlur"""
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    if sigma_y is None:
        sigma_y = 0.0
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT_101)


def to_lists(image):
    """numpy-массив """
    if image.ndim == 2:
        return image.tolist()
    return [[tuple(px) for px in row] for row in image.tolist()]


def from_lists(data):
    """Вложенные списки Python"""
    return np.array(data, dtype=np.uint8)
