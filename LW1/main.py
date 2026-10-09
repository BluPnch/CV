"""
Консольная программа: фильтр Гаусса с переменным размером ядра и параметрами
гауссианы. Обрабатывает изображение выбранным методом (или всеми сразу),
сохраняет результат и печатает время работы.

    python main.py images/input/town.jpg -k 7 -s 2
    python main.py images/input/rubezh_vekov.jpg -k 15 9 -s 4 2 --method all
"""

import argparse
import os
import time

import cv2
import numpy as np

from gaussian_native import gaussian_blur_2d, gaussian_blur_separable
from gaussian_opencv import from_lists, gaussian_blur_opencv, to_lists

METHODS = {
    "opencv": "OpenCV (cv2.GaussianBlur)",
    "native2d": "Python, 2D-свёртка",
    "separable": "Python, сепарабельная свёртка",
}


def run(image, method, ksize, sigma_x, sigma_y):
    """Возвращает (результат, время в секундах). Время — только фильтрация."""
    if method == "opencv":
        start = time.perf_counter()
        result = gaussian_blur_opencv(image, ksize, sigma_x, sigma_y)
        return result, time.perf_counter() - start

    func = gaussian_blur_2d if method == "native2d" else gaussian_blur_separable
    data = to_lists(image)
    start = time.perf_counter()
    result = func(data, ksize, sigma_x, sigma_y)
    elapsed = time.perf_counter() - start
    return from_lists(result), elapsed


def parse_args():
    p = argparse.ArgumentParser(description="Усредняющий фильтр Гаусса")
    p.add_argument("input", help="путь к входному изображению")
    p.add_argument("-k", "--ksize", type=int, nargs="+", default=[5],
                   help="размер ядра: одно нечётное число или два (по X и по Y)")
    p.add_argument("-s", "--sigma", type=float, nargs="+", default=[0.0],
                   help="sigma: одно число или два (sigmaX sigmaY); 0 — по размеру ядра")
    p.add_argument("-m", "--method", choices=[*METHODS, "all"], default="all")
    p.add_argument("-g", "--gray", action="store_true", help="перевести в оттенки серого")
    p.add_argument("-o", "--outdir", default="images/results")
    return p.parse_args()


def main():
    args = parse_args()
    flag = cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_COLOR
    image = cv2.imread(args.input, flag)
    if image is None:
        raise SystemExit(f"Не удалось открыть {args.input}")

    ksize = tuple(args.ksize) if len(args.ksize) == 2 else (args.ksize[0], args.ksize[0])
    if any(k < 1 or k % 2 == 0 for k in ksize):
        raise SystemExit("Размер ядра должен быть положительным нечётным числом")
    sigma_x = args.sigma[0]
    sigma_y = args.sigma[1] if len(args.sigma) > 1 else 0.0

    methods = list(METHODS) if args.method == "all" else [args.method]
    os.makedirs(args.outdir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.input))[0]
    h, w = image.shape[:2]
    print(f"{args.input}: {w}x{h}, ksize={ksize}, sigma=({sigma_x}, {sigma_y})")

    results = {}
    for m in methods:
        result, elapsed = run(image, m, ksize, sigma_x, sigma_y)
        results[m] = result
        out = os.path.join(args.outdir, f"{name}_{m}_k{ksize[0]}x{ksize[1]}_s{sigma_x:g}_{sigma_y:g}.png")
        cv2.imwrite(out, result)
        print(f"  {METHODS[m]:<32} {elapsed * 1000:10.2f} мс  -> {out}")

    if "opencv" in results:
        for m, r in results.items():
            if m != "opencv":
                diff = np.abs(r.astype(int) - results["opencv"].astype(int))
                print(f"  |{m} - opencv|: max={diff.max()}, mean={diff.mean():.4f}")


if __name__ == "__main__":
    main()
