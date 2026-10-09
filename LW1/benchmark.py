"""
Сравнение быстродействия трёх реализаций фильтра Гаусса:
  * OpenCV (cv2.GaussianBlur);
  * чистый Python, прямая 2D-свёртка;
  * чистый Python, сепарабельная свёртка.

Эксперимент 1: время от размера изображения (ядро фиксировано).
Эксперимент 2: время от размера ядра (изображение фиксировано).
Результаты: images/results/benchmark_*.csv, *.png и таблицы в benchmark.md.
"""

import csv
import os
import statistics
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gaussian_native import gaussian_blur_2d, gaussian_blur_separable
from gaussian_opencv import gaussian_blur_opencv, to_lists

OUT = "images/results"
SOURCE = "images/input/town.jpg"  # вечерний город

SIZES = [32, 64, 128, 256, 512]
SIZE_KSIZE = 5
KSIZES = [3, 5, 7, 9, 11, 15, 21, 31]
KSIZE_IMAGE = 128

NAMES = {
    "opencv": "OpenCV",
    "native2d": "Python 2D",
    "separable": "Python сепарабельный",
}
COLORS = {"opencv": "#2a78d6", "native2d": "#eb6834", "separable": "#1baf7a"}


def measure(func, repeats):
    func()  # прогрев
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        func()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000  # мс


def bench(image, ksize):
    data = to_lists(image)
    sigma = 0.0
    return {
        "opencv": measure(lambda: gaussian_blur_opencv(image, ksize, sigma), 200),
        "native2d": measure(lambda: gaussian_blur_2d(data, ksize, sigma), 3),
        "separable": measure(lambda: gaussian_blur_separable(data, ksize, sigma), 3),
    }


def save_csv(path, key, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([key, *NAMES])
        for x, r in rows:
            w.writerow([x, *(f"{r[m]:.4f}" for m in NAMES)])


def plot(rows, xlabel, title, path, xticks):
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#b5b4ae",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = [x for x, _ in rows]
    for m, name in NAMES.items():
        ys = [r[m] for _, r in rows]
        ax.plot(xs, ys, color=COLORS[m], linewidth=2, marker="o", markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=2, label=name)
        ax.annotate((f"{ys[-1]:.0f} мс" if ys[-1] >= 10 else f"{ys[-1]:.3g} мс"), (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color="#0b0b0b", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("время, мс (лог. шкала)")
    ax.set_title(title, color="#0b0b0b", loc="left")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def md_table(rows, key):
    lines = [f"| {key} | OpenCV, мс | Python 2D, мс | Python сепараб., мс | 2D / OpenCV | сепараб. / OpenCV |",
             "|---|---|---|---|---|---|"]
    for x, r in rows:
        lines.append(f"| {x} | {r['opencv']:.3f} | {r['native2d']:.1f} | {r['separable']:.1f} "
                     f"| {r['native2d'] / r['opencv']:.0f}× | {r['separable'] / r['opencv']:.0f}× |")
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    source = cv2.imread(SOURCE, cv2.IMREAD_GRAYSCALE)
    h, w = source.shape
    side = min(h, w)  # центральный квадрат, чтобы при масштабировании не искажать пропорции
    source = source[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]

    size_rows = []
    for n in SIZES:
        img = cv2.resize(source, (n, n), interpolation=cv2.INTER_AREA)
        r = bench(img, SIZE_KSIZE)
        size_rows.append((n, r))
        print(f"size {n}x{n}: " + ", ".join(f"{m}={v:.3f} мс" for m, v in r.items()))

    img = cv2.resize(source, (KSIZE_IMAGE, KSIZE_IMAGE), interpolation=cv2.INTER_AREA)
    ksize_rows = []
    for k in KSIZES:
        r = bench(img, k)
        ksize_rows.append((k, r))
        print(f"ksize {k}: " + ", ".join(f"{m}={v:.3f} мс" for m, v in r.items()))

    save_csv(f"{OUT}/benchmark_size.csv", "size", size_rows)
    save_csv(f"{OUT}/benchmark_ksize.csv", "ksize", ksize_rows)
    plot(size_rows, "сторона изображения N (изображение N×N, полутон)",
         f"Время фильтрации от размера изображения (ядро {SIZE_KSIZE}×{SIZE_KSIZE})",
         f"{OUT}/benchmark_size.png", SIZES)
    plot(ksize_rows, "размер ядра k (ядро k×k)",
         f"Время фильтрации от размера ядра (изображение {KSIZE_IMAGE}×{KSIZE_IMAGE})",
         f"{OUT}/benchmark_ksize.png", KSIZES)

    with open(f"{OUT}/benchmark.md", "w", encoding="utf-8") as f:
        f.write(f"### Зависимость от размера изображения (ядро {SIZE_KSIZE}×{SIZE_KSIZE})\n\n")
        f.write(md_table(size_rows, "N×N") + "\n\n")
        f.write(f"### Зависимость от размера ядра (изображение {KSIZE_IMAGE}×{KSIZE_IMAGE})\n\n")
        f.write(md_table(ksize_rows, "k") + "\n")


if __name__ == "__main__":
    main()
