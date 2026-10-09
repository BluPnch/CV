"""
Демонстрационные изображения для отчёта:
  1. ядра Гаусса при разных sigma;
  2. сетка «размер ядра × sigma» на зашумлённом изображении и карты разности
     её крайних ячеек (по строкам и по столбцам);
  3. сравнение OpenCV и нативной реализации + карта разности;
  4. анизотропное размытие (разные ksize и sigma по осям) на цветном изображении.
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from gaussian_native import gaussian_blur_separable, gaussian_kernel_2d
from gaussian_opencv import from_lists, gaussian_blur_opencv, to_lists

OUT = "images/results"
SINGER = "images/input/rubezh_vekov.jpg"  # солист музыкальной группы на концерте
TOWN = "images/input/town.jpg"            # вечерний город


def load(path, max_side=800):
    """Читает изображение и уменьшает до max_side по большей стороне:
    чистый Python на полноразмерном фото (4000×5328) работал бы минутами."""
    img = cv2.imread(path)
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def show(ax, img, title):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def kernels():
    sigmas = [0.5, 1.0, 2.0, 4.0]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(13, 3.6))
    for ax, s in zip(axes, sigmas):
        k = np.array(gaussian_kernel_2d(15, 15, s, s))
        im = ax.imshow(k, cmap="Blues")
        ax.set_title(f"ядро 15×15, σ = {s}\nцентр = {k.max():.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kernels.png", dpi=110)
    plt.close(fig)


def grid():
    clean = load(SINGER, 600)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 25, clean.shape)
    img = np.clip(clean + noise, 0, 255).astype(np.uint8)
    ksizes = [3, 7, 15]
    sigmas = [1.0, 3.0, 6.0]
    results = {}
    fig, axes = plt.subplots(len(ksizes), len(sigmas) + 1, figsize=(13, 13.5))
    for i, k in enumerate(ksizes):
        show(axes[i, 0], img, "исходное (с шумом)" if i == 0 else "")
        for j, s in enumerate(sigmas):
            res = from_lists(gaussian_blur_separable(to_lists(img), k, s))
            results[k, s] = res
            show(axes[i, j + 1], res, f"k = {k}, σ = {s}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/grid_ksize_sigma.png", dpi=90)
    plt.close(fig)
    return results, ksizes, sigmas


def grid_diffs(results, ksizes, sigmas):
    """Разность крайних ячеек сетки: по строкам (σ_min vs σ_max при том же k)
    и по столбцам (k_min vs k_max при той же σ). Цвет — |A − B|, усреднённый
    по каналам BGR; шкала общая для всех карт, чтобы их можно было сравнивать."""
    s_lo, s_hi, k_lo, k_hi = sigmas[0], sigmas[-1], ksizes[0], ksizes[-1]
    pairs = ([((k, s_lo), (k, s_hi)) for k in ksizes]
             + [((k_lo, s), (k_hi, s)) for s in sigmas])
    diffs = [np.abs(results[a].astype(int) - results[b].astype(int)) for a, b in pairs]
    # верх шкалы — 99.5-й процентиль: иначе редкие пиксели на резких границах
    # (до ~150) делают все карты почти чёрными
    vmax = float(np.percentile(np.stack([d.mean(axis=2) for d in diffs]), 99.5))

    fig, axes = plt.subplots(2, len(ksizes), figsize=(13, 10.5))
    stats = []
    for ax, (a, b), d in zip(axes.flat, pairs, diffs):
        im = ax.imshow(d.mean(axis=2), cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"|(k={a[0]}, σ={a[1]:g}) − (k={b[0]}, σ={b[1]:g})|\n"
                     f"среднее = {d.mean():.2f}, max = {d.max()}", fontsize=10)
        ax.axis("off")
        stats.append((a, b, float(d.mean()), int(d.max())))
    axes[0, 0].text(-0.08, 0.5, f"по строкам:\nσ = {s_lo:g} vs σ = {s_hi:g}", rotation=90,
                    transform=axes[0, 0].transAxes, ha="center", va="center", fontsize=11)
    axes[1, 0].text(-0.08, 0.5, f"по столбцам:\nk = {k_lo} vs k = {k_hi}", rotation=90,
                    transform=axes[1, 0].transAxes, ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, label="|разность|, уровней яркости (шкала обрезана сверху)", extend="max")
    fig.savefig(f"{OUT}/grid_diffs.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return stats


def compare():
    img = load(SINGER)
    k, s = 9, 2.5
    a = gaussian_blur_opencv(img, k, s)
    b = from_lists(gaussian_blur_separable(to_lists(img), k, s))
    # разность берём максимальной по каналам BGR
    diff = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    show(axes[0], img, "исходное")
    show(axes[1], a, f"OpenCV, k = {k}, σ = {s}")
    show(axes[2], b, "нативная реализация")
    axes[3].imshow(diff, cmap="Reds", vmin=0, vmax=max(1, diff.max()))
    axes[3].set_title(f"|разность|: max = {diff.max()}, "
                      f"доля ≠0: {np.count_nonzero(diff) / diff.size:.2%}", fontsize=10)
    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/compare_opencv_native.png", dpi=90)
    plt.close(fig)
    return diff


def anisotropic():
    img = load(TOWN)
    cases = [((1, 1), (0, 0), "исходное"),
             ((21, 1), (8, 0.1), "kx = 21, ky = 1, σx = 8"),
             ((1, 21), (0.1, 8), "kx = 1, ky = 21, σy = 8"),
             ((11, 11), (4, 4), "k = 11, σ = 4")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    for ax, (k, s, title) in zip(axes, cases):
        res = img if k == (1, 1) else from_lists(
            gaussian_blur_separable(to_lists(img), k, s[0], s[1]))
        show(ax, res, title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/anisotropic.png", dpi=90)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    kernels()
    for a, b, mean, mx in grid_diffs(*grid()):
        print(f"|{a} - {b}|: mean={mean:.2f}, max={mx}")
    diff = compare()
    anisotropic()
    print(f"готово; расхождение с OpenCV: max={diff.max()}, mean={diff.mean():.4f}")


if __name__ == "__main__":
    main()
