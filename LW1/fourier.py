"""
Частотный анализ фильтра Гаусса (преобразование Фурье).

  1. АЧХ ядер: Фурье-образ гауссова ядра — гауссиана (сравнение с формулой
     exp(-2π²σ²f²)); у box-фильтра той же «ширины» — sinc с отрицательными
     лепестками.
  2. Одномерный сигнал (ступеньки + мелкая текстура + шум) до и после
     фильтрации и его амплитудный спектр. Для сравнения — идеальный
     низкочастотный фильтр (обнуление спектра выше частоты среза), который
     даёт «звон» (эффект Гиббса), а фильтр Гаусса — нет.
  3. Двумерный спектр изображения до и после фильтра Гаусса и спектр ядра:
     по теореме о свёртке спектр результата = спектр исходного × спектр ядра.

Сама фильтрация выполняется нативной реализацией (gaussian_native.py),
БПФ — numpy.fft.
"""

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from gaussian_native import gaussian_blur_separable, gaussian_kernel_1d
from gaussian_opencv import to_lists

OUT = "images/results"
SINGER = "images/input/rubezh_vekov.jpg"

SIGMA = 2.0
KSIZE = 13          # ≈ 6σ + 1
BOX = 7             # box той же дисперсии: (w² - 1) / 12 = σ² -> w = 7

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def blur_1d(signal, ksize, sigma):
    """Сигнал как изображение 1×N -> нативный фильтр Гаусса только по X (без округления)."""
    res = gaussian_blur_separable([list(map(float, signal))], (ksize, 1), sigma, 0.1,
                                  round_result=False)
    return np.array(res[0])


def frequency_response(kernel, n=2048):
    """Комплексная частотная характеристика H(f) ядра (центрированного в 0)."""
    k = np.asarray(kernel, dtype=float)
    r = len(k) // 2
    padded = np.zeros(n)
    padded[:r + 1] = k[r:]          # отсчёты 0..r
    padded[-r:] = k[:r]             # отсчёты -r..-1 (циклически)
    h = np.fft.rfft(padded)
    f = np.fft.rfftfreq(n)          # циклов на пиксель, 0..0.5
    return f, h.real                # ядро симметрично -> H(f) вещественная


def kernels_spectrum():
    g = gaussian_kernel_1d(KSIZE, SIGMA)
    box = [1 / BOX] * BOX
    f, hg = frequency_response(g)
    _, hb = frequency_response(box)
    analytic = np.exp(-2 * np.pi ** 2 * SIGMA ** 2 * f ** 2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    xs = np.arange(-(KSIZE // 2), KSIZE // 2 + 1)
    a1.bar(xs - 0.2, g, width=0.4, color=C1, label=f"Гаусс, k = {KSIZE}, σ = {SIGMA:g}")
    a1.bar(np.arange(-(BOX // 2), BOX // 2 + 1) + 0.2, box, width=0.4, color=C2,
           label=f"box, k = {BOX}")
    a1.set_title("Ядра в пространственной области")
    a1.set_xlabel("сдвиг от центра, пикселей")
    a1.set_ylabel("вес")
    a1.legend(frameon=False)
    style(a1)

    a2.axhline(0, color=INK2, linewidth=1)
    a2.plot(f, hg, color=C1, linewidth=2, label="Гаусс (БПФ ядра)")
    a2.plot(f, analytic, color=INK, linewidth=1.2, linestyle=(0, (4, 3)),
            label="exp(−2π²σ²f²) — аналитически")
    a2.plot(f, hb, color=C2, linewidth=2, label="box (sinc)")
    lobe = np.argmin(hb)
    a2.annotate(f"отрицательный лепесток {hb[lobe]:.2f}:\nчастоты проходят с инверсией",
                (f[lobe], hb[lobe]), xytext=(f[lobe] + 0.07, hb[lobe] - 0.02),
                arrowprops={"arrowstyle": "-", "color": INK2}, color=INK, fontsize=9)
    a2.set_title("Частотная характеристика H(f)")
    a2.set_xlabel("частота f, циклов на пиксель")
    a2.set_ylabel("H(f)")
    a2.legend(frameon=False)
    style(a2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fourier_kernels.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return float(np.abs(hg - analytic).max()), float(hb[lobe])


def signal_spectrum():
    n = 512
    x = np.arange(n)
    rng = np.random.default_rng(1)
    steps = np.select([x < 128, x < 256, x < 384], [60, 180, 100], 200).astype(float)
    texture = 15 * np.sin(2 * np.pi * 0.3 * x)          # мелкая деталь, f = 0.3
    signal = steps + texture + rng.normal(0, 4, n)

    gauss = blur_1d(signal, KSIZE, SIGMA)

    # идеальный ФНЧ с той же частотой среза, где у Гаусса H(f) = 0.5
    cutoff = np.sqrt(np.log(2) / (2 * np.pi ** 2 * SIGMA ** 2))
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n)
    ideal = np.fft.irfft(np.where(freqs <= cutoff, spec, 0), n)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.4), facecolor=SURFACE)
    a1.plot(x, signal, color="#b5b4ae", linewidth=1, label="исходный сигнал")
    a1.plot(x, ideal, color=C2, linewidth=2, label=f"идеальный ФНЧ (срез f = {cutoff:.3f})")
    a1.plot(x, gauss, color=C1, linewidth=2, label=f"фильтр Гаусса (k = {KSIZE}, σ = {SIGMA:g})")
    a1.set_xlim(0, n - 1)
    peak = 128 + int(np.argmax(ideal[128:160]))
    a1.annotate("«звон» (эффект Гиббса)", (peak, ideal[peak]), xytext=(150, 235),
                arrowprops={"arrowstyle": "-", "color": INK2}, color=INK, fontsize=9)
    a1.set_title("Сигнал до и после фильтрации")
    a1.set_xlabel("отсчёт (пиксель)")
    a1.set_ylabel("яркость")
    a1.legend(frameon=False, loc="lower right", ncol=3)
    style(a1)

    # окно Ханна: без него скачок между концом (200) и началом (60) сигнала,
    # который БПФ считает периодическим, даёт «утечку» на все частоты
    win = np.hanning(n)
    amp = lambda s: np.abs(np.fft.rfft(s * win)) / win.sum() * 2  # noqa: E731
    a2.semilogy(freqs, amp(signal), color="#b5b4ae", linewidth=1, label="исходный сигнал")
    a2.semilogy(freqs, amp(ideal) + 1e-6, color=C2, linewidth=1.5, label="идеальный ФНЧ")
    a2.semilogy(freqs, amp(gauss), color=C1, linewidth=1.5, label="фильтр Гаусса")
    a2.axvline(cutoff, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    a2.annotate("текстура f = 0.3", (0.3, amp(signal)[np.argmin(abs(freqs - 0.3))]),
                xytext=(0.33, 10), arrowprops={"arrowstyle": "-", "color": INK2},
                color=INK, fontsize=9)
    a2.set_ylim(1e-6, 300)
    a2.set_xlim(0, 0.5)
    a2.set_title("Амплитудный спектр |X(f)| (окно Ханна, лог. шкала)")
    a2.set_xlabel("частота f, циклов на пиксель")
    a2.set_ylabel("амплитуда")
    a2.legend(frameon=False, loc="upper right")
    style(a2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fourier_signal.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    # Выброс за пределы ступенек на «чистом» сигнале (без шума и текстуры):
    # насколько результат выходит за диапазон исходных значений в окрестности ±40.
    clean_g = blur_1d(steps, KSIZE, SIGMA)
    spec_s = np.fft.rfft(steps)
    clean_i = np.fft.irfft(np.where(freqs <= cutoff, spec_s, 0), n)
    lo = np.array([steps[max(0, i - 40):i + 41].min() for i in range(n)])
    hi = np.array([steps[max(0, i - 40):i + 41].max() for i in range(n)])
    inner = slice(40, n - 40)  # без краёв: ИФНЧ через БПФ считает сигнал периодическим

    def overshoot(r):
        return float(max((r - hi)[inner].max(), (lo - r)[inner].max(), 0))
    return cutoff, overshoot(clean_g), overshoot(clean_i)


def spectrum(img):
    """|F| с окном Ханна: окно убирает «крест» от разрыва на краях изображения
    (БПФ считает изображение периодическим)."""
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(img.astype(float) * win))
    return np.abs(f)


def radial_freq(shape):
    h, w = shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    return np.hypot(fx, fy)


def high_freq_share(amp, radius=0.1):
    """Доля энергии спектра на частотах выше radius (циклов на пиксель)."""
    r = radial_freq(amp.shape)
    energy = amp ** 2
    energy[r == 0] = 0  # без постоянной составляющей (средней яркости)
    return float(energy[r > radius].sum() / energy.sum())


def image_spectrum():
    img = cv2.imread(SINGER, cv2.IMREAD_GRAYSCALE)
    scale = 600 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # без округления до uint8, иначе шум квантования «поднимает» высокие частоты
    blurred = np.array(gaussian_blur_separable(to_lists(img), KSIZE, SIGMA, round_result=False))

    before, after = spectrum(img), spectrum(blurred)
    # спектр ядра того же размера, что изображение (ядро дополнено нулями)
    g = np.array(gaussian_kernel_1d(KSIZE, SIGMA))
    kernel = np.zeros(img.shape)
    kernel[:KSIZE, :KSIZE] = np.outer(g, g)
    h_kernel = np.abs(np.fft.fftshift(np.fft.fft2(kernel)))

    vmax = np.log1p(before).max()
    fig, axes = plt.subplots(2, 3, figsize=(14, 12), facecolor="white")
    axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("исходное изображение")
    axes[0, 1].imshow(np.clip(blurred, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"после фильтра Гаусса (k = {KSIZE}, σ = {SIGMA:g})")
    axes[0, 2].imshow(h_kernel, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("|H| — спектр ядра (гауссиана)")
    axes[1, 0].imshow(np.log1p(before), cmap="magma", vmin=0, vmax=vmax)
    axes[1, 0].set_title("log(1 + |F|) исходного")
    axes[1, 1].imshow(np.log1p(after), cmap="magma", vmin=0, vmax=vmax)
    axes[1, 1].set_title("log(1 + |F|) после фильтра")
    axes[1, 2].imshow(np.log1p(before * h_kernel), cmap="magma", vmin=0, vmax=vmax)
    axes[1, 2].set_title("log(1 + |F|·|H|) — теорема о свёртке")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fourier_image.png", dpi=80)
    plt.close(fig)

    # Проверка теоремы о свёртке: |F_после| / |F_до| должно совпасть с H(f).
    # Усредняем отношение по кольцам одинаковой частоты |f|.
    r = radial_freq(img.shape)
    bins = np.linspace(0, 0.5, 51)
    centers = (bins[:-1] + bins[1:]) / 2
    ratio = [np.median(after[(r >= a) & (r < b)] / before[(r >= a) & (r < b)])
             for a, b in zip(bins[:-1], bins[1:])]
    analytic = np.exp(-2 * np.pi ** 2 * SIGMA ** 2 * centers ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.plot(centers, analytic, color=INK, linewidth=1.2, linestyle=(0, (4, 3)),
            label="exp(−2π²σ²f²) — Фурье-образ гауссианы")
    ax.plot(centers, ratio, color=C1, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, label="|F после| / |F до| (медиана по кольцу)")
    ax.set_xlabel("радиальная частота |f|, циклов на пиксель")
    ax.set_ylabel("коэффициент ослабления")
    ax.set_title("Во сколько раз фильтр ослабил каждую частоту изображения")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fourier_ratio.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)

    low = centers < 0.2
    max_err = float(np.max(np.abs(np.array(ratio)[low] - analytic[low])))
    return high_freq_share(before), high_freq_share(after), max_err


def main():
    os.makedirs(OUT, exist_ok=True)
    err, lobe = kernels_spectrum()
    print(f"АЧХ: max |БПФ ядра - exp(-2π²σ²f²)| = {err:.4f}; минимум у box = {lobe:.3f}")
    cutoff, ov_g, ov_i = signal_spectrum()
    print(f"сигнал: срез ФНЧ f={cutoff:.4f}; выброс над ступенькой: Гаусс {ov_g:.1f}, идеальный ФНЧ {ov_i:.1f}")
    hf_before, hf_after, err_img = image_spectrum()
    print(f"изображение: доля энергии (без DC) на |f| > 0.1: до {hf_before:.2%}, после {hf_after:.2%}; "
          f"max |отношение спектров - exp(...)| при |f| < 0.2: {err_img:.3f}")


if __name__ == "__main__":
    main()
