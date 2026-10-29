"""Иллюстрации для отчёта: раскадровки, GIF, графики ошибки, визуализация соответствий.
Запускать после make_test_videos.py и evaluate.py."""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # без GUI: Qt-бэкенд конфликтует с OpenCV
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from PIL import Image

from tracker import KeypointTracker

ROOT = Path(__file__).parent
RES = ROOT / "results"
IMG = ROOT / "images"
VIDEOS = ("test1_easy", "test2_perspective", "test3_hard")
REAL = ("mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit")
SRC_COLORS = {"detect": "#2a78d6", "flow": "#1baf7a", "predict": "#eda100", "": "#e6e5e0"}
SRC_NAMES = {"detect": "по эталону", "flow": "оптический поток", "predict": "предсказание", "": "не найден"}
COLORS = {"sift": "#2a78d6", "orb": "#eb6834", "akaze": "#1baf7a", "brisk": "#eda100"}


def read_frames(path, idx=None, step=1, width=None):
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if (idx is None and i % step == 0) or (idx is not None and i in idx):
            if width:
                f = cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    return frames


def storyboard(video, idx, out, cols=3, width=640):
    frames = read_frames(video, idx, width=width)
    rows = [np.hstack(frames[r * cols:(r + 1) * cols]) for r in range(len(frames) // cols)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def gif(video, out, step=4, width=400, fps=8):
    frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in read_frames(video, step=step, width=width)]
    frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)


def error_plot(out):
    fig, axes = plt.subplots(len(VIDEOS), 1, figsize=(11, 8.5), sharex=False)
    for ax, name in zip(axes, VIDEOS):
        rows = list(csv.DictReader(open(RES / f"per_frame_{name}.csv")))
        frame = np.array([int(r["frame"]) for r in rows])
        vis = np.array([float(r["visible"]) for r in rows])
        # серым - кадры, где объект почти не виден (< 20 %)
        for i in np.flatnonzero(vis < 0.2):
            ax.axvspan(i - 0.5, i + 0.5, color="#e6e5e0", lw=0)
        for det, c in COLORS.items():
            if f"{det}_hybrid_err" not in rows[0]:
                continue
            err = np.array([float(r[f"{det}_hybrid_err"]) for r in rows])
            ax.plot(frame[1:], err[1:], color=c, lw=1.5, label=det.upper())
        ax.set_yscale("log")
        ax.set_ylim(0.05, 100)
        ax.set_xlim(0, frame[-1])
        ax.set_title(name, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_ylabel("ошибка углов, px", color="#52514e")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b0afa8")
        ax.tick_params(colors="#52514e")
    axes[-1].set_xlabel("номер кадра", color="#52514e")
    axes[0].legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(0.25, 1.22))
    fig.text(0.99, 0.005, "разрыв линии - объект не найден; серый фон - объект виден менее чем на 20 %",
             ha="right", fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def sources_plot(out, det="sift"):
    """Для каждого видео из задания: каким способом получена рамка на каждом кадре."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    labels = []
    y = 0
    for name in REAL:
        rows = list(csv.DictReader(open(RES / f"per_frame_{name}.csv")))
        for mode in ("detect", "hybrid"):
            src = [r[f"{det}_{mode}_src"] for r in rows]
            for kind, c in SRC_COLORS.items():
                xs = [i for i, s in enumerate(src) if s == kind]
                ax.broken_barh([(x, 1) for x in xs], (y - 0.38, 0.76), facecolors=c, lw=0)
            found = sum(s != "" for s in src) / len(src)
            labels.append(f"{name}
{'только детектирование' if mode == 'detect' else 'гибрид'} - {100 * found:.0f} %")
            y += 1
        y += 0.4
    ticks = [0, 1, 2.4, 3.4, 4.8, 5.8]
    ax.set_yticks(ticks, labels, fontsize=8.5, color="#0b0b0b")
    ax.invert_yaxis()
    ax.set_xlim(0, len(rows))
    ax.set_xlabel("номер кадра", color="#52514e")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b0afa8")
    ax.tick_params(colors="#52514e", left=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SRC_COLORS.values()]
    ax.legend(handles, SRC_NAMES.values(), ncol=4, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def compare_with_sample(out, idx=(60, 200, 300, 480)):
    """Сверху - образец из задания (simple_track.avi), снизу - результат этой программы."""
    a = read_frames(ROOT / "test-videos" / "simple_track.avi", idx, width=400)
    b = read_frames(RES / "mona-lisa_sift.mp4", idx, width=400)
    cv2.imwrite(str(out), np.vstack([np.hstack(a), np.hstack(b)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def matches_figure(out, video, frame_idx, scale=0.6):
    """Эталон слева, кадр справа: инлаеры RANSAC (зелёные) и отброшенные соответствия (красные)."""
    cap = cv2.VideoCapture(str(video))
    ok, tpl = cap.read()
    for _ in range(frame_idx):
        ok, frame = cap.read()
    t = KeypointTracker("sift", max_side=1280, use_flow=False, max_predict=0)
    t.init(tpl)
    res = t.update(frame)
    # те же шаги вручную, чтобы получить и выбросы
    kp, desc = t.detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    knn = t.matcher.knnMatch(desc, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    src = t.tpl_pts[[m.trainIdx for m in good]]
    dst = np.float32([kp[m.queryIdx].pt for m in good])
    err = np.linalg.norm(cv2.perspectiveTransform(src[None], res.homography)[0] - dst, axis=1)
    canvas = np.hstack([tpl, frame])
    rng = np.random.default_rng(0)
    for i in rng.permutation(len(good))[:250]:
        p, q = src[i], dst[i] + [tpl.shape[1], 0]
        c = (60, 200, 60) if err[i] < 4 else (40, 40, 230)
        cv2.line(canvas, tuple(p.astype(int)), tuple(q.astype(int)), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p.astype(int)), 3, c, -1)
        cv2.circle(canvas, tuple(q.astype(int)), 3, c, -1)
    quad = (res.corners + [tpl.shape[1], 0]).astype(np.int32)
    cv2.polylines(canvas, [quad.reshape(-1, 1, 2)], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(out), cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"кадр {frame_idx}: {len(kp)} точек, {len(good)} соответствий, {res.n_inliers} инлаеров")


def main():
    IMG.mkdir(exist_ok=True)
    # видео из задания
    picks = {"mona-lisa": [0, 60, 130, 250, 330, 410, 450, 500, 557],
             "mona-lisa-blur": [8, 12, 18, 24, 28, 30],
             "mona-lisa-blur-extra-credit": [12, 20, 28, 389, 393, 398]}
    for name, idx in picks.items():
        storyboard(RES / f"{name}_sift.mp4", idx, IMG / f"{name}_frames.jpg", width=480)
    gif(RES / "mona-lisa-blur-extra-credit_sift.mp4", IMG / "mona-lisa.gif", step=3, width=400, fps=10)
    sources_plot(IMG / "sources.png")
    compare_with_sample(IMG / "sample_vs_ours.jpg")
    matches_figure(IMG / "matches_mona_lisa.jpg", ROOT / "test-videos" / "mona-lisa.avi", 60, scale=1.0)
    # синтетические видео
    picks = {"test1_easy": [0, 30, 70, 110, 160, 220],
             "test2_perspective": [20, 55, 100, 150, 200, 260],
             "test3_hard": [40, 80, 150, 207, 240, 330]}
    for name, idx in picks.items():
        storyboard(RES / f"{name}_sift.mp4", idx, IMG / f"{name}_frames.jpg")
    gif(RES / "test2_perspective_sift.mp4", IMG / "test2_perspective.gif")
    error_plot(IMG / "corner_error.png")
    matches_figure(IMG / "matches.jpg", ROOT / "videos" / "test2_perspective.mp4", 120)
    cv2.imwrite(str(IMG / "object.jpg"), cv2.resize(cv2.imread(str(ROOT / "videos" / "object.png")), (640, 360)))


if __name__ == "__main__":
    main()
