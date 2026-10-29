"""Оценка трекера.

1) Синтетические видео с известной разметкой (make_test_videos.py):
   recall      - доля кадров, где объект виден (>= 20 % площади), и рамка есть
   success@10  - доля таких кадров, где ошибка углов < 10 px
   false pos.  - кадры, где объекта нет (< 5 % видно), а рамка нарисована
   err median / p95 - средняя по 4 углам ошибка рамки, px

2) Видео из задания (test-videos/mona-lisa*.avi), разметки нет:
   доля кадров с рамкой (по источникам: эталон / поток / предсказание), скорость.
   mona-lisa-blur* - это mona-lisa.avi с искусственно размытыми кадрами, поэтому
   рамки трекера на чётком видео служат опорной разметкой для размытых версий.

Режимы: detect - только детектирование по эталону на каждом кадре;
        hybrid - детектирование + оптический поток + предсказание (по умолчанию).
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from tracker import DETECTORS, KeypointTracker, draw_hud, draw_result

ROOT = Path(__file__).parent
SYNTH = ("test1_easy", "test2_perspective", "test3_hard")
REAL = ("mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit")
MODES = {"detect": dict(use_flow=False, max_predict=0), "hybrid": dict(),
         "hybrid30": dict(max_predict=30)}  # для сравнения: предсказание до 1 с


def run(video: Path, detector: str, mode: str, save_to: Path | None = None, label="Object"):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    tracker = KeypointTracker(detector, **MODES[mode])
    tracker.init(frame)
    writer = None
    if save_to:
        writer = cv2.VideoWriter(str(save_to), cv2.VideoWriter_fourcc(*"mp4v"),
                                 cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
    sources, corners, inliers, times = [], [], [], []
    idx = 0
    while ok:
        t0 = time.perf_counter()
        res = tracker.update(frame)
        times.append(time.perf_counter() - t0)
        sources.append(res.source if res.found else "")
        corners.append(res.corners if res.found else np.full((4, 2), np.nan))
        inliers.append(res.n_inliers)
        if writer:
            vis = draw_result(frame, res, label, draw_points=True)
            draw_hud(vis, [f"frame {idx}  {detector.upper()}",
                           f"found ({res.source}), {res.n_inliers} inliers" if res.found else "not found"])
            writer.write(vis)
        ok, frame = cap.read()
        idx += 1
    if writer:
        writer.release()
    return np.array(sources), np.array(corners), np.array(inliers), np.array(times)


def corner_error(a, b):
    return np.linalg.norm(a - b, axis=2).mean(1)  # nan, где рамки нет


def eval_synthetic(detectors, save_video, out):
    rows = []
    for name in SYNTH:
        gt = np.load(ROOT / "videos" / f"{name}_gt.npz")
        vis_frac = gt["visible"]
        visible, absent = vis_frac >= 0.2, vis_frac < 0.05
        per_frame = {"frame": np.arange(len(vis_frac)), "visible": vis_frac.round(3)}
        for det in detectors:
            for mode in MODES:
                save = out / f"{name}_{det}.mp4" if det in save_video and mode == "hybrid" else None
                src, corners, inl, times = run(ROOT / "videos" / f"{name}.mp4", det, mode, save)
                found = src != ""
                err = corner_error(corners, gt["corners"])
                ok = found & visible
                rows.append(dict(
                    video=name, detector=det, mode=mode,
                    recall=ok.sum() / visible.sum(),
                    success10=(ok & (err < 10)).sum() / visible.sum(),
                    false_pos=int((found & absent).sum()), absent=int(absent.sum()),
                    err_median=np.median(err[ok]), err_p95=np.percentile(err[ok], 95),
                    by_flow=int((src == "flow").sum()), by_predict=int((src == "predict").sum()),
                    fps=1 / times[1:].mean()))
                per_frame[f"{det}_{mode}_err"] = err.round(2)
                per_frame[f"{det}_{mode}_src"] = src
                r = rows[-1]
                print(f"{name:18s} {det:6s} {mode:6s} recall={r['recall']:.3f} succ@10={r['success10']:.3f} "
                      f"FP={r['false_pos']}/{r['absent']} err_med={r['err_median']:.2f} "
                      f"p95={r['err_p95']:.2f} flow={r['by_flow']} pred={r['by_predict']} "
                      f"fps={r['fps']:.1f}", flush=True)
        write_csv(out / f"per_frame_{name}.csv", per_frame)

    md = ["| Видео | Детектор | Режим | Recall | Success@10px | Ложные срабатывания "
          "| Ошибка углов, медиана, px | Ошибка углов, 95-й перц., px | Кадров потоком / предсказано | FPS |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fp = f"{r['false_pos']} / {r['absent']}" if r["absent"] else "—"
        md.append(f"| {r['video']} | {r['detector'].upper()} | {r['mode']} | {100 * r['recall']:.1f} % | "
                  f"{100 * r['success10']:.1f} % | {fp} | {r['err_median']:.2f} | {r['err_p95']:.2f} | "
                  f"{r['by_flow']} / {r['by_predict']} | {r['fps']:.1f} |")
    (out / "metrics_synthetic.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_rows(out / "metrics_synthetic.csv", rows)


def eval_real(detectors, save_video, out):
    rows = []
    ref = {}
    for name in REAL:
        per_frame = {}
        for det in detectors:
            for mode in MODES:
                save = out / f"{name}_{det}.mp4" if det in save_video and mode == "hybrid" else None
                src, corners, inl, times = run(ROOT / "test-videos" / f"{name}.avi", det, mode, save, "Mona Lisa")
                if name == "mona-lisa":
                    ref[det, mode] = corners
                n = len(src)
                per_frame.setdefault("frame", np.arange(n))
                per_frame[f"{det}_{mode}_src"] = src
                per_frame[f"{det}_{mode}_inliers"] = inl
                r = dict(video=name, detector=det, mode=mode, frames=n,
                         found=(src != "").sum() / n,
                         by_detect=int((src == "detect").sum()), by_flow=int((src == "flow").sum()),
                         by_predict=int((src == "predict").sum()), fps=1 / times[1:].mean(),
                         err_blur_mean=np.nan, err_blur_max=np.nan)
                if name != "mona-lisa":
                    # опорная разметка - тот же трекер на чётком видео; ошибка на размытых кадрах
                    blurred = blurred_frames(name)
                    e = corner_error(corners, ref[det, "hybrid"])[blurred]
                    per_frame[f"{det}_{mode}_err_vs_sharp"] = corner_error(corners, ref[det, "hybrid"]).round(2)
                    r["blurred"] = int(blurred.sum())
                    r["found_blurred"] = int(np.isfinite(e).sum())
                    if np.isfinite(e).any():
                        r["err_blur_mean"], r["err_blur_max"] = np.nanmean(e), np.nanmax(e)
                rows.append(r)
                print(f"{name:28s} {det:6s} {mode:6s} found={100 * r['found']:.1f}% "
                      f"(detect {r['by_detect']}, flow {r['by_flow']}, predict {r['by_predict']}) "
                      f"blur err={r['err_blur_mean']:.1f}/{r['err_blur_max']:.1f} fps={r['fps']:.1f}", flush=True)
        write_csv(out / f"per_frame_{name}.csv", per_frame)

    md = ["| Видео | Детектор | Режим | Кадров с рамкой | по эталону / потоком / предсказано "
          "| Размытых кадров с рамкой | Ошибка на размытых кадрах, средняя / макс., px | FPS |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        blur = f"{r['found_blurred']} / {r['blurred']}" if "blurred" in r else "—"
        err = f"{r['err_blur_mean']:.1f} / {r['err_blur_max']:.1f}" if np.isfinite(r["err_blur_mean"]) else "—"
        md.append(f"| {r['video']} | {r['detector'].upper()} | {r['mode']} | {100 * r['found']:.1f} % | "
                  f"{r['by_detect']} / {r['by_flow']} / {r['by_predict']} | {blur} | {err} | {r['fps']:.1f} |")
    (out / "metrics_test_videos.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_rows(out / "metrics_test_videos.csv", rows)


_blur_cache = {}


def blurred_frames(name):
    """Номера кадров, которые в размытой версии отличаются от исходного видео."""
    if name not in _blur_cache:
        a = cv2.VideoCapture(str(ROOT / "test-videos" / "mona-lisa.avi"))
        b = cv2.VideoCapture(str(ROOT / "test-videos" / f"{name}.avi"))
        diff = []
        while True:
            ok1, fa = a.read()
            ok2, fb = b.read()
            if not (ok1 and ok2):
                break
            diff.append(np.abs(fa.astype(np.int16) - fb).mean())
        # в extra-credit все кадры немного отличаются (перекодирование, ~2), размытые - на порядок
        _blur_cache[name] = np.array(diff) > 5
    return _blur_cache[name]


def write_csv(path, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols.keys())
        w.writerows(zip(*cols.values()))


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--save-video", nargs="*", default=["sift"], help="для каких детекторов сохранять видео")
    ap.add_argument("--only", choices=("synthetic", "real"))
    a = ap.parse_args()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    if a.only != "synthetic":
        eval_real(a.detectors, a.save_video, out)
    if a.only != "real":
        eval_synthetic(a.detectors, a.save_video, out)


if __name__ == "__main__":
    main()
