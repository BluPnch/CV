"""Полный эксперимент: четыре разрешённых видео, четыре детектора, три режима.
Первый кадр служит эталоном и исключается из итоговых метрик.
"""
import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from experiment_config import ROOT, VIDEO_FILES, validate_video
from tracker import DETECTORS, KeypointTracker, draw_hud, draw_result

MODES = {
    'detect': dict(use_flow=False, max_predict=0),
    'flow': dict(use_flow=True, max_predict=0),
    'hybrid': dict(use_flow=True, max_predict=15),
}


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def longest_run(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def run(video, detector, mode, save_to=None):
    validate_video(video)
    cv2.setRNGSeed(42)
    cap = cv2.VideoCapture(str(video))
    writer = None
    records, corners = [], []
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Не удалось прочитать первый кадр {video}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tracker = KeypointTracker(detector, **MODES[mode])
        n_template = tracker.init(frame)
        if save_to:
            writer = cv2.VideoWriter(str(save_to), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f'Не удалось создать {save_to}')
        idx = 0
        while ok:
            t0 = time.perf_counter()
            res = tracker.update(frame)
            elapsed = 1000 * (time.perf_counter() - t0)
            quad = res.corners if res.found else np.full((4, 2), np.nan)
            corners.append(quad.copy())
            records.append(dict(
                frame=idx, time_s=idx / fps, evaluated=int(idx > 0),
                found=int(res.found), measured=int(res.found and res.source != 'predict'),
                source=res.source if res.found else '', keypoints=res.n_keypoints,
                matches=res.n_matches, inliers=res.n_inliers, processing_ms=elapsed,
                **{f'{axis}{i}': float(quad[i, j])
                   for i in range(4) for j, axis in enumerate('xy')},
            ))
            if writer:
                label = 'Book' if video.stem == 'my_video' else 'Mona Lisa'
                vis = draw_result(frame, res, label)
                status = res.source if res.found else 'lost'
                draw_hud(vis, [f'{detector.upper()} / {mode} / frame {idx}',
                               f'{status} / {res.n_inliers} inliers'])
                writer.write(vis)
            ok, frame = cap.read()
            idx += 1
        if len(records) < 2:
            raise RuntimeError(f'Для оценки нужно хотя бы два кадра: {video}')
        return records, np.asarray(corners), fps, n_template
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def blur_difference(sharp, blurred):
    """Различие синхронных версий; порог 5 — эвристика, не разметка блюра."""
    a, b = cv2.VideoCapture(str(sharp)), cv2.VideoCapture(str(blurred))
    difference = []
    try:
        if not np.isclose(a.get(cv2.CAP_PROP_FPS), b.get(cv2.CAP_PROP_FPS)):
            raise ValueError('Версии видео имеют разную частоту кадров')
        while True:
            oka, fa = a.read()
            okb, fb = b.read()
            if oka != okb:
                raise ValueError('Версии видео имеют разную длину')
            if not oka:
                break
            if fa.shape != fb.shape:
                raise ValueError('Версии видео имеют разные размеры кадров')
            difference.append(float(np.abs(fa.astype(np.int16) - fb).mean()))
    finally:
        a.release()
        b.release()
    return np.asarray(difference)


def summarize(records, video, det, mode, fps, n_template):
    rows = records[1:]
    sources = np.array([r['source'] for r in rows])
    ms = np.array([r['processing_ms'] for r in rows])
    measured = np.array([r['measured'] for r in rows], bool)
    return dict(
        video=video.stem, detector=det, mode=mode, frames=len(records),
        evaluated_frames=len(rows), input_fps=fps, template_keypoints=n_template,
        found=float(np.mean(sources != '')), measured=float(measured.mean()),
        by_detect=int((sources == 'detect').sum()), by_flow=int((sources == 'flow').sum()),
        by_predict=int((sources == 'predict').sum()), fps=1000 / ms.mean(),
        latency_median_ms=float(np.median(ms)), latency_p95_ms=float(np.percentile(ms, 95)),
        longest_lost=longest_run(sources == ''), longest_unmeasured=longest_run(~measured),
        keypoints_mean=float(np.mean([r['keypoints'] for r in rows])),
    )


def eval_videos(detectors, save_video, out):
    videos = [validate_video(ROOT / 'videos' / name) for name in VIDEO_FILES]
    logs = out / 'per_frame'
    logs.mkdir(parents=True, exist_ok=True)
    differences = {video.stem: blur_difference(videos[0], video) for video in videos[1:3]}
    for name, diff in differences.items():
        write_rows(out / f'blur_mask_{name}.csv', [
            dict(frame=i, mean_abs_difference=value, changed=int(value > 5))
            for i, value in enumerate(diff)])
    metrics, references = [], {}
    for video in videos:
        combined = None
        for det in detectors:
            for mode in MODES:
                save = out / f'{video.stem}_{det}.mp4' if det in save_video and mode == 'hybrid' else None
                records, corners, fps, n_tpl = run(video, det, mode, save)
                row = summarize(records, video, det, mode, fps, n_tpl)
                if video == videos[0]:
                    references[det, mode] = corners
                elif video.stem in differences:
                    reference = references[det, mode]
                    if len(reference) != len(corners):
                        raise ValueError('Несовпадение кадров опорного и размытого видео')
                    errors = np.linalg.norm(corners - reference, axis=2).mean(axis=1)
                    mask = differences[video.stem] > 5
                    mask[0] = False
                    paired = mask & np.isfinite(errors)
                    measured = np.array([r['measured'] for r in records], bool)
                    row.update(blurred=int(mask.sum()), paired_blurred=int(paired.sum()),
                               found_blurred=int(sum(r['found'] for r, m in zip(records, mask) if m)),
                               measured_blurred=int((measured & mask).sum()),
                               err_blur_mean=float(errors[paired].mean()) if paired.any() else np.nan,
                               err_blur_max=float(errors[paired].max()) if paired.any() else np.nan)
                    for record, error in zip(records, errors):
                        record['err_vs_sharp'] = float(error)
                write_rows(logs / f'{video.stem}_{det}_{mode}.csv', records)
                if combined is None:
                    combined = [dict(frame=r['frame']) for r in records]
                for dst, record in zip(combined, records):
                    dst.update({f'{det}_{mode}_{k}': v for k, v in record.items() if k != 'frame'})
                metrics.append(row)
                write_rows(out / 'metrics_test_videos.csv', metrics)
                print(f"{video.stem:28s} {det:5s} {mode:6s} "
                      f"frame={100*row['found']:.1f}% measured={100*row['measured']:.1f}% "
                      f"FPS={row['fps']:.1f}", flush=True)
        write_rows(out / f'per_frame_{video.stem}.csv', combined)
    header = ['| Видео | Детектор | Режим | Кадров с рамкой, % | Измерено, % | Прогноз, кадров | FPS | P95, мс | Макс. потеря, кадров |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in metrics:
        header.append(f"| {r['video']} | {r['detector'].upper()} | {r['mode']} | "
                      f"{100*r['found']:.2f} | {100*r['measured']:.2f} | {r['by_predict']} | "
                      f"{r['fps']:.2f} | {r['latency_p95_ms']:.2f} | {r['longest_lost']} |")
    (out / 'metrics_test_videos.md').write_text('\n'.join(header) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detectors', nargs='+', choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument('--save-video', nargs='*', choices=DETECTORS, default=list(DETECTORS))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                    platform=platform.platform(), processor=platform.processor(), threads=1,
                    rng_seed=42, ratio=0.75, min_inliers=15, max_side=960,
                    modes=MODES, detectors=args.detectors, save_video=args.save_video,
                    excluded_template_frame=0,
                    videos=[dict(name=name, bytes=(ROOT / 'videos' / name).stat().st_size)
                            for name in VIDEO_FILES])
    (out / 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    eval_videos(args.detectors, args.save_video, out)


if __name__ == '__main__':
    main()
