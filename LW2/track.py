# Запуск трекера

import argparse
import csv
import time
from pathlib import Path

import cv2

from tracker import DETECTORS, KeypointTracker, draw_hud, draw_result
from experiment_config import validate_video


def parse_args():
    p = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам")
    p.add_argument("video", help="входное видео; первый кадр - крупное изображение объекта")
    p.add_argument("-o", "--output", help="выходное видео (по умолчанию results/manual/<имя>_<детектор>.mp4)")
    p.add_argument("--detector", default="sift", choices=DETECTORS)
    p.add_argument("--label", default="Object", help="подпись у рамки")
    p.add_argument("--roi", help="область объекта на первом кадре: x,y,w,h (по умолчанию весь кадр)")
    p.add_argument("--select", action="store_true", help="выбрать область объекта мышью на первом кадре")
    p.add_argument("--max-side", type=int, default=960, help="размер обработки по большей стороне")
    p.add_argument("--ratio", type=float, default=0.75, help="порог ratio-теста Лоу")
    p.add_argument("--min-inliers", type=int, default=15)
    p.add_argument("--no-flow", action="store_true",
                   help="только детектирование, без сопровождения оптическим потоком")
    p.add_argument("--max-predict", type=int, default=15,
                   help="сколько кадров предсказывать положение потерянного объекта (0 - выкл.)")
    p.add_argument("--points", action="store_true", help="рисовать точки-инлаеры")
    p.add_argument("--show", action="store_true", help="показывать результат в окне (q/Esc - выход)")
    p.add_argument("--csv", help="сохранить покадровый лог (углы рамки, число точек) в CSV")
    return p.parse_args()


def main():
    a = parse_args()
    try:
        validate_video(a.video)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Видео пустое")

    roi = None
    if a.select:
        r = cv2.selectROI("Select object", first, showCrosshair=False)
        cv2.destroyWindow("Select object")
        roi = tuple(r) if r[2] > 0 and r[3] > 0 else None
    elif a.roi:
        roi = tuple(int(v) for v in a.roi.split(","))

    tracker = KeypointTracker(a.detector, ratio=a.ratio, min_inliers=a.min_inliers,
                              max_side=a.max_side, use_flow=not a.no_flow,
                              max_predict=0 if a.no_flow else a.max_predict)
    n_tpl = tracker.init(first, roi)
    print(f"{a.video}: {first.shape[1]}x{first.shape[0]}, {n_total} кадров, {fps:.1f} fps; "
          f"эталон: {n_tpl} ключевых точек ({a.detector})")

    out_path = Path(a.output) if a.output else \
        Path(__file__).parent / "results" / "manual" / f"{Path(a.video).stem}_{a.detector}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Не удалось создать видео {out_path}")

    log = []
    frame, idx, t_sum = first, 0, 0.0
    n_found = {"detect": 0, "flow": 0, "predict": 0}
    while True:
        t0 = time.perf_counter()
        res = tracker.update(frame)
        dt = time.perf_counter() - t0
        t_sum += dt
        if res.found:
            n_found[res.source] += 1

        vis = draw_result(frame, res, a.label, draw_points=a.points)
        status = f"found ({res.source}), {res.n_inliers} inliers" if res.found else "not found"
        draw_hud(vis, [f"frame {idx}  {a.detector.upper()}  {1 / dt:.0f} fps", status])
        writer.write(vis)

        c = [round(float(v), 1) for v in res.corners.ravel()] if res.found else [""] * 8
        log.append([idx, int(res.found), res.source if res.found else "", res.n_keypoints, res.n_matches, res.n_inliers, *c])

        if a.show:
            k = max(vis.shape[:2])
            cv2.imshow("tracking", vis if k <= 1280 else cv2.resize(vis, None, fx=1280 / k, fy=1280 / k))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    n = idx + 1
    total = sum(n_found.values())
    print(f"объект найден на {total}/{n} кадрах ({100 * total / n:.1f} %; по эталону {n_found['detect']}, "
          f"потоком {n_found['flow']}, предсказано {n_found['predict']}), "
          f"средняя скорость {n / t_sum:.1f} кадр/с -> {out_path}")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "found", "source", "keypoints", "matches", "inliers",
                        "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
            w.writerows(log)


if __name__ == "__main__":
    main()
