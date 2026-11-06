"""Парное сравнение cv2.RANSAC и cv2.USAC_MAGSAC на одинаковых соответствиях."""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_config import ROOT, VIDEO_FILES, VIDEO_LABELS, validate_video
from tracker import DETECTORS, KeypointTracker, TrackResult, draw_result

OUT = ROOT / 'results' / 'estimators'
IMG = ROOT / 'images' / 'estimators'
METHODS = {'ransac': cv2.RANSAC, 'magsac': cv2.USAC_MAGSAC}
NAMES = {'ransac': 'RANSAC', 'magsac': 'MAGSAC++'}
REPEATS = 3
THRESHOLD = 4.0
MAX_ITERS = 5000
CONFIDENCE = .999


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def correspondences(tracker, frame):
    gray, scale = tracker._prepare(frame)
    kp, desc = tracker.detector.detectAndCompute(gray, None)
    n_keypoints = 0 if kp is None else len(kp)
    if desc is None or n_keypoints < tracker.min_inliers:
        return n_keypoints, np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), scale
    knn = tracker.matcher.knnMatch(desc, k=2)
    good = [pair[0] for pair in knn if len(pair) == 2
            and pair[0].distance < tracker.ratio * pair[1].distance]
    src = tracker.tpl_pts[[match.trainIdx for match in good]].reshape(-1, 2)
    dst = np.float32([kp[match.queryIdx].pt for match in good]).reshape(-1, 2)
    return n_keypoints, src, dst, scale


def estimate(tracker, src, dst, method, seed, scale):
    result = dict(attempted=0, found=0, inliers=0, accepted_inliers=0,
                  estimation_median_ms=0.0, reprojection_rmse_px='', reason='мало соответствий',
                  **{f'{axis}{i}': '' for i in range(4) for axis in 'xy'})
    if len(src) < tracker.min_inliers:
        return result
    result['attempted'] = 1
    samples = []
    H = mask = None
    for repeat in range(REPEATS):
        cv2.setRNGSeed(seed)
        start = time.perf_counter_ns()
        current_H, current_mask = cv2.findHomography(
            src, dst, METHODS[method], THRESHOLD, maxIters=MAX_ITERS, confidence=CONFIDENCE)
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if repeat == 0:
            H, mask = current_H, current_mask
        elif (H is None) != (current_H is None) or (
                H is not None and (not np.allclose(H, current_H) or not np.array_equal(mask, current_mask))):
            raise RuntimeError('Одинаковые входы и seed дали разные повторные оценки H')
    result['estimation_median_ms'] = float(np.median(samples))
    if H is None or mask is None:
        result['reason'] = 'гомография не найдена'
        return result
    inliers = mask.ravel().astype(bool)
    result['inliers'] = int(inliers.sum())
    corners = cv2.perspectiveTransform(tracker.tpl_corners[None], H)[0]
    reason = tracker._check(H, corners, result['inliers'], len(src), tracker.min_inliers)
    result['reason'] = reason
    if reason:
        return result
    result['found'] = 1
    result['accepted_inliers'] = result['inliers']
    projected = cv2.perspectiveTransform(src[inliers][None], H)[0]
    result['reprojection_rmse_px'] = float(np.sqrt(np.mean(np.sum((projected - dst[inliers]) ** 2, axis=1))) / scale)
    for i, point in enumerate(corners / scale):
        result[f'x{i}'], result[f'y{i}'] = map(float, point)
    return result


def run_pair(filename, detector):
    path = validate_video(ROOT / 'videos' / filename)
    video = path.stem
    cap = cv2.VideoCapture(str(path))
    rows, sources, destinations, offsets = [], [], [], [0]
    try:
        ok, template = cap.read()
        if not ok:
            raise RuntimeError(f'Нет первого кадра: {filename}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.setRNGSeed(42)
        tracker = KeypointTracker(detector, use_flow=False, max_predict=0)
        tracker.init(template)
        # Прогрев обеих реализаций вне измерений; кадр 0 не оценивается.
        for method in METHODS.values():
            cv2.findHomography(tracker.tpl_pts, tracker.tpl_pts, method, THRESHOLD,
                               maxIters=MAX_ITERS, confidence=CONFIDENCE)
        frame_index = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seed = 42 + frame_index
            cv2.setRNGSeed(seed)
            common_start = time.perf_counter_ns()
            n_keypoints, src, dst, scale = correspondences(tracker, frame)
            common_ms = (time.perf_counter_ns() - common_start) / 1e6
            digest = hashlib.sha256(src.tobytes() + dst.tobytes()).hexdigest()
            sources.append(src)
            destinations.append(dst)
            offsets.append(offsets[-1] + len(src))
            # Чередуем порядок вызова, чтобы один метод не всегда шёл первым.
            order = list(METHODS) if frame_index % 2 else list(reversed(METHODS))
            for method in order:
                result = estimate(tracker, src, dst, method, seed, scale)
                rows.append(dict(video=video, detector=detector, estimator=method,
                                 frame=frame_index, time_s=frame_index / fps, seed=seed,
                                 scale=scale, keypoints=n_keypoints, matches=len(src),
                                 matches_sha256=digest, common_ms=common_ms, **result))
            if frame_index % 100 == 0:
                print(f'{video} / {detector.upper()}: кадр {frame_index}', flush=True)
            frame_index += 1
    finally:
        cap.release()
    if not rows:
        raise RuntimeError(f'Нет оцениваемых кадров: {filename}')
    write_csv(OUT / 'per_frame' / f'{video}_{detector}.csv', rows)
    np.savez_compressed(OUT / 'matches' / f'{video}_{detector}.npz',
                        src=np.concatenate(sources), dst=np.concatenate(destinations),
                        offsets=np.array(offsets), frames=np.arange(1, frame_index))
    return rows


def summarise(rows):
    summary = []
    for filename in VIDEO_FILES:
        video = Path(filename).stem
        for detector in DETECTORS:
            group = [row for row in rows if row['video'] == video and row['detector'] == detector]
            pairs = {method: {int(row['frame']): row for row in group if row['estimator'] == method}
                     for method in METHODS}
            for method in METHODS:
                current = list(pairs[method].values())
                other = pairs['magsac' if method == 'ransac' else 'ransac']
                calls = [float(row['estimation_median_ms']) for row in current if int(row['attempted'])]
                summary.append(dict(video=video, detector=detector, estimator=method,
                                    frames=len(current), found=sum(int(r['found']) for r in current),
                                    coverage=sum(int(r['found']) for r in current) / len(current),
                                    attempted=len(calls),
                                    accepted_inliers_mean=float(np.mean([int(r['accepted_inliers']) for r in current])),
                                    estimation_mean_ms=float(np.mean(calls)) if calls else 0.0,
                                    estimation_p95_ms=float(np.percentile(calls, 95)) if calls else 0.0,
                                    only_this_found=sum(int(r['found']) and not int(other[int(r['frame'])]['found']) for r in current)))
    return summary


def draw_frame(video, row):
    filename = next(name for name in VIDEO_FILES if Path(name).stem == video)
    cap = cv2.VideoCapture(str(ROOT / 'videos' / filename))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame']))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Нет кадра {row["frame"]}: {video}')
    finally:
        cap.release()
    corners = np.array([[float(row[f'x{i}']), float(row[f'y{i}'])] for i in range(4)]) if int(row['found']) else None
    return draw_result(frame, TrackResult(bool(int(row['found'])), corners=corners,
                                         n_inliers=int(row['inliers']), source='detect'),
                       'Book' if video == 'my_video' else 'Mona Lisa')


def render():
    IMG.mkdir(exist_ok=True, parents=True)
    summary = read_csv(OUT / 'summary.csv')
    index = {(r['video'], r['detector'], r['estimator']): r for r in summary}
    selected = []
    for detector in DETECTORS:
        fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
        x = np.arange(len(VIDEO_FILES))
        for offset, (method, color) in zip((-.19, .19), [('ransac', '#2463b4'), ('magsac', '#df681c')]):
            values = [100 * float(index[Path(name).stem, detector, method]['coverage']) for name in VIDEO_FILES]
            bars = ax.bar(x + offset, values, .36, label=NAMES[method], color=color)
            ax.bar_label(bars, fmt='%.2f%%', fontsize=9, padding=3)
        ax.set(xticks=x, xticklabels=['Исходное', 'blur', 'extra credit', 'Своё видео'],
               ylabel='Кадров с принятой рамкой, %', ylim=(0, 109),
               title=f'{detector.upper()}: RANSAC и MAGSAC++\nОдинаковые соответствия, поток и прогноз выключены')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left')
        fig.savefig(IMG / f'{detector}.png', dpi=150, facecolor='white')
        plt.close(fig)

        fig, axes = plt.subplots(4, 2, figsize=(12, 17), layout='constrained')
        fig.get_layout_engine().set(h_pad=.12, hspace=.08)
        for axes_row, filename in zip(axes, VIDEO_FILES):
            video = Path(filename).stem
            rows = read_csv(OUT / 'per_frame' / f'{video}_{detector}.csv')
            pairs = {method: {r['frame']: r for r in rows if r['estimator'] == method} for method in METHODS}
            def difference(frame):
                a, b = pairs['ransac'][frame], pairs['magsac'][frame]
                distance = 0.0
                if int(a['found']) and int(b['found']):
                    corners = [np.array([[float(r[f'x{i}']), float(r[f'y{i}'])] for i in range(4)]) for r in (a, b)]
                    distance = float(np.mean(np.linalg.norm(corners[0] - corners[1], axis=1)))
                return (int(a['found']) != int(b['found']), distance, abs(int(a['inliers']) - int(b['inliers'])))
            frame = max(pairs['ransac'], key=difference)
            selected.append(dict(video=video, detector=detector, frame=int(frame),
                                 acceptance_differs=difference(frame)[0], corners_distance=difference(frame)[1],
                                 decisions={method: dict(found=int(pairs[method][frame]['found']),
                                                         reason=pairs[method][frame]['reason']) for method in METHODS}))
            for ax, method in zip(axes_row, METHODS):
                row = pairs[method][frame]
                ax.imshow(cv2.cvtColor(draw_frame(video, row), cv2.COLOR_BGR2RGB))
                status = 'рамка принята' if int(row['found']) else 'рамки нет'
                ax.set_title(f'{VIDEO_LABELS[video]} — {NAMES[method]}\nКадр {frame}: {status}, инлаеров {row["inliers"]}', fontsize=9)
                ax.axis('off')
        fig.suptitle(f'{detector.upper()}: один и тот же кадр, два способа оценки H', fontsize=12)
        fig.savefig(IMG / f'{detector}_frames.jpg', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    (OUT / 'selected_frames.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')


def report_lines():
    if not (OUT / 'summary.csv').exists():
        return []
    metadata = json.loads((OUT / 'experiment.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete':
        raise RuntimeError('Сравнение RANSAC / MAGSAC++ ещё не завершено')
    rows = read_csv(OUT / 'summary.csv')
    index = {(r['video'], r['detector'], r['estimator']): r for r in rows}
    lines = ['### 3.9. Дополнительно: RANSAC и MAGSAC++ для четырёх детекторов', '',
             'Здесь SIFT, ORB, AKAZE и BRISK остаются детекторами и дескрипторами, а RANSAC и MAGSAC++ сравниваются как два способа оценки гомографии по уже найденным соответствиям. Цель — проверить, как смена оценщика влияет на доступность рамки и время вычисления H.', '',
             '**Условия сравнения.** Исследованы все четыре разрешённых видео: 4 детектора × 4 видео × 2 оценщика = 32 комбинации. Первый кадр служит эталоном и исключён из метрик. Работает только поиск по эталону (`detect`), поток и прогноз выключены. На каждом кадре точки, дескрипторы, FLANN kNN и ratio-тест 0.75 вычисляются один раз; одни и те же пары в том же порядке передаются в `cv2.findHomography` с флагами `cv2.RANSAC` и `cv2.USAC_MAGSAC`. Хеши пар и сами координаты сохранены для проверки.', '',
             'Для обоих методов заданы `ransacReprojThreshold=4.0` px в масштабе обработки, `maxIters=5000`, `confidence=0.999`. Размер ограничен большей стороной 960 px. Рамка принимается по тем же проверкам, что и в основном трекере: минимум 15 инлаеров, их доля не меньше 25% и проверка геометрии четырёхугольника. Одинаковые параметры не означают одинаковые случайные выборки или одинаковый критерий качества моделей. В OpenCV порог используется и для остановки MAGSAC++; его не следует считать полностью беспороговым вызовом. См. [описание USAC в OpenCV](https://docs.opencv.org/4.13.0/de/d3e/tutorial_usac.html) и [параметры findHomography](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).', '',
             'Сравниваются готовые реализации OpenCV: классический `RANSAC` и `USAC_MAGSAC`. Они различаются не только оценкой инлаеров, но и внутренними процедурами проверки, уточнения модели и остановки. Поэтому измеренный эффект относится к смене всей реализации оценки H, а не к изолированной замене одной формулы.', '',
             '**Метрики.** На каждом графике показан один детектор: два столбца на каждое видео — доля кадров с принятой рамкой. «Инлаеры» в таблицах — среднее число инлаеров принятой рамки по всем оцениваемым кадрам; без рамки считается 0. «Время H» — среднее из медиан трёх повторных вызовов `findHomography` на каждом кадре, где было достаточно соответствий. Детектирование, сопоставление, проверки рамки, чтение и рисование в это время не включены; это не время обработки полного кадра и не FPS.', '',
             f"**Воспроизводимость.** Отдельный парный прогон: Python {metadata['python']}, OpenCV {metadata['opencv']}, NumPy {metadata['numpy']}; один поток OpenCV. Seed для кадра — `42 + номер кадра`; он сбрасывается перед сопоставлением и перед каждым повтором оценки. Порядок вызова методов чередуется по кадрам. Дата начала UTC: {metadata['started_utc']}. Это один проход по видео с тремя повторами оценки H, без подбора порога и доверительных интервалов. Результаты относятся к этому прогону, а не к старым CSV разделов 3.2–3.3.", '']
    total_advantages = []
    for detector in DETECTORS:
        lines += [f'#### {detector.upper()}: RANSAC и MAGSAC++', '',
                  f'![{detector.upper()}: сравнение RANSAC и MAGSAC++](images/estimators/{detector}.png)', '',
                  '| Видео | Рамка RANSAC / MAGSAC++ | Инлаеры RANSAC / MAGSAC++ | Время H RANSAC / MAGSAC++, мс | Только RANSAC / только MAGSAC++, кадров |',
                  '|---|---:|---:|---:|---:|']
        advantages = []
        for filename in VIDEO_FILES:
            video = Path(filename).stem
            a, b = [index[video, detector, method] for method in METHODS]
            advantages.append((video, int(b['found']) - int(a['found'])))
            lines.append(f"| {VIDEO_LABELS[video]} | {100*float(a['coverage']):.2f}% / {100*float(b['coverage']):.2f}% | "
                         f"{float(a['accepted_inliers_mean']):.2f} / {float(b['accepted_inliers_mean']):.2f} | "
                         f"{float(a['estimation_mean_ms']):.3f} / {float(b['estimation_mean_ms']):.3f} | {a['only_this_found']} / {b['only_this_found']} |")
        gained = sum(max(delta, 0) for _, delta in advantages)
        lost = sum(max(-delta, 0) for _, delta in advantages)
        equal = sum(delta == 0 for _, delta in advantages)
        a_rows = [index[Path(name).stem, detector, 'ransac'] for name in VIDEO_FILES]
        b_rows = [index[Path(name).stem, detector, 'magsac'] for name in VIDEO_FILES]
        elapsed = [sum(float(r['estimation_mean_ms']) * int(r['attempted']) for r in group)
                   / sum(int(r['attempted']) for r in group) for group in (a_rows, b_rows)]
        speed = 'MAGSAC++' if elapsed[1] < elapsed[0] else 'RANSAC'
        lines += ['', f'**Вывод для {detector.upper()}.** При переходе от RANSAC к MAGSAC++ сумма приростов числа кадров с рамкой по видео составляет {gained}, сумма снижений — {lost}; одинаковое итоговое покрытие получено на {equal} из 4 видео. Среднее время оценки H с учётом числа вызовов — {elapsed[0]:.3f} мс у RANSAC и {elapsed[1]:.3f} мс у MAGSAC++; быстрее в этом замере {speed}. Равное покрытие не означает совпадения рамок или наборов инлаеров.', '',
                  f'![{detector.upper()}: примеры рамок RANSAC и MAGSAC++ на одинаковых кадрах](images/estimators/{detector}_frames.jpg)', '',
                  'Слева RANSAC, справа MAGSAC++. Для каждого видео выбран один кадр с различием принятия рамки; если такого нет — с наибольшим расхождением углов среди общих рамок, затем по различию числа инлаеров. Это иллюстрации различий, а не независимая проверка правильности рамки.', '']
        total_advantages.extend(advantages)
    better = sum(delta > 0 for _, delta in total_advantages)
    worse = sum(delta < 0 for _, delta in total_advantages)
    equal = sum(delta == 0 for _, delta in total_advantages)
    net = sum(delta for _, delta in total_advantages)
    lines += [f'**Общий вывод.** На 16 сочетаниях «детектор × видео» MAGSAC++ даёт больше кадров с принятой рамкой в {better} случаях, меньше — в {worse}, столько же — в {equal}. Суммарная разница числа принятых рамок MAGSAC++ минус RANSAC равна {net:+d}. Это изменение доступности оценки при фиксированных соответствиях и настройках. Независимой разметки углов нет: большее покрытие и больше инлаеров не доказывают более высокую истинную точность. Разница времени относится только к оценке H. Общий вывод о превосходстве метода для любых сцен или порогов по этому опыту сделать нельзя.', '',
              'Воспроизведение: `python compare_estimators.py`, затем `python report.py`. Перерисовать иллюстрации из сохранённых результатов без повторного анализа видео: `python compare_estimators.py --render-only`. Полный цикл `run_experiments.py` также включает это сравнение.', '',
              'Данные: [сводная CSV](results/estimators/summary.csv), [условия прогона](results/estimators/experiment.json), [выбранные кадры](results/estimators/selected_frames.json). Подробные логи: `results/estimators/per_frame/<видео>_<детектор>.csv`; архивы общих соответствий: `results/estimators/matches/<видео>_<детектор>.npz`.', '']
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        render()
        return
    for path in (OUT, OUT / 'per_frame', OUT / 'matches', IMG):
        path.mkdir(exist_ok=True, parents=True)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                    threads=1, seed='42 + frame', threshold=THRESHOLD, max_iters=MAX_ITERS,
                    confidence=CONFIDENCE, ratio=.75, min_inliers=15, min_inlier_ratio=.25,
                    max_side=960, estimation_repeats=REPEATS, videos=list(VIDEO_FILES),
                    detectors=list(DETECTORS), methods=METHODS, excluded_frame=0)
    meta_path = OUT / 'experiment.json'
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for detector in DETECTORS:
        for filename in VIDEO_FILES:
            print(f'Начало: {detector.upper()} / {filename}', flush=True)
            current = run_pair(filename, detector)
            rows.extend(current)
            print(f'Завершено: {detector.upper()} / {filename}, {len(current)//2} кадров', flush=True)
    summary = summarise(rows)
    write_csv(OUT / 'summary.csv', summary)
    render()
    metadata.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started, combinations=len(summary),
                    evaluated_pairs=len(rows) // 2)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Готово: {len(summary)} комбинации, четыре графика.', flush=True)


if __name__ == '__main__':
    main()
