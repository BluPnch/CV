"""Кадры характерных экстремумов из сохранённых логов, без повторного трекинга."""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_config import ROOT, VIDEO_FILES, MONA_VIDEOS, VIDEO_LABELS
from tracker import DETECTORS, TrackResult, draw_result

OUT = ROOT / 'images' / 'events'
MANIFEST = ROOT / 'results' / 'frame_examples.json'
LABELS = {'coverage': 'Доля кадров с рамкой', 'inliers': 'Принятые инлаеры',
          'keypoints': 'Ключевые точки', 'latency': 'Время обработки'}
FIELDS = {'coverage': 'found', 'inliers': 'inliers', 'keypoints': 'keypoints',
          'latency': 'processing_ms'}


def extrema(values):
    """Заметные повороты кривой; равные значения объединяются в плато."""
    values = np.asarray(values)
    cuts = np.flatnonzero(np.abs(np.diff(values)) > 1e-9) + 1
    runs = list(zip(np.r_[0, cuts], np.r_[cuts - 1, len(values) - 1]))
    candidates = {'min': [], 'max': []}
    for start, end in runs:
        value = values[start]
        left = values[start - 1] if start else None
        right = values[end + 1] if end + 1 < len(values) else None
        for kind, sign in [('min', -1), ('max', 1)]:
            if left is not None and sign * (value - left) <= 0:
                continue
            if right is not None and sign * (value - right) <= 0:
                continue
            if left is None and right is None:
                continue
            changes = []
            if left is not None:
                changes.append(np.max(sign * (value - values[max(0, start - 60):start])))
            if right is not None:
                changes.append(np.max(sign * (value - values[end + 1:end + 61])))
            # Для плато берём его начало (минимум) / конец (максимум),
            # чтобы кадр был ближе к переходу, а не к середине длинной потери.
            pos = int(start if kind == 'min' else end)
            candidates[kind].append((float(min(changes)), pos, start == 0 or end == len(values) - 1))
    selected = []
    for kind in ('min', 'max'):
        options = [item for item in candidates[kind] if not item[2]] or candidates[kind]
        if options:
            score, pos, boundary = max(options, key=lambda item: (item[0], -item[1]))
        else:
            pos = int(np.argmin(values) if kind == 'min' else np.argmax(values))
            boundary = True
        selected.append((kind, pos, bool(boundary)))
    return selected


@lru_cache(maxsize=256)
def input_frame(video, index):
    filename = next(name for name in VIDEO_FILES if Path(name).stem == video)
    cap = cv2.VideoCapture(str(ROOT / 'videos' / filename))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Не прочитан кадр {index}: {filename}')
        return frame
    finally:
        cap.release()


def make_event(video, detector, mode, row, label, **extra):
    return dict(video=video, detector=detector, mode=mode, frame=int(row['frame']),
                time_s=float(row['time_s']), source=row['source'] or 'lost',
                found=int(row['found']), keypoints=int(row['keypoints']),
                inliers=int(row['inliers']) if row['found'] == '1' else 0,
                processing_ms=float(row['processing_ms']),
                corners=[[float(row[f'x{i}']), float(row[f'y{i}'])] for i in range(4)]
                if row['found'] == '1' else None, label=label, **extra)


def event_image(event):
    frame = input_frame(event['video'], event['frame'])
    result = TrackResult(bool(event['found']),
                         corners=np.array(event['corners']) if event['found'] else None,
                         n_inliers=event['inliers'], source=event['source'])
    return draw_result(frame, result, 'Book' if event['video'] == 'my_video' else 'Mona Lisa')


def render(events, filename, title, columns=2):
    rows = (len(events) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 4.4 + .4),
                             squeeze=False, layout='constrained')
    for ax, event in zip(axes.flat, events):
        ax.imshow(cv2.cvtColor(event_image(event), cv2.COLOR_BGR2RGB))
        context = f"{event['detector'].upper()} / {event['mode']} / {event['source']}"
        values = f"\nГрафик: {event['plotted']:.2f}; на кадре: {event['raw']:.2f}" if 'plotted' in event else ''
        ax.set_title(f"{event['label']}\nКадр {event['frame']} ({event['time_s']:.2f} с), {context}{values}", fontsize=9)
        ax.axis('off')
    for ax in list(axes.flat)[len(events):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.get_layout_engine().set(h_pad=.12, w_pad=.06, hspace=.08, wspace=.04)
    path = OUT / filename
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path.relative_to(ROOT).as_posix()


def curve_events(data, series, video, detector, metric, curve):
    rows = data(video, detector)
    values = series(rows, metric)
    events = []
    for kind, pos, boundary in extrema(values):
        tag = 'Минимум' if kind == 'min' else 'Максимум'
        if boundary:
            tag += ' у края графика'
        event = make_event(video, detector, 'detect', rows[pos], f'{curve}: {tag.lower()}',
                           metric=metric, curve=curve, kind=kind, boundary=boundary,
                           plotted=float(values[pos]), raw=float(rows[pos][FIELDS[metric]]))
        if metric == 'inliers' and not event['found']:
            event['raw'] = 0.0
        events.append(event)
    return events


def longest_interval(rows, field):
    best, current = [], []
    for row in rows:
        if row[field] == '0':
            current.append(row)
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []
    return best


def build_examples(data, series):
    OUT.mkdir(exist_ok=True, parents=True)
    manifest = {'graphs': {}, 'sections': {}}
    for video in MONA_VIDEOS:
        for metric in LABELS:
            events = [event for det in DETECTORS
                      for event in curve_events(data, series, video, det, metric, det.upper())]
            key = f'detectors/{video}_{metric}'
            image = render(events, f'{video}_{metric}.jpg', f'{VIDEO_LABELS[video]} — {LABELS[metric]}')
            manifest['graphs'][key] = dict(image=image, events=events)
    for det in DETECTORS:
        for metric in ('coverage', 'inliers', 'keypoints'):
            events = [event for video in MONA_VIDEOS
                      for event in curve_events(data, series, video, det, metric, VIDEO_LABELS[video])]
            key = f'blur/{det}_{metric}'
            image = render(events, f'{det}_{metric}.jpg', f'{det.upper()} — {LABELS[metric]}')
            manifest['graphs'][key] = dict(image=image, events=events)

    # Сводные диаграммы не имеют оси времени: иллюстрируем конкретные
    # события из логов, не называя их экстремумами столбчатого графика.
    events = []
    for video in VIDEO_LABELS:
        hybrid = data(video, 'sift', 'hybrid')
        for source in ('flow', 'predict'):
            choices = [r for r in hybrid if r['source'] == source]
            if not choices:
                continue
            row = choices[len(choices) // 2]
            for mode in ('detect', 'flow', 'hybrid'):
                matched = next(r for r in data(video, 'sift', mode) if r['frame'] == row['frame'])
                events.append(make_event(video, 'sift', mode, matched,
                                         f'{VIDEO_LABELS[video]}: {source} в hybrid'))
    manifest['sections']['3.4'] = dict(image=render(events, 'modes.jpg', 'Один кадр в трёх режимах', 3), events=events)

    events = []
    for video in MONA_VIDEOS[1:]:
        mask_path = ROOT / 'results' / f'blur_mask_{video}.csv'
        import csv
        with mask_path.open(encoding='utf-8-sig', newline='') as file:
            changed = {r['frame'] for r in csv.DictReader(file) if r['changed'] == '1'}
        for det in DETECTORS:
            choices = [r for r in data(video, det, 'hybrid')
                       if r['frame'] in changed and r['err_vs_sharp']
                       and np.isfinite(float(r['err_vs_sharp']))]
            row = max(choices, key=lambda r: float(r['err_vs_sharp']))
            error = float(row['err_vs_sharp'])
            original = next(r for r in data('mona-lisa', det, 'hybrid') if r['frame'] == row['frame'])
            for name, chosen in [('mona-lisa', original), (video, row)]:
                events.append(make_event(name, det, 'hybrid', chosen, VIDEO_LABELS[name],
                                         error=error, comparison=video))
    manifest['sections']['3.5'] = dict(image=render(events, 'blur_error.jpg', 'Кадры максимального расхождения углов', 2), events=events)

    events = []
    for video in VIDEO_LABELS:
        choices = []
        for det in DETECTORS:
            for mode in ('detect', 'flow', 'hybrid'):
                rows = data(video, det, mode)
                interval = longest_interval(rows, 'measured')
                if interval:
                    choices.append((len(interval), det, mode, interval, rows))
        length, det, mode, interval, rows = max(choices, key=lambda item: item[0])
        start, end = int(interval[0]['frame']), int(interval[-1]['frame'])
        for target in (max(1, start - 1), start, end):
            row = next(r for r in rows if int(r['frame']) == target)
            events.append(make_event(video, det, mode, row,
                                     f'{VIDEO_LABELS[video]}: потеря {start}–{end}',
                                     interval=[start, end], length=length, event_type='loss'))
    manifest['sections']['loss'] = dict(image=render(events, 'loss.jpg', 'Начало и конец самых длинных серий без измерения', 3), events=events)

    events = []
    for video in VIDEO_LABELS:
        choices = [(float(row['processing_ms']), det, mode, row)
                   for det in DETECTORS for mode in ('detect', 'flow', 'hybrid')
                   for row in data(video, det, mode)]
        for title, selected in [('Минимальное время', min(choices, key=lambda item: item[0])),
                                ('Максимальное время', max(choices, key=lambda item: item[0]))]:
            elapsed, det, mode, row = selected
            events.append(make_event(video, det, mode, row,
                                     f'{VIDEO_LABELS[video]}: {title.lower()}', elapsed=elapsed))
    manifest['sections']['latency'] = dict(image=render(events, 'latency.jpg', 'Кадры минимального и максимального времени обработки', 2), events=events)

    events = []
    for det in DETECTORS:
        rows = data('my_video', det)
        missing = [r for r in rows if r['found'] == '0']
        interval = longest_interval(rows, 'found')
        minimum = interval[len(interval) // 2] if interval else min(rows, key=lambda r: int(r['inliers']))
        maximum = max(rows, key=lambda r: int(r['inliers']) if r['found'] == '1' else 0)
        for row, tag in [(minimum, 'Нет рамки' if missing else 'Мало инлаеров'), (maximum, 'Больше всего инлаеров')]:
            events.append(make_event('my_video', det, 'detect', row, f'{det.upper()}: {tag.lower()}'))
    manifest['sections']['3.7'] = dict(image=render(events, 'own_video.jpg', 'Собственное видео: потеря рамки и уверенное сопоставление', 2), events=events)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Кадры событий: {len(manifest["graphs"])} сравнений и {len(manifest["sections"])} дополнительных подборок.')
    return manifest


def graph_markdown(manifest, key):
    graph = manifest['graphs'][key]
    lines = ['**Кадры характерных минимумов и максимумов.**', '',
             f'![Кадры экстремумов: {key}]({graph["image"]})', '',
             '| Линия графика | Минимум: кадр / значение | Максимум: кадр / значение |',
             '|---|---|---|']
    events = graph['events']
    for minimum, maximum in zip(events[::2], events[1::2]):
        def cell(event):
            edge = ' (край)' if event['boundary'] else ''
            return f"≈{event['frame']} / {event['plotted']:.2f}{edge}"
        lines.append(f'| {minimum["curve"]} | {cell(minimum)} | {cell(maximum)} |')
    return lines + ['']


def section_markdown(manifest, key):
    section = manifest['sections'][key]
    lines = [f'![Кадры событий исследования {key}]({section["image"]})', '']
    if key == '3.4':
        lines += ['В каждой строке показан один и тот же кадр в режимах `detect`, `flow` и `hybrid` (слева направо). Источник положения подписан над кадром; `lost` означает отсутствие принятой рамки. Столбчатая диаграмма не имеет оси времени: эти примеры показывают события, формирующие итоговое покрытие.', '']
    elif key == '3.5':
        lines += ['Слева — исходное видео, справа — изменённое в тот же момент. Для каждого детектора выбран изменённый кадр с максимальным расхождением углов в режиме `hybrid`.', '',
                  '| Вариант | Детектор | Кадр | Расхождение углов, px |', '|---|---|---:|---:|']
        for event in section['events'][1::2]:
            lines.append(f"| {VIDEO_LABELS[event['video']]} | {event['detector'].upper()} | {event['frame']} | {event['error']:.2f} |")
        lines += ['', 'Расхождение измеряется относительно рамки трекера на исходном видео; это не ошибка относительно ручной разметки.', '']
    elif key == 'loss':
        lines += ['В каждой строке — кадр перед потерей, первый и последний кадры самой длинной серии без измерения для данного видео (выбор среди всех детекторов и режимов).', '',
                  '| Видео | Детектор / режим | Серия без измерения | Длина, кадров |', '|---|---|---|---:|']
        for event in section['events'][::3]:
            start, end = event['interval']
            lines.append(f"| {VIDEO_LABELS[event['video']]} | {event['detector'].upper()} / {event['mode']} | {start}–{end} | {event['length']} |")
        lines += ['']
    elif key == 'latency':
        lines += ['Для каждого видео показаны кадры минимального и максимального времени обработки среди всех детекторов и режимов. Это отдельные замеры; высота столбца FPS или P95 не соответствует одному кадру. Задержка зависит также от нагрузки компьютера, поэтому по содержимому фотографии нельзя установить причину скачка.', '',
                  '| Видео | Кадр | Детектор / режим | Время, мс |', '|---|---:|---|---:|']
        for event in section['events']:
            lines.append(f"| {VIDEO_LABELS[event['video']]} | {event['frame']} | {event['detector'].upper()} / {event['mode']} | {event['elapsed']:.2f} |")
        lines += ['']
    elif key == '3.7':
        lines += ['Для каждого детектора слева показан кадр из самой длинной серии без рамки в режиме `detect`, справа — кадр с максимальным числом принятых инлаеров. Это примеры к итоговым показателям покрытия; отсутствие рамки не означает отсутствия объекта на изображении.', '']
    return lines


if __name__ == '__main__':
    from figures import data, series
    build_examples(data, series)
