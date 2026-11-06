"""24 основных сравнительных графика и дополнительные исследования из CSV.
Запуск: python evaluate.py, затем python figures.py. Видео повторно не анализируются.
"""
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_config import ROOT, VIDEO_FILES, MONA_VIDEOS, VIDEO_LABELS
from tracker import DETECTORS
from evaluate import MODES

RES, IMG = ROOT / 'results', ROOT / 'images'
COLORS = dict(zip(DETECTORS, ['#2463b4', '#df681c', '#148665', '#a249a7']))
BLUR_COLORS = dict(zip(MONA_VIDEOS, ['#2463b4', '#df681c', '#a249a7']))
MODE_LABELS = {'detect': 'Детектор', 'flow': 'Детектор + поток', 'hybrid': 'Детектор + поток + прогноз'}
METRICS = {
    'coverage': ('Накопленная доля кадров с рамкой, %', 'found'),
    'inliers': ('Принятые инлаеры (среднее за 21 кадр)', 'inliers'),
    'keypoints': ('Ключевые точки (среднее за 21 кадр)', 'keypoints'),
    'latency': ('Обработка, мс (среднее за 21 кадр)', 'processing_ms'),
}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def data(video, det, mode='detect'):
    return read_csv(RES / 'per_frame' / f'{video}_{det}_{mode}.csv')[1:]


def numeric(rows, field):
    return np.array([float(r[field]) for r in rows])


def series(rows, metric):
    key = METRICS[metric][1]
    values = numeric(rows, key)
    if metric == 'coverage':
        return 100 * np.cumsum(values) / np.arange(1, len(values) + 1)
    if metric == 'inliers':
        values = values * numeric(rows, 'found')
    window = min(21, len(values))
    weights = np.ones(window)
    return np.convolve(values, weights, 'same') / np.convolve(np.ones(len(values)), weights, 'same')


def finish(fig, path):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def shade_changed(ax, video):
    if video == 'mona-lisa':
        return
    mask = read_csv(RES / f'blur_mask_{video}.csv')
    changed = [int(r['frame']) for r in mask if r['changed'] == '1']
    for i in changed:
        ax.axvspan(i - .5, i + .5, color='#bbbbbb', alpha=.23, lw=0)


def comparisons():
    gallery = ['# Основные сравнения', '',
               'Режим detect: поток и прогноз выключены. Первый кадр исключён.', '',
               'Серые интервалы — кадры с отличием от исходного ролика > 5 уровней яркости.', '',
               '## 1. Разные детекторы на одном видео: 4 метрики × 3 видео = 12 графиков', '']
    for video in MONA_VIDEOS:
        gallery.extend([f'### {VIDEO_LABELS[video]}', ''])
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(10, 4))
            for det in DETECTORS:
                rows = data(video, det)
                ax.plot(numeric(rows, 'frame'), series(rows, metric), color=COLORS[det], label=det.upper(), lw=1.4)
            shade_changed(ax, video)
            ax.set(xlabel='Номер кадра', ylabel=METRICS[metric][0],
                   title=f'{VIDEO_LABELS[video]} — сравнение детекторов')
            if metric == 'coverage':
                ax.set_ylim(0, 105)
            ax.legend(ncol=4, fontsize=9)
            name = f'detectors/{video}_{metric}.png'
            finish(fig, IMG / name)
            gallery.extend([f'![{METRICS[metric][0]}]({name})', ''])
    gallery.extend(['## 2. Варианты размытия для одного детектора: 3 метрики × 4 детектора = 12 графиков', '',
                    'Сравниваются исходный ролик, blur и extra credit; названия не задают конкретный тип ядра размытия.', ''])
    for det in DETECTORS:
        gallery.extend([f'### {det.upper()}', ''])
        for metric in ('coverage', 'inliers', 'keypoints'):
            fig, ax = plt.subplots(figsize=(10, 4))
            for video in MONA_VIDEOS:
                rows = data(video, det)
                ax.plot(numeric(rows, 'frame'), series(rows, metric), color=BLUR_COLORS[video],
                        label=VIDEO_LABELS[video], lw=1.4)
            ax.set(xlabel='Номер кадра', ylabel=METRICS[metric][0],
                   title=f'{det.upper()} — устойчивость к вариантам размытия')
            if metric == 'coverage':
                ax.set_ylim(0, 105)
            ax.legend(fontsize=8)
            name = f'blur/{det}_{metric}.png'
            finish(fig, IMG / name)
            gallery.extend([f'![{METRICS[metric][0]}]({name})', ''])
    (IMG / 'comparisons.md').write_text('\n'.join(gallery), encoding='utf-8')


def additional(metrics):
    videos = list(VIDEO_LABELS)
    index = {(r['video'], r['detector'], r['mode']): r for r in metrics}
    # Отдельно показываем подтверждённые измерения и прогноз.
    x = np.arange(len(DETECTORS))
    fig, ax = plt.subplots(figsize=(10, 4))
    values = np.array([[100 * float(index[v, d, 'detect']['found']) for d in DETECTORS] for v in videos])
    heat = ax.imshow(values, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(x, [d.upper() for d in DETECTORS])
    ax.set_yticks(np.arange(4), [VIDEO_LABELS[v] for v in videos])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]:.1f}%', ha='center', va='center',
                    color='white' if values[i,j] > 60 else 'black')
    ax.set_title('Доля кадров с рамкой: только детектор, без потока и прогноза')
    fig.colorbar(heat, ax=ax, label='% кадров')
    finish(fig, IMG / 'found_rate.png')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for ax, video in zip(axes.flat, videos):
        for i, mode in enumerate(MODES):
            rows = [index[video, det, mode] for det in DETECTORS]
            measured = [100 * float(r['measured']) for r in rows]
            predicted = [100 * (float(r['found']) - float(r['measured'])) for r in rows]
            pos = x + (i - 1) * .25
            ax.bar(pos, measured, .24, label=MODE_LABELS[mode], color=['#2463b4', '#148665', '#a249a7'][i])
            ax.bar(pos, predicted, .24, bottom=measured, facecolor='#ffc756', hatch='///', edgecolor='#7f641d')
        ax.set(title=VIDEO_LABELS[video], ylabel='Кадров, %', xticks=x,
               xticklabels=[d.upper() for d in DETECTORS], ylim=(0, 108))
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Вклад потока и прогноза; жёлтая штриховка — только предсказание')
    finish(fig, IMG / 'mode_ablation.png')

    for field, title, name in [
        ('longest_unmeasured', 'Максимальная серия без измерения (прогноз считается отсутствием измерения)', 'lost_intervals.png'),
        ('latency_p95_ms', '95-й перцентиль времени обработки кадра', 'latency_p95.png'),
        ('fps', 'Скорость вычислений трекера без чтения, рисования и записи', 'fps_comparison.png'),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for ax, video in zip(axes.flat, videos):
            for i, mode in enumerate(MODES):
                vals = [float(index[video, det, mode][field]) for det in DETECTORS]
                bars = ax.bar(x + (i-1)*.25, vals, .24, label=MODE_LABELS[mode])
                ax.bar_label(bars, fmt='%.0f', fontsize=6, padding=2)
            ax.set(title=VIDEO_LABELS[video], xticks=x, xticklabels=[d.upper() for d in DETECTORS],
                   ylabel='Кадров' if field == 'longest_unmeasured' else ('мс' if field == 'latency_p95_ms' else 'FPS'))
            ax.margins(y=.18)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title)
        finish(fig, IMG / name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, video in zip(axes, MONA_VIDEOS[1:]):
        for i, mode in enumerate(MODES):
            rows = [index[video, det, mode] for det in DETECTORS]
            errors = [float(r['err_blur_mean']) for r in rows]
            for j, (row, error) in enumerate(zip(rows, errors)):
                xpos = j + (i-1)*.25
                if np.isfinite(error):
                    ax.bar(xpos, error, .24, color=['#2463b4', '#148665', '#a249a7'][i])
                ax.annotate(f"{int(row['paired_blurred'])}/{int(row['blurred'])}",
                            (xpos, error if np.isfinite(error) else 0),
                            xytext=(0, 4), textcoords='offset points', ha='center', fontsize=7, rotation=90)
        ax.set(title=VIDEO_LABELS[video], xticks=x, xticklabels=[d.upper() for d in DETECTORS],
               ylabel='Среднее расхождение углов, px')
        ax.margins(y=.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['#2463b4', '#148665', '#a249a7']]
    axes[0].legend(handles, list(MODE_LABELS.values()), loc='upper left', fontsize=8)
    fig.suptitle('Расхождение с трекером на исходном видео; подпись: пары / изменённые кадры')
    finish(fig, IMG / 'blur_error.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, field, label in zip(axes, ['found', 'fps', 'longest_lost'],
                                ['Кадров с рамкой, %', 'FPS', 'Максимальная потеря, кадров']):
        for i, mode in enumerate(MODES):
            vals = [float(index['my_video', det, mode][field]) * (100 if field == 'found' else 1) for det in DETECTORS]
            ax.bar(x + (i-1)*.25, vals, .24, label=MODE_LABELS[mode])
        ax.set(xticks=x, xticklabels=[d.upper() for d in DETECTORS], ylabel=label)
    axes[0].legend(fontsize=7)
    fig.suptitle('Собственное видео: перенос на другой объект и условия съёмки')
    finish(fig, IMG / 'own_video.png')


def examples():
    # На каждом видео выбираем подтверждённый кадр ближе к середине ролика.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, filename in zip(axes.flat, VIDEO_FILES):
        video = Path(filename).stem
        rows = data(video, 'sift', 'hybrid')
        candidates = [r for r in rows if r['measured'] == '1']
        if not candidates:
            raise RuntimeError(f'Нет измеренных рамок для {video}')
        row = min(candidates, key=lambda r: abs(int(r['frame']) - len(rows) // 2))
        idx = int(row['frame'])
        cap = cv2.VideoCapture(str(RES / f'{video}_sift.mp4'))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Нет кадра {idx} в результате {video}')
        finally:
            cap.release()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{VIDEO_LABELS[video]}\nКадр {idx}, источник: {row['source']}", fontsize=10)
        ax.axis('off')
    fig.suptitle('Результат с рамкой: SIFT, гибридный режим')
    finish(fig, IMG / 'tracking_examples.png')


def frame_100_examples():
    frame_index = 100
    fig, axes = plt.subplots(1, len(MONA_VIDEOS), figsize=(15, 4))
    for ax, video in zip(axes, MONA_VIDEOS):
        row = next(r for r in data(video, 'sift', 'hybrid') if int(r['frame']) == frame_index)
        cap = cv2.VideoCapture(str(RES / f'{video}_sift.mp4'))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Нет кадра {frame_index} в результате {video}')
        finally:
            cap.release()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{VIDEO_LABELS[video]}\nКадр {frame_index}, источник: {row['source']}", fontsize=10)
        ax.axis('off')
    fig.suptitle('Кадр 100 на тестовых видео: SIFT, гибридный режим')
    finish(fig, IMG / 'tracking_frame_100.png')


def main():
    for path in [IMG, IMG / 'detectors', IMG / 'blur']:
        path.mkdir(exist_ok=True)
    rows = read_csv(RES / 'metrics_test_videos.csv')
    expected = {(Path(v).stem, d, m) for v in VIDEO_FILES for d in DETECTORS for m in MODES}
    actual = {(r['video'], r['detector'], r['mode']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError('Нужен полный актуальный прогон evaluate.py: 48 комбинаций')
    comparisons()
    additional(rows)
    examples()
    frame_100_examples()
    from frame_examples import build_examples
    build_examples(data, series)
    print('Готово: 24 основных графика, 9 дополнительных иллюстраций и кадры событий в images/events/.')


if __name__ == '__main__':
    main()
