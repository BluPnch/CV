"""Рассчитывает метрики сохранённых PNG и строит отчёт без загрузки SUPIR."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import Metrics
from experiment import NOISES, sha, write_rows


def main():
    """Добавляет метрики, сравнительные изображения и графики шумов."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with (args.output / 'metrics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    metrics = Metrics(args.device)
    for row in rows:
        path = args.output / row['result']
        if sha(path) != row['sha256']:
            raise ValueError(f'Изменён результат: {path}')
        if row['lpips'] and row['clipiqa']:
            continue
        reference = Image.open(args.output / 'images' / row['image'] / 'reference.png').convert('RGB')
        result = Image.open(path).convert('RGB')
        values = metrics(reference, result)
        if not all(math.isfinite(v) or (k == 'psnr' and v == float('inf')) for k, v in values.items()):
            raise FloatingPointError(f'Некорректные метрики: {path}')
        row.update(values)
        write_rows(args.output / 'metrics.csv', rows)
    report = ['# Результаты SUPIR\n',
              'Метрики полного кадра. PSNR ↑, LPIPS ↓, CLIP-IQA ↑. '
              'Частичные маски включают неизменённый фон; zero должен совпадать с baseline.\n',
              '| Фото | Шум | Маска | Seed | Control | PSNR | LPIPS | CLIP-IQA |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append('| ' + ' | '.join([row['image'], row['degradation'], row['mask'],
            str(row['seed']), str(row['control'])] + [f'{float(row[k]):.4f}' for k in ('psnr', 'lpips', 'clipiqa')]) + ' |')
    for photo in sorted({r['image'] for r in rows}):
        chosen = {(r['degradation'], r['mask']): r for r in rows
                  if r['image'] == photo and int(r['seed']) == 42 and float(r['control']) == 1}
        noises = [n for n in NOISES if (n, 'baseline') in chosen]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for ax, key, title in zip(axes, ('psnr', 'lpips', 'clipiqa'), ('PSNR ↑', 'LPIPS ↓', 'CLIP-IQA ↑')):
            for j, mask in enumerate(('baseline', 'zero', 'one', 'mixed')):
                ax.bar(np.arange(len(noises)) + (j - 1.5) * .2,
                       [float(chosen[n, mask][key]) for n in noises], width=.19, label=mask)
            ax.set_ylabel(title)
            ax.grid(axis='y', alpha=.25)
            ax.set_axisbelow(True)
        axes[0].legend(ncol=4)
        axes[-1].set_xticks(np.arange(len(noises)), noises, rotation=30, ha='right')
        fig.suptitle(photo)
        fig.tight_layout()
        name = f'noise_{Path(photo).stem}.png'
        fig.savefig(args.output / name, dpi=150)
        plt.close(fig)
        report.append(f'\n![Метрики {photo}]({name})\n')
        for noise in noises:
            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            reference = args.output / 'images' / photo / 'reference.png'
            panels = [('reference', reference)] + [(m, args.output / chosen[noise, m]['result'])
                                                    for m in ('baseline', 'zero', 'one', 'mixed')]
            for ax, (title, path) in zip(axes, panels):
                with Image.open(path) as image:
                    ax.imshow(image)
                ax.set_title(title)
                ax.axis('off')
            fig.suptitle(f'{photo}: {noise}')
            fig.tight_layout()
            name = f'comparison_{Path(photo).stem}_{noise}.png'
            fig.savefig(args.output / name, dpi=130)
            plt.close(fig)
            report.append(f'\n![{photo}: {noise}]({name})\n')
    (args.output / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(args.output / 'REPORT.md')


if __name__ == '__main__':
    main()
