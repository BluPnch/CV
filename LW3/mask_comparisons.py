"""Сравнение восьми вариантов маски на трёх фото и двух деградациях."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiment import masks


def make_mask_atlas(output, photo):
    """Показывает бинарные маски и их области на конкретной фотографии."""
    with Image.open(output / 'images' / photo / 'reference.png') as image:
        reference = np.asarray(image.convert('RGB'))
    variants = masks((reference.shape[1], reference.shape[0]))
    fig, axes = plt.subplots(4, 4, figsize=(14, 15), layout='constrained')
    for index, (name, mask) in enumerate(variants.items()):
        row, column = (index // 4) * 2, index % 4
        axes[row, column].imshow(mask, cmap='gray', vmin=0, vmax=1,
                                 interpolation='nearest')
        axes[row, column].set_title(f'{name}: ROI {mask.mean():.1%}', fontsize=14)
        axes[row + 1, column].imshow(reference)
        overlay = np.zeros((*mask.shape, 4))
        overlay[..., 0] = 1
        overlay[..., 3] = mask * .4
        axes[row + 1, column].imshow(overlay)
        axes[row + 1, column].set_title(f'{name}: область обработки', fontsize=13)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[2:, 3]:
        ax.set_visible(False)
    fig.suptitle(f'{photo}: внешний вид масок\n'
                 'Белое / красное — обрабатывать; чёрное / без наложения — сохранять',
                 fontsize=16)
    target = output / f'mask_atlas_{Path(photo).stem}.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(target)


def main():
    """Строит 48 позиций из сохранённых результатов без запуска модели."""
    root = Path(__file__).resolve().parent
    labels = ('baseline', 'zero', 'one', 'mixed', 'left_half', 'circle', 'stripes', 'blocks')
    conditions = (('bicubic', 'Bicubic: без дополнительного шума'),
                  ('gaussian_15', 'Гауссовский шум: σ=15'))
    for directory in ('results', 'results_background'):
        output = root / directory
        with (output / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for photo in sorted({r['image'] for r in rows}):
            make_mask_atlas(output, photo)
            selected = {(r['degradation'], r['mask']): r for r in rows
                        if r['image'] == photo and int(r['seed']) == 42
                        and float(r['strength']) == .35}
            columns = 4 if photo == 'background.jpg' else 8
            rows_per_condition = len(labels) // columns
            fig, axes = plt.subplots(2 * rows_per_condition, columns,
                                     figsize=(16, 12) if columns == 4 else (24, 10),
                                     layout='constrained')
            for row_index, (noise, title) in enumerate(conditions):
                for column, label in enumerate(labels):
                    ax = axes[row_index * rows_per_condition + column // columns, column % columns]
                    record = selected[noise, label]
                    if record['status'] == 'ok':
                        with Image.open(output / record['result']) as image:
                            ax.imshow(image)
                    else:
                        ax.text(.5, .5, 'Отклонено моделью\nНет изображения',
                                ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(label, fontsize=15)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if column % columns == 0:
                        ax.set_ylabel(title, fontsize=14)
            fig.suptitle(f'{photo}: восемь вариантов × две деградации\n'
                         'seed=42, strength=0.35', fontsize=19)
            target = output / f'masks_photos_{Path(photo).stem}.png'
            fig.savefig(target, dpi=150)
            plt.close(fig)
            print(target)


if __name__ == '__main__':
    main()
