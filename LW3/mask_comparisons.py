"""Сравнение восьми вариантов маски на трёх фото и двух деградациях."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


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
