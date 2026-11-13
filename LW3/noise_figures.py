"""Plot measured noise studies from CSV without running inference."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NOISES = ('bicubic', 'gaussian_5', 'gaussian_15', 'gaussian_30',
          'poisson_30', 'poisson_10', 'impulse_0.01', 'impulse_0.05',
          'blur_noise_jpeg')
LABELS = ('Bicubic', 'Gaussian 5', 'Gaussian 15', 'Gaussian 30',
          'Poisson 30', 'Poisson 10', 'Impulse 1%', 'Impulse 5%',
          'Blur + noise\n+ JPEG')
METRICS = (('psnr', 'PSNR, dB ↑'), ('lpips', 'LPIPS ↓'),
           ('clipiqa', 'CLIP-IQA ↑'))
COLORS = {'baseline': '#475569', 'one': '#2563eb', 'mixed': '#d97706'}


def main():
    root = Path(__file__).resolve().parent
    for directory in ('results', 'results_background'):
        output = root / directory
        with (output / 'metrics.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for photo in sorted({r['image'] for r in rows}):
            selected = {(r['degradation'], r['mask']): r for r in rows
                        if r['image'] == photo and int(r['seed']) == 42
                        and float(r['strength']) == .35}
            fig, axes = plt.subplots(3, 2, figsize=(17, 12), sharex=True)
            positions = np.arange(len(NOISES))
            for index, (metric, title) in enumerate(METRICS):
                absolute, delta = axes[index]
                for offset, mask in enumerate(('baseline', 'one', 'mixed')):
                    values = np.array([
                        float(selected[noise, mask][metric])
                        if selected[noise, mask]['status'] == 'ok' else np.nan
                        for noise in NOISES])
                    absolute.bar(positions + (offset - 1) * .25, values,
                                 width=.24, color=COLORS[mask], label=mask)
                    if mask == 'baseline':
                        baseline = values
                        continue
                    # Positive values always mean improvement over the same input.
                    gains = (baseline - values if metric == 'lpips'
                             else values - baseline)
                    delta.bar(positions + (offset - 1.5) * .34, gains,
                              width=.32, color=COLORS[mask], label=mask)
                    for j in np.flatnonzero(np.isnan(values)):
                        for ax, x in ((absolute, j + (offset - 1) * .25),
                                      (delta, j + (offset - 1.5) * .34)):
                            ax.text(x, .02, '×', color=COLORS[mask], ha='center',
                                    transform=ax.get_xaxis_transform(), fontsize=13)
                absolute.set_ylabel(title)
                delta.set_ylabel('Improvement, dB' if metric == 'psnr'
                                 else 'Improvement')
                delta.axhline(0, color='#334155', linewidth=1)
                for ax in (absolute, delta):
                    ax.set_axisbelow(True)
                    ax.grid(axis='y', alpha=.25)
                    ax.set_xticks(positions, LABELS, rotation=35, ha='right')
            axes[0, 0].set_title('Measured quality (full image)')
            axes[0, 1].set_title('Change vs baseline: positive = better')
            axes[0, 0].legend(ncol=3)
            axes[0, 1].legend(ncol=2)
            fig.suptitle(f'{photo} | seed=42, strength=0.35, steps=30, guidance=5\n'
                         '× = safety_rejected; missing values are not zero. '
                         'Zero mask equals baseline.', fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .94))
            target = output / f'noise_detail_{Path(photo).stem}'
            fig.savefig(target.with_suffix('.png'), dpi=160)
            fig.savefig(target.with_suffix('.svg'))
            plt.close(fig)
            print(target)


if __name__ == '__main__':
    main()
