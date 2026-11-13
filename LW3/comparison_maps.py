"""Visualize measured restorations, absolute differences and error improvement."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_comparison_maps(root):
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    labels = ('baseline', 'zero', 'one', 'mixed', 'circle', 'blocks')
    for name in config['inputs_sha256']:
        folder = root / 'images' / name
        condition = folder / 'gaussian_15' / f"seed_{config['seed']}"
        reference = np.asarray(Image.open(folder / 'reference.png').convert('RGB'), dtype=float)
        source = np.asarray(Image.open(condition / 'input.png').convert('RGB'), dtype=float)
        input_error = np.mean(np.abs(source - reference), axis=2)
        fig, axes = plt.subplots(4, len(labels), figsize=(18, 14), layout='constrained')
        for column, label in enumerate(labels):
            path = condition / f"strength_{config['strength']:g}" / f'{label}.png'
            if not path.exists():
                for ax in axes[:, column]:
                    ax.text(.5, .5, 'No accepted output', ha='center', va='center', transform=ax.transAxes)
                    ax.set_axis_off()
                continue
            result = np.asarray(Image.open(path).convert('RGB'), dtype=float)
            change = np.mean(np.abs(result - source), axis=2)
            error = np.mean(np.abs(result - reference), axis=2)
            gain = input_error - error
            axes[0, column].imshow(result.astype('uint8'))
            axes[0, column].set_title(label, fontsize=14)
            axes[1, column].imshow(change, cmap='magma', vmin=0, vmax=100)
            absolute = axes[2, column].imshow(error, cmap='magma', vmin=0, vmax=100)
            improvement = axes[3, column].imshow(gain, cmap='RdYlGn', vmin=-50, vmax=50)
            for ax in axes[:, column]:
                ax.set_xticks([])
                ax.set_yticks([])
        for index, label in enumerate(('Output', 'Change vs input', 'Error vs reference', 'Error reduction')):
            axes[index, 0].set_ylabel(label, fontsize=13)
        fig.colorbar(absolute, ax=axes[1:3, :].ravel().tolist(), shrink=.7, extend='max',
                     label='Mean absolute RGB difference (levels)')
        fig.colorbar(improvement, ax=axes[3, :].tolist(), shrink=.7, extend='both',
                     label='Error reduction (levels)')
        fig.suptitle(f'{name}: Gaussian sigma=15, seed={config["seed"]}, strength={config["strength"]}', fontsize=16)
        fig.savefig(root / f'difference_maps_{name}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    make_comparison_maps(Path(__file__).resolve().parent / 'results')
