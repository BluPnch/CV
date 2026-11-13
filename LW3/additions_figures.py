"""Explanatory scientific figures from measured user photographs; no inference."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiment import masks
from research import NOISES


def main():
    root = Path(__file__).resolve().parent / 'results'
    output = root / 'additions'
    output.mkdir(exist_ok=True)
    photo = root / 'images/rubezh_vekov.jpg'
    reference = np.asarray(Image.open(photo / 'reference.png'))
    variants = masks((reference.shape[1], reference.shape[0]))
    fig, axes = plt.subplots(2, 7, figsize=(18, 8))
    for column, (name, mask) in enumerate(variants.items()):
        axes[0, column].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[0, column].set_title(f'{name}\nROI {mask.mean():.1%}')
        axes[1, column].imshow(reference)
        overlay = np.zeros((*mask.shape, 4))
        overlay[..., 0] = 1
        overlay[..., 3] = mask * .4
        axes[1, column].imshow(overlay)
        for row in range(2):
            axes[row, column].axis('off')
    fig.tight_layout()
    fig.savefig(output / 'mask_atlas.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 11))
    images = [('reference', photo / 'reference.png')] + [(name, photo / name / 'seed_42/input.png') for name in NOISES]
    for ax, (name, path) in zip(axes.flat, images):
        ax.imshow(Image.open(path))
        ax.set_title(name)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'noise_atlas.png', dpi=150)
    plt.close(fig)

    folder = photo / 'gaussian_15/seed_42'
    source = np.asarray(Image.open(folder / 'input.png'))
    result = np.asarray(Image.open(folder / 'strength_0.35/mixed.png'))
    mask = variants['mixed'].astype(bool)
    difference = np.max(np.abs(result.astype(float) - source.astype(float)), axis=2)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, title, pixels in zip(axes[:3], ['Input', 'Mixed mask', 'Restored'], [source, mask, result]):
        ax.imshow(pixels, cmap='gray' if pixels.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    display = axes[3].imshow(difference, cmap='magma', vmin=0, vmax=255)
    axes[3].set_title('Max RGB change: output vs input')
    axes[3].axis('off')
    fig.colorbar(display, ax=axes[3], shrink=.65)
    fig.tight_layout()
    fig.savefig(output / 'outside_preservation.png', dpi=150)
    plt.close(fig)

    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler.from_pretrained(str(Path(__file__).resolve().parent / '.cache/sd15/scheduler'))
    scheduler.set_timesteps(30)
    times = scheduler.timesteps.cpu().numpy()
    alpha = scheduler.alphas_cumprod.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.sqrt(alpha), label='Signal coefficient')
    axes[0].plot(np.sqrt(1 - alpha), label='Noise coefficient')
    axes[0].set_xlabel('Training timestep t')
    axes[0].legend()
    for strength in (.2, .35, .5):
        selected = times[30 - int(30 * strength):]
        axes[1].plot(np.arange(len(selected)), selected, 'o-', label=f'strength={strength}, steps={len(selected)}')
    axes[1].set_xlabel('Actual denoising step')
    axes[1].set_ylabel('Training timestep t')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / 'schedule.png', dpi=150)
    plt.close(fig)
    print(f'Saved four figures to {output}')


if __name__ == '__main__':
    main()
