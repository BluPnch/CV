"""Обрабатывает одну фотографию SUPIR с пользовательской маской."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from masked_supir import MaskedSUPIR, validate_mask


def main():
    """Читает фотографию и бинарную маску PNG/NPY и сохраняет восстановление."""
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('mask', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path('/kaggle/working/supir_lab'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--control', type=float, default=1.)
    args = parser.parse_args()
    image = Image.open(args.image).convert('RGB')
    if args.mask.suffix.lower() == '.npy':
        mask = np.load(args.mask, allow_pickle=False)
    else:
        mask = np.asarray(Image.open(args.mask).convert('L'))
        if not np.isin(mask, [0, 255]).all():
            raise ValueError('PNG-маска должна содержать только 0 и 255')
        mask = mask // 255
    mask = validate_mask(mask, image.size)
    result = image.copy() if not mask.any() else MaskedSUPIR(
        args.root / 'SUPIR', args.root / 'supir.yaml').restore(
            image, mask, steps=args.steps, seed=args.seed, control=args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == '__main__':
    main()
