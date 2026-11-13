"""Общие деградации, маски и метрики из исходной лабораторной LW3."""
import io
import math

import numpy as np
from PIL import Image, ImageFilter


def masks(size):
    w, h = size
    mixed = np.zeros((h, w), dtype=np.uint8)
    mixed[h // 4:3 * h // 4, w // 4:3 * w // 4] = 1
    yy, xx = np.indices((h, w))
    blocks = np.random.default_rng(17).random(((h + 15) // 16, (w + 15) // 16)) < .25
    return {'zero': np.zeros_like(mixed), 'one': np.ones_like(mixed), 'mixed': mixed,
            'left_half': (xx < w // 2).astype('uint8'),
            'circle': (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) / 3) ** 2).astype('uint8'),
            'stripes': ((xx // 32) % 2 == 0).astype('uint8'),
            'blocks': np.repeat(np.repeat(blocks, 16, 0), 16, 1)[:h, :w].astype('uint8')}


def degrade(image, kind, seed):
    w, h = image.size
    if min(w, h) < 4:
        raise ValueError('Images must be at least 4×4')
    source = image.filter(ImageFilter.GaussianBlur(1.2)) if kind == 'blur_noise_jpeg' else image
    low = source.resize((w // 4, h // 4), Image.Resampling.BICUBIC)
    rng = np.random.default_rng(seed)
    if kind.startswith('gaussian_'):
        sigma = float(kind.split('_')[1])
        low = Image.fromarray(np.clip(np.asarray(low, dtype=float) + rng.normal(0, sigma, (low.height, low.width, 3)), 0, 255).astype('uint8'))
    elif kind.startswith('poisson_'):
        peak = float(kind.split('_')[1])
        low = Image.fromarray(np.clip(rng.poisson(np.asarray(low, dtype=float) / 255 * peak) / peak * 255, 0, 255).astype('uint8'))
    elif kind.startswith('impulse_'):
        probability = float(kind.split('_')[1])
        pixels = np.array(low)
        uniform = rng.random((low.height, low.width))
        pixels[uniform < probability / 2] = 0
        pixels[uniform > 1 - probability / 2] = 255
        low = Image.fromarray(pixels)
    elif kind == 'blur_noise_jpeg':
        low = Image.fromarray(np.clip(np.asarray(low, dtype=float) + rng.normal(0, 5, (low.height, low.width, 3)), 0, 255).astype('uint8'))
        buffer = io.BytesIO()
        low.save(buffer, format='JPEG', quality=60)
        buffer.seek(0)
        low = Image.open(buffer).convert('RGB')
    elif kind != 'bicubic':
        raise ValueError(kind)
    return low.resize(image.size, Image.Resampling.BICUBIC)


def psnr(reference, image, mask=None):
    error = (np.asarray(reference, dtype=float) - np.asarray(image, dtype=float)) ** 2
    if mask is not None:
        error = error[np.asarray(mask, dtype=bool)]
    if not error.size:
        return None
    mse = error.mean()
    return float('inf') if mse == 0 else 10 * math.log10(255 ** 2 / mse)


class Metrics:
    def __init__(self, device):
        import pyiqa
        self.device = device
        self.lpips = pyiqa.create_metric('lpips', device=device)
        self.clipiqa = pyiqa.create_metric('clipiqa', device=device)

    def __call__(self, reference, result):
        import torch
        def tensor(image):
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float().to(self.device) / 255
        with torch.inference_mode():
            x, y = tensor(result), tensor(reference)
            return {'psnr': psnr(reference, result), 'lpips': self.lpips(x, y).item(),
                    'clipiqa': self.clipiqa(x).item()}
