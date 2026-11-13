"""Локальное восстановление SUPIR с ограничением каждого шага EDM."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def validate_mask(mask, size):
    """Проверяет размер и бинарность маски."""
    mask = np.asarray(mask)
    if mask.shape != (size[1], size[0]) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Маска должна быть бинарной H×W и совпадать с изображением')
    return mask.astype(bool)


def composite(source, generated, mask):
    """Гарантирует точное сохранение входа вне маски."""
    mask = validate_mask(mask, source.size)
    return Image.fromarray(np.where(mask[..., None], np.asarray(generated), np.asarray(source)).astype('uint8'))


def project_latents(proposal, original, noise, sigma, mask):
    """Вне ROI возвращает траекторию z_input + sigma * epsilon в координатах EDM."""
    level = torch.as_tensor(sigma, device=proposal.device, dtype=proposal.dtype)
    while level.ndim < proposal.ndim:
        level = level[..., None]
    return torch.where(mask.bool(), proposal, original + level * noise)


def sampler_class():
    """Создаёт наследника штатного EDM, сохраняя формулы шага SUPIR."""
    from sgm.modules.diffusionmodules.sampling import RestoreEDMSampler

    class MaskedRestoreEDMSampler(RestoreEDMSampler):
        def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
            """Фиксирует шум траектории и ограничивает начальное состояние."""
            self.fixed_noise = x.clone()
            values = list(super().prepare_sampling_loop(x, cond, uc, num_steps))
            values[0] = project_latents(values[0], self.original, self.fixed_noise,
                                        values[2][0], self.roi)
            self.step_count = 0
            return tuple(values)

        def sampler_step(self, sigma, next_sigma, *args, **kwargs):
            """После штатного шага ограничивает изменения областью маски."""
            proposal = super().sampler_step(sigma, next_sigma, *args, **kwargs)
            self.step_count += 1
            return project_latents(proposal, self.original, self.fixed_noise,
                                   next_sigma, self.roi)

    return MaskedRestoreEDMSampler


class MaskedSUPIR:
    def __init__(self, repo, config, device='cuda:0'):
        """Загружает SUPIR; компоненты последовательно переносятся на одну GPU."""
        if not torch.cuda.is_available():
            raise RuntimeError('Для SUPIR нужна CUDA GPU; включите Accelerator в Kaggle')
        sys.path.insert(0, str(Path(repo).resolve()))
        from SUPIR.util import create_SUPIR_model
        self.device = torch.device(device)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            self.model = create_SUPIR_model(str(config)).eval().requires_grad_(False).half().cpu()
        finally:
            torch.set_default_dtype(previous_dtype)
        self.model.first_stage_model.float()
        self.model.denoiser.float().to(self.device)
        self.model.ae_dtype = torch.float32
        self.model.model.dtype = torch.float16
        self.model.init_tile_vae(encoder_tile_size=256, decoder_tile_size=32)
        self.last_steps = 0

    def release(self, module):
        """Освобождает видеопамять после завершения этапа."""
        module.cpu()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def restore(self, image, mask, steps=30, seed=42, cfg=4.0, control=1.0,
                prompt='a detailed natural photograph'):
        """Восстанавливает вход без изменения размера и пикселей вне маски."""
        image = image.convert('RGB')
        mask = validate_mask(mask, image.size)
        if steps < 2 or not 0 <= control <= 2 or cfg < 0:
            raise ValueError('Нужны steps >= 2, 0 <= control <= 2, cfg >= 0')
        self.last_steps = 0
        if not mask.any():
            return image.copy()
        from pytorch_lightning import seed_everything
        from omegaconf import OmegaConf
        seed_everything(seed)
        model = self.model
        w, h = image.size
        pad = ((0, (-h) % 64), (0, (-w) % 64))
        pixels = np.pad(np.asarray(image), (*pad, (0, 0)), mode='edge')
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float().to(self.device) / 127.5 - 1
        model.first_stage_model.to(self.device)
        original = model.encode_first_stage(tensor)
        condition_latent = model.encode_first_stage_with_denoise(tensor, use_sample=False)
        stage1 = model.decode_first_stage(condition_latent)
        center = model.encode_first_stage(stage1)
        self.release(model.first_stage_model)
        roi = torch.from_numpy(np.pad(mask, pad, mode='edge').copy()).to(self.device)[None, None].float()
        roi = F.interpolate(roi, size=original.shape[-2:], mode='nearest')
        model.conditioner.to(self.device)
        with torch.autocast('cuda', dtype=torch.float16):
            c, uc = model.prepare_condition(condition_latent, [prompt], '',
                                            'blurry, low quality, artifacts, invented details', 1)
        self.release(model.conditioner)
        configuration = model.sampler_config.copy()
        configuration.params.num_steps = steps
        configuration.params.s_churn = 0
        configuration.params.restore_cfg = -1
        configuration.params.guider_config.params.scale = cfg
        configuration.params.guider_config.params.scale_min = cfg
        sampler = sampler_class()(**OmegaConf.to_container(configuration.params, resolve=True))
        sampler.original, sampler.roi = original, roi
        model.model.to(self.device)

        def denoiser(x, sigma, conditioning, control_scale):
            """Вызывает обученную сеть SUPIR с условием испорченного входа."""
            return model.denoiser(model.model, x, sigma, conditioning, control_scale)

        try:
            with torch.autocast('cuda', dtype=torch.float16):
                latent = sampler(denoiser, torch.randn_like(condition_latent), cond=c, uc=uc,
                                 x_center=center, control_scale=control)
            self.last_steps = sampler.step_count
        finally:
            self.release(model.model)
        model.first_stage_model.to(self.device)
        try:
            decoded = model.decode_first_stage(latent)
            if not torch.isfinite(decoded).all():
                raise FloatingPointError('SUPIR вернул NaN/Inf')
            pixels = ((decoded[0].float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype('uint8')
        finally:
            self.release(model.first_stage_model)
        generated = Image.fromarray(pixels).crop((0, 0, w, h))
        return composite(image, generated, mask)
