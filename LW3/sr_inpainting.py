"""Латентная диффузия с маской и фиксированной траекторией шума вне области обработки."""
from pathlib import Path

import numpy as np
from PIL import Image


class SafetyCheckRejected(RuntimeError):
    """Проверка модели отклонила результат; изображение не возвращается."""


def validate_mask(mask, size):
    """Проверка размерности и бинарности маски, возвращает логический массив."""
    mask = np.asarray(mask)
    if mask.shape != (size[1], size[0]) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask must be a binary H×W array matching the input image')
    return mask.astype(bool)


def composite(source, generated, mask):
    """Сборка результата, сохраняя исходные пиксели вне маски."""
    mask = validate_mask(mask, source.size)
    return Image.fromarray(np.where(mask[..., None], np.asarray(generated), np.asarray(source)).astype('uint8'))


def blend_latents(proposal, reference, mask):
    """Новый латент внутри маски и опорный латент снаружи."""
    return mask * proposal + (1 - mask) * reference


class MaskedDiffusion:
    def __init__(self, model='stable-diffusion-v1-5/stable-diffusion-v1-5', device='auto'):
        """Загружает модель и DDIM, настраивает устройство и расход памяти."""
        import torch
        from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        cuda = self.device.startswith('cuda')
        # На GTX 16xx свёртки U-Net в FP16 могут давать NaN.
        self.offload = cuda and 'GTX 16' in torch.cuda.get_device_name(self.device)
        dtype = torch.float16 if cuda and not self.offload else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=dtype)
        # FP32 предотвращает переполнение при кодировании и декодировании VAE.
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        if self.offload:
            # В видеопамяти 6 ГБ держим по одному компоненту FP32, включая проверку безопасности.
            self.pipe.model_cpu_offload_seq += '->safety_checker'
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)

    def restore(self, image, mask, *, prompt='a detailed sharp natural photograph',
                negative_prompt='blurry, low quality, artifacts', steps=30, strength=0.35,
                guidance=5.0, seed=42):
        """Восстанавливает область маски, ограничивая каждый шаг DDIM и итоговые пиксели."""
        import torch
        import torch.nn.functional as F
        image = image.convert('RGB')
        mask = validate_mask(mask, image.size)
        if steps < 1 or not 0 < strength <= 1 or int(steps * strength) < 1:
            raise ValueError('Require steps >= 1 and 1/steps <= strength <= 1')
        if not mask.any():
            return image.copy()
        p = self.pipe
        # Дополнение краёв до кратности 8, сохраняя координаты пикселей маски.
        w, h = image.size
        pad = ((0, (-h) % 8), (0, (-w) % 8))
        pixels = np.pad(np.asarray(image), (*pad, (0, 0)), mode='edge')
        padded_mask = np.pad(mask, pad, mode='edge')
        with torch.inference_mode():
            tensor = p.image_processor.preprocess(Image.fromarray(pixels)).to(self.device, p.vae.dtype)
            # Латент 
            original = p.vae.encode(tensor).latent_dist.mode() * p.vae.config.scaling_factor
            original = original.to(getattr(p.unet, 'dtype', original.dtype))
            if not torch.isfinite(original).all():
                raise FloatingPointError('Non-finite VAE encoding')
            
            roi = torch.tensor(padded_mask.copy(), device=self.device, dtype=original.dtype)[None, None]
            roi = F.interpolate(roi, size=original.shape[-2:], mode='nearest')
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(original.shape, generator=generator, device=self.device, dtype=original.dtype)
            
            p.scheduler.set_timesteps(steps, device=self.device)
            timesteps = p.scheduler.timesteps[steps - int(steps * strength):]
            latents = p.scheduler.add_noise(original, noise, timesteps[:1])
            positive, negative = p.encode_prompt(prompt, self.device, 1, True, negative_prompt)
            embeddings = torch.cat([negative, positive])
            for index, timestep in enumerate(timesteps):
                model_input = p.scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
                
                unconditional, conditional = p.unet(model_input, timestep, encoder_hidden_states=embeddings).sample.chunk(2)
                if not torch.isfinite(conditional).all() or not torch.isfinite(unconditional).all():
                    raise FloatingPointError('Non-finite U-Net prediction')
                prediction = unconditional + guidance * (conditional - unconditional)
               
                proposal = p.scheduler.step(prediction, timestep, latents, eta=0).prev_sample
                reference = (p.scheduler.add_noise(original, noise, timesteps[index + 1:index + 2])
                             if index + 1 < len(timesteps) else original)
                
                latents = blend_latents(proposal, reference, roi)

            decoded = p.vae.decode((latents / p.vae.config.scaling_factor).to(p.vae.dtype), return_dict=False)[0]
            if not torch.isfinite(decoded).all():
                raise FloatingPointError('Non-finite VAE decoding')
            # Сохраняем проверку безопасности в изменённом процессе восстановления.
            decoded, flags = p.run_safety_checker(decoded, self.device, embeddings.dtype)
            generated = p.image_processor.postprocess(decoded, output_type='pil')[0].crop((0, 0, w, h))
            if getattr(self, 'offload', False):
                p.maybe_free_model_hooks()
            if flags is not None and any(flags):
                raise SafetyCheckRejected('Model safety checker rejected the generated image')
        return composite(image, generated, mask)


def main():
    """Читает аргументы команды, восстанавливает одну фотографию и сохраняет результат."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('mask', type=Path, help='.npy binary HxW array or black/white PNG')
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--strength', type=float, default=.35)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', default='a detailed sharp natural photograph')
    args = parser.parse_args()
    image = Image.open(args.image).convert('RGB')
    mask = np.load(args.mask, allow_pickle=False) if args.mask.suffix == '.npy' else np.asarray(Image.open(args.mask).convert('L'))
    if args.mask.suffix != '.npy':
        if not np.isin(mask, [0, 255]).all():
            raise ValueError('PNG mask must contain only 0 and 255')
        mask = mask // 255
    validate_mask(mask, image.size)
    result = (MaskedDiffusion(args.model, args.device).restore(image, mask, steps=args.steps,
              strength=args.strength, seed=args.seed, prompt=args.prompt) if mask.any() else image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)


if __name__ == '__main__':
    main()
