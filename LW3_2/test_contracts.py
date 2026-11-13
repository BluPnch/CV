"""Проверки ограничений маски и протокола без весов SUPIR."""
import unittest
import ast
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from common import degrade, masks
from experiment import plan
from masked_supir import MaskedSUPIR, composite, project_latents, sampler_class, validate_mask


class Contracts(unittest.TestCase):
    def test_upstream_edm_loop(self):
        """Проверяет адаптер с настоящими методами цикла и шага SUPIR, без весов."""
        path = Path(__file__).parent / 'vendor/SUPIR/sgm/modules/diffusionmodules/sampling.py'
        if not path.exists():
            path = Path('/kaggle/working/supir_lab/SUPIR/sgm/modules/diffusionmodules/sampling.py')
        if not path.exists():
            self.skipTest('Исходники SUPIR ещё не скачаны')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'BaseDiffusionSampler', 'SingleStepDiffusionSampler', 'RestoreEDMSampler'}
        classes = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[])

        class Identity:
            def prepare_inputs(self, x, sigma, c, uc):
                return x, sigma, c

            def __call__(self, value, sigma):
                return value

        def append_dims(value, ndim):
            return value.reshape(*value.shape, *((1,) * (ndim - value.ndim)))

        namespace = dict(torch=torch, Dict=dict, Union=__import__('typing').Union,
                         ListConfig=list, OmegaConf=dict, DEFAULT_GUIDER={},
                         default=lambda x, other: other if x is None else x,
                         instantiate_from_config=lambda x: x,
                         append_dims=append_dims,
                         to_d=lambda x, sigma, denoised: (x - denoised) / append_dims(sigma, x.ndim))
        exec(compile(classes, str(path), 'exec'), namespace)
        module = ModuleType('sgm.modules.diffusionmodules.sampling')
        module.RestoreEDMSampler = namespace['RestoreEDMSampler']
        original = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        outside = (~mask.bool()).expand_as(original)
        observed = []

        def denoiser(x, sigma, conditioning, control):
            expected = original + sigma[:, None, None, None] * sampler.fixed_noise
            torch.testing.assert_close(x[outside], expected[outside])
            observed.append(sigma.clone())
            return torch.ones_like(x) * .25

        with patch.dict('sys.modules', {module.__name__: module}):
            sampler = sampler_class()(discretization_config=lambda n, device: torch.tensor([3., 1., .1, 0.]),
                                      guider_config=Identity(), device='cpu', num_steps=3, restore_cfg=-1)
        sampler.original, sampler.roi = original, mask
        result = sampler(denoiser, torch.randn_like(original), cond={})
        self.assertEqual(sampler.step_count, 3)
        self.assertEqual(len(observed), 3)
        torch.testing.assert_close(result[outside], original[outside])
        inside = mask.bool().expand_as(original)
        torch.testing.assert_close(result[inside], torch.full_like(result[inside], .25))

    def test_projection_at_all_noise_levels(self):
        """Вне маски сохраняется нужная траектория, при sigma=0 — чистый латент входа."""
        original = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(original)
        proposal = torch.randn_like(original)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        inside = mask.bool().expand_as(original)
        for sigma in (14.6, 3., .1, 0.):
            result = project_latents(proposal, original, noise, sigma, mask)
            torch.testing.assert_close(result[inside], proposal[inside])
            torch.testing.assert_close(result[~inside], (original + sigma * noise)[~inside])

    def test_binary_mask_and_pixel_preservation(self):
        """Проверяет точность композиции для всех форм масок."""
        source = Image.fromarray(np.random.default_rng(1).integers(0, 256, (65, 73, 3), dtype='uint8'))
        generated = Image.new('RGB', source.size, 'red')
        for mask in masks(source.size).values():
            result = np.asarray(composite(source, generated, mask))
            np.testing.assert_array_equal(result[mask == 0], np.asarray(source)[mask == 0])
            np.testing.assert_array_equal(result[mask == 1], np.asarray(generated)[mask == 1])
        for mask in (np.zeros((2, 2)), np.full((65, 73), .5)):
            with self.assertRaises(ValueError):
                validate_mask(mask, source.size)

    def test_zero_without_model(self):
        """Нулевая маска возвращает вход, даже если модель не загружена."""
        model = MaskedSUPIR.__new__(MaskedSUPIR)
        image = Image.new('RGB', (65, 73), 'blue')
        result = model.restore(image, np.zeros((73, 65)))
        np.testing.assert_array_equal(result, image)

    def test_plan_and_noise_reproducibility(self):
        """Протокол охватывает девять деградаций и воспроизводимые входы."""
        from experiment import NOISES
        cases = plan(['photo.jpg'], True)
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual({c[1] for c in cases}, set(NOISES))
        image = Image.new('RGB', (96, 128), (90, 120, 150))
        for noise in NOISES:
            np.testing.assert_array_equal(degrade(image, noise, 42), degrade(image, noise, 42))


if __name__ == '__main__':
    unittest.main()
