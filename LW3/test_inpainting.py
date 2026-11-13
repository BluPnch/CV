"""CPU contract tests; no downloaded weights required."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from experiment import degrade, masks, psnr
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected, blend_latents, composite, validate_mask
from research import NOISES, plan


class Contracts(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (65, 73, 3), dtype=np.uint8))

    def test_masks_and_preservation(self):
        generated = Image.new('RGB', self.image.size, 'red')
        for name, mask in masks(self.image.size).items():
            output = np.asarray(composite(self.image, generated, mask))
            np.testing.assert_array_equal(output[mask == 0], np.asarray(self.image)[mask == 0])
            np.testing.assert_array_equal(output[mask == 1], np.asarray(generated)[mask == 1])

    def test_invalid_masks(self):
        for mask in (np.zeros((2, 2)), np.full((65, 73), .5), np.full((65, 73), np.nan)):
            with self.assertRaises(ValueError):
                validate_mask(mask, self.image.size)

    def test_zero_skips_model(self):
        restorer = MaskedDiffusion.__new__(MaskedDiffusion)
        result = restorer.restore(self.image, np.zeros((65, 73)))
        np.testing.assert_array_equal(result, self.image)

    def test_every_sampling_step_preserves_reference(self):
        roi = torch.tensor([[[[0., 1.], [1., 0.]]]])
        latent = torch.zeros(1, 4, 2, 2)
        for step in range(10):
            reference = torch.full_like(latent, float(step))
            proposal = latent + 123
            latent = blend_latents(proposal, reference, roi)
            torch.testing.assert_close(latent * (1 - roi), reference * (1 - roi))
            torch.testing.assert_close(latent * roi, proposal * roi)

    def test_degradations_reproducible(self):
        for kind in NOISES:
            a = degrade(self.image, kind, 42)
            self.assertEqual(a.size, self.image.size)
            np.testing.assert_array_equal(a, degrade(self.image, kind, 42))

    def test_research_plan(self):
        cases = plan(['town.jpg', 'rubezh_vekov.jpg'])
        keys = [(c['image'], c['degradation'], c['mask'], c['seed'], c['strength']) for c in cases]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(cases), 104)
        for case in cases:
            self.assertTrue(any(c['mask'] == 'baseline' and c['image'] == case['image']
                                and c['degradation'] == case['degradation'] and c['seed'] == case['seed'] for c in cases))
        self.assertEqual({s for c in cases for s in c['studies']}, {'masks', 'noise', 'seeds', 'strength'})

    def test_noise_level_changes_result(self):
        self.assertFalse(np.array_equal(degrade(self.image, 'gaussian_5', 42), degrade(self.image, 'gaussian_30', 42)))
        self.assertFalse(np.array_equal(degrade(self.image, 'gaussian_15', 42), degrade(self.image, 'gaussian_15', 123)))

    def test_psnr(self):
        self.assertEqual(psnr(self.image, self.image), float('inf'))
        self.assertIsNone(psnr(self.image, self.image, np.zeros((65, 73))))
        self.assertAlmostEqual(psnr(Image.new('RGB', (8, 8)), Image.new('RGB', (8, 8), (255, 255, 255))), 0)

    def test_sampling_loop_with_fake_components(self):
        # Exercise padding, timestep selection, CFG, next-time reference,
        # safety checker and final composition without a network download.
        p = SimpleNamespace()
        p.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=1.0))
        p.vae.encode = lambda x: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: torch.zeros(1, 4, 9, 10)))
        p.vae.decode = lambda x, **kw: (torch.ones(1, 3, 72, 80),)
        p.image_processor = SimpleNamespace(
            preprocess=lambda x: torch.zeros(1, 3, x.height, x.width),
            postprocess=lambda x, **kw: [Image.new('RGB', (80, 72), 'red')])
        p.encode_prompt = lambda *args: (torch.ones(1, 2, 4), torch.zeros(1, 2, 4))
        p.unet = lambda x, t, **kw: SimpleNamespace(sample=torch.zeros_like(x))
        p.run_safety_checker = Mock(return_value=(torch.ones(1, 3, 72, 80), [False]))
        scheduler = SimpleNamespace(timesteps=torch.tensor([3, 2, 1, 0]))
        scheduler.set_timesteps = lambda *args, **kw: None
        scheduler.scale_model_input = lambda x, t: x
        scheduler.add_noise = Mock(side_effect=lambda x, n, t: x + float(t[0]))
        scheduler.step = Mock(side_effect=lambda prediction, t, x, **kw: SimpleNamespace(prev_sample=x + 10))
        p.scheduler = scheduler
        restorer = MaskedDiffusion.__new__(MaskedDiffusion)
        restorer.device, restorer.pipe = 'cpu', p
        mask = masks(self.image.size)['mixed']
        output = restorer.restore(self.image, mask, steps=4, strength=.75)
        self.assertEqual(output.size, self.image.size)
        self.assertEqual(scheduler.step.call_count, 3)
        self.assertEqual([int(call.args[2][0]) for call in scheduler.add_noise.call_args_list], [2, 1, 0])
        np.testing.assert_array_equal(np.asarray(output)[mask == 0], np.asarray(self.image)[mask == 0])
        p.run_safety_checker.assert_called_once()
        p.run_safety_checker.reset_mock()
        p.unet = lambda x, t, **kw: SimpleNamespace(sample=torch.full_like(x, float('nan')))
        with self.assertRaises(FloatingPointError):
            restorer.restore(self.image, mask, steps=4, strength=.75)
        p.run_safety_checker.assert_not_called()
        p.unet = lambda x, t, **kw: SimpleNamespace(sample=torch.zeros_like(x))
        p.run_safety_checker.return_value = (torch.ones(1, 3, 72, 80), [True])
        p.maybe_free_model_hooks = Mock()
        restorer.offload = True
        with self.assertRaises(SafetyCheckRejected):
            restorer.restore(self.image, mask, steps=4, strength=.75)
        p.maybe_free_model_hooks.assert_called_once()


if __name__ == '__main__':
    unittest.main()
