"""Проверки чистого фона, начального состояния и каждого шага без скачивания весов."""
import unittest
import csv
from pathlib import Path
import tempfile
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
import run_all as runner

from experiment import masks
from run_all import make_mask, plan, statistics
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected, validate_mask


class CleanBackground(unittest.TestCase):
    def setUp(self):
        self.dirty = Image.new('RGB', (16,16), (25,25,25))
        self.clean = Image.new('RGB', (16,16), (200,200,200))
        self.mask = np.zeros((16,16), dtype='uint8')
        self.mask[:,8:] = 1

    def fake_model(self):
        """Задаёт разные латенты чистого и грязного входа, фиксируя все шаги."""
        p = SimpleNamespace()
        p.vae = SimpleNamespace(dtype=torch.float32, config=SimpleNamespace(scaling_factor=1.))
        p.vae.encode = Mock(side_effect=lambda x: SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda: torch.full((1,4,2,2), float(x.mean())))))
        p.vae.decode = Mock(return_value=(torch.zeros(1,3,16,16),))
        p.image_processor = SimpleNamespace(preprocess=lambda im: torch.tensor(np.asarray(im).copy()).float(),
            postprocess=lambda x, **kw: [Image.new('RGB',(16,16),'red')])
        p.encode_prompt = lambda *args: (torch.ones(1,2,4),torch.zeros(1,2,4))
        p.unet = Mock(side_effect=lambda x,t,**kw: SimpleNamespace(sample=torch.zeros_like(x)))
        p.unet.dtype = torch.float32
        p.run_safety_checker = Mock(side_effect=lambda decoded,*args: (decoded,[False]))
        p.scheduler = SimpleNamespace(timesteps=torch.tensor([3,2,1,0]))
        p.scheduler.set_timesteps = lambda *args,**kw: None
        p.scheduler.scale_model_input = Mock(side_effect=lambda x,t: x)
        p.scheduler.add_noise = Mock(side_effect=lambda x,n,t: x+float(t[0]))
        p.scheduler.step = Mock(side_effect=lambda pred,t,x,**kw: SimpleNamespace(prev_sample=x+10))
        model = MaskedDiffusion.__new__(MaskedDiffusion)
        model.device,model.pipe,model.offload = 'cpu',p,False
        return model,p

    def test_zero_returns_clean_without_model(self):
        model = MaskedDiffusion.__new__(MaskedDiffusion)
        output = model.restore(self.dirty, np.zeros((16,16)), clean_image=self.clean)
        np.testing.assert_array_equal(output,self.clean)
        self.assertFalse(np.array_equal(output,self.dirty))

    def test_initial_and_all_steps_use_clean_latent(self):
        model,p = self.fake_model()
        result = model.restore(self.dirty,self.mask,clean_image=self.clean,steps=4,strength=.75)
        self.assertEqual(p.vae.encode.call_count,2)
        self.assertEqual(p.scheduler.step.call_count,3)
        for call,t in zip(p.scheduler.scale_model_input.call_args_list,(2,1,0)):
            x=call.args[0]
            torch.testing.assert_close(x[:,:,:,0],torch.full_like(x[:,:,:,0],200.+t))
        first = p.scheduler.scale_model_input.call_args_list[0].args[0]
        torch.testing.assert_close(first[:,:,:,1],torch.full_like(first[:,:,:,1],27.))
        decoded_latent=p.vae.decode.call_args.args[0]
        torch.testing.assert_close(decoded_latent[:,:,:,0],torch.full_like(decoded_latent[:,:,:,0],200.))
        np.testing.assert_array_equal(np.asarray(result)[self.mask==0],np.asarray(self.clean)[self.mask==0])
        np.testing.assert_array_equal(np.asarray(result)[self.mask==1],np.full((128,3),[255,0,0]))

    def test_full_mask_does_not_use_clean_latent(self):
        model,p = self.fake_model()
        model.restore(self.dirty,np.ones((16,16)),clean_image=self.clean,steps=4,strength=.75)
        self.assertEqual(p.vae.encode.call_count,1)
        first=p.scheduler.scale_model_input.call_args_list[0].args[0]
        torch.testing.assert_close(first,torch.full_like(first,27.))

    def test_mismatched_size_and_invalid_mask(self):
        model = MaskedDiffusion.__new__(MaskedDiffusion)
        with self.assertRaises(ValueError):
            model.restore(self.dirty,self.mask,clean_image=Image.new('RGB',(8,8)))
        for bad in (np.zeros((3,4)),np.full((16,16),.5),np.full((16,16),np.nan)):
            with self.assertRaises(ValueError):
                validate_mask(bad,(16,16))

    def test_safety_rejection_and_nan(self):
        model,p=self.fake_model()
        p.run_safety_checker.side_effect=lambda decoded,*args:(decoded,[True])
        with self.assertRaises(SafetyCheckRejected):
            model.restore(self.dirty,self.mask,clean_image=self.clean,steps=4,strength=.75)
        model,p=self.fake_model()
        p.unet.side_effect=lambda x,t,**kw:SimpleNamespace(sample=torch.full_like(x,float('nan')))
        with self.assertRaises(FloatingPointError):
            model.restore(self.dirty,self.mask,clean_image=self.clean,steps=4,strength=.75)

    def test_plan_masks_and_perfect_psnr(self):
        cases=plan(['background.jpg','town.jpg','rubezh_vekov.jpg'])
        self.assertEqual(len(cases),585)
        self.assertEqual(len({(c['image'],c['degradation'],c['mask'],c['seed'],c['strength']) for c in cases}),585)
        self.assertEqual(sum('noise' in c['studies'] for c in cases),540)
        for shape in ('rectangle','disk'):
            variants=[make_mask(f'{shape}_{n:03}',(512,288)) for n in (100,80,60,40,20)]
            for large,small in zip(variants,variants[1:]):
                self.assertTrue((small<=large).all())
        self.assertTrue(make_mask('rectangle_100',(512,288)).all())
        self.assertEqual(statistics([float('inf')]*5),(float('inf'),None))

    def test_summary_perfect_zero_and_rejected_pairs(self):
        records=[]
        for photo in ('a.jpg','b.jpg','c.jpg'):
            for noise in runner.NOISES:
                for i,seed in enumerate(runner.SEEDS,1):
                    for mask in ('baseline','zero','one','mixed'):
                        rejected=(photo=='a.jpg' and ((noise=='gaussian_15' and mask=='one' and i==1)
                                  or (noise=='poisson_10' and mask=='mixed' and i<5)))
                        base=dict(psnr=i*10,lpips=i/10,clipiqa=i/10)
                        value={k:v+(1 if k=='psnr' else .01) for k,v in base.items()}
                        if mask=='baseline': value=base.copy()
                        if mask=='zero': base=value=dict(psnr=float('inf'),lpips=0.,clipiqa=.7)
                        row=dict(image=photo,degradation=noise,mask=mask,seed=seed,strength=.35,
                                 studies='noise',status='safety_rejected' if rejected else 'ok')
                        row.update({k:'' if rejected else v for k,v in value.items()})
                        row.update({'control_'+k:v for k,v in base.items()})
                        records.append(row)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            runner.save_csv(output/'metrics.csv',records)
            with patch.object(runner,'OUTPUT',output): runner.aggregate()
            with (output/'noise_summary.csv').open(encoding='utf-8') as handle:
                rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),108)
        lookup={(r['image'],r['degradation'],r['mask']):r for r in rows}
        row=lookup['a.jpg','gaussian_15','one']
        self.assertEqual(float(row['psnr_mean']),36.)
        self.assertEqual(float(row['psnr_control_mean']),35.)
        self.assertEqual(float(row['psnr_gain_mean']),1.)
        row=lookup['a.jpg','poisson_10','mixed']
        self.assertEqual(int(row['n']),1)
        self.assertEqual(row['psnr_std'],'')
        row=lookup['a.jpg','bicubic','zero']
        self.assertEqual(float(row['psnr_mean']),float('inf'))
        self.assertEqual(row['psnr_std'],'')
        self.assertEqual(float(row['psnr_gain_mean']),0.)

    def test_atomic_image_write_preserves_other_hardlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.png',Path(directory)/'b.png'
            black=Image.new('RGB',(8,8),'black')
            white=Image.new('RGB',(8,8),'white')
            runner.save_image(black,a)
            os.link(a,b)
            runner.save_image(black,a)
            self.assertTrue(os.path.samefile(a,b))
            runner.save_image(white,a)
            np.testing.assert_array_equal(Image.open(a),white)
            np.testing.assert_array_equal(Image.open(b),black)
            self.assertFalse(os.path.samefile(a,b))


if __name__=='__main__':
    unittest.main()
