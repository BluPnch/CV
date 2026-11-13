"""Пять повторов исследования шумов, средние метрики и парные изменения."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiment import Metrics, degrade, masks, psnr
from research import NOISES
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results_noise_repeats'
SEEDS = (42, 123, 204, 285, 366)
METRICS = ('psnr', 'lpips', 'clipiqa')
VARIANTS = ('baseline', 'one', 'mixed')
LABELS = ('Bicubic', 'Gaussian σ=5', 'Gaussian σ=15', 'Gaussian σ=30',
          'Poisson peak=30', 'Poisson peak=10', 'Salt-and-pepper 1%',
          'Salt-and-pepper 5%', 'Blur + noise + JPEG')


def resident_unet(model):
    """Держит U-Net на GPU, а остальные компоненты переносит только на время вызова."""
    import torch
    model.pipe.remove_all_hooks()
    model.pipe.to('cpu')
    model.pipe.unet.to(model.device, dtype=torch.float32)
    model.pipe.vae.float()
    for name in ('encode', 'decode'):
        original = getattr(model.pipe.vae, name)

        def call(*args, _original=original, **kwargs):
            model.pipe.vae.to(model.device)
            try:
                return _original(*args, **kwargs)
            finally:
                model.pipe.vae.to('cpu')
                torch.cuda.empty_cache()
        setattr(model.pipe.vae, name, call)
    for module in (model.pipe.text_encoder, model.pipe.safety_checker):
        if module is not None:
            module.float()
            def activate(m, args):
                m.to(model.device, dtype=torch.float32)
            def release(m, args, result):
                m.to('cpu')
            module.register_forward_pre_hook(activate)
            module.register_forward_hook(release)
    model.offload = False
    gc.collect()


def sha(path):
    """Возвращает контрольную сумму PNG."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Атомарно сохраняет сведения о завершённом опыте."""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    """Сохраняет CSV без потери предыдущего файла при прерывании записи."""
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prepare():
    """Создаёт план и входы; на одном seed все маски получают одинаковый шум."""
    OUTPUT.mkdir(exist_ok=True)
    references, cases = {}, []
    config = dict(seeds=SEEDS, noises=NOISES, masks=VARIANTS, steps=30, strength=.35,
                  guidance=5., eta=0, prompt='a detailed sharp natural photograph',
                  negative_prompt='blurry, low quality, artifacts', references={})
    for directory in ('results', 'results_background'):
        old = ROOT / directory
        old_config = json.loads((old / 'config.json').read_text(encoding='utf-8'))
        for photo in old_config['inputs_sha256']:
            path = old / 'images' / photo / 'reference.png'
            references[photo] = Image.open(path).convert('RGB')
            config['references'][photo] = sha(path)
            folder = OUTPUT / 'images' / photo
            folder.mkdir(parents=True, exist_ok=True)
            references[photo].save(folder / 'reference.png')
            for noise in NOISES:
                for seed in SEEDS:
                    condition = folder / noise / f'seed_{seed}'
                    condition.mkdir(parents=True, exist_ok=True)
                    degrade(references[photo], noise, seed).save(condition / 'input.png')
                    for variant in VARIANTS:
                        cases.append(dict(image=photo, degradation=noise, seed=seed, mask=variant,
                                          source_directory=directory))
    config['model_provenance'] = json.loads((ROOT / '.cache/sd15/download_provenance.json').read_text(encoding='utf-8'))
    config = json.loads(json.dumps(config))
    path = OUTPUT / 'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError('Конфигурация повторов изменилась')
    write_json(path, config)
    write_json(OUTPUT / 'plan.json', cases)
    return references, cases


def locations(case):
    """Определяет пути входа и восстановления."""
    folder = OUTPUT / 'images' / case['image'] / case['degradation'] / f'seed_{case["seed"]}'
    return folder, folder / f'{case["mask"]}.png'


def restore(references, cases):
    """Выполняет недостающие повторы; проверенные старые измерения переиспользуются."""
    model = None
    for index, case in enumerate(cases, 1):
        folder, path = locations(case)
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['case'] == case and (meta['status'] == 'safety_rejected'
                    or (path.exists() and sha(path) == meta['sha256'])):
                continue
        source = Image.open(folder / 'input.png').convert('RGB')
        mask = (np.zeros((source.height, source.width), dtype='uint8')
                if case['mask'] == 'baseline' else masks(source.size)[case['mask']])
        Image.fromarray(mask * 255).save(folder / f'{case["mask"]}_mask.png')
        old_folder = ROOT / case['source_directory'] / 'images' / case['image'] / case['degradation'] / f'seed_{case["seed"]}'
        old_path = old_folder / 'strength_0.35' / path.name
        old_meta_path = old_path.with_suffix('.json')
        reused = False
        if old_meta_path.exists() and (old_folder / 'input.png').exists():
            old_meta = json.loads(old_meta_path.read_text(encoding='utf-8'))
            old_case = old_meta['case']
            matches = all(old_case[k] == case[k] for k in ('image', 'degradation', 'seed', 'mask'))
            matches = matches and old_case['strength'] == .35 and sha(old_folder / 'input.png') == sha(folder / 'input.png')
            if matches:
                if old_meta.get('status') == 'safety_rejected':
                    meta = dict(case=case, status='safety_rejected', sha256=None, reused=True)
                    reused = True
                elif old_path.exists() and sha(old_path) == old_meta['sha256']:
                    shutil.copyfile(old_path, path)
                    meta = dict(case=case, status='ok', sha256=sha(path), reused=True)
                    reused = True
        if not reused:
            print(f'[{index}/{len(cases)}] {case}', flush=True)
            start = time.perf_counter()
            try:
                if case['mask'] == 'baseline':
                    result = source.copy()
                else:
                    if model is None:
                        import torch
                        from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
                        model = MaskedDiffusion.__new__(MaskedDiffusion)
                        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        if model.device == 'cpu':
                            model = MaskedDiffusion(str(ROOT / '.cache/sd15'), 'cpu')
                        else:
                            # Веса изначально FP16; перед вычислением компоненты переводятся в FP32.
                            model.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                                str(ROOT / '.cache/sd15'), torch_dtype=torch.float16)
                            model.pipe.scheduler = DDIMScheduler.from_config(model.pipe.scheduler.config)
                            model.pipe.enable_attention_slicing()
                            resident_unet(model)
                        write_json(OUTPUT / 'runtime.json', dict(gpu=torch.cuda.get_device_name(0),
                            torch=torch.__version__, dtype=str(model.pipe.unet.dtype), offload=model.offload,
                            memory_strategy='resident_fp32_unet_and_staged_components'))
                    result = model.restore(source, mask, seed=case['seed'], steps=30, strength=.35, guidance=5.)
                result.save(path)
                meta = dict(case=case, status='ok', sha256=sha(path), reused=False)
            except SafetyCheckRejected:
                meta = dict(case=case, status='safety_rejected', sha256=None, reused=False)
            meta['seconds'] = time.perf_counter() - start
        write_json(meta_path, meta)
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(references, cases):
    """Рассчитывает метрики каждого повтора с проверкой целостности и фона."""
    import torch
    metric = Metrics('cuda' if torch.cuda.is_available() else 'cpu')
    rows, cache = [], {}
    previous = {}
    if (OUTPUT / 'metrics.csv').exists():
        with (OUTPUT / 'metrics.csv').open(encoding='utf-8') as handle:
            previous = {(r['image'], r['degradation'], int(r['seed']), r['mask']): r for r in csv.DictReader(handle)}
    for index, case in enumerate(cases, 1):
        folder, path = locations(case)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        row = dict(image=case['image'], degradation=case['degradation'], seed=case['seed'],
                   mask=case['mask'], status=meta['status'], psnr=None, lpips=None, clipiqa=None,
                   outside_max_error=None, mask_fraction=None, sha256=meta['sha256'],
                   result=str(path.relative_to(OUTPUT)).replace('\\', '/') if meta['status'] == 'ok' else '')
        if meta['status'] == 'ok':
            if sha(path) != meta['sha256']:
                raise ValueError(f'Изменён результат: {path}')
            result = Image.open(path).convert('RGB')
            source = np.asarray(Image.open(folder / 'input.png')).astype(int)
            mask = (np.zeros(source.shape[:2], dtype='uint8') if case['mask'] == 'baseline'
                    else masks(result.size)[case['mask']])
            difference = np.abs(np.asarray(result).astype(int) - source)[mask == 0]
            outside = int(difference.max()) if difference.size else 0
            assert outside == 0
            key = (case['image'], case['degradation'], case['seed'], case['mask'])
            old = previous.get(key)
            digest_key = (case['image'], meta['sha256'])
            if old and old['sha256'] == meta['sha256'] and all(old[k] for k in METRICS):
                values = {k: float(old[k]) for k in METRICS}
            elif digest_key in cache:
                values = cache[digest_key]
            else:
                values = metric(references[case['image']], result)
            assert all(np.isfinite(v) for v in values.values())
            cache[digest_key] = values
            row.update(values, outside_max_error=outside, mask_fraction=float(mask.mean()))
        rows.append(row)
        write_csv(OUTPUT / 'metrics.csv', rows)
        print(f'Метрики {index}/{len(cases)}: {case["image"]}/{case["degradation"]}/{case["seed"]}/{case["mask"]}', flush=True)


def aggregate():
    """Усредняет метрики и изменения по парным успешным seed; отклонения не заменяет нулями."""
    with (OUTPUT / 'metrics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(r['image'], r['degradation'], int(r['seed']), r['mask']): r for r in rows}
    assert len(rows) == len(lookup) == 405
    summary = []
    for photo in sorted({r['image'] for r in rows}):
        for noise in NOISES:
            for variant in VARIANTS:
                accepted = [lookup[photo, noise, seed, variant] for seed in SEEDS
                            if lookup[photo, noise, seed, variant]['status'] == 'ok']
                result = dict(image=photo, degradation=noise, mask=variant, n=len(accepted),
                              attempted=5, rejected=5-len(accepted))
                for key in METRICS:
                    values = np.array([float(r[key]) for r in accepted])
                    deltas = np.array([(1 if key != 'lpips' else -1) *
                        (float(r[key]) - float(lookup[photo, noise, int(r['seed']), 'baseline'][key]))
                        for r in accepted])
                    result.update({f'{key}_mean': float(values.mean()) if len(values) else None,
                                   f'{key}_std': float(values.std(ddof=1)) if len(values)>1 else (0. if len(values)==1 else None),
                                   f'{key}_delta_mean': float(deltas.mean()) if len(values) else None,
                                   f'{key}_delta_std': float(deltas.std(ddof=1)) if len(values)>1 else (0. if len(values)==1 else None)})
                summary.append(result)
    write_csv(OUTPUT / 'summary.csv', summary)


def main():
    """Запускает повторы с возобновлением и обновляет итоговый раздел отчёта."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'restore', 'evaluate', 'report'], default='all')
    args = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    references, cases = prepare()
    if args.stage in ('all', 'restore'):
        restore(references, cases)
    if args.stage in ('all', 'evaluate'):
        evaluate(references, cases)
    if args.stage in ('all', 'evaluate', 'report'):
        aggregate()
        from noise_repeats_report import main as report
        report()


if __name__ == '__main__':
    main()
