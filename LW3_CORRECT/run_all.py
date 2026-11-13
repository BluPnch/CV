"""Полный протокол исследований с чистым эталоном вне маски."""
import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageOps

from experiment import Metrics, degrade, masks, psnr
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected, composite

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'LW3'
OUTPUT = ROOT / 'results'
NOISES = ('bicubic', 'gaussian_5', 'gaussian_15', 'gaussian_30', 'poisson_30',
          'poisson_10', 'impulse_0.01', 'impulse_0.05', 'blur_noise_jpeg')
SEEDS = (42, 123, 204, 285, 366)
MASKS = ('zero', 'one', 'mixed', 'left_half', 'circle', 'stripes', 'blocks')
KEYS = ('psnr', 'lpips', 'clipiqa')
SCALES = (1., .8, .6, .4, .2)


def sha(path):
    """Возвращает SHA-256 файла."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    """Атомарно записывает параметры или сведения об опыте."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows):
    """Атомарно сохраняет CSV с промежуточными или итоговыми измерениями."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_image(image,path):
    """Атомарно пишет PNG, не изменяя другие имена того же файла при дедупликации."""
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    payload=buffer.getvalue()
    if path.exists() and sha(path)==hashlib.sha256(payload).hexdigest():
        return
    temporary=path.with_suffix('.png.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def make_mask(name, size):
    """Создаёт стандартную маску или центральную маску заданного размера."""
    w, h = size
    if name == 'baseline':
        return np.ones((h, w), dtype='uint8')
    if name in MASKS:
        return masks(size)[name]
    shape, percent = name.rsplit('_', 1)
    scale = int(percent) / 100
    yy, xx = np.indices((h, w))
    if shape == 'rectangle':
        width, height = round(w * scale), round(h * scale)
        left, top = (w-width)//2, (h-height)//2
        mask = np.zeros((h, w), dtype='uint8')
        mask[top:top+height, left:left+width] = 1
        return mask
    if shape == 'disk':
        diameter = round(min(w, h)*scale)
        return (((xx-(w-1)/2)**2 + (yy-(h-1)/2)**2) <= (diameter/2)**2).astype('uint8')
    raise ValueError(name)


def plan(photos):
    """Объединяет исследования, не дублируя одинаковые опыты."""
    cases = {}

    def add(photo, noise, mask, seed=42, strength=.35, study='noise'):
        key = (photo, noise, mask, seed, strength)
        if key not in cases:
            cases[key] = dict(image=photo, degradation=noise, mask=mask, seed=seed,
                              strength=strength, studies=[])
        if study not in cases[key]['studies']:
            cases[key]['studies'].append(study)

    for photo in photos:
        for noise in NOISES:
            for seed in SEEDS:
                for mask in ('baseline', 'zero', 'one', 'mixed'):
                    add(photo, noise, mask, seed)
        for noise in ('bicubic', 'gaussian_15'):
            for mask in ('baseline',) + MASKS:
                add(photo, noise, mask, study='masks')
        for seed in SEEDS:
            for mask in ('baseline', 'zero', 'one', 'mixed'):
                add(photo, 'gaussian_15', mask, seed, study='seeds')
        for strength in (.2, .35, .5):
            for mask in ('one', 'mixed'):
                add(photo, 'gaussian_15', mask, strength=strength, study='strength')
    for shape, study in (('rectangle', 'rectangle_sizes'), ('disk', 'circle_sizes')):
        for scale in SCALES:
            name = 'one' if shape == 'rectangle' and scale == 1 else f'{shape}_{round(scale*100):03}'
            add('background.jpg', 'gaussian_15', name, study=study)
    return list(cases.values())


def locations(case):
    """Возвращает пути общего входа и результата конкретного опыта."""
    folder = OUTPUT / 'images' / case['image'] / case['degradation'] / f'seed_{case["seed"]}'
    path = folder / f'strength_{case["strength"]:g}' / f'{case["mask"]}.png'
    return folder, path


def prepare():
    """Готовит эталоны, деградации и маски; старый проект только читается."""
    OUTPUT.mkdir(exist_ok=True)
    references = {}
    for path in sorted((ROOT / 'input').glob('*.jpg')):
        with Image.open(path) as loaded:
            image = ImageOps.exif_transpose(loaded).convert('RGB')
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        references[path.name] = image
        folder = OUTPUT / 'images' / path.name
        folder.mkdir(parents=True, exist_ok=True)
        save_image(image,folder / 'reference.png')
    cases = plan(references)
    configuration = dict(protocol='clean_reference_outside_v1', seeds=SEEDS, steps=30,
        guidance=5., eta=0, max_side=512, initial_projection=True,
        prompt='a detailed sharp natural photograph', negative_prompt='blurry, low quality, artifacts',
        inputs={p.name: sha(p) for p in sorted((ROOT/'input').glob('*.jpg'))},
        references={p: sha(OUTPUT/'images'/p/'reference.png') for p in references},
        code={n: sha(ROOT/n) for n in ('sr_inpainting.py', 'experiment.py', 'run_all.py')},
        model=json.loads((OLD/'.cache/sd15/download_provenance.json').read_text(encoding='utf-8')))
    configuration = json.loads(json.dumps(configuration))
    config_path = OUTPUT / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != configuration:
        raise ValueError('Изменена конфигурация. Не смешивайте разные алгоритмы в одном каталоге результатов.')
    save_json(config_path, configuration)
    save_json(OUTPUT/'plan.json', cases)
    conditions = set()
    for case in cases:
        folder, result = locations(case)
        result.parent.mkdir(parents=True, exist_ok=True)
        key = (case['image'], case['degradation'], case['seed'])
        if key not in conditions:
            save_image(degrade(references[case['image']], case['degradation'], case['seed']),folder/'input.png')
            conditions.add(key)
        mask = make_mask(case['mask'], references[case['image']].size)
        save_image(Image.fromarray(mask*255),result.with_name(result.stem+'_mask.png'))
    print(f'План: {len(cases)} условий, {len(conditions)} входов, {len(references)} фото', flush=True)
    return references, cases


def create_model():
    """Загружает веса в компактном виде; все вычисления выполняются в FP32."""
    import torch
    from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
    if not torch.cuda.is_available():
        return MaskedDiffusion(str(OLD/'.cache/sd15'), 'cpu')
    model = MaskedDiffusion.__new__(MaskedDiffusion)
    model.device, model.offload = 'cuda', False
    model.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(OLD/'.cache/sd15'), torch_dtype=torch.float16)
    model.pipe.scheduler = DDIMScheduler.from_config(model.pipe.scheduler.config)
    model.pipe.enable_attention_slicing()
    model.pipe.unet.to('cuda', dtype=torch.float32)
    model.pipe.vae.float()
    for name in ('encode', 'decode'):
        method = getattr(model.pipe.vae, name)

        def call(*args, _method=method, **kwargs):
            model.pipe.vae.to('cuda')
            try:
                return _method(*args, **kwargs)
            finally:
                model.pipe.vae.to('cpu')
                torch.cuda.empty_cache()
        setattr(model.pipe.vae, name, call)
    for module in (model.pipe.text_encoder, model.pipe.safety_checker):
        if module is not None:
            module.float()
            def activate(m, args):
                m.to('cuda')
            def release(m, args, result):
                m.to('cpu')
            module.register_forward_pre_hook(activate)
            module.register_forward_hook(release)
    save_json(OUTPUT/'runtime.json', dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        precision='FP32', memory_strategy='resident_unet_staged_vae_text_safety'))
    return model


def reusable_full_mask(case, source):
    """Полная маска не использует фон: допускает проверенный результат исходной версии."""
    if case['mask'] != 'one':
        return None
    if case['strength'] == .35:
        folder = OLD/'results_noise_repeats/images'/case['image']/case['degradation']/f'seed_{case["seed"]}'
        path = folder/'one.png'
    else:
        directory = 'results_background' if case['image'] == 'background.jpg' else 'results'
        folder = OLD/directory/'images'/case['image']/case['degradation']/f'seed_{case["seed"]}'
        path = folder/f'strength_{case["strength"]:g}'/'one.png'
    if not path.with_suffix('.json').exists() or not (folder/'input.png').exists():
        return None
    if sha(source) != sha(folder/'input.png'):
        raise ValueError('Входы версий различаются; повторное использование недопустимо')
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    old_case = meta['case']
    if any(old_case[k] != case[k] for k in ('image', 'degradation', 'mask', 'seed')):
        return None
    if meta.get('status') == 'safety_rejected':
        return dict(status='safety_rejected', sha256=None, reused_from=str(path))
    if path.exists() and sha(path) == meta['sha256']:
        return dict(status='ok', sha256=sha(path), reused_from=str(path))
    return None


def restore(references, cases):
    """Заново рассчитывает все частичные маски и сохраняет чистый фон."""
    model = None
    for index, case in enumerate(cases, 1):
        folder, path = locations(case)
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['case'] == case and (meta['status'] == 'safety_rejected' or
                    (path.exists() and sha(path) == meta['sha256'])):
                continue
        clean = references[case['image']]
        source = Image.open(folder/'input.png').convert('RGB')
        mask = make_mask(case['mask'], clean.size)
        reused = reusable_full_mask(case, folder/'input.png')
        started = time.perf_counter()
        if reused is not None:
            meta = dict(case=case, **reused)
            if meta['status'] == 'ok':
                temporary=path.with_suffix('.png.tmp')
                shutil.copyfile(meta['reused_from'], temporary)
                temporary.replace(path)
        else:
            print(f'[{index}/{len(cases)}] {case}', flush=True)
            try:
                if case['mask'] == 'baseline':
                    result = source.copy()
                elif not mask.any():
                    result = clean.copy()
                else:
                    if model is None:
                        model = create_model()
                    result = model.restore(source, mask, clean_image=clean, seed=case['seed'],
                                           steps=30, strength=case['strength'], guidance=5.)
                save_image(result,path)
                meta = dict(case=case, status='ok', sha256=sha(path), reused_from=None)
            except SafetyCheckRejected:
                meta = dict(case=case, status='safety_rejected', sha256=None, reused_from=None)
        meta['seconds'] = time.perf_counter()-started
        save_json(meta_path, meta)
    del model
    gc.collect()


def evaluate(references, cases):
    """Считает метрики результата и контроля с таким же чистым фоном."""
    import torch
    metric = Metrics('cuda' if torch.cuda.is_available() else 'cpu')
    cache = {}
    cache_path = OUTPUT/'metric_cache.json'
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding='utf-8'))

    def score(photo, image):
        key = photo + '/' + hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        if key not in cache:
            values = metric(references[photo], image)
            if values['psnr'] == float('inf'):
                values['lpips'] = 0.
            if not all(math.isfinite(v) or (k == 'psnr' and v == float('inf')) for k,v in values.items()):
                raise FloatingPointError(key)
            cache[key] = values
        return cache[key]

    rows = []
    for index, case in enumerate(cases, 1):
        folder, path = locations(case)
        meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        clean = references[case['image']]
        source = Image.open(folder/'input.png').convert('RGB')
        mask = make_mask(case['mask'], clean.size)
        control = source if case['mask'] == 'baseline' else composite(clean, source, mask)
        control_path = path.with_name(path.stem+'_control.png')
        save_image(control,control_path)
        baseline = score(case['image'], control)
        row = {**{k:v for k,v in case.items() if k != 'studies'}, 'studies': ';'.join(case['studies']),
               'status': meta['status'], 'mask_fraction': float(mask.mean()),
               'psnr': None, 'lpips': None, 'clipiqa': None, 'psnr_roi': None,
               'control_psnr': baseline['psnr'], 'control_lpips': baseline['lpips'],
               'control_clipiqa': baseline['clipiqa'], 'control_psnr_roi': psnr(clean, control, mask),
               'outside_max_error': None, 'sha256': meta['sha256'],
               'result': str(path.relative_to(OUTPUT)).replace('\\','/') if meta['status']=='ok' else '',
               'control': str(control_path.relative_to(OUTPUT)).replace('\\','/'),
               'reused_from': meta.get('reused_from') or ''}
        if meta['status'] == 'ok':
            if sha(path) != meta['sha256']:
                raise ValueError(f'Повреждён результат {path}')
            result = Image.open(path).convert('RGB')
            outside = np.abs(np.asarray(result).astype(int)-np.asarray(clean).astype(int))[mask==0]
            error = int(outside.max()) if outside.size else 0
            if case['mask'] != 'baseline' and error != 0:
                raise AssertionError(f'Фон отличается от чистого эталона: {path}')
            row.update(score(case['image'], result), psnr_roi=psnr(clean,result,mask), outside_max_error=error)
        rows.append(row)
        save_csv(OUTPUT/'metrics.csv', rows)
        save_json(cache_path, cache)
        print(f'Метрики [{index}/{len(cases)}] {case["image"]}/{case["mask"]}', flush=True)


def statistics(values):
    """Учитывает бесконечный PSNR точного эталона отдельно от конечных измерений."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    if np.isposinf(values).all():
        return float('inf'), None
    if not np.isfinite(values).all():
        raise ValueError('Смешаны конечные и бесконечные значения PSNR')
    return float(values.mean()), float(values.std(ddof=1)) if len(values)>1 else None


def aggregate():
    """Усредняет пять повторов шумов, сохраняя число успешных результатов."""
    with (OUTPUT/'metrics.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    summary = []
    for photo in sorted({r['image'] for r in rows}):
        for noise in NOISES:
            for mask in ('baseline','zero','one','mixed'):
                group = [r for r in rows if r['image']==photo and r['degradation']==noise
                         and r['mask']==mask and float(r['strength'])==.35 and 'noise' in r['studies'].split(';')]
                assert len(group)==5
                accepted = [r for r in group if r['status']=='ok']
                result = dict(image=photo, degradation=noise, mask=mask, n=len(accepted), attempted=5)
                for key in KEYS:
                    mean, sd = statistics([float(r[key]) for r in accepted])
                    controls = [float(r['control_'+key]) for r in accepted]
                    cm, cs = statistics(controls)
                    # У zero результат и контроль равны эталону: inf-inf трактуется как нулевое изменение.
                    gains = [(0. if float(r[key])==float(r['control_'+key]) else
                              (float(r[key])-float(r['control_'+key])) * (-1 if key=='lpips' else 1))
                             for r in accepted]
                    gm, gs = statistics(gains)
                    for suffix,value in (('mean',mean),('std',sd),('control_mean',cm),('control_std',cs),
                                         ('gain_mean',gm),('gain_std',gs)):
                        result[f'{key}_{suffix}'] = value
                summary.append(result)
    save_csv(OUTPUT/'noise_summary.csv', summary)


def main():
    """Выполняет подготовку, восстановление, метрики и отчёт с возобновлением."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all','prepare','restore','evaluate','report'], default='all')
    args = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(OLD/'.cache/torch'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    references,cases = prepare()
    if args.stage in ('all','restore'):
        restore(references,cases)
    if args.stage in ('all','evaluate'):
        evaluate(references,cases)
    if args.stage in ('all','evaluate','report'):
        aggregate()
        from report import main as make_report
        make_report()


if __name__ == '__main__':
    main()
