"""Four controlled studies on user photographs: masks, noise, seeds, strength."""
import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiment import Metrics, degrade, masks, psnr
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected

NOISES = ('bicubic', 'gaussian_5', 'gaussian_15', 'gaussian_30', 'poisson_30',
          'poisson_10', 'impulse_0.01', 'impulse_0.05', 'blur_noise_jpeg')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plan(names, seed=42, strength=.35):
    cases = {}

    def add(image, noise, mask, run_seed, value, study):
        key = (image, noise, mask, run_seed, value)
        if key not in cases:
            cases[key] = dict(image=image, degradation=noise, mask=mask, seed=run_seed,
                              strength=value, studies=[])
        if study not in cases[key]['studies']:
            cases[key]['studies'].append(study)

    for name in names:
        for noise in ('bicubic', 'gaussian_15'):
            for mask in masks((64, 64)):
                add(name, noise, mask, seed, strength, 'masks')
        for noise in NOISES:
            for mask in ('zero', 'one', 'mixed'):
                add(name, noise, mask, seed, strength, 'noise')
        for run_seed in (seed, seed + 81):
            for mask in ('zero', 'one', 'mixed'):
                add(name, 'gaussian_15', mask, run_seed, strength, 'seeds')
        for value in (.2, strength, .5):
            for mask in ('one', 'mixed'):
                add(name, 'gaussian_15', mask, seed, value, 'strength')
    # Each degradation/seed also has an explicit unprocessed baseline.
    for case in list(cases.values()):
        for study in case['studies']:
            add(case['image'], case['degradation'], 'baseline', case['seed'], strength, study)
    return list(cases.values())


def locations(output, case):
    folder = output / 'images' / case['image'] / case['degradation'] / f"seed_{case['seed']}"
    result = folder / f"strength_{case['strength']:g}" / f"{case['mask']}.png"
    return folder, result


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hr', type=Path, default=Path('LW3/input'))
    parser.add_argument('--output', type=Path, default=Path('LW3/results'))
    parser.add_argument('--model', default='LW3/.cache/sd15')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--strength', type=float, default=.35)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', default='a detailed sharp natural photograph')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--metrics-only', action='store_true')
    parser.add_argument('--baseline-only', action='store_true', help='Measure only degraded inputs and zero masks')
    args = parser.parse_args()
    if args.max_side < 64 or args.steps * .2 < 1 or args.steps * args.strength < 1 or not 0 < args.strength <= 1:
        parser.error('Require max-side >= 64, steps >= 5 and 0 < strength <= 1')
    paths = sorted(p for p in args.hr.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'})
    if not paths:
        parser.error('No user photographs found')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cases = plan([p.name for p in paths], args.seed, args.strength)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ('prepare_only', 'metrics_only', 'baseline_only', 'device')}
    config.update(inputs_sha256={p.name: sha(p) for p in paths}, protocol=2,
                  guidance=5.0, negative_prompt='blurry, low quality, artifacts', eta=0)
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing results use a different configuration; choose a new --output directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output / 'plan.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    references = {}
    dimensions = {}
    for path in paths:
        with Image.open(path) as loaded:
            hr = ImageOps.exif_transpose(loaded).convert('RGB')
        original_size = hr.size
        hr.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        references[path.name] = hr
        dimensions[path.name] = {'original': original_size, 'reference': hr.size}
        folder = output / 'images' / path.name
        folder.mkdir(parents=True, exist_ok=True)
        hr.save(folder / 'reference.png')
        for name, mask in masks(hr.size).items():
            Image.fromarray(mask * 255).save(folder / f'mask_{name}.png')
    conditions = {}
    for case in cases:
        key = (case['image'], case['degradation'], case['seed'])
        if key not in conditions:
            source = degrade(references[case['image']], case['degradation'], case['seed'])
            conditions[key] = source
            folder, _ = locations(output, case)
            folder.mkdir(parents=True, exist_ok=True)
            source.save(folder / 'input.png')
    print(f'{len(paths)} photographs, {len(cases)} cases, {len(conditions)} degraded inputs', flush=True)
    runtime_path = output / 'runtime.json'
    previous_runtime = json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.exists() else {}
    runtime = {**previous_runtime, 'python': platform.python_version(), 'dimensions': dimensions,
               'packages': {name: version(name) for name in ('numpy', 'Pillow', 'torch', 'diffusers', 'pyiqa', 'setuptools')}}
    if args.prepare_only:
        (output / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
        return
    if args.baseline_only:
        cases = [case for case in cases if case['mask'] in ('baseline', 'zero')]
    provenance = Path(args.model) / 'download_provenance.json'
    if provenance.exists():
        runtime['model_provenance'] = json.loads(provenance.read_text(encoding='utf-8'))
    restorer = None
    for index, case in enumerate(cases, 1):
        folder, result_path = locations(output, case)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = result_path.with_suffix('.json')
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding='utf-8'))
            if previous['case'] == case and previous.get('status') == 'safety_rejected':
                continue
        if result_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['case'] == case and meta['sha256'] == sha(result_path):
                continue
        if args.metrics_only:
            raise FileNotFoundError(f'Missing validated restoration: {result_path}')
        source = conditions[(case['image'], case['degradation'], case['seed'])]
        mask = np.ones((source.height, source.width), dtype='uint8') if case['mask'] == 'baseline' else masks(source.size)[case['mask']]
        print(f"[{index}/{len(cases)}] {case}", flush=True)
        started = time.perf_counter()
        if case['mask'] == 'baseline' or not mask.any():
            restored = source.copy()
        else:
            if restorer is None:
                restorer = MaskedDiffusion(args.model, args.device)
                runtime.update(actual_device=restorer.device, scheduler=type(restorer.pipe.scheduler).__name__,
                               weight_dtype=str(restorer.pipe.unet.dtype), vae_dtype=str(restorer.pipe.vae.dtype),
                               memory_strategy='model_cpu_offload' if restorer.offload else 'resident')
                runtime_path.write_text(json.dumps(runtime, indent=2), encoding='utf-8')
            started = time.perf_counter()
            try:
                restored = restorer.restore(source, mask, seed=case['seed'], steps=args.steps,
                                            strength=case['strength'], prompt=args.prompt)
            except SafetyCheckRejected:
                meta_path.write_text(json.dumps({'case': case, 'status': 'safety_rejected',
                    'seconds': time.perf_counter() - started, 'sha256': None}, indent=2), encoding='utf-8')
                print('  Rejected by model safety checker; continuing', flush=True)
                continue
        restored.save(result_path)
        meta_path.write_text(json.dumps({'case': case, 'seconds': time.perf_counter() - started,
                                        'sha256': sha(result_path)}, indent=2), encoding='utf-8')
    del restorer
    gc.collect()
    import torch
    runtime['cuda_runtime'] = torch.version.cuda
    runtime['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    runtime['cpu'] = platform.processor()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    runtime['metrics_device'] = device
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    metric = Metrics(device)
    cache = {}
    rows = []
    for index, case in enumerate(cases, 1):
        _, result_path = locations(output, case)
        source = conditions[(case['image'], case['degradation'], case['seed'])]
        mask = np.ones((source.height, source.width), dtype='uint8') if case['mask'] == 'baseline' else masks(source.size)[case['mask']]
        meta = json.loads(result_path.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('status') == 'safety_rejected':
            rows.append({**case, 'studies': ';'.join(case['studies']), 'status': 'safety_rejected',
                         'psnr': None, 'lpips': None, 'clipiqa': None, 'psnr_roi': None,
                         'mask_fraction': float(mask.mean()), 'outside_max_error': None,
                         'seconds': meta['seconds'], 'result': ''})
            write_csv(output / ('baseline_metrics.csv' if args.baseline_only else 'metrics.csv'), rows)
            continue
        result = Image.open(result_path).convert('RGB')
        background = np.abs(np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16))[mask == 0]
        outside = int(background.max()) if background.size else 0
        if outside:
            raise AssertionError(f'Changed pixels outside mask: {result_path}')
        key = (case['image'], sha(result_path))
        if key not in cache:
            cache[key] = metric(references[case['image']], result)
        rows.append({**case, 'studies': ';'.join(case['studies']), 'status': 'ok', **cache[key],
                     'psnr_roi': psnr(references[case['image']], result, mask),
                     'mask_fraction': float(mask.mean()), 'outside_max_error': outside,
                     'seconds': meta['seconds'], 'result': str(result_path.relative_to(output)).replace('\\', '/')})
        write_csv(output / ('baseline_metrics.csv' if args.baseline_only else 'metrics.csv'), rows)
        print(f'Metrics [{index}/{len(cases)}]: {case["image"]}/{case["mask"]}', flush=True)
    if not args.baseline_only:
        report(rows, output)


def report(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = sorted({r['image'] for r in rows})
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    seed, strength = config['seed'], config['strength']
    rejected = sum(r.get('status') == 'safety_rejected' for r in rows)
    def fmt(value, precision=4):
        return '—' if value is None else f'{value:.{precision}f}'
    lines = ['# Исследования на пользовательских фотографиях', '',
             'PSNR ↑, LPIPS ↓, CLIP-IQA ↑ измерены по полному RGB-кадру. PSNR ROI приведён отдельно в CSV.', '',
             f"Все эталоны получены уменьшением пользовательских JPEG до длинной стороны {config['max_side']} px (точные размеры — runtime.json). "
             'Это опорные фотографии для контролируемых деградаций, а не доказанно чистые HR-эталоны.', '']
    lines += [f'Всего опытов: {len(rows)}; успешных измерений: {len(rows) - rejected}; отклонено штатной проверкой модели: {rejected}. '
              'Для отклонений метрики отсутствуют; чёрные заглушки не оцениваются.', '']
    for study, title in [('masks', 'Форма и площадь маски'), ('noise', 'Тип и интенсивность шума'),
                         ('seeds', 'Повтор с другим seed'), ('strength', 'Глубина диффузионного уточнения')]:
        lines += [f'## {title}', '', '| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|']
        selected = [r for r in rows if study in r['studies'].split(';')]
        for r in selected:
            lines.append(f"| {r['image']} | {r['degradation']} | {r['mask']} | {r['seed']} | {r['strength']:g} | "
                         f"{r['mask_fraction']:.3f} | {fmt(r['psnr'], 3)} | {fmt(r['lpips'])} | {fmt(r['clipiqa'])} |")
        lines += ['']
    lines += ['## Контроль сохранения фона', '',
              f"Максимальное изменение вне маски по успешным измерениям: {max(r['outside_max_error'] for r in rows if r['outside_max_error'] is not None)} уровней яркости.", '',
              'Сравнение разных форм масок одновременно зависит от их площади. Полные метрики учитывают неизменённый фон; '
              'это ограничивает сравнение частичной и полной маски. Два seed показывают чувствительность конкретного опыта, '
              'но не дают надёжного статистического вывода.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    for study, variable, filename in [('masks', 'mask', 'masks'), ('noise', 'degradation', 'noise'),
                                      ('strength', 'strength', 'strength'), ('seeds', 'seed', 'seeds')]:
        fig, axes = plt.subplots(len(names), 3, figsize=(16, 4.5 * len(names)), squeeze=False)
        for row_index, name in enumerate(names):
            selected = [r for r in rows if r['image'] == name and study in r['studies'].split(';')
                        and (study == 'seeds' or r['seed'] == seed)]
            if study == 'masks':
                selected = [r for r in selected if r['degradation'] == 'gaussian_15']
                series = [('masks', selected)]
            else:
                series = [(mask, [r for r in selected if r['mask'] == mask]) for mask in ('zero', 'one', 'mixed')]
            for column, key in enumerate(('psnr', 'lpips', 'clipiqa')):
                ax = axes[row_index, column]
                for label, group in series:
                    if not group:
                        continue
                    if study in ('strength', 'seeds'):
                        group = sorted(group, key=lambda r: r[variable])
                    elif study == 'noise':
                        group = sorted(group, key=lambda r: NOISES.index(r['degradation']))
                    if study == 'masks':
                        ax.bar([str(r[variable]) for r in group], [r[key] if r[key] is not None else np.nan for r in group], label=label)
                    else:
                        ax.plot([str(r[variable]) for r in group], [r[key] if r[key] is not None else np.nan for r in group], 'o-', label=label)
                ax.set_title(f'{name}: {key}')
                ax.tick_params(axis='x', rotation=60)
                ax.grid(alpha=.25)
                ax.legend()
        fig.tight_layout()
        fig.savefig(output / f'{filename}.png', dpi=140)
        plt.close(fig)
    for name in names:
        subset = [r for r in rows if r['image'] == name and r['degradation'] == 'gaussian_15' and r['seed'] == seed and r['strength'] == strength]
        by_mask = {r['mask']: r for r in subset}
        labels = ['reference', 'baseline', 'zero', 'one', 'mixed', 'circle', 'blocks']
        fig, axes = plt.subplots(2, 4, figsize=(20, 12), layout='constrained')
        for ax in axes.flat:
            ax.axis('off')
        for ax, label in zip(axes.flat, labels):
            if label != 'reference' and by_mask[label].get('status') == 'safety_rejected':
                ax.text(.5, .5, 'Rejected', ha='center', va='center')
                ax.axis('off')
                continue
            path = output / 'images' / name / 'reference.png' if label == 'reference' else output / by_mask[label]['result']
            ax.imshow(Image.open(path), interpolation='nearest')
            ax.set_title(label, fontsize=18)
            ax.axis('off')
        fig.savefig(output / f'comparison_{name}.png', dpi=180)
        plt.close(fig)
    from comparison_maps import make_comparison_maps
    make_comparison_maps(output)


if __name__ == '__main__':
    main()
