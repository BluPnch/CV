"""Исследования масок и шумов SUPIR с возобновлением и отдельным расчётом метрик."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import degrade, masks, psnr

NOISES = ('bicubic', 'gaussian_5', 'gaussian_15', 'gaussian_30', 'poisson_30',
          'poisson_10', 'impulse_0.01', 'impulse_0.05', 'blur_noise_jpeg')


def sha(path):
    """Возвращает хеш файла для проверки исходников и сохранённых результатов."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(photos, full):
    """Составляет основной протокол или короткую проверку перед полным запуском."""
    cases = set()
    for photo in photos:
        for noise in (NOISES if full else ('gaussian_15',)):
            for mask in ('baseline', 'zero', 'one', 'mixed'):
                cases.add((photo, noise, mask, 42, 1.0))
        if full:
            for mask in masks((64, 64)):
                cases.add((photo, 'gaussian_15', mask, 42, 1.0))
            for mask in ('baseline', 'zero', 'one', 'mixed'):
                cases.add((photo, 'gaussian_15', mask, 123, 1.0))
            for control in (.8, 1.2):
                for mask in ('one', 'mixed'):
                    cases.add((photo, 'gaussian_15', mask, 42, control))
    return sorted(cases)


def write_rows(path, rows):
    """Атомарно сохраняет CSV после каждого завершённого опыта."""
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    """Запускает восстановление, подготовку данных или пересчёт метрик."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('input'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--root', type=Path, default=Path('/kaggle/working/supir_lab'))
    parser.add_argument('--max-side', type=int, default=512)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args()
    if args.max_side < 64 or args.steps < 2:
        parser.error('Нужны max-side >= 64 и steps >= 2')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not paths:
        parser.error('В папке input нет изображений')
    configuration = dict(inputs={p.name: sha(p) for p in paths}, max_side=args.max_side,
                         steps=args.steps, full=args.full, cfg=4.0, s_churn=0,
                         restore_cfg=-1, color_fix='None', prompt='a detailed natural photograph',
                         code={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')})
    for name in ('setup.json', 'weights.json'):
        if (args.root / name).exists():
            configuration[name] = json.loads((args.root / name).read_text(encoding='utf-8'))
    config_path = args.output / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != configuration:
            raise ValueError('Конфигурация изменилась: выберите новую папку --output')
    else:
        config_path.write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    references = {}
    for path in paths:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            if max(image.size) > args.max_side:
                scale = args.max_side / max(image.size)
                image = image.resize(tuple(round(v * scale) for v in image.size), Image.Resampling.LANCZOS)
            references[path.name] = image.copy()
    cases = plan(list(references), args.full)
    (args.output / 'plan.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    csv_path = args.output / 'metrics.csv'
    existing = {}
    if csv_path.exists():
        with csv_path.open(encoding='utf-8') as handle:
            existing = {r['id']: r for r in csv.DictReader(handle)}
    restorer = None
    rows = []
    for photo, noise, mask_name, seed, control in cases:
        identity = f'{photo}/{noise}/seed_{seed}/control_{control:g}/{mask_name}'
        reference = references[photo]
        folder = args.output / 'images' / photo / noise / f'seed_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        reference.save(args.output / 'images' / photo / 'reference.png')
        source = degrade(reference, noise, seed)
        source.save(folder / 'input.png')
        mask = (np.zeros((source.height, source.width), dtype='uint8') if mask_name == 'baseline'
                else masks(source.size)[mask_name])
        result_path = args.output / 'images' / f'{identity}.png'
        result_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask * 255).save(result_path.with_name(mask_name + '_mask.png'))
        old = existing.get(identity)
        if old and result_path.exists() and old['sha256'] == sha(result_path):
            rows.append(old)
            continue
        if args.metrics_only:
            raise ValueError(f'Нет проверенного результата {identity}; сначала запустите восстановление')
        if args.prepare_only:
            continue
        if mask.any():
            if restorer is None:
                from masked_supir import MaskedSUPIR
                restorer = MaskedSUPIR(args.root / 'SUPIR', args.root / 'supir.yaml')
            result = restorer.restore(source, mask, steps=args.steps, seed=seed, control=control)
            actual_steps = restorer.last_steps
        else:
            result, actual_steps = source.copy(), 0
        result.save(result_path)
        outside = ~mask.astype(bool)
        difference = np.abs(np.asarray(result).astype(int) - np.asarray(source).astype(int))
        outside_error = int(difference[outside].max()) if outside.any() else 0
        assert outside_error == 0
        rows.append(dict(id=identity, image=photo, degradation=noise, mask=mask_name,
                         seed=seed, control=control, steps=actual_steps, mask_fraction=float(mask.mean()),
                         outside_max_error=outside_error, psnr=psnr(reference, result),
                         psnr_roi=psnr(reference, result, mask), lpips='', clipiqa='',
                         result=str(result_path.relative_to(args.output)), sha256=sha(result_path)))
        write_rows(csv_path, rows)
        print(f'Готово {len(rows)}/{len(cases)}: {identity}', flush=True)
    if args.prepare_only:
        print(f'Подготовлено {len(cases)} условий; модель не загружалась')
        return
    # Метрики запускаются отдельным процессом, чтобы освободить всю память SUPIR.
    write_rows(csv_path, rows)
    print('Восстановление завершено. Запустите evaluate.py для LPIPS, CLIP-IQA и отчёта.')


if __name__ == '__main__':
    main()
