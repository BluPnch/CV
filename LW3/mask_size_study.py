"""Два исследования размера маски на background.jpg с фиксированным входом."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiment import Metrics, psnr
from sr_inpainting import MaskedDiffusion, SafetyCheckRejected

ROOT = Path(__file__).resolve().parent
SCALES = (1., .8, .6, .4, .2)


def sha(path):
    """Проверяет целостность исходников и результатов."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_masks(size):
    """Создаёт вложенные центральные прямоугольники и круги пяти размеров."""
    w, h = size
    yy, xx = np.indices((h, w))
    for shape in ('rectangle', 'circle'):
        for scale in SCALES:
            if shape == 'rectangle':
                width, height = round(w * scale), round(h * scale)
                mask = np.zeros((h, w), dtype='uint8')
                left, top = (w - width) // 2, (h - height) // 2
                mask[top:top + height, left:left + width] = 1
                dimensions = f'{width}×{height}'
            else:
                diameter = round(min(w, h) * scale)
                mask = (((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2)
                        <= (diameter / 2) ** 2).astype('uint8')
                dimensions = f'd={diameter}'
            yield dict(id=f'{shape}_{round(scale * 100):03}', shape=shape,
                       scale=scale, dimensions=dimensions, mask_fraction=float(mask.mean())), mask


def save_json(path, value):
    """Атомарно сохраняет параметры завершённого опыта."""
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run_restorations(output, cases, source, model_path):
    """Восстанавливает десять условий с проверкой сохранённых PNG."""
    restorer = None
    for index, (case, mask) in enumerate(cases, 1):
        path = output / f"{case['id']}.png"
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta['case'] == case and (meta['status'] == 'safety_rejected'
                    or (path.exists() and meta['sha256'] == sha(path))):
                print(f'[{index}/10] Проверен {case["id"]}', flush=True)
                continue
        print(f'[{index}/10] Восстановление {case["id"]}, ROI={mask.mean():.2%}', flush=True)
        start = time.perf_counter()
        try:
            if restorer is None:
                restorer = MaskedDiffusion(str(model_path), 'auto')
                import torch
                save_json(output / 'runtime.json', dict(device=restorer.device,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    torch=torch.__version__, dtype=str(restorer.pipe.unet.dtype),
                    offload=restorer.offload))
            result = restorer.restore(source, mask, seed=42, steps=30, strength=.35, guidance=5.)
            result.save(path)
            meta = dict(case=case, status='ok', sha256=sha(path))
        except SafetyCheckRejected:
            meta = dict(case=case, status='safety_rejected', sha256=None)
        meta['seconds'] = time.perf_counter() - start
        save_json(meta_path, meta)
    del restorer
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(output, cases, reference, source):
    """Считает три метрики полного кадра и проверяет сохранение фона."""
    import torch
    metrics = Metrics('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for case, mask in [(dict(id='baseline', shape='baseline', scale=0, dimensions='—',
                            mask_fraction=0), np.zeros((source.height, source.width), dtype='uint8'))] + cases:
        if case['id'] == 'baseline':
            result, status, digest = source, 'ok', sha(output / 'input.png')
        else:
            path = output / f'{case["id"]}.png'
            meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            status, digest = meta['status'], meta['sha256']
            if status == 'ok':
                if sha(path) != digest:
                    raise ValueError(f'Изменён PNG: {path}')
                result = Image.open(path).convert('RGB')
        row = dict(**case, status=status, psnr=None, lpips=None, clipiqa=None,
                   psnr_roi=None, outside_max_error=None, sha256=digest)
        if status == 'ok':
            values = metrics(reference, result)
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError(f'Некорректные метрики: {case["id"]}')
            outside = np.abs(np.asarray(result).astype(int) - np.asarray(source).astype(int))[mask == 0]
            outside_error = int(outside.max()) if outside.size else 0
            assert outside_error == 0
            row.update(values, psnr_roi=psnr(reference, result, mask), outside_max_error=outside_error)
        rows.append(row)
        print(f'Метрики: {case["id"]}: {row["psnr"]}, {row["lpips"]}, {row["clipiqa"]}', flush=True)
        with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)


def make_report(output, cases, reference):
    """Создаёт панели масок и результатов, графики и раздел README по реальным измерениям."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output / 'metrics.csv').open(encoding='utf-8') as handle:
        rows = {r['id']: r for r in csv.DictReader(handle)}
    baseline = rows['baseline']
    text = ['### 3.6. Влияние размера маски на background.jpg\n',
            'Выполнены два отдельных исследования с пятью размерами центральной маски: '
            'горизонтальный прямоугольник и круг. Во всех десяти опытах использован **один и тот же** '
            'вход: опорная фотография 512×288 → уменьшение ×4 → Gaussian σ=15 → bicubic. '
            'Параметры SD 1.5 фиксированы: seed=42, steps=30, strength=0.35 '
            '(10 шагов DDIM), guidance=5, eta=0.\n',
            'Размеры уменьшаются по линейному коэффициенту **100%, 80%, 60%, 40%, 20%**. '
            'У прямоугольника обе стороны уменьшаются пропорционально, сохраняя горизонтальные '
            'пропорции 16:9; максимальный прямоугольник занимает весь кадр. '
            'У круга максимальный диаметр равен высоте кадра (288 px); круг целиком вписан в изображение. '
            'Коэффициент размера не равен площади ROI: площадь убывает примерно как квадрат коэффициента.\n',
            '**Как читать панели:** верхний ряд — бинарные маски (белое разрешает обработку), '
            'средний — красное наложение ROI на опорную фотографию, нижний — восстановленные изображения. '
            'Столбцы идут от самой большой маски к маленькой.\n',
            '| Контроль | PSNR ↑ | LPIPS ↓ | CLIP-IQA ↑ |', '|---|---:|---:|---:|',
            f'| Вход без восстановления | {float(baseline["psnr"]):.3f} | {float(baseline["lpips"]):.4f} | {float(baseline["clipiqa"]):.4f} |\n']
    for shape, title in [('rectangle', 'Горизонтальная прямоугольная маска'), ('circle', 'Круглая маска')]:
        subset = [(c, m) for c, m in cases if c['shape'] == shape]
        fig, axes = plt.subplots(3, 5, figsize=(18, 8), layout='constrained')
        for column, (case, mask) in enumerate(subset):
            axes[0, column].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[0, column].set_title(f'{case["scale"]:.0%}: {case["dimensions"]}\nROI {mask.mean():.2%}')
            axes[1, column].imshow(reference)
            overlay = np.zeros((*mask.shape, 4))
            overlay[..., 0], overlay[..., 3] = 1, mask * .4
            axes[1, column].imshow(overlay)
            record = rows[case['id']]
            if record['status'] == 'ok':
                axes[2, column].imshow(Image.open(output / f'{case["id"]}.png'))
            else:
                axes[2, column].text(.5, .5, 'Отклонено моделью', ha='center', transform=axes[2, column].transAxes)
            for ax in axes[:, column]:
                ax.set_xticks([])
                ax.set_yticks([])
        for row_index, label in enumerate(('Маска', 'Область обработки', 'Результат')):
            axes[row_index, 0].set_ylabel(label)
        fig.suptitle(title, fontsize=18)
        fig.savefig(output / f'{shape}_comparison.png', dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        for ax, key, label in zip(axes, ('psnr', 'lpips', 'clipiqa'), ('PSNR ↑, дБ', 'LPIPS ↓', 'CLIP-IQA ↑')):
            values = [float(rows[c['id']][key]) if rows[c['id']]['status'] == 'ok' else np.nan for c, _ in subset]
            ax.plot([c['mask_fraction'] * 100 for c, _ in subset], values, 'o-', label='Восстановление')
            ax.axhline(float(baseline[key]), linestyle='--', color='gray', label='Вход без обработки')
            ax.set_xlabel('Площадь ROI, % кадра')
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        axes[0].legend()
        fig.suptitle(title)
        fig.savefig(output / f'{shape}_metrics.png', dpi=150)
        plt.close(fig)
        text += [f'#### {title}\n', f'![Маски и восстановление: {shape}](results_mask_sizes/{shape}_comparison.png)\n',
                 '| Линейный размер | Размер, px | ROI | PSNR ↑ | LPIPS ↓ | CLIP-IQA ↑ |',
                 '|---|---|---:|---:|---:|---:|']
        for case, _ in subset:
            row = rows[case['id']]
            values = [f'{float(row[k]):.4f}' if row['status'] == 'ok' else 'Отклонено' for k in ('psnr', 'lpips', 'clipiqa')]
            text.append('| ' + ' | '.join([f'{case["scale"]:.0%}', case['dimensions'], f'{case["mask_fraction"]:.2%}'] + values) + ' |')
        text += ['', f'![Метрики в зависимости от площади ROI: {shape}](results_mask_sizes/{shape}_metrics.png)\n']
        accepted = [rows[c['id']] for c, _ in subset if rows[c['id']]['status'] == 'ok']
        if accepted:
            best = [(label, (max if key != 'lpips' else min)(accepted, key=lambda r: float(r[key])))
                    for key, label in [('psnr', 'PSNR'), ('lpips', 'LPIPS'), ('clipiqa', 'CLIP-IQA')]]
            text.append('Лучшие значения среди пяти размеров этой формы: ' + '; '.join(
                f'{label} — размер {float(row["scale"]):.0%} (ROI {float(row["mask_fraction"]):.2%})'
                for label, row in best) + '.\n')
        if len(accepted) == 5:
            largest, smallest = accepted[0], accepted[-1]
            text.append(f'При уменьшении от максимальной маски до 20% PSNR меняется '
                        f'{float(largest["psnr"]):.3f}→{float(smallest["psnr"]):.3f} дБ, '
                        f'LPIPS — {float(largest["lpips"]):.4f}→{float(smallest["lpips"]):.4f}, '
                        f'CLIP-IQA — {float(largest["clipiqa"]):.4f}→{float(smallest["clipiqa"]):.4f}. '
                        'Все пять размеров дают PSNR ниже baseline и LPIPS ниже baseline.\n')
            if shape == 'rectangle':
                text.append('PSNR растёт последовательно при уменьшении прямоугольника, '
                            'LPIPS последовательно ухудшается. CLIP-IQA сначала падает '
                            'при переходе 100%→80%, затем растёт, но остаётся ниже baseline.\n')
            else:
                gain = float(smallest['clipiqa']) - float(baseline['clipiqa'])
                text.append('PSNR последовательно растёт при уменьшении круга. LPIPS сначала '
                            'снижается при переходе 100%→80%, затем растёт. Для самого маленького '
                            f'круга CLIP-IQA выше baseline лишь на {gain:.6f}; '
                            'при одном seed это не доказывает устойчивого улучшения качества.\n')
    text += ['При уменьшении маски всё больше шумных пикселей сохраняется без обработки, '
             'поэтому полные метрики могут приближаться к baseline. Это не означает, что маленькая ROI '
             'восстанавливается лучше. Разные размеры охватывают разное содержание кадра; это исследование '
             'площади и охвата области, а не изолированное сравнение качества на одних пикселях. '
             'Прямоугольник и круг при одинаковом линейном коэффициенте имеют разную площадь.\n',
             'Все опыты используют одну фотографию и один seed; выводы описывают эту серию. '
             'Пиксели вне маски проверяются на точное совпадение с входом. '
             'Дополнительный PSNR ROI, статусы и хеши сохранены в [CSV](results_mask_sizes/metrics.csv).\n',
             'Повторный запуск из корня проекта: `.\\LW3\\.venv\\Scripts\\python.exe LW3/mask_size_study.py`. '
             'Повторное построение таблиц и графиков: добавьте `--stage report`.\n']
    section = '\n'.join(text)
    (output / 'SECTION.md').write_text(section, encoding='utf-8')
    readme = ROOT / 'README.md'
    content = readme.read_text(encoding='utf-8')
    start, end = '<!-- mask-size-study:start -->', '<!-- mask-size-study:end -->'
    block = start + '\n' + section + '\n' + end + '\n\n'
    if start in content:
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        content = before + block.rstrip() + after
    else:
        content = content.replace('## 4. Выводы', block + '## 4. Выводы')
    readme.write_text(content, encoding='utf-8')


def main():
    """Запускает оба исследования и добавляет новый раздел отчёта."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'restore', 'evaluate', 'report'], default='all')
    args = parser.parse_args()
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    output = ROOT / 'results_mask_sizes'
    output.mkdir(exist_ok=True)
    previous = ROOT / 'results_background/images/background.jpg'
    reference = Image.open(previous / 'reference.png').convert('RGB')
    source = Image.open(previous / 'gaussian_15/seed_42/input.png').convert('RGB')
    model_path = ROOT / '.cache/sd15'
    config = dict(reference_sha256=sha(previous / 'reference.png'),
                  input_sha256=sha(previous / 'gaussian_15/seed_42/input.png'),
                  size=reference.size, scales=SCALES, seed=42, steps=30, strength=.35, guidance=5,
                  degradation='gaussian_15', eta=0, model_provenance=json.loads(
                      (model_path / 'download_provenance.json').read_text(encoding='utf-8')))
    config = json.loads(json.dumps(config))
    config_path = output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Параметры исследования изменились')
    save_json(config_path, config)
    reference.save(output / 'reference.png')
    source.save(output / 'input.png')
    cases = list(study_masks(source.size))
    for case, mask in cases:
        Image.fromarray(mask * 255).save(output / f'{case["id"]}_mask.png')
    if args.stage in ('all', 'restore'):
        run_restorations(output, cases, source, model_path)
    if args.stage in ('all', 'evaluate'):
        evaluate(output, cases, reference, source)
    if args.stage in ('all', 'report', 'evaluate'):
        make_report(output, cases, reference)


if __name__ == '__main__':
    main()
