"""Итоговые графики и таблицы исследования шумов по пяти seed."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from noise_repeats import OUTPUT, ROOT, NOISES, LABELS, METRICS, VARIANTS, SEEDS

COLORS = ('#475569', '#2563eb', '#d97706')
TITLES = ('PSNR ↑, dB', 'LPIPS ↓', 'CLIP-IQA ↑')


def make_noise_atlas():
    """Показывает все деградации двух фотографий в порядке оси графика шумов."""
    from PIL import Image
    fig, axes = plt.subplots(2, len(NOISES), figsize=(27, 9), layout='constrained')
    for row, (photo, title) in enumerate((('rubezh_vekov.jpg', 'Рубеж веков'),
                                        ('town.jpg', 'Town'))):
        for column, (noise, label) in enumerate(zip(NOISES, LABELS)):
            path = ROOT / 'results/images' / photo / noise / 'seed_42/input.png'
            with Image.open(path) as image:
                axes[row, column].imshow(image)
            axes[row, column].set_title(label.replace(' + ', '\n+ '), fontsize=12)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column == 0:
                axes[row, column].set_ylabel(title, fontsize=17)
    fig.suptitle('Исследуемые деградации: входы до восстановления, seed=42', fontsize=20)
    fig.savefig(OUTPUT / 'noise_inputs_two_rows.png', dpi=160)
    plt.close(fig)


ATLAS_TEXT = ('#### Как выглядят исследуемые деградации\n\n'
              'Верхняя строка — **«Рубеж веков»**, нижняя — **town**. '
              'В каждой строке все девять входов расположены слева направо в том же порядке, '
              'что на графике `results/noise.png`: **bicubic → Gaussian σ=5 → σ=15 → σ=30 '
              '→ Poisson peak=30 → peak=10 → salt-and-pepper 1% → 5% → blur+noise+JPEG**. '
              'Это сохранённые изображения **до восстановления**, пример для seed=42; '
              'итоговые графики метрик выше усреднены по пяти seed. '
              'Bicubic — контроль без дополнительного шума.\n\n'
              '![Все девять деградаций: Рубеж веков сверху, town снизу]'
              '(results_noise_repeats/noise_inputs_two_rows.png)\n\n'
              '[Открыть панель в полном размере](results_noise_repeats/noise_inputs_two_rows.png).\n')


def main():
    """Обновляет только раздел 3.2, сохраняя остальные исследования."""
    with (OUTPUT / 'summary.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    with (OUTPUT / 'metrics.csv').open(encoding='utf-8') as handle:
        raw = list(csv.DictReader(handle))
    photos = list(dict.fromkeys(r['image'] for r in raw))
    lookup = {(r['image'], r['degradation'], r['mask']): r for r in rows}
    make_noise_atlas()

    def values(photo, variant, key, suffix):
        return np.array([float(lookup[photo, n, variant][key + suffix])
                         if lookup[photo, n, variant][key + suffix] else np.nan for n in NOISES])

    def draw(ax, photo, key, delta=False):
        positions = np.arange(len(NOISES))
        chosen = VARIANTS[1:] if delta else VARIANTS
        for offset, variant in enumerate(chosen):
            suffix = '_delta_mean' if delta else '_mean'
            std_suffix = '_delta_std' if delta else '_std'
            means = values(photo, variant, key, suffix)
            stds = values(photo, variant, key, std_suffix)
            xs = positions + (offset - (len(chosen)-1)/2) * .25
            ax.bar(xs, means, yerr=stds, capsize=2, width=.24,
                   color=COLORS[VARIANTS.index(variant)], label=variant,
                   error_kw={'elinewidth': .9})
            for x, noise, mean in zip(xs, NOISES, means):
                count = int(lookup[photo, noise, variant]['n'])
                if count < 5:
                    ax.text(x, .02, f'{count}/5', rotation=90, fontsize=7,
                            ha='center', transform=ax.get_xaxis_transform())
        if delta:
            ax.axhline(0, color='#334155', linewidth=.8)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
        ax.set_xticks(positions, [s.replace(' ', '\n', 1) for s in LABELS], rotation=35, ha='right', fontsize=8)

    fig, axes = plt.subplots(3, 3, figsize=(21, 15), layout='constrained')
    for index, photo in enumerate(photos):
        for j, (key, title) in enumerate(zip(METRICS, TITLES)):
            draw(axes[index, j], photo, key)
            axes[index, j].set_title(f'{photo}: {title}')
        axes[index, 0].legend(ncol=3)
    fig.suptitle('Noise study: mean ± sample SD over 5 seeds; n/5 shown where outputs were rejected', fontsize=16)
    fig.savefig(OUTPUT / 'noise_means.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
    for ax, key, title in zip(axes, METRICS, TITLES):
        for variant, color in zip(VARIANTS, COLORS):
            chosen = [lookup['town.jpg', f'gaussian_{sigma}', variant] for sigma in (5, 15, 30)]
            means = [float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in chosen]
            stds = [float(r[key+'_std']) if r[key+'_std'] else np.nan for r in chosen]
            ax.errorbar((5, 15, 30), means, yerr=stds, marker='o', capsize=4,
                        color=color, label=variant)
        ax.set_title(title)
        ax.set_xlabel('Gaussian σ (0–255)')
        ax.set_xticks((5, 15, 30))
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle('town.jpg: Gaussian noise, mean ± SD over 5 seeds')
    fig.savefig(OUTPUT / 'town_gaussian_means.png', dpi=150)
    plt.close(fig)
    for photo in photos:
        fig, axes = plt.subplots(3, 2, figsize=(17, 12), layout='constrained')
        for i, (key, title) in enumerate(zip(METRICS, TITLES)):
            draw(axes[i, 0], photo, key)
            draw(axes[i, 1], photo, key, delta=True)
            axes[i, 0].set_ylabel(title)
            axes[i, 1].set_ylabel('Improvement, dB' if key == 'psnr' else 'Improvement')
        axes[0, 0].set_title('Quality: mean ± SD')
        axes[0, 1].set_title('Paired change vs baseline: positive = better')
        axes[0, 0].legend(ncol=3)
        axes[0, 1].legend(ncol=2)
        fig.suptitle(f'{photo} | 5 seeds: {SEEDS}\n'
                     'Error bars = sample SD; n/5 = accepted repeats; strength=0.35')
        target = OUTPUT / f'noise_detail_{Path(photo).stem}'
        fig.savefig(target.with_suffix('.png'), dpi=150)
        fig.savefig(target.with_suffix('.svg'))
        plt.close(fig)
    for noise, label in zip(NOISES, LABELS):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
        for ax, key, title in zip(axes, METRICS, TITLES):
            for offset, variant in enumerate(VARIANTS):
                chosen = [lookup[p, noise, variant] for p in photos]
                means = [float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in chosen]
                stds = [float(r[key+'_std']) if r[key+'_std'] else np.nan for r in chosen]
                xs = np.arange(len(photos)) + (offset-1) * .25
                ax.bar(xs, means, yerr=stds, capsize=3, width=.24, color=COLORS[offset], label=variant)
                for x, record in zip(xs, chosen):
                    if int(record['n']) < 5:
                        ax.text(x, .02, f'{record["n"]}/5', rotation=90, fontsize=8,
                                ha='center', transform=ax.get_xaxis_transform())
            ax.set_title(title)
            ax.set_xticks(np.arange(len(photos)), [Path(p).stem for p in photos])
            ax.grid(axis='y', alpha=.25)
            ax.set_axisbelow(True)
        axes[0].legend()
        fig.suptitle(f'{label} | mean ± SD, 5 seeds; n/5 = accepted repeats')
        fig.savefig(OUTPUT / f'{noise}.png', dpi=150)
        plt.close(fig)
    rejected = sum(r['status'] != 'ok' for r in raw)
    text = ['### 3.2. Результаты для разных шумов\n',
            '**Итоговое исследование по пяти повторам каждого условия.**\n',
            'Ранее графики этого раздела описывали один запуск с seed=42. '
            'Теперь для каждого из девяти условий на каждой из трёх фотографий выполнено '
            '**пять повторов** с seed **42, 123, 204, 285, 366**. '
            'В каждом повторе меняются и реализация входного шума, и начальный шум диффузии. '
            'Для baseline, one и mixed внутри одного повтора используется один и тот же вход. '
            'Проверенные результаты старых seed переиспользованы, остальные рассчитаны заново.\n',
            'Параметры фиксированы: уменьшение ×4, опорная длинная сторона 512 px, '
            'SD 1.5, steps=30, strength=0.35 (10 шагов DDIM), guidance=5, eta=0. '
            'Для каждого результата один раз рассчитаны PSNR, LPIPS и CLIP-IQA, затем '
            'усреднены **метрики пяти повторов**, а не сами изображения. '
            'Пять одинаковых запусков с одним seed не давали бы оценки разброса.\n',
            '**Все итоговые графики ниже показывают среднее ± выборочное стандартное отклонение '
            '(SD, ddof=1).** Это разброс повторов, не доверительный интервал. '
            'Статусы отклонения модели не заменяются нулями: при отклонениях среднее считается '
            'по доступным результатам, а число успешных повторов указано как n/5. '
            'При n=0 среднее отсутствует; при n=1 разброс оценить нельзя '
            '(нулевая длина полосы не означает отсутствие случайности).\n',
            f'Всего **405 условий**: 3 фотографии × 9 деградаций × 5 seed × '
            f'3 варианта (baseline/one/mixed). Успешных измерений: **{405-rejected}**, '
            f'отклонений: **{rejected}**. Zero отдельно не запускается: '
            'нулевая маска возвращает baseline без вызова модели. '
            'В bicubic нет случайного входного шума: пять baseline одинаковы, '
            'и их SD=0; случайность сохраняется в восстановлении.\n',
            '![Средние метрики и разброс по пяти повторам](results_noise_repeats/noise_means.png)\n',
            ATLAS_TEXT]
    a, b = (lookup['town.jpg', f'gaussian_{s}', 'baseline'] for s in (15, 30))
    text += [f'**Проверка прежнего наблюдения для town.** При Gaussian σ=15 baseline CLIP-IQA '
             f'равна {float(a["clipiqa_mean"]):.4f} ± {float(a["clipiqa_std"]):.4f}, '
             f'при σ=30 — {float(b["clipiqa_mean"]):.4f} ± {float(b["clipiqa_std"]):.4f}. '
             f'Средний PSNR меняется с {float(a["psnr_mean"]):.3f} до {float(b["psnr_mean"]):.3f} дБ. '
             + ('Рост среднего CLIP-IQA при усилении шума сохраняется и после усреднения. '
                if float(b['clipiqa_mean']) > float(a['clipiqa_mean']) else
                'Рост CLIP-IQA из одиночного запуска после усреднения не подтверждается. ')
             + 'Это описательный результат пяти повторов; одного сравнения средних '
             'недостаточно для вывода о статистической значимости или точности реальных деталей.\n',
             '![Town: средние метрики и разброс при Gaussian σ=5/15/30](results_noise_repeats/town_gaussian_means.png)\n',
             '#### Средние значения по фотографиям\n',
             'Слева на панелях — средние значения метрик, справа — **среднее парное изменение '
             'относительно baseline того же seed**. Выше нуля всегда лучше: '
             'результат − baseline для PSNR и CLIP-IQA, baseline − результат для LPIPS. '
             'При отклонениях для парного сравнения используются только успешные пары, '
             'а не разность средних по разным наборам seed.\n',
             'В таблицах в каждой ячейке: **PSNR / LPIPS / CLIP-IQA**, каждое значение '
             'в формате **среднее ± SD**, затем число успешных повторов **n/5**.\n']
    for photo in photos:
        stem = Path(photo).stem
        text += [f'**{photo}**\n',
                 f'![Средние метрики и парные изменения: {photo}](results_noise_repeats/noise_detail_{stem}.png)\n',
                 f'[Векторная версия SVG](results_noise_repeats/noise_detail_{stem}.svg).\n',
                 '| Деградация | Baseline / zero | One | Mixed |', '|---|---|---|---|']
        for noise, label in zip(NOISES, LABELS):
            cells = []
            for variant in VARIANTS:
                record = lookup[photo, noise, variant]
                cell = ' / '.join(f'{float(record[k+"_mean"]):.4f} ± {float(record[k+"_std"]):.4f}'
                                  for k in METRICS) if int(record['n']) else 'Нет результатов'
                cells.append(cell + f' (n={record["n"]}/5)')
            text.append('| ' + ' | '.join([label] + cells) + ' |')
        text.append('')
    text += ['#### Каждый шум: средние метрики и пример входа\n',
             'Графики рассчитаны по пяти seed. Фотографии под каждым графиком — '
             '**пример входного шума для seed=42**, а не усреднённая картинка.\n']
    for noise, label in zip(NOISES, LABELS):
        text += [f'##### {label}\n',
                 f'![Средние метрики: {label}](results_noise_repeats/{noise}.png)\n',
                 f'**Шум на входе: {label}, пример seed=42.**\n',
                 f'![Примеры зашумлённых входов: {label}](results/noise_studies/{noise}_inputs.png)\n']
    text += ['#### Ограничения и воспроизводимость\n',
             'Нужно учитывать отклонения штатной проверкой модели: при n<5 средние '
             'условны по успешным результатам и могут быть смещены. Повторы показывают '
             'чувствительность к случайности на этих трёх фотографиях, а не качество на большом датасете. '
             'Полные метрики частичной маски учитывают неизменённый шумный фон. '
             'Разные семейства шума не имеют общей числовой шкалы силы; составной '
             'blur+noise+JPEG не позволяет отделить вклад каждого компонента.\n',
             '[Все 405 индивидуальных условий](results_noise_repeats/metrics.csv), '
             '[81 строка средних, SD и парных изменений](results_noise_repeats/summary.csv), '
             '[Конфигурация и источники весов](results_noise_repeats/config.json), '
             '[План](results_noise_repeats/plan.json). '
             'Исходные одиночные измерения сохранены в `results` и `results_background`; '
             'другие разделы отчёта по-прежнему описывают указанные в них seed.\n',
             'Повторный запуск: `.\\LW3\\.venv\\Scripts\\python.exe LW3/noise_repeats.py`. '
             'Только пересборка средних и графиков: добавьте `--stage report`.\n']
    section = '\n'.join(text)
    (OUTPUT / 'SECTION.md').write_text(section, encoding='utf-8')
    readme = ROOT / 'README.md'
    content = readme.read_text(encoding='utf-8')
    start = content.index('### 3.2.')
    end = content.index('### 3.3. Результаты повторов', start)
    content = content[:start] + section + '\n' + content[end:]
    note = ('В разделе 3.2 исследование шумов дополнительно повторено по пяти seed для каждой '
            'из трёх фотографий. Итоговые графики этого раздела показывают средние и выборочные SD; '
            'средние при отклонениях рассчитаны только по успешным результатам с указанием n/5.\n\n')
    if note not in content:
        content = content.replace('## 4. Выводы\n\n', '## 4. Выводы\n\n' + note)
    readme.write_text(content, encoding='utf-8')
    print('Раздел 3.2 обновлён по пяти повторам', flush=True)


if __name__ == '__main__':
    main()
