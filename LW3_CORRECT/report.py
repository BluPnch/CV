"""Графики, панели и отчёт новой лабораторной по фактически измеренным данным."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_all import ROOT, OUTPUT, OLD, NOISES, MASKS, SEEDS, KEYS, make_mask, sha

LABELS = ('Bicubic','Gaussian σ=5','Gaussian σ=15','Gaussian σ=30','Poisson 30',
          'Poisson 10','Impulse 1%','Impulse 5%','Blur+noise+JPEG')
COLORS = ('#64748b','#16a34a','#2563eb','#d97706')
TITLES = ('PSNR ↑, дБ','LPIPS ↓','CLIP-IQA ↑')


def fmt(value, digits=4):
    """Отображает идеальный PSNR как ∞, а отсутствующие измерения — прочерком."""
    if value is None or value=='':
        return '—'
    value=float(value)
    return '∞' if value==float('inf') else f'{value:.{digits}f}'


def mean_sd(row,key):
    """Отдельно подписывает бесконечный PSNR с неопределённым SD."""
    if row[key+'_mean'] and math.isinf(float(row[key+'_mean'])):
        return '∞ (SD не опр.)'
    return fmt(row[key+'_mean'])+' ± '+fmt(row[key+'_std'])


def image_panel(ax,path,title):
    """Рисует изображение или явно отмечает отсутствие принятого результата."""
    if path is not None and Path(path).exists():
        with Image.open(path) as image:
            ax.imshow(image)
    else:
        ax.text(.5,.5,'Отклонено моделью',ha='center',va='center',transform=ax.transAxes,fontsize=10)
    ax.set_title(title,fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def reference_path(photo):
    return OUTPUT/'images'/photo/'reference.png'


def make_mask_panels(photos,lookup):
    """Показывает все формы масок и результаты двух деградаций для каждого фото."""
    variants=('baseline',)+MASKS
    for photo in photos:
        stem=Path(photo).stem
        clean=Image.open(reference_path(photo)).convert('RGB')
        fig,axes=plt.subplots(4,4,figsize=(14,15),layout='constrained')
        for i,name in enumerate(MASKS):
            row,column=i//4*2,i%4
            mask=make_mask(name,clean.size)
            axes[row,column].imshow(mask,cmap='gray',vmin=0,vmax=1)
            axes[row,column].set_title(f'{name}: ROI {mask.mean():.1%}')
            axes[row+1,column].imshow(clean)
            overlay=np.zeros((*mask.shape,4)); overlay[...,0]=1; overlay[...,3]=mask*.4
            axes[row+1,column].imshow(overlay)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[2:,3]:
            ax.set_visible(False)
        fig.suptitle(f'{photo}: белое/красное — восстанавливать; снаружи — чистый эталон')
        fig.savefig(OUTPUT/f'mask_atlas_{stem}.png',dpi=140); plt.close(fig)
        columns=4 if photo=='background.jpg' else 8
        rows_per=8//columns
        fig,axes=plt.subplots(rows_per*2,columns,figsize=(16,12) if columns==4 else (24,10),layout='constrained')
        for row,noise in enumerate(('bicubic','gaussian_15')):
            for i,name in enumerate(variants):
                record=lookup[photo,noise,name,42,.35]
                ax=axes[row*rows_per+i//columns,i%columns]
                image_panel(ax,OUTPUT/record['result'] if record['result'] else None,
                            f'{name}\nPSNR {fmt(record["psnr"],2)}')
                if i%columns==0:
                    ax.set_ylabel('Bicubic' if row==0 else 'Gaussian σ=15')
        fig.suptitle(f'{photo}: результаты с чистым фоном, seed=42, strength=0.35')
        fig.savefig(OUTPUT/f'mask_results_{stem}.png',dpi=150); plt.close(fig)
        condition=OUTPUT/'images'/photo/'gaussian_15/seed_42'
        fig,axes=plt.subplots(1,5,figsize=(17,5),layout='constrained')
        mixed=lookup[photo,'gaussian_15','mixed',42,.35]
        panels=[('Чистый эталон',reference_path(photo)),('Грязный вход',condition/'input.png'),
                ('Контроль: чистый фон',OUTPUT/mixed['control']),
                ('Mixed: восстановление',OUTPUT/mixed['result'] if mixed['result'] else None),
                ('Zero: чистый эталон',OUTPUT/lookup[photo,'gaussian_15','zero',42,.35]['result'])]
        for ax,(title,path) in zip(axes,panels):
            image_panel(ax,path,title)
        fig.suptitle(f'{photo}: что изменилось при использовании чистого фона')
        fig.savefig(OUTPUT/f'clean_context_{stem}.png',dpi=150); plt.close(fig)
        # Карты ошибки результата относительно чистого эталона: нули вне ROI обязательны.
        chosen=('baseline','zero','one','mixed','circle','blocks')
        fig,axes=plt.subplots(4,6,figsize=(19,14),layout='constrained')
        for i,name in enumerate(chosen):
            record=lookup[photo,'gaussian_15',name,42,.35]
            image_panel(axes[0,i],OUTPUT/record['result'] if record['result'] else None,name)
            if record['result']:
                pixels=np.asarray(Image.open(OUTPUT/record['result'])).astype(float)
                control=np.asarray(Image.open(OUTPUT/record['control'])).astype(float)
                change=np.mean(np.abs(pixels-control),axis=2)
                error=np.mean(np.abs(pixels-np.asarray(clean).astype(float)),axis=2)
                gain=np.mean(np.abs(control-np.asarray(clean).astype(float)),axis=2)-error
                axes[1,i].imshow(change,cmap='magma',vmin=0,vmax=100)
                display=axes[2,i].imshow(error,cmap='magma',vmin=0,vmax=100)
                reduction=axes[3,i].imshow(gain,cmap='RdYlGn',vmin=-50,vmax=50)
            else:
                for ax in axes[1:,i]:
                    ax.text(.5,.5,'Нет результата',ha='center',transform=ax.transAxes)
            for ax in axes[1:,i]: ax.set_xticks([]); ax.set_yticks([])
        for index,label in enumerate(('Результат','Изменение к контролю','Ошибка к чистому эталону','Уменьшение ошибки')):
            axes[index,0].set_ylabel(label)
        fig.colorbar(display,ax=axes[1:3,:].ravel().tolist(),shrink=.6,label='Среднее абсолютное RGB-различие')
        fig.colorbar(reduction,ax=axes[3,:].tolist(),shrink=.6,label='Уменьшение ошибки: зелёное лучше, красное хуже')
        fig.savefig(OUTPUT/f'error_maps_{stem}.png',dpi=140); plt.close(fig)


def plot_mask_metrics(photos,lookup):
    """Сопоставляет формы масок с соответствующим контролем без нейросети."""
    variants=('baseline',)+MASKS
    fig,axes=plt.subplots(3,3,figsize=(18,13),layout='constrained')
    for i,photo in enumerate(photos):
        chosen=[lookup[photo,'gaussian_15',m,42,.35] for m in variants]
        for ax,key,title in zip(axes[i],KEYS,TITLES):
            x=np.arange(8)
            for offset,prefix,label,color in ((-.18,'','Результат','#2563eb'),(.18,'control_','Контроль','#94a3b8')):
                values=np.array([float(r[prefix+key]) if r[prefix+key] else np.nan for r in chosen])
                finite=np.isfinite(values)
                ax.bar((x+offset)[finite],values[finite],width=.34,label=label,color=color)
                for xx in (x+offset)[np.isinf(values)]:
                    ax.text(xx,.98,'∞',ha='center',va='top',color=color,transform=ax.get_xaxis_transform())
            ax.set_title(f'{photo}: {title}')
            ax.set_xticks(x,variants,rotation=40,ha='right')
            ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
        axes[i,0].legend()
    fig.suptitle('Формы маски: Gaussian σ=15, seed=42, strength=0.35; zero PSNR=∞')
    fig.savefig(OUTPUT/'masks_metrics.png',dpi=140); plt.close(fig)


def plot_repeated(photos,summary):
    """Показывает средние шумов и парные изменения с контролем чистого фона."""
    variants=('baseline','zero','one','mixed')
    for photo in photos:
        fig,axes=plt.subplots(3,2,figsize=(18,12),layout='constrained')
        x=np.arange(9)
        for row,(key,title) in enumerate(zip(KEYS,TITLES)):
            for i,mask in enumerate(variants):
                records=[summary[photo,n,mask] for n in NOISES]
                mean=np.array([float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in records])
                sd=np.array([float(r[key+'_std']) if r[key+'_std'] else 0 for r in records])
                xs=x+(i-1.5)*.18
                finite=np.isfinite(mean)
                if finite.any():
                    axes[row,0].bar(xs[finite],mean[finite],yerr=sd[finite],capsize=2,width=.17,label=mask,color=COLORS[i])
                if np.isinf(mean).any():
                    axes[row,0].plot([],[],color=COLORS[i],label='zero: PSNR=∞')
                    for xx in xs[np.isinf(mean)]:
                        axes[row,0].text(xx,.97,'∞',color=COLORS[i],ha='center',va='top',transform=axes[row,0].get_xaxis_transform())
                if mask in ('one','mixed'):
                    gains=[float(r[key+'_gain_mean']) if r[key+'_gain_mean'] else np.nan for r in records]
                    deviations=[float(r[key+'_gain_std']) if r[key+'_gain_std'] else 0 for r in records]
                    gx=x+(-.12 if mask=='one' else .12)
                    axes[row,1].bar(gx,gains,yerr=deviations,capsize=2,width=.23,color=COLORS[i],label=mask)
                for j,r in enumerate(records):
                    if int(r['n'])<5:
                        axes[row,0].text(xs[j],.02,f'{r["n"]}/5',fontsize=7,rotation=90,ha='center',transform=axes[row,0].get_xaxis_transform())
            axes[row,0].set_ylabel(title)
            axes[row,1].set_ylabel('Улучшение к контролю, дБ' if key=='psnr' else 'Улучшение к контролю')
            axes[row,1].axhline(0,color='black',linewidth=.7)
            for ax in axes[row]:
                ax.set_xticks(x,LABELS,rotation=35,ha='right',fontsize=8)
                ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
        axes[0,0].legend(fontsize=9,ncol=3); axes[0,1].legend()
        axes[0,0].set_title('Метрики: среднее ± SD; ∞ не заменяется конечным столбцом')
        axes[0,1].set_title('Парное изменение к входу с тем же чистым фоном; выше нуля лучше')
        fig.suptitle(f'{photo}: пять seed, чистый фон вне маски',fontsize=17)
        target=OUTPUT/f'noise_means_{Path(photo).stem}'
        fig.savefig(target.with_suffix('.png'),dpi=150)
        fig.savefig(target.with_suffix('.svg')); plt.close(fig)
    for noise,label in zip(NOISES,LABELS):
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax,key,title in zip(axes,KEYS,TITLES):
            for i,mask in enumerate(variants):
                records=[summary[p,noise,mask] for p in photos]
                mean=np.array([float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in records])
                sd=np.array([float(r[key+'_std']) if r[key+'_std'] else 0 for r in records])
                xs=np.arange(len(photos))+(i-1.5)*.2
                finite=np.isfinite(mean)
                if finite.any():
                    ax.bar(xs[finite],mean[finite],yerr=sd[finite],width=.19,capsize=3,color=COLORS[i],label=mask)
                elif np.isinf(mean).any():
                    ax.plot([],[],color=COLORS[i],label='zero: PSNR=∞')
                for xx in xs[np.isinf(mean)]:
                    ax.text(xx,.97,'∞',ha='center',va='top',color=COLORS[i],transform=ax.get_xaxis_transform())
            ax.set_title(title); ax.set_xticks(np.arange(len(photos)),[Path(p).stem for p in photos])
            ax.grid(axis='y',alpha=.25); ax.set_axisbelow(True)
        axes[0].legend(fontsize=9)
        fig.suptitle(f'{label}: среднее ± SD, 5 повторов; zero — точный эталон')
        fig.savefig(OUTPUT/f'noise_{noise}.png',dpi=140); plt.close(fig)
        fig,axes=plt.subplots(1,len(photos),figsize=(12,5),layout='constrained')
        for ax,photo in zip(axes,photos):
            image_panel(ax,OUTPUT/'images'/photo/noise/'seed_42/input.png',photo)
        fig.suptitle(f'{label}: примеры входа до восстановления, seed=42')
        fig.savefig(OUTPUT/f'input_{noise}.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(2,9,figsize=(27,9),layout='constrained')
    for row,photo in enumerate(('rubezh_vekov.jpg','town.jpg')):
        for column,(noise,label) in enumerate(zip(NOISES,LABELS)):
            image_panel(axes[row,column],OUTPUT/'images'/photo/noise/'seed_42/input.png',label)
            if column==0: axes[row,column].set_ylabel(photo)
    fig.suptitle('Примеры деградаций до восстановления, seed=42')
    fig.savefig(OUTPUT/'noise_inputs.png',dpi=150); plt.close(fig)


def plot_parameter(photos,rows,study,parameter,filename):
    """Строит зависимости от seed или strength для one/mixed."""
    fig,axes=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for i,photo in enumerate(photos):
        for j,(key,title) in enumerate(zip(KEYS,TITLES)):
            ax=axes[i,j]
            for mask,color in (('one',COLORS[2]),('mixed',COLORS[3])):
                chosen=sorted([r for r in rows if r['image']==photo and r['mask']==mask
                               and study in r['studies'].split(';')],key=lambda r:float(r[parameter]))
                values=[float(r[key]) if r['status']=='ok' else np.nan for r in chosen]
                ax.plot([float(r[parameter]) for r in chosen],values,'o-',color=color,label=mask)
                controls=[float(r['control_'+key]) for r in chosen]
                ax.plot([float(r[parameter]) for r in chosen],controls,'--',color=color,alpha=.45,label=f'контроль {mask}')
            ax.set_title(f'{photo}: {title}'); ax.set_xlabel(parameter); ax.grid(alpha=.25)
        axes[i,0].legend(fontsize=8)
    fig.savefig(OUTPUT/filename,dpi=140); plt.close(fig)


def size_panels(rows,lookup):
    """Пять размеров прямоугольника и круга: маски, результат, графики и контроль."""
    photo='background.jpg'; clean=Image.open(reference_path(photo)).convert('RGB')
    for study,shape,title in (('rectangle_sizes','rectangle','Прямоугольник'),('circle_sizes','disk','Круг')):
        chosen=[r for r in rows if study in r['studies'].split(';')]
        chosen.sort(key=lambda r:-float(r['mask_fraction']))
        fig,axes=plt.subplots(3,5,figsize=(19,9),layout='constrained')
        for column,r in enumerate(chosen):
            mask=make_mask(r['mask'],clean.size)
            axes[0,column].imshow(mask,cmap='gray',vmin=0,vmax=1)
            percent=(100 if r['mask']=='one' else int(r['mask'].split('_')[-1]))
            axes[0,column].set_title(f'{percent}%: ROI {float(r["mask_fraction"]):.2%}')
            image_panel(axes[1,column],OUTPUT/r['control'],'Контроль: чистый фон')
            image_panel(axes[2,column],OUTPUT/r['result'] if r['result'] else None,'Восстановление')
        for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
        axes[0,0].set_ylabel('Маска'); axes[1,0].set_ylabel('Контроль'); axes[2,0].set_ylabel('Результат')
        fig.suptitle(f'background.jpg: {title}, пять размеров, чистый фон')
        fig.savefig(OUTPUT/f'{shape}_sizes.png',dpi=150); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for ax,key,label in zip(axes,KEYS,TITLES):
            x=[float(r['mask_fraction'])*100 for r in chosen]
            y=[float(r[key]) if r['status']=='ok' else np.nan for r in chosen]
            ax.plot(x,y,'o-',label='Восстановление')
            ax.plot(x,[float(r['control_'+key]) for r in chosen],'o--',label='Контроль с чистым фоном')
            ax.set_xlabel('Площадь ROI, %'); ax.set_ylabel(label); ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{title}: метрики и контроль для каждой площади')
        fig.savefig(OUTPUT/f'{shape}_size_metrics.png',dpi=150); plt.close(fig)


def table(rows,include_control=True):
    """Формирует таблицу трёх метрик и PSNR ROI с явными статусами."""
    text=['| Фото / условие | Маска | ROI | PSNR ↑ | LPIPS ↓ | CLIP-IQA ↑ | PSNR ROI ↑ |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        label=f'{r["image"]}, {r["degradation"]}, seed={r["seed"]}, strength={r["strength"]}'
        values=[fmt(r[k]) for k in KEYS]+[fmt(r['psnr_roi'])]
        if r['status']!='ok': values=['Отклонено']*4
        text.append('| '+' | '.join([label,r['mask'],f'{float(r["mask_fraction"]):.2%}']+values)+' |')
        if include_control and r['mask'] not in ('baseline','zero'):
            values=[fmt(r['control_'+k]) for k in KEYS]+[fmt(r['control_psnr_roi'])]
            text.append('| '+' | '.join(['↳ контроль: та же ROI без нейросети','—','—']+values)+' |')
    return '\n'.join(text)+'\n'


def main():
    """Собирает всю лабораторную заново без переноса прежних выводов."""
    with (OUTPUT/'metrics.csv').open(encoding='utf-8') as handle: rows=list(csv.DictReader(handle))
    with (OUTPUT/'noise_summary.csv').open(encoding='utf-8') as handle: summaries=list(csv.DictReader(handle))
    photos=list(dict.fromkeys(r['image'] for r in rows))
    lookup={(r['image'],r['degradation'],r['mask'],int(r['seed']),float(r['strength'])):r for r in rows}
    summary={(r['image'],r['degradation'],r['mask']):r for r in summaries}
    make_mask_panels(photos,lookup)
    plot_mask_metrics(photos,lookup)
    plot_repeated(photos,summary)
    plot_parameter(photos,rows,'seeds','seed','seeds.png')
    plot_parameter(photos,rows,'strength','strength','strength.png')
    size_panels(rows,lookup)
    rejected=sum(r['status']!='ok' for r in rows)
    reused=sum(bool(r['reused_from']) for r in rows)
    text=['# Лабораторная работа №3: SR-inpainting с чистым фоном\n',
          '**Отдельная версия `LW3_CORRECT`. Исходная `LW3` не изменяется.** '
          'В отличие от первоначальной версии, вне маски используются латенты и пиксели '
          '**чистой опорной фотографии до искусственного уменьшения ×4, шума и JPEG**. '
          'Внутри маски восстанавливается деградированное изображение.\n',
          '## 1. Теоретическая база и постановка\n',
          'Это эксперимент с известным чистым контекстом: алгоритм получает эталон для фона. '
          'Он отличается от восстановления полностью неизвестного чистого изображения. '
          'VAE кодирует весь эталон; его пространственный охват может переносить сведения '
          'из ROI в латентные значения снаружи. Это не строгая постановка неизвестной чистой ROI. '
          'Копирование чистого фона повышает полные метрики само по себе, поэтому качество '
          'нейросети нужно сравнивать с контролем с таким же фоном, а не только с полностью грязным входом.\n',
          'Чистый эталон — исходная пользовательская фотография, приведённая к длинной стороне '
          '512 px. Это «идеальное состояние» относительно искусственных деградаций: '
          'естественная дымка background.jpg и исходные JPEG-артефакты не удаляются.\n',
          'Для входа y, чистой фотографии x и маски M (1 — восстанавливать) VAE кодирует '
          'два латента: z_y=E(y) и z_x=E(x). Перед первым шагом и после каждого шага DDIM '
          'вне ROI используется зашумлённая версия **z_x**, а не z_y:\n',
          '```text\n'
          'reference(t) = sqrt(alpha_bar[t])*z_x + sqrt(1-alpha_bar[t])*epsilon_fixed\n'
          'z_next = M_latent*z_proposal + (1-M_latent)*reference(t_next)\n'
          'result = M*generated + (1-M)*x\n```\n',
          'На последнем шаге reference=z_x без дополнительного диффузионного шума. '
          'Один фиксированный epsilon используется на всей траектории. Маска уменьшается '
          'nearest-neighbor до латентного размера; финальная композиция выполняется по исходной маске. '
          'Фон в начальном состоянии также ограничен чистым латентом. '
          '**Zero возвращает x без вызова модели; one не использует чистый фон.**\n',
          '## 2. Разработанная система и запуск\n',
          '| Файл | Назначение |\n|---|---|\n'
          '| [sr_inpainting.py](sr_inpainting.py) | Основной алгоритм и CLI с обязательным чистым эталоном |\n'
          '| [experiment.py](experiment.py) | Маски, деградации, PSNR, LPIPS, CLIP-IQA |\n'
          '| [run_all.py](run_all.py) | Единый план, возобновление, контроль фона и метрики |\n'
          '| [report.py](report.py) | Все таблицы, панели и графики |\n'
          '| [test_correct.py](test_correct.py) | Проверки чистого латента на каждом шаге и пикселей |\n'
          '| [validate.py](validate.py) | Аудит всех PNG, метрик, средних и сохранности LW3 |\n'
          '| [deduplicate.py](deduplicate.py) | Экономия диска для идентичных PNG внутри новой папки |\n'
          '| [additions.md](additions.md) | Формулы метрик и объяснение контролей |\n',
          'Веса SD 1.5, кэш метрик и Python читаются из уже подготовленной `LW3`; '
          'скрипты, входные копии, результаты и отчёт новой версии находятся в `LW3_CORRECT`. '
          'Это не SUPIR: здесь переделан основной вариант лабораторной на SD 1.5. '
          'Вычисления FP32, U-Net на GPU, VAE и текстовый энкодер переносятся по этапам; '
          'проверки NaN/Inf и штатный safety checker сохранены.\n',
          '```powershell\n'
          '.\\LW3\\.venv\\Scripts\\python.exe -m unittest discover -s LW3_CORRECT -p test_correct.py -v\n'
          '.\\LW3\\.venv\\Scripts\\python.exe LW3_CORRECT/run_all.py --stage restore\n'
          '.\\LW3\\.venv\\Scripts\\python.exe LW3_CORRECT/run_all.py --stage evaluate\n'
          '.\\LW3\\.venv\\Scripts\\python.exe LW3_CORRECT/validate.py\n'
          '# Только пересборка отчёта:\n'
          '.\\LW3\\.venv\\Scripts\\python.exe LW3_CORRECT/run_all.py --stage report\n```\n',
          'Для одной фотографии: `python LW3_CORRECT/sr_inpainting.py dirty.png clean.png '
          'mask.npy result.png --model LW3/.cache/sd15`. Вход, эталон и маска должны иметь одинаковый размер.\n',
          '## 3. Исследования\n',
          f'Выполнен план из **{len(rows)} уникальных условий**: успешных **{len(rows)-rejected}**, '
          f'отклонений **{rejected}**. Повторы шумов: 540 условий; формы масок: 48; '
          'seed: 60; strength: 18; размеры прямоугольника и круга: по 5. '
          'Одинаковые условия между исследованиями переиспользуются.\n',
          f'**Все частичные маски пересчитаны новым алгоритмом.** У {reused} условий '
          'полной маски переиспользован проверенный результат/статус исходной версии: '
          'при M=1 фон математически не участвует. Источник указан в CSV и JSON каждого такого опыта; '
          'совпадение входа и SHA-256 проверено. Baseline — полностью деградированный вход; '
          'zero — чистый эталон. Это разные контроли, они больше не совпадают.\n',
          '### Метрики и контроль\n',
          'PSNR ↑, LPIPS ↓ и CLIP-IQA ↑ рассчитаны по полному кадру. '
          'Для каждого опыта дополнительно сохранён **контроль без нейросети**: '
          '`control=M*y+(1-M)*x`. Он содержит такую же ROI с повреждениями и такой же чистый фон. '
          'В таблицах и графиках парные изменения сравниваются именно с этим контролем. '
          'PSNR ROI оценивает только восстанавливаемые пиксели; у zero он отсутствует. '
          'У zero PSNR=∞ и LPIPS=0; CLIP-IQA равен оценке самого эталона, а не обязательно 1. '
          'Бесконечность в графиках отмечена символом ∞ и не заменяется конечным числом.\n',
          'Для бесконечного PSNR нулевой маски SD не определено и отмечается прочерком; '
          'оно не подменяется числом 0. Для конечных одинаковых LPIPS и CLIP-IQA SD равно нулю.\n',
          '### 3.1. Форма и площадь маски\n',
          'Seed=42, strength=0.35, steps=30, guidance=5. Восемь вариантов: baseline и семь масок. '
          'В панелях результатов верхняя строка — bicubic, нижняя — Gaussian σ=15; '
          'для background каждая деградация занимает два ряда по четыре картинки. '
          'Доля ROI и содержание изменяются вместе, поэтому это не изолированное сравнение геометрических форм.\n']
    text += ['На картах различий четыре ряда: результат; изменение относительно контроля с тем же чистым фоном; '
             'ошибка к эталону; уменьшение ошибки относительно контроля. В последнем ряду зелёный '
             'означает улучшение, красный — ухудшение. Шкалы одинаковы для всех трёх фотографий: '
             'абсолютные различия 0–100, уменьшение ошибки −50…50 уровней RGB.\n']
    text += ['![Метрики форм масок и соответствующие контроли](results/masks_metrics.png)\n']
    for photo in photos:
        stem=Path(photo).stem
        text += [f'**{photo}**\n',f'![Маски и наложения: {photo}](results/mask_atlas_{stem}.png)\n',
                 f'![Все результаты масок: {photo}](results/mask_results_{stem}.png)\n',
                 f'![Чистый фон и контроль: {photo}](results/clean_context_{stem}.png)\n',
                 f'![Карты ошибки: {photo}](results/error_maps_{stem}.png)\n']
    text += [table([r for r in rows if 'masks' in r['studies'].split(';') and r['degradation']=='gaussian_15']),
             '### 3.2. Разные деградации: пять повторов\n',
             'Во всех девяти условиях и на трёх фотографиях использованы seed '
             '**42, 123, 204, 285, 366**. Меняются входной и диффузионный шум; '
             'внутри одного seed вход одинаков для всех масок. Показаны средние ± выборочные SD '
             '(ddof=1), не доверительные интервалы. При отклонениях усредняются только '
             'успешные результаты и указывается n/5. При n=1 SD не оценивается, полоса разброса отсутствует. '
             'У bicubic нет случайного входного шума, '
             'его baseline одинаков в пяти повторах. Zero одинаков при всех деградациях.\n',
             '![Деградации: Рубеж веков сверху, town снизу](results/noise_inputs.png)\n']
    for photo in photos:
        text += [f'**{photo}**\n',f'![Средние и изменения к контролю: {photo}](results/noise_means_{Path(photo).stem}.png)\n',
                 'В ячейке: PSNR / LPIPS / CLIP-IQA, каждое среднее ± SD.\n',
                 '| Деградация | Baseline | Zero | One | Mixed |','|---|---|---|---|---|']
        for noise,label in zip(NOISES,LABELS):
            cells=[]
            for mask in ('baseline','zero','one','mixed'):
                r=summary[photo,noise,mask]
                cells.append(' / '.join(mean_sd(r,k) for k in KEYS)+f' (n={r["n"]}/5)')
            text.append('| '+' | '.join([label]+cells)+' |')
        text.append('')
    text += ['#### Каждый тип деградации\n']
    for noise,label in zip(NOISES,LABELS):
        text += [f'**{label}**\n',f'![Средние для {label}](results/noise_{noise}.png)\n',
                 f'![Как выглядит {label}, seed=42](results/input_{noise}.png)\n']
    text += ['### 3.3. Чувствительность к seed\n',
             'Gaussian σ=15, one/mixed, пять seed. Пунктир — соответствующий контроль с чистым фоном; '
             'он тоже меняется из-за новой реализации входного шума. Zero остаётся чистым эталоном.\n',
             '![Пять seed](results/seeds.png)\n',
             table([r for r in rows if 'seeds' in r['studies'].split(';') and r['mask'] in ('one','mixed')]),
             '### 3.4. Влияние strength\n',
             'Gaussian σ=15, seed=42, one/mixed; strength=0.2/0.35/0.5 даёт 6/10/15 шагов DDIM. '
             'Пунктирные контроли не зависят от strength: без нейросети их пиксели одинаковы.\n',
             '![Strength с чистым фоном](results/strength.png)\n',
             table([r for r in rows if 'strength' in r['studies'].split(';')]),
             '### 3.5. Размер горизонтальной прямоугольной маски\n',
             'Background 512×288, Gaussian σ=15, seed=42, strength=0.35. '
             'Маски вложенные центральные: обе стороны уменьшаются пропорционально '
             '100/80/60/40/20%, начиная со всего кадра. Вверху маски, затем контроль с чистым фоном '
             'и результаты восстановления.\n',
             '![Пять прямоугольников](results/rectangle_sizes.png)\n',
             '![Метрики прямоугольников](results/rectangle_size_metrics.png)\n',
             table([r for r in rows if 'rectangle_sizes' in r['studies'].split(';')]),
             '### 3.6. Размер круглой маски\n',
             'Те же параметры. Диаметры 288/230/173/115/58 px; максимальный круг вписан '
             'по высоте изображения. Вне круга берётся чистый эталон.\n',
             '![Пять кругов](results/disk_sizes.png)\n',
             '![Метрики кругов](results/disk_size_metrics.png)\n',
             table([r for r in rows if 'circle_sizes' in r['studies'].split(';')]),
             '## 4. Выводы по новым измерениям\n']
    for study,title in (('rectangle_sizes','Прямоугольники'),('circle_sizes','Круги')):
        group=sorted([r for r in rows if study in r['studies'].split(';') and r['status']=='ok'],key=lambda r:-float(r['mask_fraction']))
        if group:
            large,small=group[0],group[-1]
            text.append(f'**{title}:** при переходе от ROI {float(large["mask_fraction"]):.2%} '
                        f'к {float(small["mask_fraction"]):.2%} PSNR меняется '
                        f'{fmt(large["psnr"],3)}→{fmt(small["psnr"],3)} дБ, LPIPS '
                        f'{fmt(large["lpips"])}→{fmt(small["lpips"])}, CLIP-IQA '
                        f'{fmt(large["clipiqa"])}→{fmt(small["clipiqa"])}. '
                        f'Контроль без нейросети на этих же площадях имеет PSNR '
                        f'{fmt(large["control_psnr"],3)}→{fmt(small["control_psnr"],3)} дБ.\n')
    accepted=[r for r in rows if 'noise' in r['studies'].split(';') and r['mask'] in ('one','mixed') and r['status']=='ok']
    counts={k:sum((float(r[k])<float(r['control_'+k]) if k=='lpips' else float(r[k])>float(r['control_'+k])) for r in accepted) for k in KEYS}
    text += [f'Среди **{len(accepted)} успешных** восстановлений one/mixed в исследовании шумов '
             f'по сравнению с контролем той же ROI PSNR улучшился в **{counts["psnr"]}**, '
             f'LPIPS — в **{counts["lpips"]}**, CLIP-IQA — в **{counts["clipiqa"]}** случаях. '
             'Это описательные подсчёты отдельных повторов, не тест статистической значимости.\n',
             'Zero имеет идеальное совпадение с эталоном по определению, а не благодаря нейросети. '
             'При уменьшении ROI чистый фон занимает всё большую часть кадра, поэтому улучшение '
             'полных метрик нельзя целиком приписывать восстановлению. Для оценки модели нужны '
             'парный контроль и PSNR ROI. Нельзя напрямую сравнивать частичные маски этой версии '
             'с прежней, не учитывая изменение доступной модели информации.\n',
             'Все успешные результаты вне ROI совпадают с чистым эталоном пиксель в пиксель. '
             'Отклонения модели не заменяются чёрными изображениями или нулевыми метриками. '
             'Одна фотография background уже содержит дымку; эксперименты не доказывают её удаление.\n',
             '## 5. Данные, воспроизводимость и источники\n',
             '[Все измерения](results/metrics.csv), [средние по пяти повторам](results/noise_summary.csv), '
             '[план 585 условий](results/plan.json), [конфигурация и SHA-256](results/config.json), '
             '[проверка результатов](results/validation.json).\n',
             '- [Задание](../LW3-2.pdf).\n'
             '- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).\n'
             '- [DDIM](https://arxiv.org/abs/2010.02502).\n'
             '- [LPIPS](https://arxiv.org/abs/1801.03924).\n'
             '- [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).\n']
    (ROOT/'README.md').write_text('\n'.join(text),encoding='utf-8')
    print('Новый README и все графики готовы',flush=True)


if __name__=='__main__': main()
