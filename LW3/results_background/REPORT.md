# Исследования на пользовательских фотографиях

PSNR ↑, LPIPS ↓, CLIP-IQA ↑ измерены по полному RGB-кадру. PSNR ROI приведён отдельно в CSV.

Все эталоны получены уменьшением пользовательских JPEG до длинной стороны 512 px (точные размеры — runtime.json). Это опорные фотографии для контролируемых деградаций, а не доказанно чистые HR-эталоны.

Всего опытов: 52; успешных измерений: 50; отклонено штатной проверкой модели: 2. Для отклонений метрики отсутствуют; чёрные заглушки не оцениваются.

## Форма и площадь маски

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| background.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 32.369 | 0.3896 | 0.3988 |
| background.jpg | bicubic | one | 42 | 0.35 | 1.000 | 25.012 | 0.5035 | 0.3281 |
| background.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 28.723 | 0.4200 | 0.3162 |
| background.jpg | bicubic | left_half | 42 | 0.35 | 0.500 | 27.146 | 0.4546 | 0.3521 |
| background.jpg | bicubic | circle | 42 | 0.35 | 0.196 | 29.420 | 0.4111 | 0.3116 |
| background.jpg | bicubic | stripes | 42 | 0.35 | 0.500 | 27.875 | 0.4239 | 0.2575 |
| background.jpg | bicubic | blocks | 42 | 0.35 | 0.248 | 30.077 | 0.3823 | 0.2435 |
| background.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.472 | 0.7541 | 0.3450 |
| background.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 21.556 | 0.7239 | 0.1742 |
| background.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.186 | 0.7369 | 0.2539 |
| background.jpg | gaussian_15 | left_half | 42 | 0.35 | 0.500 | 22.699 | 0.7406 | 0.2937 |
| background.jpg | gaussian_15 | circle | 42 | 0.35 | 0.196 | 24.427 | 0.7387 | 0.2789 |
| background.jpg | gaussian_15 | stripes | 42 | 0.35 | 0.500 | 23.463 | 0.7260 | 0.2360 |
| background.jpg | gaussian_15 | blocks | 42 | 0.35 | 0.248 | 24.755 | 0.7116 | 0.2234 |
| background.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 32.369 | 0.3896 | 0.3988 |
| background.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.472 | 0.7541 | 0.3450 |

## Тип и интенсивность шума

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| background.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 32.369 | 0.3896 | 0.3988 |
| background.jpg | bicubic | one | 42 | 0.35 | 1.000 | 25.012 | 0.5035 | 0.3281 |
| background.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 28.723 | 0.4200 | 0.3162 |
| background.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.472 | 0.7541 | 0.3450 |
| background.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 21.556 | 0.7239 | 0.1742 |
| background.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.186 | 0.7369 | 0.2539 |
| background.jpg | gaussian_5 | zero | 42 | 0.35 | 0.000 | 30.766 | 0.5102 | 0.3325 |
| background.jpg | gaussian_5 | one | 42 | 0.35 | 1.000 | 24.385 | 0.5584 | 0.2165 |
| background.jpg | gaussian_5 | mixed | 42 | 0.35 | 0.250 | 27.728 | 0.5292 | 0.2950 |
| background.jpg | gaussian_30 | zero | 42 | 0.35 | 0.000 | 20.355 | 0.8381 | 0.2254 |
| background.jpg | gaussian_30 | one | 42 | 0.35 | 1.000 | 19.201 | 0.7778 | 0.1637 |
| background.jpg | gaussian_30 | mixed | 42 | 0.35 | 0.250 | 20.015 | 0.8261 | 0.1714 |
| background.jpg | poisson_30 | zero | 42 | 0.35 | 0.000 | 19.581 | 0.8485 | 0.2153 |
| background.jpg | poisson_30 | one | 42 | 0.35 | 1.000 | 19.162 | 0.8091 | 0.1526 |
| background.jpg | poisson_30 | mixed | 42 | 0.35 | 0.250 | 19.277 | 0.8309 | 0.1608 |
| background.jpg | poisson_10 | zero | 42 | 0.35 | 0.000 | 15.558 | 0.8862 | 0.3421 |
| background.jpg | poisson_10 | one | 42 | 0.35 | 1.000 | 15.926 | 0.8930 | 0.1960 |
| background.jpg | poisson_10 | mixed | 42 | 0.35 | 0.250 | — | — | — |
| background.jpg | impulse_0.01 | zero | 42 | 0.35 | 0.000 | 25.422 | 0.6125 | 0.4427 |
| background.jpg | impulse_0.01 | one | 42 | 0.35 | 1.000 | 22.762 | 0.6146 | 0.3506 |
| background.jpg | impulse_0.01 | mixed | 42 | 0.35 | 0.250 | 24.409 | 0.6173 | 0.3756 |
| background.jpg | impulse_0.05 | zero | 42 | 0.35 | 0.000 | 19.778 | 0.7768 | 0.4016 |
| background.jpg | impulse_0.05 | one | 42 | 0.35 | 1.000 | 19.109 | 0.7581 | 0.3852 |
| background.jpg | impulse_0.05 | mixed | 42 | 0.35 | 0.250 | 19.442 | 0.7770 | 0.3151 |
| background.jpg | blur_noise_jpeg | zero | 42 | 0.35 | 0.000 | 28.469 | 0.6257 | 0.2661 |
| background.jpg | blur_noise_jpeg | one | 42 | 0.35 | 1.000 | 24.578 | 0.6062 | 0.2241 |
| background.jpg | blur_noise_jpeg | mixed | 42 | 0.35 | 0.250 | 26.761 | 0.6076 | 0.2174 |
| background.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 32.369 | 0.3896 | 0.3988 |
| background.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.472 | 0.7541 | 0.3450 |
| background.jpg | gaussian_5 | baseline | 42 | 0.35 | 1.000 | 30.766 | 0.5102 | 0.3325 |
| background.jpg | gaussian_30 | baseline | 42 | 0.35 | 1.000 | 20.355 | 0.8381 | 0.2254 |
| background.jpg | poisson_30 | baseline | 42 | 0.35 | 1.000 | 19.581 | 0.8485 | 0.2153 |
| background.jpg | poisson_10 | baseline | 42 | 0.35 | 1.000 | 15.558 | 0.8862 | 0.3421 |
| background.jpg | impulse_0.01 | baseline | 42 | 0.35 | 1.000 | 25.422 | 0.6125 | 0.4427 |
| background.jpg | impulse_0.05 | baseline | 42 | 0.35 | 1.000 | 19.778 | 0.7768 | 0.4016 |
| background.jpg | blur_noise_jpeg | baseline | 42 | 0.35 | 1.000 | 28.469 | 0.6257 | 0.2661 |

## Повтор с другим seed

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| background.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.472 | 0.7541 | 0.3450 |
| background.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 21.556 | 0.7239 | 0.1742 |
| background.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.186 | 0.7369 | 0.2539 |
| background.jpg | gaussian_15 | zero | 123 | 0.35 | 0.000 | 25.446 | 0.7596 | 0.3788 |
| background.jpg | gaussian_15 | one | 123 | 0.35 | 1.000 | 21.497 | 0.6723 | 0.2532 |
| background.jpg | gaussian_15 | mixed | 123 | 0.35 | 0.250 | 23.760 | 0.7377 | 0.3078 |
| background.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.472 | 0.7541 | 0.3450 |
| background.jpg | gaussian_15 | baseline | 123 | 0.35 | 1.000 | 25.446 | 0.7596 | 0.3788 |

## Глубина диффузионного уточнения

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| background.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 21.556 | 0.7239 | 0.1742 |
| background.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.186 | 0.7369 | 0.2539 |
| background.jpg | gaussian_15 | one | 42 | 0.2 | 1.000 | 23.942 | 0.6721 | 0.1769 |
| background.jpg | gaussian_15 | mixed | 42 | 0.2 | 0.250 | 24.890 | 0.7296 | 0.2904 |
| background.jpg | gaussian_15 | one | 42 | 0.5 | 1.000 | 20.581 | 0.6277 | 0.2110 |
| background.jpg | gaussian_15 | mixed | 42 | 0.5 | 0.250 | — | — | — |
| background.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.472 | 0.7541 | 0.3450 |

## Контроль сохранения фона

Максимальное изменение вне маски по успешным измерениям: 0.0 уровней яркости.

Сравнение разных форм масок одновременно зависит от их площади. Полные метрики учитывают неизменённый фон; это ограничивает сравнение частичной и полной маски. Два seed показывают чувствительность конкретного опыта, но не дают надёжного статистического вывода.
