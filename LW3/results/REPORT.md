# Исследования на пользовательских фотографиях

PSNR ↑, LPIPS ↓, CLIP-IQA ↑ измерены по полному RGB-кадру. PSNR ROI приведён отдельно в CSV.

Все эталоны получены уменьшением пользовательских JPEG до длинной стороны 512 px (точные размеры — runtime.json). Это опорные фотографии для контролируемых деградаций, а не доказанно чистые HR-эталоны.

Всего опытов: 104; успешных измерений: 99; отклонено штатной проверкой модели: 5. Для отклонений метрики отсутствуют; чёрные заглушки не оцениваются.

## Форма и площадь маски

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rubezh_vekov.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 29.614 | 0.2244 | 0.6599 |
| rubezh_vekov.jpg | bicubic | one | 42 | 0.35 | 1.000 | 24.318 | 0.2964 | 0.4153 |
| rubezh_vekov.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 27.815 | 0.2424 | 0.5710 |
| rubezh_vekov.jpg | bicubic | left_half | 42 | 0.35 | 0.500 | 26.272 | 0.2650 | 0.4667 |
| rubezh_vekov.jpg | bicubic | circle | 42 | 0.35 | 0.262 | 27.828 | 0.2412 | 0.5546 |
| rubezh_vekov.jpg | bicubic | stripes | 42 | 0.35 | 0.500 | 26.903 | 0.2414 | 0.4287 |
| rubezh_vekov.jpg | bicubic | blocks | 42 | 0.35 | 0.243 | 28.276 | 0.2329 | 0.4555 |
| rubezh_vekov.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.232 | 0.5761 | 0.4438 |
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.071 | 0.5434 | 0.2581 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.369 | 0.5651 | 0.2381 |
| rubezh_vekov.jpg | gaussian_15 | left_half | 42 | 0.35 | 0.500 | 23.376 | 0.5556 | 0.2277 |
| rubezh_vekov.jpg | gaussian_15 | circle | 42 | 0.35 | 0.262 | 24.349 | 0.5652 | 0.2428 |
| rubezh_vekov.jpg | gaussian_15 | stripes | 42 | 0.35 | 0.500 | 23.703 | 0.5543 | 0.2391 |
| rubezh_vekov.jpg | gaussian_15 | blocks | 42 | 0.35 | 0.243 | 24.504 | 0.5556 | 0.2626 |
| town.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 27.213 | 0.4007 | 0.3800 |
| town.jpg | bicubic | one | 42 | 0.35 | 1.000 | 24.390 | 0.4971 | 0.2455 |
| town.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 26.336 | 0.4307 | 0.3254 |
| town.jpg | bicubic | left_half | 42 | 0.35 | 0.500 | 25.290 | 0.4457 | 0.3054 |
| town.jpg | bicubic | circle | 42 | 0.35 | 0.280 | 26.249 | 0.4268 | 0.3178 |
| town.jpg | bicubic | stripes | 42 | 0.35 | 0.532 | 25.895 | 0.4276 | 0.2390 |
| town.jpg | bicubic | blocks | 42 | 0.35 | 0.244 | 26.591 | 0.4057 | 0.2877 |
| town.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 24.071 | 0.8368 | 0.2111 |
| town.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.816 | 0.7711 | 0.1404 |
| town.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 23.721 | 0.8160 | 0.1683 |
| town.jpg | gaussian_15 | left_half | 42 | 0.35 | 0.500 | 23.109 | 0.7998 | 0.1702 |
| town.jpg | gaussian_15 | circle | 42 | 0.35 | 0.280 | 23.724 | 0.8140 | 0.1541 |
| town.jpg | gaussian_15 | stripes | 42 | 0.35 | 0.532 | 23.492 | 0.7803 | 0.1549 |
| town.jpg | gaussian_15 | blocks | 42 | 0.35 | 0.244 | 23.800 | 0.7986 | 0.1742 |
| rubezh_vekov.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 29.614 | 0.2244 | 0.6599 |
| rubezh_vekov.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.232 | 0.5761 | 0.4438 |
| town.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 27.213 | 0.4007 | 0.3800 |
| town.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 24.071 | 0.8368 | 0.2111 |

## Тип и интенсивность шума

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rubezh_vekov.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 29.614 | 0.2244 | 0.6599 |
| rubezh_vekov.jpg | bicubic | one | 42 | 0.35 | 1.000 | 24.318 | 0.2964 | 0.4153 |
| rubezh_vekov.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 27.815 | 0.2424 | 0.5710 |
| rubezh_vekov.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.232 | 0.5761 | 0.4438 |
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.071 | 0.5434 | 0.2581 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.369 | 0.5651 | 0.2381 |
| rubezh_vekov.jpg | gaussian_5 | zero | 42 | 0.35 | 0.000 | 28.809 | 0.3440 | 0.5191 |
| rubezh_vekov.jpg | gaussian_5 | one | 42 | 0.35 | 1.000 | 24.065 | 0.3491 | 0.3542 |
| rubezh_vekov.jpg | gaussian_5 | mixed | 42 | 0.35 | 0.250 | 27.163 | 0.3485 | 0.3917 |
| rubezh_vekov.jpg | gaussian_30 | zero | 42 | 0.35 | 0.000 | 20.801 | 0.7029 | 0.4031 |
| rubezh_vekov.jpg | gaussian_30 | one | 42 | 0.35 | 1.000 | — | — | — |
| rubezh_vekov.jpg | gaussian_30 | mixed | 42 | 0.35 | 0.250 | 20.537 | 0.6905 | 0.2586 |
| rubezh_vekov.jpg | poisson_30 | zero | 42 | 0.35 | 0.000 | 20.727 | 0.7072 | 0.3912 |
| rubezh_vekov.jpg | poisson_30 | one | 42 | 0.35 | 1.000 | — | — | — |
| rubezh_vekov.jpg | poisson_30 | mixed | 42 | 0.35 | 0.250 | 20.476 | 0.6962 | 0.2770 |
| rubezh_vekov.jpg | poisson_10 | zero | 42 | 0.35 | 0.000 | 16.830 | 0.7640 | 0.4605 |
| rubezh_vekov.jpg | poisson_10 | one | 42 | 0.35 | 1.000 | — | — | — |
| rubezh_vekov.jpg | poisson_10 | mixed | 42 | 0.35 | 0.250 | — | — | — |
| rubezh_vekov.jpg | impulse_0.01 | zero | 42 | 0.35 | 0.000 | 23.796 | 0.4819 | 0.5811 |
| rubezh_vekov.jpg | impulse_0.01 | one | 42 | 0.35 | 1.000 | 22.260 | 0.4699 | 0.5081 |
| rubezh_vekov.jpg | impulse_0.01 | mixed | 42 | 0.35 | 0.250 | 23.277 | 0.4802 | 0.4926 |
| rubezh_vekov.jpg | impulse_0.05 | zero | 42 | 0.35 | 0.000 | 18.586 | 0.6584 | 0.4466 |
| rubezh_vekov.jpg | impulse_0.05 | one | 42 | 0.35 | 1.000 | 18.890 | 0.6210 | 0.4705 |
| rubezh_vekov.jpg | impulse_0.05 | mixed | 42 | 0.35 | 0.250 | 18.603 | 0.6545 | 0.3814 |
| rubezh_vekov.jpg | blur_noise_jpeg | zero | 42 | 0.35 | 0.000 | 25.788 | 0.4246 | 0.5958 |
| rubezh_vekov.jpg | blur_noise_jpeg | one | 42 | 0.35 | 1.000 | — | — | — |
| rubezh_vekov.jpg | blur_noise_jpeg | mixed | 42 | 0.35 | 0.250 | 24.837 | 0.4242 | 0.3243 |
| town.jpg | bicubic | zero | 42 | 0.35 | 0.000 | 27.213 | 0.4007 | 0.3800 |
| town.jpg | bicubic | one | 42 | 0.35 | 1.000 | 24.390 | 0.4971 | 0.2455 |
| town.jpg | bicubic | mixed | 42 | 0.35 | 0.250 | 26.336 | 0.4307 | 0.3254 |
| town.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 24.071 | 0.8368 | 0.2111 |
| town.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.816 | 0.7711 | 0.1404 |
| town.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 23.721 | 0.8160 | 0.1683 |
| town.jpg | gaussian_5 | zero | 42 | 0.35 | 0.000 | 26.653 | 0.5708 | 0.2836 |
| town.jpg | gaussian_5 | one | 42 | 0.35 | 1.000 | 23.957 | 0.6011 | 0.1345 |
| town.jpg | gaussian_5 | mixed | 42 | 0.35 | 0.250 | 25.855 | 0.5834 | 0.1841 |
| town.jpg | gaussian_30 | zero | 42 | 0.35 | 0.000 | 20.491 | 0.9270 | 0.2478 |
| town.jpg | gaussian_30 | one | 42 | 0.35 | 1.000 | 20.841 | 0.8752 | 0.1798 |
| town.jpg | gaussian_30 | mixed | 42 | 0.35 | 0.250 | 20.504 | 0.9154 | 0.2051 |
| town.jpg | poisson_30 | zero | 42 | 0.35 | 0.000 | 20.819 | 0.8992 | 0.2262 |
| town.jpg | poisson_30 | one | 42 | 0.35 | 1.000 | 20.754 | 0.8324 | 0.1038 |
| town.jpg | poisson_30 | mixed | 42 | 0.35 | 0.250 | 20.718 | 0.8841 | 0.1616 |
| town.jpg | poisson_10 | zero | 42 | 0.35 | 0.000 | 17.123 | 0.9446 | 0.3024 |
| town.jpg | poisson_10 | one | 42 | 0.35 | 1.000 | 17.842 | 0.9206 | 0.2208 |
| town.jpg | poisson_10 | mixed | 42 | 0.35 | 0.250 | 17.159 | 0.9524 | 0.2386 |
| town.jpg | impulse_0.01 | zero | 42 | 0.35 | 0.000 | 23.245 | 0.5627 | 0.4744 |
| town.jpg | impulse_0.01 | one | 42 | 0.35 | 1.000 | 22.458 | 0.5850 | 0.2774 |
| town.jpg | impulse_0.01 | mixed | 42 | 0.35 | 0.250 | 22.917 | 0.5714 | 0.4286 |
| town.jpg | impulse_0.05 | zero | 42 | 0.35 | 0.000 | 18.520 | 0.7262 | 0.3781 |
| town.jpg | impulse_0.05 | one | 42 | 0.35 | 1.000 | 19.010 | 0.7143 | 0.3866 |
| town.jpg | impulse_0.05 | mixed | 42 | 0.35 | 0.250 | 18.494 | 0.7214 | 0.3654 |
| town.jpg | blur_noise_jpeg | zero | 42 | 0.35 | 0.000 | 25.454 | 0.6933 | 0.2734 |
| town.jpg | blur_noise_jpeg | one | 42 | 0.35 | 1.000 | 23.486 | 0.6785 | 0.1845 |
| town.jpg | blur_noise_jpeg | mixed | 42 | 0.35 | 0.250 | 24.777 | 0.6876 | 0.2279 |
| rubezh_vekov.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 29.614 | 0.2244 | 0.6599 |
| rubezh_vekov.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.232 | 0.5761 | 0.4438 |
| rubezh_vekov.jpg | gaussian_5 | baseline | 42 | 0.35 | 1.000 | 28.809 | 0.3440 | 0.5191 |
| rubezh_vekov.jpg | gaussian_30 | baseline | 42 | 0.35 | 1.000 | 20.801 | 0.7029 | 0.4031 |
| rubezh_vekov.jpg | poisson_30 | baseline | 42 | 0.35 | 1.000 | 20.727 | 0.7072 | 0.3912 |
| rubezh_vekov.jpg | poisson_10 | baseline | 42 | 0.35 | 1.000 | 16.830 | 0.7640 | 0.4605 |
| rubezh_vekov.jpg | impulse_0.01 | baseline | 42 | 0.35 | 1.000 | 23.796 | 0.4819 | 0.5811 |
| rubezh_vekov.jpg | impulse_0.05 | baseline | 42 | 0.35 | 1.000 | 18.586 | 0.6584 | 0.4466 |
| rubezh_vekov.jpg | blur_noise_jpeg | baseline | 42 | 0.35 | 1.000 | 25.788 | 0.4246 | 0.5958 |
| town.jpg | bicubic | baseline | 42 | 0.35 | 1.000 | 27.213 | 0.4007 | 0.3800 |
| town.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 24.071 | 0.8368 | 0.2111 |
| town.jpg | gaussian_5 | baseline | 42 | 0.35 | 1.000 | 26.653 | 0.5708 | 0.2836 |
| town.jpg | gaussian_30 | baseline | 42 | 0.35 | 1.000 | 20.491 | 0.9270 | 0.2478 |
| town.jpg | poisson_30 | baseline | 42 | 0.35 | 1.000 | 20.819 | 0.8992 | 0.2262 |
| town.jpg | poisson_10 | baseline | 42 | 0.35 | 1.000 | 17.123 | 0.9446 | 0.3024 |
| town.jpg | impulse_0.01 | baseline | 42 | 0.35 | 1.000 | 23.245 | 0.5627 | 0.4744 |
| town.jpg | impulse_0.05 | baseline | 42 | 0.35 | 1.000 | 18.520 | 0.7262 | 0.3781 |
| town.jpg | blur_noise_jpeg | baseline | 42 | 0.35 | 1.000 | 25.454 | 0.6933 | 0.2734 |

## Повтор с другим seed

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rubezh_vekov.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 25.232 | 0.5761 | 0.4438 |
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.071 | 0.5434 | 0.2581 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.369 | 0.5651 | 0.2381 |
| rubezh_vekov.jpg | gaussian_15 | zero | 123 | 0.35 | 0.000 | 25.238 | 0.5782 | 0.3539 |
| rubezh_vekov.jpg | gaussian_15 | one | 123 | 0.35 | 1.000 | 22.422 | 0.5468 | 0.2548 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 123 | 0.35 | 0.250 | 24.514 | 0.5745 | 0.2733 |
| town.jpg | gaussian_15 | zero | 42 | 0.35 | 0.000 | 24.071 | 0.8368 | 0.2111 |
| town.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.816 | 0.7711 | 0.1404 |
| town.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 23.721 | 0.8160 | 0.1683 |
| town.jpg | gaussian_15 | zero | 123 | 0.35 | 0.000 | 24.088 | 0.8358 | 0.2265 |
| town.jpg | gaussian_15 | one | 123 | 0.35 | 1.000 | 22.736 | 0.7877 | 0.1543 |
| town.jpg | gaussian_15 | mixed | 123 | 0.35 | 0.250 | 23.636 | 0.8096 | 0.1929 |
| rubezh_vekov.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.232 | 0.5761 | 0.4438 |
| rubezh_vekov.jpg | gaussian_15 | baseline | 123 | 0.35 | 1.000 | 25.238 | 0.5782 | 0.3539 |
| town.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 24.071 | 0.8368 | 0.2111 |
| town.jpg | gaussian_15 | baseline | 123 | 0.35 | 1.000 | 24.088 | 0.8358 | 0.2265 |

## Глубина диффузионного уточнения

| Фото | Деградация | Маска | seed | strength | Доля ROI | PSNR | LPIPS | CLIP-IQA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.071 | 0.5434 | 0.2581 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 24.369 | 0.5651 | 0.2381 |
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.2 | 1.000 | 23.793 | 0.5222 | 0.2244 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.2 | 0.250 | 24.820 | 0.5630 | 0.2779 |
| rubezh_vekov.jpg | gaussian_15 | one | 42 | 0.5 | 1.000 | 19.869 | 0.4965 | 0.6344 |
| rubezh_vekov.jpg | gaussian_15 | mixed | 42 | 0.5 | 0.250 | 23.824 | 0.5633 | 0.2454 |
| town.jpg | gaussian_15 | one | 42 | 0.35 | 1.000 | 22.816 | 0.7711 | 0.1404 |
| town.jpg | gaussian_15 | mixed | 42 | 0.35 | 0.250 | 23.721 | 0.8160 | 0.1683 |
| town.jpg | gaussian_15 | one | 42 | 0.2 | 1.000 | 23.641 | 0.7577 | 0.1200 |
| town.jpg | gaussian_15 | mixed | 42 | 0.2 | 0.250 | 23.947 | 0.8116 | 0.1551 |
| town.jpg | gaussian_15 | one | 42 | 0.5 | 1.000 | 20.977 | 0.6480 | 0.4436 |
| town.jpg | gaussian_15 | mixed | 42 | 0.5 | 0.250 | 23.131 | 0.8322 | 0.1695 |
| rubezh_vekov.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 25.232 | 0.5761 | 0.4438 |
| town.jpg | gaussian_15 | baseline | 42 | 0.35 | 1.000 | 24.071 | 0.8368 | 0.2111 |

## Контроль сохранения фона

Максимальное изменение вне маски по успешным измерениям: 0 уровней яркости.

Сравнение разных форм масок одновременно зависит от их площади. Полные метрики учитывают неизменённый фон; это ограничивает сравнение частичной и полной маски. Два seed показывают чувствительность конкретного опыта, но не дают надёжного статистического вывода.
