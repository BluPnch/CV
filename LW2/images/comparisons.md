# Основные сравнения

Режим detect: поток и прогноз выключены. Первый кадр исключён.

Серые интервалы — кадры с отличием от исходного ролика > 5 уровней яркости.

## 1. Разные детекторы на одном видео: 4 метрики × 3 видео = 12 графиков

### Mona Lisa: исходное

![Накопленная доля кадров с рамкой, %](detectors/mona-lisa_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](detectors/mona-lisa_inliers.png)

![Ключевые точки (среднее за 21 кадр)](detectors/mona-lisa_keypoints.png)

![Обработка, мс (среднее за 21 кадр)](detectors/mona-lisa_latency.png)

### Mona Lisa: blur

![Накопленная доля кадров с рамкой, %](detectors/mona-lisa-blur_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](detectors/mona-lisa-blur_inliers.png)

![Ключевые точки (среднее за 21 кадр)](detectors/mona-lisa-blur_keypoints.png)

![Обработка, мс (среднее за 21 кадр)](detectors/mona-lisa-blur_latency.png)

### Mona Lisa: extra credit

![Накопленная доля кадров с рамкой, %](detectors/mona-lisa-blur-extra-credit_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](detectors/mona-lisa-blur-extra-credit_inliers.png)

![Ключевые точки (среднее за 21 кадр)](detectors/mona-lisa-blur-extra-credit_keypoints.png)

![Обработка, мс (среднее за 21 кадр)](detectors/mona-lisa-blur-extra-credit_latency.png)

## 2. Варианты размытия для одного детектора: 3 метрики × 4 детектора = 12 графиков

Сравниваются исходный ролик, blur и extra credit; названия не задают конкретный тип ядра размытия.

### SIFT

![Накопленная доля кадров с рамкой, %](blur/sift_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](blur/sift_inliers.png)

![Ключевые точки (среднее за 21 кадр)](blur/sift_keypoints.png)

### ORB

![Накопленная доля кадров с рамкой, %](blur/orb_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](blur/orb_inliers.png)

![Ключевые точки (среднее за 21 кадр)](blur/orb_keypoints.png)

### AKAZE

![Накопленная доля кадров с рамкой, %](blur/akaze_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](blur/akaze_inliers.png)

![Ключевые точки (среднее за 21 кадр)](blur/akaze_keypoints.png)

### BRISK

![Накопленная доля кадров с рамкой, %](blur/brisk_coverage.png)

![Принятые инлаеры (среднее за 21 кадр)](blur/brisk_inliers.png)

![Ключевые точки (среднее за 21 кадр)](blur/brisk_keypoints.png)
