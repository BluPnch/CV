"""Приложение README с превью и ссылками на 16 выходных видео."""
from pathlib import Path

import cv2

from experiment_config import ROOT, VIDEO_FILES, VIDEO_LABELS
from tracker import DETECTORS

HEADER = '## Приложение. Видео с результатами трекинга'


def preview(video, detector):
    path = ROOT / 'results' / f'{video}_{detector}.mp4'
    poster = ROOT / 'images' / 'videos' / f'{video}_{detector}.jpg'
    if not path.is_file():
        raise FileNotFoundError(f'Нет выходного видео: {path}')
    if not poster.is_file() or poster.stat().st_mtime < path.stat().st_mtime:
        cap = cv2.VideoCapture(str(path))
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            index = min(100, frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Не удалось прочитать кадр {index}: {path}')
        finally:
            cap.release()
        height, width = frame.shape[:2]
        scale = min(320 / width, 240 / height, 1.0)
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        poster.parent.mkdir(exist_ok=True, parents=True)
        if not cv2.imwrite(str(poster), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f'Не удалось сохранить превью: {poster}')
    return path.relative_to(ROOT).as_posix(), poster.relative_to(ROOT).as_posix()


def appendix_lines():
    lines = [HEADER, '',
             'Полные выходные видео с рамкой объекта: три варианта Mona Lisa и собственное видео, каждый обработан методами SIFT, ORB, AKAZE и BRISK. Всего **4 видео × 4 детектора = 16 роликов**.', '',
             'Все ролики получены в режиме `hybrid`: поиск по эталону, оптический поток и краткосрочный прогноз. Измеренная рамка зелёная, прогнозируемая — жёлтая с пометкой `predicted`.', '',
             '**Нажмите на превью или ссылку «Открыть MP4», чтобы открыть видео.** Если просмотрщик не воспроизводит файл, его можно скачать и открыть в видеоплеере.', '']
    for number, filename in enumerate(VIDEO_FILES, 1):
        video = Path(filename).stem
        entries = [preview(video, detector) for detector in DETECTORS]
        lines += [f'### П.{number}. {VIDEO_LABELS[video]}', '',
                  f'Исходный файл: `{filename}`. Превью соответствуют кадру с индексом 100.', '',
                  '| SIFT | ORB | AKAZE | BRISK |', '|---|---|---|---|',
                  '| ' + ' | '.join(f'[![{VIDEO_LABELS[video]} — {det.upper()}]({poster})]({path})'
                                    for det, (path, poster) in zip(DETECTORS, entries)) + ' |',
                  '| ' + ' | '.join(f'[Открыть MP4]({path})' for path, _ in entries) + ' |', '']
    return lines


def main():
    path = ROOT / 'README.md'
    original = path.read_text(encoding='utf-8-sig')
    body = original.split(HEADER, 1)[0].rstrip()
    path.write_text(body + '\n\n' + '\n'.join(appendix_lines()), encoding='utf-8')
    print('Приложение: 16 видео с превью и ссылками.')


if __name__ == '__main__':
    main()
