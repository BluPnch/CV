"""Единый список входов для CLI, экспериментов и иллюстраций."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_FILES = (
    "mona-lisa.avi",
    "mona-lisa-blur.avi",
    "mona-lisa-blur-extra-credit.avi",
    "my_video.mp4",
)
MONA_VIDEOS = tuple(Path(name).stem for name in VIDEO_FILES[:3])
VIDEO_LABELS = {
    "mona-lisa": "Mona Lisa: исходное",
    "mona-lisa-blur": "Mona Lisa: blur",
    "mona-lisa-blur-extra-credit": "Mona Lisa: extra credit",
    "my_video": "Собственное видео",
}


def validate_video(path):
    path = Path(path).resolve()
    allowed = [(ROOT / "videos" / name).resolve() for name in VIDEO_FILES]
    if path not in allowed:
        raise ValueError("Допустимы только четыре входа из experiment_config.VIDEO_FILES")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
