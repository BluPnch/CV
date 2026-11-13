"""Собирает архив с кодом и фотографиями для загрузки в Kaggle Dataset."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    """Исключает веса, окружения, сторонний репозиторий и результаты."""
    root = Path(__file__).resolve().parent
    target = root / 'LW3_2_kaggle.zip'
    files = sorted(p for p in root.iterdir() if p.suffix in ('.py', '.md', '.txt', '.ipynb'))
    files += sorted((root / 'input').glob('*'))
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, str(Path('LW3_2') / path.relative_to(root)))
    print(target)


if __name__ == '__main__':
    main()
