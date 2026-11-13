"""Готовит отдельную среду SUPIR, не меняя системный PyTorch Kaggle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(*args):
    """Выполняет команду с немедленной остановкой при ошибке."""
    subprocess.run([str(a) for a in args], check=True)


def main():
    """Устанавливает зависимости и сохраняет точную версию исходников SUPIR."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/kaggle/working/supir_lab'))
    parser.add_argument('--revision', default='bda91af2000042f8bedfec8897d92917e67c1d88',
                        help='Проверенная версия исходников SUPIR')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run(sys.executable, '-m', 'pip', 'install', 'uv==0.6.17')
    run(sys.executable, '-m', 'uv', 'venv', '--python', '3.10', '--seed', root / 'venv')
    python = root / 'venv/bin/python'
    run(python, '-m', 'pip', 'install', 'torch==2.2.2', 'torchvision==0.17.2',
        '--index-url', 'https://download.pytorch.org/whl/cu121')
    constraints = root / 'constraints.txt'
    constraints.write_text('torch==2.2.2\ntorchvision==0.17.2\nxformers==0.0.25.post1\n', encoding='utf-8')
    run(python, '-m', 'pip', 'install', '-c', constraints, 'xformers==0.0.25.post1',
        '-r', Path(__file__).with_name('requirements.txt'))
    repo = root / 'SUPIR'
    if not repo.exists():
        run('git', 'clone', 'https://github.com/Fanghua-Yu/SUPIR.git', repo)
        run('git', '-C', repo, 'checkout', args.revision)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    # Отключаем пути разработчиков; CLIP загружается из общедоступных репозиториев.
    (repo / 'CKPT_PTH.py').write_text(
        'LLAVA_CLIP_PATH = None\nLLAVA_MODEL_PATH = None\n'
        'SDXL_CLIP1_PATH = None\nSDXL_CLIP2_CKPT_PTH = None\n', encoding='utf-8')
    (root / 'setup.json').write_text(json.dumps({'supir_commit': commit,
        'python': str(python), 'repo': str(repo)}, indent=2), encoding='utf-8')
    with (root / 'environment.txt').open('w', encoding='utf-8') as handle:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=handle, check=True)
    print(f'Готово: {python}; SUPIR revision={commit}')


if __name__ == '__main__':
    main()
