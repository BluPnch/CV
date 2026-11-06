"""Полный цикл обработки строго четырёх видео и обновления отчёта."""
import subprocess
import sys
from experiment_config import ROOT

if __name__ == '__main__':
    for script in ['evaluate.py', 'compare_estimators.py', 'figures.py', 'report.py']:
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
