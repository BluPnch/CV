"""Экономит диск, объединяя идентичные PNG только внутри новой лабораторной."""
import os
from pathlib import Path

from run_all import ROOT,OUTPUT,sha


def main():
    root=OUTPUT.resolve()
    canonical={}
    reclaimed=0
    count=0
    for path in sorted(root.rglob('*.png')):
        if not path.resolve().is_relative_to(root):
            raise ValueError(path)
        digest=sha(path)
        if digest not in canonical:
            canonical[digest]=path
            continue
        target=canonical[digest]
        if os.path.samefile(path,target):
            continue
        size=path.stat().st_size
        temporary=path.with_suffix('.link.tmp')
        os.link(target,temporary)
        os.replace(temporary,path)
        reclaimed+=size
        count+=1
    print(f'Объединено {count} одинаковых PNG внутри LW3_CORRECT: {reclaimed/2**20:.1f} MiB')


if __name__=='__main__': main()
