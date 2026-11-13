"""Независимая проверка результатов, чистого фона и неизменности LW3."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from run_all import ROOT, OLD, OUTPUT, KEYS, SEEDS, make_mask, sha, save_json


def check_original():
    """Сравнивает содержимое старой лабораторной со снимком до начала работы."""
    manifest=json.loads((ROOT/'original_lw3_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        path=OLD/name
        if not path.exists() or sha(path)!=digest:
            raise AssertionError(f'Изменён исходный файл LW3: {path}')
    return len(manifest)


def close(a,b):
    """Сравнивает численные поля, включая математически бесконечный PSNR."""
    if a==float('inf') or b==float('inf'):
        assert a==b
    else:
        assert np.isclose(a,b,rtol=1e-9,atol=1e-9),(a,b)


def psnr_array(clean,result,mask=None):
    """Пересчитывает PSNR непосредственно из пикселей для аудита CSV."""
    error=(clean.astype(float)-result.astype(float))**2
    if mask is not None: error=error[mask.astype(bool)]
    if not error.size: return None
    mse=float(error.mean())
    return float('inf') if mse==0 else 10*math.log10(255**2/mse)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-only',action='store_true')
    args=parser.parse_args()
    original_count=check_original()
    if args.source_only:
        print(f'LW3 не изменена: проверено {original_count} файлов')
        return
    with (OUTPUT/'metrics.csv').open(encoding='utf-8') as handle: rows=list(csv.DictReader(handle))
    with (OUTPUT/'noise_summary.csv').open(encoding='utf-8') as handle: summary=list(csv.DictReader(handle))
    assert len(rows)==585 and len(summary)==108
    keys={(r['image'],r['degradation'],r['mask'],r['seed'],r['strength']) for r in rows}
    assert len(keys)==585
    references={r['image']:np.asarray(Image.open(OUTPUT/'images'/r['image']/'reference.png')).copy() for r in rows}
    successful=rejected=zero=reused=0
    for r in rows:
        clean=references[r['image']]
        size=(clean.shape[1],clean.shape[0]); mask=make_mask(r['mask'],size)
        control=np.asarray(Image.open(OUTPUT/r['control']))
        source=np.asarray(Image.open(OUTPUT/'images'/r['image']/r['degradation']/f'seed_{r["seed"]}'/'input.png'))
        np.testing.assert_array_equal(control,np.where(mask[...,None],source,clean))
        close(float(r['control_psnr']),psnr_array(clean,control))
        if r['reused_from']:
            assert r['mask']=='one'
            reused+=1
        if r['status']!='ok':
            rejected+=1
            assert all(not r[k] for k in KEYS)
            continue
        successful+=1
        path=OUTPUT/r['result']
        assert sha(path)==r['sha256']
        result=np.asarray(Image.open(path))
        if r['mask']!='baseline':
            np.testing.assert_array_equal(result[mask==0],clean[mask==0])
        if r['mask']=='zero':
            np.testing.assert_array_equal(result,clean)
            assert float(r['psnr'])==float('inf') and float(r['lpips'])==0
            zero+=1
        close(float(r['psnr']),psnr_array(clean,result))
        roi=psnr_array(clean,result,mask)
        if roi is None: assert not r['psnr_roi']
        else: close(float(r['psnr_roi']),roi)
        assert all(np.isfinite(float(r[k])) for k in ('lpips','clipiqa'))
    for s in summary:
        group=[r for r in rows if all(r[k]==s[k] for k in ('image','degradation','mask'))
               and float(r['strength'])==.35 and 'noise' in r['studies'].split(';')]
        assert len(group)==5 and {int(r['seed']) for r in group}==set(SEEDS)
        accepted=[r for r in group if r['status']=='ok']
        assert len(accepted)==int(s['n'])
        for key in KEYS:
            values=[float(r[key]) for r in accepted]
            if not values:
                assert not s[key+'_mean']
                continue
            close(float(s[key+'_mean']),float('inf') if np.isposinf(values).all() else float(np.mean(values)))
            if len(values)>1:
                if np.isposinf(values).all(): assert not s[key+'_std']
                else: close(float(s[key+'_std']),float(np.std(values,ddof=1)))
            else: assert not s[key+'_std']
            gains=[0. if r[key]==r['control_'+key] else
                   (float(r[key])-float(r['control_'+key]))*(-1 if key=='lpips' else 1) for r in accepted]
            close(float(s[key+'_gain_mean']),float(np.mean(gains)))
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in link: continue
        path=ROOT/link.split('#')[0]
        if path==OUTPUT/'validation.json': continue
        assert path.exists(),path
    report=dict(cases=585,successful=successful,rejected=rejected,zero_exact=zero,
                full_masks_reused=reused,noise_groups=108,old_files_unchanged=original_count,
                outside_reference='clean',outside_max_error=0,
                psnr_recomputed_from_pixels=True,matched_control_verified=True,
                mean_and_sd_verified=True,readme_links_verified=True,
                code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')})
    save_json(OUTPUT/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
