"""Download official SD 1.5 weights with resumable HTTP ranges and SHA-256 checks.

Useful when a proxy stalls on an unbounded multi-gigabyte HTTP response.
Weights are stored under canonical Diffusers filenames in a local model folder.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import threading

import requests
from huggingface_hub import hf_hub_download, model_info


REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'


def download(destination, workers=8):
    destination.mkdir(parents=True, exist_ok=True)
    info = model_info(REPO, files_metadata=True)
    components = {'unet', 'vae', 'text_encoder', 'safety_checker'}
    weights = [s for s in info.siblings if s.rfilename.endswith('.fp16.safetensors')
               and s.rfilename.split('/')[0] in components]
    configs = ['model_index.json', 'scheduler/scheduler_config.json',
               'feature_extractor/preprocessor_config.json', 'tokenizer/vocab.json',
               'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json',
               'tokenizer/special_tokens_map.json', *[f'{c}/config.json' for c in sorted(components)]]
    for filename in configs:
        source = hf_hub_download(REPO, filename, revision=info.sha)
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    provenance = {'repo': REPO, 'revision': info.sha, 'files': {}}
    for entry in weights:
        expected_hash = entry.lfs.sha256
        filename = entry.rfilename.replace('.fp16.safetensors', '.safetensors')
        target = destination / filename
        provenance['files'][filename] = {'source': entry.rfilename, 'sha256': expected_hash}
        if target.exists() and hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() == expected_hash:
            print(f'Already verified: {filename}', flush=True)
            continue
        old_part_dir = destination / '.parts' / entry.rfilename.replace('/', '_')
        part_dir = old_part_dir.with_name(old_part_dir.name + '.1m')
        part_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = 1024 * 1024
        # Reuse complete chunks made by the earlier 4 MiB downloader.
        if old_part_dir.is_dir():
            for old_part in old_part_dir.iterdir():
                if not old_part.name.isdigit():
                    continue
                start = int(old_part.name)
                data = old_part.read_bytes()
                if len(data) != min(4 * chunk_size, entry.size - start):
                    continue
                for offset in range(0, len(data), chunk_size):
                    (part_dir / str(start + offset)).write_bytes(data[offset:offset + chunk_size])
        offsets = list(range(0, entry.size, chunk_size))
        sessions = threading.local()

        def fetch(start):
            end = min(start + chunk_size, entry.size) - 1
            part = part_dir / str(start)
            if part.exists() and part.stat().st_size == end - start + 1:
                return
            url = f'https://huggingface.co/{REPO}/resolve/{info.sha}/{entry.rfilename}?download=true'
            if not hasattr(sessions, 'http'):
                sessions.http = requests.Session()
            for attempt in range(10):
                try:
                    response = sessions.http.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(60, 90))
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{entry.size}':
                        raise ValueError(f'Invalid range response: {response.status_code} {response.headers.get("Content-Range")}')
                    if len(response.content) != end - start + 1:
                        raise ValueError('Truncated range')
                    part.write_bytes(response.content)
                    return
                except (requests.RequestException, ValueError) as error:
                    print(f'  Retry offset={start}: {type(error).__name__}', flush=True)
                    if attempt == 9:
                        raise
                    time.sleep(min(2 ** attempt, 30))

        print(f'Downloading {entry.rfilename}: {entry.size / 1e6:.1f} MB', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, start) for start in offsets]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 10 == 0 or index == len(offsets):
                    print(f'  {index}/{len(offsets)} parts', flush=True)
        temporary = target.with_suffix('.assembling')
        with temporary.open('wb') as handle:
            for start in offsets:
                with (part_dir / str(start)).open('rb') as part:
                    shutil.copyfileobj(part, handle)
        with temporary.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected_hash:
            raise ValueError(f'SHA-256 mismatch: {filename}')
        temporary.replace(target)
        # Remove only the individually known downloaded parts, after verification.
        for start in offsets:
            (part_dir / str(start)).unlink()
        part_dir.rmdir()
        print(f'Verified: {filename}', flush=True)
    (destination / 'download_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('LW3/.cache/sd15'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    download(args.output, args.workers)
