"""Resumable range downloads for the official LPIPS and CLIP-IQA backbones."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import threading

import requests
from torch.hub import get_dir
from pyiqa.archs.clip_model import _MODELS


def download(url, target, expected_sha256, workers=64):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest().startswith(expected_sha256):
                print(f'Already verified: {target.name}', flush=True)
                return
    response = requests.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=(20, 60))
    response.raise_for_status()
    if response.status_code != 206:
        response.close()
        raise ValueError('Server must support HTTP ranges')
    size = int(response.headers['Content-Range'].split('/')[-1])
    response.close()
    parts = target.parent / (target.name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk_size = 1024 * 1024
    offsets = list(range(0, size, chunk_size))
    sessions = threading.local()

    def fetch(start):
        end = min(size, start + chunk_size) - 1
        path = parts / str(start)
        if path.exists() and path.stat().st_size == end - start + 1:
            return
        if not hasattr(sessions, 'http'):
            sessions.http = requests.Session()
        for attempt in range(10):
            try:
                response = sessions.http.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=(60, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Invalid Content-Range')
                if len(response.content) != end - start + 1:
                    raise ValueError('Truncated part')
                path.write_bytes(response.content)
                return
            except (requests.RequestException, ValueError):
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 30))

    print(f'Downloading {target.name}: {size / 1e6:.1f} MB', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(fetch, start) for start in offsets]), 1):
            future.result()
            if index % 20 == 0 or index == len(offsets):
                print(f'  {index}/{len(offsets)} parts', flush=True)
    temporary = target.with_suffix('.assembling')
    with temporary.open('wb') as handle:
        for start in offsets:
            handle.write((parts / str(start)).read_bytes())
    with temporary.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if not actual.startswith(expected_sha256):
        raise ValueError(f'SHA-256 mismatch: {target.name}')
    temporary.replace(target)
    for start in offsets:
        (parts / str(start)).unlink()
    parts.rmdir()
    print(f'Verified: {target.name}', flush=True)


if __name__ == '__main__':
    root = Path(get_dir())
    download('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
             root / 'checkpoints/alexnet-owt-7be5be79.pth', '7be5be79')
    url = _MODELS['RN50']
    download(url, root / 'clip/RN50.pt', url.split('/')[-2])
