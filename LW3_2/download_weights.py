"""Скачивает веса SUPIR и SDXL и создаёт конфигурацию локальных путей."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    """Сохраняет пути, ревизии и SHA-256 весов для воспроизводимости."""
    from huggingface_hub import HfApi, hf_hub_download
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/kaggle/working/supir_lab'))
    parser.add_argument('--variant', choices=['F', 'Q'], default='F')
    args = parser.parse_args()
    root = args.root.resolve()
    files = [('sdxl', 'stabilityai/stable-diffusion-xl-base-1.0', 'sd_xl_base_1.0_0.9vae.safetensors'),
             ('supir', 'ashleykleynhans/SUPIR', f'SUPIR-v0{args.variant}.ckpt')]
    record = {}
    for key, repo, filename in files:
        revision = HfApi().model_info(repo).sha
        path = Path(hf_hub_download(repo, filename, revision=revision, cache_dir=root / 'weights'))
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        record[key] = dict(repo=repo, revision=revision, filename=filename,
                           path=str(path), sha256=digest.hexdigest())
    config = OmegaConf.load(root / 'SUPIR/options/SUPIR_v0.yaml')
    config.SDXL_CKPT = record['sdxl']['path']
    config.SUPIR_CKPT = record['supir']['path']
    config.SUPIR_CKPT_F = config.SUPIR_CKPT_Q = None
    OmegaConf.save(config, root / 'supir.yaml')
    (root / 'weights.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(root / 'supir.yaml')


if __name__ == '__main__':
    main()
