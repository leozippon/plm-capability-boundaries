#!/usr/bin/env python3
"""Check all declared generation inputs without loading model weights."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.capability.generation.generation_replication import resolve_checkpoint, sha256, write_json
from src.capability.models.progen3 import PROGEN3_SOURCE


def active_weight_files(path):
    """Use the installed loader's selection, not every alternate-format index."""
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    files, _ = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=str(path), subfolder='', variant=None,
        gguf_file=None, from_tf=False, from_flax=False, use_safetensors=None,
        cache_dir=None, force_download=False, proxies=None, local_files_only=True,
        token=None, user_agent={}, revision='main', commit_hash=None, is_remote_code=False)
    return [Path(p) for p in files or []]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=(Path(__file__).resolve().parents[3] / 'configs/generation_replication_manifest.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    records = []
    failures = []
    for arm in sorted({c['arm'] for c in manifest['cells']}):
        path = resolve_checkpoint(arm)
        weights = active_weight_files(path)
        configs = list(path.glob('*.json'))
        tokens = [p for p in path.iterdir() if p.is_file() and any(w in p.name for w in ['token', 'vocab', 'merges'])] if path.exists() else []
        okay = bool(weights) and (path / 'config.json').is_file() and (bool(tokens) or arm.startswith('progen3-'))
        okay = okay and all(p.is_file() for p in weights)
        if not okay:
            failures.append(arm)
        records.append({'arm': arm, 'path': str(path), 'weights_bytes': sum(p.stat().st_size for p in weights),
                        'weights_files': [p.name for p in weights], 'tokenizer_files': [p.name for p in tokens],
                        'config_sha256': {p.name: sha256(p) for p in configs}, 'passed': okay})
    for name in ['progen3/generator.py', 'progen3/batch_preparer.py', 'progen3/modeling.py']:
        if not (PROGEN3_SOURCE / name).is_file():
            failures.append(name)
    import torch
    import transformers
    result = {'manifest_sha256': sha256(a.manifest), 'checkpoints': records,
              'failures': failures, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'cuda_devices': torch.cuda.device_count(), 'passed': not failures}
    write_json(a.out / 'generation_preflight.json', result)
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
