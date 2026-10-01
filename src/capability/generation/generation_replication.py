"""Independent sampling streams at the frozen operating points.

Sampling reuses the historical drivers. A passive generate wrapper retains token
IDs without drawing random numbers or changing generation arguments.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from . import conditioned_generation as cg
from . import generation_evidence as ge
from . import unconditional_generation as ug


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    ge.write_immutable(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def token_trace(ids: list[int], eos: Any, maximum: int) -> dict:
    eos_ids = [] if eos is None else ([int(eos)] if isinstance(eos, int) else [int(i) for i in eos])
    first = next((i for i, value in enumerate(ids) if value in eos_ids), None)
    actual = ids if first is None else ids[:first + 1]
    if first is not None:
        stop = 'eos'
    elif len(ids) == maximum:
        stop = 'max_new_tokens'
    else:
        raise ValueError(f'neither EOS nor declared token limit: {len(ids)} != {maximum}')
    return dict(generated_token_ids=actual, padded_output_token_ids=ids,
                effective_eos_token_ids=eos_ids, effective_max_new_tokens=maximum,
                generated_tokens=len(actual), decoder_stop=stop)


@contextmanager
def capture_generate(model: Any, traces: list[dict]):
    """Observe the exact model.generate output; never reconstruct tokens from text."""
    model = cg.ensure_generate(model)
    original = model.generate

    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        sequences = output.sequences if hasattr(output, 'sequences') else output
        inputs = kwargs.get('input_ids', args[0] if args else None)
        if inputs is None:
            raise ValueError('generation trace needs explicit input_ids')
        config = kwargs.get('generation_config', model.generation_config)
        eos = kwargs.get('eos_token_id', config.eos_token_id)
        maximum = int(kwargs.get('max_new_tokens', config.max_new_tokens))
        for row in sequences:
            traces.append(token_trace(row[int(inputs.shape[1]):].tolist(), eos, maximum))
        return output

    model.generate = observed
    try:
        yield
    finally:
        model.generate = original


def resolve_checkpoint(arm: str) -> Path:
    if arm == 'progen3-3b':
        import os
        from ..core.arms import MODEL_ROOT
        return Path(os.environ.get('MODEL_ROOT', MODEL_ROOT)) / 'progen3-3b'
    if arm == 'zymctrl':
        from ..core.arms import arm_spec
        return Path(arm_spec(arm).path)
    return ug.checkpoint_for_arm(arm)


def checkpoint_receipt(path: Path) -> dict:
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.suffix in ('.json', '.txt', '.model', '.safetensors', '.bin', '.py')
        or p.name in ('vocab.json', 'merges.txt')))
    if not files or not any(p.suffix in ('.bin', '.safetensors') for p in files):
        raise ValueError(f'incomplete checkpoint: {path}')
    return {'path': str(path), 'files_sha256': {p.name: sha256(p) for p in files}}


def extract_sequence(cell: dict, raw: str, compiled: str | None = None) -> tuple[str, str]:
    """Use the original arm-specific extractor, including native compiler checks."""
    arm = cell['arm']
    if arm.startswith('progen3-'):
        from .progen3_generation import generation_row

        return generation_row(raw, compiled, 0)['sequence'], '<eos>'
    if cell['condition'] == 'requested':
        close = cg.ARMS[arm].end_delimiter
        return cg.extract_protein(raw, end_delimiter=close), close
    return ug.extract_residues(arm, raw), ug.ADMITTED[arm].close_token


def load_cell(cell: dict, device: str):
    arm = cell['arm']
    if arm.startswith('progen3-'):
        import torch
        from ..models.progen3 import load_progen3
        from .progen3_generation import generation_self_check, make_generator
        pg = load_progen3(resolve_checkpoint(arm), device=device, dtype=torch.bfloat16)
        check = generation_self_check(pg)
        return pg.model, pg.tokenizer, make_generator(pg), check
    if arm == 'zymctrl':
        from ..core.arms import load_arm
        handle = load_arm(arm, device=device, dtype='bfloat16')
        return handle.model, handle.tokenizer, None, None
    model, tokenizer = ug.load_generator(arm, device=device)
    return model, tokenizer, None, None


def generate_cell(manifest_path: Path, cell_name: str, campaign_id: str, out: Path,
                  device: str) -> dict:
    import torch
    manifest = json.loads(manifest_path.read_text())
    cell = next(c for c in manifest['cells'] if c['cell'] == cell_name)
    campaign = next(c for c in manifest['campaigns'] if c['id'] == campaign_id)
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    snapshot = Path(__file__).resolve().parents[3]
    code_manifest = snapshot / 'CODE_CONTENT_SHA256SUMS'
    if not code_manifest.is_file():
        raise ValueError('scientific generation requires the official frozen snapshot')
    receipt = {'manifest_sha256': sha256(manifest_path), 'cell': cell,
               'campaign': campaign, 'checkpoint': checkpoint_receipt(resolve_checkpoint(cell['arm'])),
               'snapshot': snapshot.name, 'code_sha256': sha256(code_manifest),
               'torch': torch.__version__, 'device_name': torch.cuda.get_device_name(device)}
    if cell['arm'].startswith('progen3-'):
        from ..models.progen3 import PROGEN3_SOURCE
        receipt['upstream_source_sha256'] = {
            str(p.relative_to(PROGEN3_SOURCE)): sha256(p)
            for p in sorted(PROGEN3_SOURCE.rglob('*.py'))}
        receipt['upstream_tokenizer_sha256'] = {
            str(p.relative_to(PROGEN3_SOURCE)): sha256(p)
            for p in sorted(PROGEN3_SOURCE.rglob('*.json'))}
    digest = ge.digest(receipt)
    write_json(out / 'run_manifest.json', {**receipt, 'digest': digest})
    model, tokenizer, official, check = load_cell(cell, device)
    if model.training:
        raise RuntimeError('generation requires evaluation mode')
    if check is not None:
        check_path = out / 'interface_check.json'
        if not check_path.exists():
            write_json(check_path, check)
    arm = cell['arm']
    groups = cell.get('classes', [{'class_key': None, 'label': None}])
    rows = []
    for group in groups:
        key = group['class_key']
        total = cell.get('attempts_per_class', cell['attempts'])
        seed = (cg.cell_seed(seed=campaign['seed'], arm_name=arm, class_key=key,
                             condition='requested') if key is not None else campaign['seed'])
        prompt = (cg.prompt_for(None, cg.ARMS[arm], group['label']) if key is not None else cell['prompt'])
        batch_size = cell['batch_size']
        # The historical forward driver reseeds every attempt, not every batch.
        if arm in ('rita-xl', 'proteinglm-7b-clm'):
            batch_size = 1
        for begin in range(0, total, batch_size):
            end = min(begin + batch_size, total)
            batch_path = out / 'batches' / f'{key or "unconditioned"}_{begin:05d}.json'
            if batch_path.exists():
                saved = json.loads(batch_path.read_text())
                if saved['run_digest'] != digest or saved['rows_digest'] != ge.digest(saved['rows']):
                    raise ValueError(f'resume digest mismatch: {batch_path}')
                batch_rows = saved['rows']
                if [r['source_sample_index'] for r in batch_rows] != list(range(begin, end)):
                    raise ValueError(f'resume indices mismatch: {batch_path}')
            else:
                traces = []
                compiled = None
                if official is not None:
                    torch.manual_seed(seed + begin // batch_size)
                    with capture_generate(model, traces):
                        sampled = list(official.generate(prompt, end - begin, 0, 400))
                    raw = [r.generation for r in sampled]
                    compiled = [r.sequence for r in sampled]
                elif arm in ('rita-xl', 'proteinglm-7b-clm'):
                    raw = ug._sample_by_forward(ug.ADMITTED[arm], model, tokenizer,
                                                n=end - begin, seed=seed + begin,
                                                token_traces=traces)
                else:
                    with capture_generate(model, traces):
                        raw = cg.sample_continuations(model, tokenizer, prompt, n=end - begin,
                            seed=seed + begin // batch_size, batch_size=batch_size,
                            max_new_tokens=cell['max_new_tokens'], temperature=cell['temperature'],
                            top_p=cell['top_p'], top_k=cell['top_k'], use_cache=cell['use_cache'],
                            add_special_tokens=cell['add_special_tokens'])
                if len(raw) != end - begin or len(traces) != len(raw):
                    raise RuntimeError('missing generation or exact token trace')
                batch_rows = []
                for offset, (text, trace) in enumerate(zip(raw, traces)):
                    index = begin + offset
                    sequence, close = extract_sequence(cell, text,
                        compiled[offset] if compiled is not None else None)
                    row = ge._row(sequence, arm=arm, class_key=key, condition=cell['condition'],
                        role='generation', source_label=cell_name, source_sample_index=index,
                        primary_class=True, source_key=f'{campaign_id}|{cell_name}|{key}')
                    row.update(trace)
                    row.update(campaign=campaign_id, campaign_seed=campaign['seed'], raw_continuation=text,
                               native_delimiter_observed=close in text,
                               comparison_selected=key is None or index in group['comparison_indices'],
                               exact_duplicate_group=f'{campaign_id}|{cell_name}|sha256:{row["sequence_sha256"]}')
                    if compiled is not None:
                        row['official_compilation_valid'] = compiled[offset] is not None
                    batch_rows.append(row)
                write_json(batch_path, {'run_digest': digest, 'rows': batch_rows,
                                       'rows_digest': ge.digest(batch_rows)})
            rows.extend(batch_rows)
            print(f'{campaign_id} {cell_name} {key} {end}/{total}', flush=True)
    if len(rows) != cell['attempts'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('attempt census/identity mismatch')
    ge.write_immutable(out / 'attempts.jsonl', ge.jsonl_bytes(rows))
    summary = {'run_digest': digest, 'manifest_sha256': sha256(manifest_path),
               'cell': cell_name, 'campaign': campaign_id, 'attempts': len(rows),
               'comparison_attempts': sum(r['comparison_selected'] for r in rows),
               'attempts_sha256': sha256(out / 'attempts.jsonl'),
               'elapsed_seconds': time.monotonic() - start_time,
               'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device)}
    write_json(out / 'generation_replication.json', summary)
    return summary
