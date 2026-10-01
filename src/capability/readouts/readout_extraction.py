"""Frozen block-output summaries using the established native forward."""
from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import re
import numpy as np
import torch
from src.capability.context import context_homologue as ch

TEXT_EXTRA_ARM = 'qwen2.5-0.5b-instruct'


def text_readout_names():
    from src.capability.models.text_aa_cohort import text_aa_model_names
    return (*text_aa_model_names(), TEXT_EXTRA_ARM)


def text_boundary(arm):
    return (getattr(arm, 'serving_provenance', None) or {}).get('text_aa_boundary')


def load_readout_arm(name, stage46, *, device=None, dtype='float32'):
    """Use validated native protein or literal-AA text loading, without widening panels."""
    if name == 'zymctrl':
        from src.capability.core.arms import Arm, arm_spec, load_arm
        if dtype != 'float32':
            raise ValueError('Conditioned ZymCTRL readout requires float32')
        if device is not None:
            return load_arm(name, device=device, dtype=dtype)
        from transformers import AutoTokenizer
        spec = arm_spec(name)
        tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True)
        return Arm(spec=spec, model=None, tokenizer=tokenizer, device='cpu', dtype='none')
    if name not in text_readout_names():
        if device is None:
            return ch.tokenizer_arm(name)
        return stage46.load_scorable_arm(name, device=device, dtype=dtype)
    if dtype != 'float32':
        raise ValueError('Literal-AA text readout requires float32')
    from src.capability.core.arms import Arm, arm_spec, TEXT_MODEL_BASE
    from src.capability.models.text_aa_cohort import load_text_aa_boundary_table
    from src.capability.models.text_aa_fitness import resolve_text_aa_boundary, load_text_aa_tokenizer, load_text_aa_scorer
    source_name = 'qwen2.5-0.5b' if name == TEXT_EXTRA_ARM else name
    boundary = resolve_text_aa_boundary(source_name, load_text_aa_boundary_table())
    spec = arm_spec(source_name)
    if name == TEXT_EXTRA_ARM:
        spec = replace(spec, name=name, path=TEXT_MODEL_BASE/'Qwen2.5-0.5B-Instruct')
        # Same literal document separator as Base; actual Instruct EOS is im_end.
        boundary = replace(boundary, name=name, eos_token_id=151645)
    if device is None:
        loaded = load_text_aa_tokenizer(spec.path, boundary=boundary, name=name)
        if loaded.hard_context is not None and loaded.hard_context < 1024:
            raise ValueError('Text checkpoint context is below the frozen 1024-position budget')
        return Arm(spec=spec, model=None, tokenizer=loaded.tokenizer, device='cpu', dtype='none',
                   serving_provenance={'text_aa_boundary': boundary})
    scorer = load_text_aa_scorer(spec.path, boundary=boundary, application_window_tokens=1024, device=device, name=name)
    return Arm(spec=spec, model=scorer.model, tokenizer=scorer.tokenizer, device=device, dtype=dtype,
               serving_provenance={'text_aa_boundary': boundary, 'text_aa_scorer': scorer})


def text_ids(arm, sequence):
    from src.capability.models.text_aa_fitness import encode_text_aa
    return encode_text_aa(arm.tokenizer, sequence, text_boundary(arm))


EC_SELECTION_RULE = 'lexicographically first full four-field EC among exact-WT matches; no measured labels'


def validate_ec_conditioning(payload, cohort, cohort_sha256):
    """Bind genuine annotation records to exact assays and WT bytes, without defaults."""
    if payload.get('schema_version') != 'readout_ec_conditioning_v1' or payload.get('cohort_sha256') != cohort_sha256:
        raise ValueError('EC conditioning schema or cohort hash mismatch')
    if not isinstance(payload.get('selection_rule'), str) or not payload['selection_rule'].strip():
        raise ValueError('Undeclared EC selection rule')
    source = payload.get('annotation_source', {})
    for key in ('swissprot_xml_sha256', 'ec_fasta_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(source.get(key, ''))):
            raise ValueError('EC conditioning requires hashed XML and FASTA provenance')
    rows = {r['assay']: r for r in cohort['assays']}
    entries = payload.get('assays')
    if not isinstance(entries, dict) or not set(entries) <= set(rows):
        raise ValueError('EC conditioning contains unknown assay records')
    result = {}
    for assay, entry in entries.items():
        expected = sha256(rows[assay]['wildtype'].encode()).hexdigest()
        if entry.get('wildtype_sha256') != expected:
            raise ValueError(f'{assay}: EC annotation wild-type hash mismatch')
        candidates = entry.get('ec_numbers')
        if not isinstance(candidates, list) or not candidates or any(not isinstance(ec, str) or not re.fullmatch(r'[1-7]\.\d+\.\d+\.\d+', ec) for ec in candidates):
            raise ValueError(f'{assay}: EC annotation must contain full four-field EC numbers')
        if entry.get('ec') != min(candidates):
            raise ValueError(f'{assay}: EC annotation violates the label-independent selection rule')
        if not entry.get('accessions'):
            raise ValueError(f'{assay}: EC annotation lacks source accessions')
        result[assay] = entry['ec']
    return result


def bind_readout_ec(arm, ec):
    if arm.name != 'zymctrl' or not isinstance(ec, str) or not re.fullmatch(r'[1-7]\.\d+\.\d+\.\d+', ec):
        raise ValueError('Only a full assay-bound EC can condition ZymCTRL')
    if arm.serving_provenance is None:
        arm.serving_provenance = {}
    arm.serving_provenance['readout_ec'] = ec


def pack_conditioned_sequence(arm, sequence):
    from src.capability.core.arms import Cohort, conditioning_boundary_ids, AA20
    ec = (arm.serving_provenance or {}).get('readout_ec')
    if not isinstance(ec, str) or not re.fullmatch(r'[1-7]\.\d+\.\d+\.\d+', ec):
        raise ValueError('ZymCTRL requires an assay-bound EC before packing')
    if not sequence or not set(sequence) <= set(AA20):
        raise ValueError('ZymCTRL sequence must contain only AA20 residues')
    cohort = Cohort(name='readout_variant', kind='protein', records=[sequence], min_symbols=0, max_symbols=0, metadata={'ec_labels':[ec]})
    rendered = cohort.input_strings(arm)[0]
    ids = list(arm.tokenizer(rendered, return_tensors=None)['input_ids'])
    if ''.join(arm.tokenizer.convert_ids_to_tokens(ids)) != rendered:
        raise ValueError('ZymCTRL tokenizer cannot exactly represent the EC-conditioned native string')
    start, end = conditioning_boundary_ids(arm)
    if ids.count(start) != 1 or ids.count(end) != 1:
        raise ValueError('ZymCTRL rendering lacks unique native start/end markers')
    left, right = ids.index(start)+1, ids.index(end)
    if left >= right or arm.tokenizer.convert_ids_to_tokens(ids[left:right]) != list(sequence):
        raise ValueError('ZymCTRL residue span does not match the supplied sequence')
    return ids, (left, right), (left, right)


FEATURE_NAMES = ('middle_mean', 'middle_last', 'final_mean', 'final_last')


def representation_blocks(arm):
    if arm.name == 'proteinglm-7b-clm':
        return arm.model.transformer.encoder.layers
    if arm.spec.architecture in {'progen3', 'qwen3'} or arm.name == 'protgpt3-1.3b':
        return arm.model.model.layers
    if arm.name == 'rita-xl':
        return arm.model.transformer.layers
    return arm.blocks()


def pack_sequence(arm, sequence):
    """Full native target; representation includes all residue-bearing tokens."""
    if getattr(arm, 'name', None) == 'zymctrl':
        return pack_conditioned_sequence(arm, sequence)
    boundary = text_boundary(arm)
    if boundary is not None:
        ids = text_ids(arm, sequence)
        if len(ids) < 2:
            raise ValueError('Literal-AA scoring requires at least one predicted token')
        return ids, (1, len(ids)), (0 if boundary.conditioning_id is None else 1, len(ids))
    item = ch.item_ids(arm, sequence, modality='protein')
    prefix = ch.row_prefix_ids(arm)
    start, end = ch.target_span(arm, item, record=sequence)
    left, trailing = ch._item_affixes(arm, item, record=sequence)
    return prefix + item, (len(prefix)+start, len(prefix)+end), (len(prefix)+left, len(prefix)+len(item)-trailing)


def representation_positions(arm, sequences, packed):
    """ProtGPT2 states exclude pure FASTA newlines, retaining mixed BPE tokens."""
    if arm.name != 'protgpt2':
        return None
    positions = []
    for sequence, (ids, _, (left, right)) in zip(sequences, packed, strict=True):
        pieces = [arm.tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in ids[left:right]]
        expected = '\n'.join(sequence[i:i+60] for i in range(0, len(sequence), 60))
        if ''.join(pieces) != expected or any(not piece or not set(piece) <= set(ch.AA20+'\n') for piece in pieces):
            raise ValueError('ProtGPT2 target tokens do not exactly reconstruct the native FASTA content')
        selected = [left+i for i,piece in enumerate(pieces) if any(aa in ch.AA20 for aa in piece)]
        if not selected:
            raise ValueError('ProtGPT2 target has no residue-bearing tokens')
        positions.append(selected)
    return positions


def pool_hidden(hidden, spans, *, time_first=False, positions=None):
    if time_first:
        hidden = hidden.transpose(0, 1)
    if hidden.ndim != 3 or hidden.shape[0] != len(spans):
        raise ValueError('Block output must have batch, time, hidden dimensions')
    pooled = []
    for i, (left, right) in enumerate(spans):
        if not 0 <= left < right <= hidden.shape[1]:
            raise ValueError('Invalid residue-bearing token span')
        if positions is None:
            values = hidden[i, left:right].float()
        else:
            selected = positions[i]
            if not selected or selected != sorted(set(selected)) or not all(left <= p < right for p in selected):
                raise ValueError('Invalid residue-bearing token positions')
            values = hidden[i, selected].float()
        pooled.append(torch.stack((values.mean(0), values[-1])))
    result = torch.stack(pooled).detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite representation')
    return result


def forward_readout_rows(arm, rows, stage46):
    if text_boundary(arm) is None and arm.name != 'zymctrl':
        return stage46._forward_rows(arm, rows)
    from src.capability.core.precision_policy import fp32_matmul_context
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        pad = arm.tokenizer.eos_token_id
    if pad is None:
        raise ValueError('Text readout requires a declared padding or EOS ID')
    ids = torch.full((len(rows), max(map(len, rows))), int(pad), dtype=torch.long, device=arm.device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=arm.device)
        mask[i, :len(row)] = 1
    with fp32_matmul_context(torch), torch.no_grad():
        logits = arm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    if logits.dtype != torch.float32:
        raise ValueError('Text readout logits must remain float32')
    return logits, ids


def hooked_block_indices(block_count: int, extra=None) -> tuple[int, ...]:
    """Which transformer blocks one forward pass hooks, in the order it hooks them.

    With no ``extra`` this is exactly the pair every admitted extraction used --
    the block after half the stack and the last block -- so the default is
    unchanged for the 33 arms of completed extraction and for every published cell
    that rests on them. That property is asserted by test against both real
    extractors rather than left to inspection.

    ``extra`` adds further zero-based indices, deduplicated against the admitted
    pair and against each other and sorted ascending, and appended **after** it.
    Appending rather than interleaving is deliberate: the admitted pair stays at
    feature positions 0 to 3, so an archive with extra depths still presents the
    admitted four blocks where a reader of the admitted layout expects them, and a
    class-level recomputation reads it unchanged.

    An index outside the stack is refused rather than clamped. Clamping would
    silently hook a block the caller did not ask for and record it as the one it
    did, which is the failure a depth-resolved archive cannot tolerate.
    """

    if block_count < 1:
        raise ValueError('a transformer stack has at least one block')
    admitted = [(block_count - 1) // 2, block_count - 1]
    requested = sorted({int(index) for index in (extra or ())})
    for index in requested:
        if not 0 <= index < block_count:
            raise ValueError(f'block index {index} is outside 0..{block_count - 1}; a requested '
                             'depth outside the stack is refused rather than clamped')
    return tuple(admitted) + tuple(index for index in requested if index not in admitted)


def feature_block_names(block_indices) -> list[str]:
    """The retained feature axis, one name per hooked block and pooling rule.

    Recorded in the receipt so a later reader can tell which block and which
    pooling rule each slice of an archive holds, rather than inferring it from a
    count. The first four names of a default extraction are the admitted
    ``middle_mean``, ``middle_last``, ``final_mean``, ``final_last``.
    """

    names = []
    for position, index in enumerate(block_indices):
        stem = ('middle', 'final')[position] if position < 2 else f'block{index:03d}'
        names.extend(f'{stem}_{rule}' for rule in ('mean', 'last'))
    return names


def extract_batch(arm, sequences, stage46, block_indices=None, *, position_terms=None):
    """Block summaries and native likelihood for one batch of packed sequences.

    ``position_terms``, when a list, additionally receives one record per
    sequence carrying the per-scored-token vector whose sum is that sequence's
    returned likelihood, together with the packed ids and scored span needed to
    attribute it to residues. The default is unchanged and no arithmetic on the
    returned values differs, so an archive extracted with the flag absent is the
    archive it always was.
    """
    packed = [pack_sequence(arm, s) for s in sequences]
    blocks = representation_blocks(arm)
    indices = (hooked_block_indices(len(blocks)) if block_indices is None
               else tuple(int(index) for index in block_indices))
    if len(set(indices)) != len(indices):
        raise ValueError(f'hooked block indices repeat: {indices}')
    for index in indices:
        if not 0 <= index < len(blocks):
            raise ValueError(f'block index {index} is outside 0..{len(blocks)-1}')
    labels = [f'h{position}' for position in range(len(indices))]
    captured, handles = {}, []
    spans = [p[2] for p in packed]
    positions = representation_positions(arm, sequences, packed)
    def capture(label):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[label] = pool_hidden(hidden, spans, time_first=arm.name == 'proteinglm-7b-clm', positions=positions)
        return hook
    for label, index in zip(labels, indices):
        handles.append(blocks[index].register_forward_hook(capture(label)))
    try:
        logits, ids = forward_readout_rows(arm, [p[0] for p in packed], stage46)
        likelihood = np.array([-stage46._target_nll(logits[i:i+1], ids[i:i+1], *p[1])['nll_sum'] for i,p in enumerate(packed)])
        if position_terms is not None:
            from src.capability.position.position_terms import target_nll_terms
            for i, p in enumerate(packed):
                terms = target_nll_terms(logits[i:i+1], ids[i:i+1], *p[1])
                position_terms.append({'terms': terms.detach().cpu().numpy().astype(np.float32, copy=False),
                                       'ids': [int(v) for v in p[0]],
                                       'span': (int(p[1][0]), int(p[1][1]))})
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(labels) or not np.isfinite(likelihood).all():
        raise ValueError('Incomplete or nonfinite forward')
    return np.concatenate([captured[label] for label in labels], axis=1), likelihood


def mutation_relative_drift(batched, singleton):
    """Maximum per-mutant relative L2 across the concatenated four blocks."""
    delta = batched[1:] - batched[0]
    reference = singleton[1:] - singleton[0]
    errors = []
    for observed, expected in zip(delta, reference, strict=True):
        denominator = float(np.linalg.norm(expected))
        numerator = float(np.linalg.norm(observed-expected))
        if denominator == 0:
            if numerator != 0:
                raise ValueError('Nonzero drift for zero reference mutation vector')
            errors.append(0.)
        else:
            errors.append(numerator/denominator)
    return max(errors)
