"""Plug-and-play HuggingFace, dynamic plugin, and first-party benchmark adapters."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.capability.benchmark.protocol import (
    AA20_INDEX,
    AA20_SET,
    BaseProteinBenchmarkAdapter,
    CapabilityFlags,
    ModelMetadata,
    ProteinBenchmarkAdapter,
    pack_mid_and_last_hidden_states,
)
from src.capability.core.amino_acids import AA20


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


def _resolve_torch_dtype(dtype: str):
    import torch

    d = dtype.strip().lower()
    if d in ("fp32", "float32", "auto", "default"):
        return torch.float32
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported dtype {dtype!r}; use float32, bfloat16, or float16")


def _parse_mutation(mut_str: str) -> list[tuple[str, int, str]]:
    items: list[tuple[str, int, str]] = []
    for tok in (t.strip().upper() for t in str(mut_str).split(":") if t.strip()):
        if len(tok) < 3:
            raise ValueError(f"Invalid mutation token {tok!r}")
        wt, mut = tok[0], tok[-1]
        pos = int(tok[1:-1]) - 1
        if wt not in AA20_SET or mut not in AA20_SET or pos < 0:
            raise ValueError(f"Invalid AA20 mutation token {tok!r}")
        items.append((wt, pos, mut))
    return items


def _apply_mutations(wt: str, items: Sequence[tuple[str, int, str]]) -> str:
    chars = list(wt)
    for w_aa, pos, m_aa in items:
        if pos >= len(chars) or chars[pos] != w_aa:
            raise ValueError(f"WT mismatch at 1-based pos {pos + 1}: expected {w_aa}, found {chars[pos] if pos < len(chars) else 'OOB'}")
        chars[pos] = m_aa
    return "".join(chars)


# --------------------------------------------------------------------------- #
# 1. Generic HuggingFace Causal PLM Adapter
# --------------------------------------------------------------------------- #


class HuggingFaceCausalPLMAdapter(BaseProteinBenchmarkAdapter):
    """Plug-and-play adapter wrapping any HuggingFace ``AutoModelForCausalLM``."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        model_id: str | None = None,
        display_name: str | None = None,
        paradigm: str = "causal_protein",
        modality: str = "protein",
        device: str = "auto",
        dtype: str = "float32",
        max_positions: int = 1024,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.model_path = str(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device).eval().requires_grad_(False)

        n_params_m = float(sum(p.numel() for p in self.model.parameters())) / 1e6
        eff_id = model_id or Path(self.model_path).name
        meta = ModelMetadata(
            model_id=eff_id,
            display_name=display_name or eff_id,
            paradigm=paradigm,
            modality=modality,
            conditioning_regime="sequence_only",
            input_alignment_format="unaligned_concat",
            n_params_millions=max(n_params_m, 1e-3),
            max_positions=int(max_positions),
        )
        caps = CapabilityFlags(
            score_mutations=True,
            site_conditional_logprobs=True,
            score_sequence_nll=True,
            extract_representations_1024=True,
            generate_unconditional=True,
            continue_from_prefix=True,
        )
        super().__init__(meta, caps)

        # Single-token AA20 IDs for site-conditional AA20 distribution
        aa_ids: list[int] = []
        for aa in AA20:
            enc = self.tokenizer.encode(aa, add_special_tokens=False)
            aa_ids.append(int(enc[-1]) if enc else 0)
        self._aa20_ids_t = torch.as_tensor(aa_ids, dtype=torch.long, device=self.device)

    def _bos_prefix_ids(self) -> list[int]:
        if self.tokenizer.bos_token_id is not None:
            return [int(self.tokenizer.bos_token_id)]
        if self.tokenizer.eos_token_id is not None:
            return [int(self.tokenizer.eos_token_id)]
        return []

    def _nll_for_seq(self, full_seq: str, target_len: int) -> tuple[float, int]:
        import torch
        import torch.nn.functional as F

        toks = self.tokenizer.encode(full_seq, add_special_tokens=False)
        prefix_bos = self._bos_prefix_ids()
        ids = (prefix_bos + toks)[-self.metadata.max_positions :]
        if len(ids) < 2:
            return 0.0, max(target_len, 1)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=x).logits[0, :-1, :].float()
            targets = x[0, 1:]
            losses = F.cross_entropy(logits, targets, reduction="none")
        # Approximate target suffix token slice when context precedes target
        tgt_toks = self.tokenizer.encode(full_seq[-target_len:], add_special_tokens=False)
        n_tgt = min(len(tgt_toks), len(losses))
        nll = float(losses[-n_tgt:].sum().item()) if n_tgt > 0 else float(losses.sum().item())
        return max(0.0, nll), max(target_len, 1)

    def score_sequence_nll(
        self,
        sequence: str,
        *,
        context_items: Sequence[str] = (),
        coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
    ) -> dict[str, float]:
        full_seq = "".join(context_items) + sequence
        nll_sum, n_res = self._nll_for_seq(full_seq, len(sequence))
        return {"nll_sum": nll_sum, "nll_per_residue": nll_sum / n_res}

    def score_mutations(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        wt_nll = self.score_sequence_nll(wildtype)["nll_sum"]
        scores = np.empty(len(mutations), dtype=np.float64)
        for i, m in enumerate(mutations):
            mut_seq = _apply_mutations(wildtype, _parse_mutation(m))
            mut_nll = self.score_sequence_nll(mut_seq)["nll_sum"]
            scores[i] = wt_nll - mut_nll  # higher = more favorable
        return scores

    def site_conditional_logprobs(
        self,
        sequence: str,
        positions: Sequence[int] | None = None,
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact causal prefix-only next-token log-probabilities ``p(x_i | x_{<i})`` (strictly zero leakage)."""
        import torch
        import torch.nn.functional as F

        pos_list = list(range(len(sequence))) if positions is None else [int(p) for p in positions]
        out = np.empty((len(pos_list), 20), dtype=np.float32)
        bos = self._bos_prefix_ids()
        with torch.no_grad():
            for idx, p in enumerate(pos_list):
                pref_ids = bos + self.tokenizer.encode(sequence[:p], add_special_tokens=False)
                if not pref_ids:
                    out[idx] = np.full(20, -np.log(20.0), dtype=np.float32)
                    continue
                pref_ids = pref_ids[-self.metadata.max_positions :]
                x = torch.tensor([pref_ids], dtype=torch.long, device=self.device)
                last_logits = self.model(input_ids=x).logits[0, -1, :].float()
                lp20 = F.log_softmax(last_logits.index_select(-1, self._aa20_ids_t), dim=-1)
                out[idx] = lp20.cpu().numpy()
        return out

    def extract_representations_1024(
        self,
        sequences: Sequence[str],
        *,
        coords_list: Sequence[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        import torch

        if not sequences:
            return np.zeros((0, 1024), dtype=np.float32)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for seq in sequences:
                ids = (self._bos_prefix_ids() + self.tokenizer.encode(seq, add_special_tokens=False))[
                    : self.metadata.max_positions
                ]
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                hs = self.model(input_ids=x, output_hidden_states=True).hidden_states
                n_layers = len(hs) - 1
                mid_idx = max(1, n_layers // 2)
                h_mid = hs[mid_idx][0].float().cpu().numpy()
                h_last = hs[-1][0].float().cpu().numpy()
                rows.append(pack_mid_and_last_hidden_states(h_mid, h_last))
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    def _sample_ar(self, prefix: str, target_length: int, *, seed: int) -> str:
        import torch
        import torch.nn.functional as F

        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        seq = prefix
        bos = self._bos_prefix_ids()
        with torch.no_grad():
            while len(seq) < target_length:
                ids = (bos + self.tokenizer.encode(seq, add_special_tokens=False))[-self.metadata.max_positions :]
                if not ids:
                    ids = [int(self._aa20_ids_t[0].item())]
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=x).logits[0, -1, :].float()
                probs = F.softmax(logits.index_select(-1, self._aa20_ids_t), dim=-1)
                choice = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
                seq += AA20[choice]
        return seq[:target_length]

    def generate_unconditional(self, lengths: Sequence[int], *, seed: int = 20261124) -> list[str]:
        return [self._sample_ar("", int(L), seed=seed + i) for i, L in enumerate(lengths)]

    def continue_from_prefix(
        self,
        context_items: Sequence[str],
        target_length: int,
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        prefix = "".join(context_items)
        full = self._sample_ar(prefix, len(prefix) + int(target_length), seed=seed)
        return full[len(prefix) : len(prefix) + int(target_length)]


# --------------------------------------------------------------------------- #
# 2. Generic HuggingFace Masked PLM Adapter
# --------------------------------------------------------------------------- #


class HuggingFaceMaskedPLMAdapter(BaseProteinBenchmarkAdapter):
    """Plug-and-play adapter wrapping any HuggingFace ``AutoModelForMaskedLM`` (ESM-2, AMPLIFY, etc.)."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        model_id: str | None = None,
        display_name: str | None = None,
        paradigm: str = "masked_lm",
        modality: str = "protein",
        device: str = "auto",
        dtype: str = "float32",
        max_positions: int = 1022,
        batch_size: int = 16,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_torch_dtype(dtype)
        self.batch_size = max(1, int(batch_size))
        self.model_path = str(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            dtype=self.torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device).eval().requires_grad_(False)
        self.mask_id = int(self.tokenizer.mask_token_id)

        n_params_m = float(sum(p.numel() for p in self.model.parameters())) / 1e6
        eff_id = model_id or Path(self.model_path).name
        meta = ModelMetadata(
            model_id=eff_id,
            display_name=display_name or eff_id,
            paradigm=paradigm,
            modality=modality,
            conditioning_regime="sequence_only",
            input_alignment_format="unaligned_concat",
            n_params_millions=max(n_params_m, 1e-3),
            max_positions=int(max_positions),
        )
        caps = CapabilityFlags(
            score_mutations=True,
            site_conditional_logprobs=True,
            score_sequence_nll=True,
            extract_representations_1024=True,
            infill_local=True,
        )
        super().__init__(meta, caps)
        self._aa20_ids_t = torch.as_tensor(
            [int(self.tokenizer.convert_tokens_to_ids(aa)) for aa in AA20],
            dtype=torch.long,
            device=self.device,
        )

    def _tokenize(self, sequence: str):
        enc = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=self.metadata.max_positions + 2)
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def site_conditional_logprobs(
        self,
        sequence: str,
        positions: Sequence[int] | None = None,
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        pos_list = list(range(len(sequence))) if positions is None else [int(p) for p in positions]
        if not pos_list:
            return np.zeros((0, 20), dtype=np.float32)
        base_ids, base_mask = self._tokenize(sequence)
        max_res = base_ids.shape[1] - 2
        out = np.empty((len(pos_list), 20), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(pos_list), self.batch_size):
                chunk = pos_list[start : start + self.batch_size]
                cur = len(chunk)
                b_ids = base_ids.expand(cur, -1).clone()
                b_mask = base_mask.expand(cur, -1)
                for row_i, p in enumerate(chunk):
                    if p < max_res:
                        b_ids[row_i, 1 + p] = self.mask_id
                logits = self.model(input_ids=b_ids, attention_mask=b_mask).logits.float()
                for row_i, p in enumerate(chunk):
                    idx_tok = min(1 + p, logits.shape[1] - 1)
                    lp20 = F.log_softmax(logits[row_i, idx_tok, :].index_select(-1, self._aa20_ids_t), dim=-1)
                    out[start + row_i] = lp20.cpu().numpy()
        return out

    def score_mutations(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        parsed = [_parse_mutation(m) for m in mutations]
        sites = sorted({pos for items in parsed for _, pos, _ in items})
        lp_map = {
            pos: row
            for pos, row in zip(sites, self.site_conditional_logprobs(wildtype, positions=sites))
        }
        scores = np.empty(len(parsed), dtype=np.float64)
        for i, items in enumerate(parsed):
            scores[i] = sum(float(lp_map[pos][AA20_INDEX[m_aa]] - lp_map[pos][AA20_INDEX[w_aa]]) for w_aa, pos, m_aa in items)
        return scores

    def score_sequence_nll(
        self,
        sequence: str,
        *,
        context_items: Sequence[str] = (),
        coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
    ) -> dict[str, float]:
        """8-fold stride-masked PLL on ``sequence`` conditioned on optional ``context_items``."""
        import torch
        import torch.nn.functional as F

        max_ctx = max(0, self.metadata.max_positions - len(sequence))
        ctx = "".join(context_items)[:max_ctx]
        full_seq = ctx + sequence
        tgt_start = len(ctx)
        base_ids, base_mask = self._tokenize(full_seq)
        k = min(8, len(sequence))
        b_ids = base_ids.expand(k, -1).clone()
        b_mask = base_mask.expand(k, -1)
        fold_pos = [list(range(f, len(sequence), k)) for f in range(k)]
        for f, pf in enumerate(fold_pos):
            for p in pf:
                if 1 + tgt_start + p < b_ids.shape[1] - 1:
                    b_ids[f, 1 + tgt_start + p] = self.mask_id
        with torch.no_grad():
            logits = self.model(input_ids=b_ids, attention_mask=b_mask).logits.float()
            lp20 = F.log_softmax(logits.index_select(-1, self._aa20_ids_t), dim=-1).cpu().numpy()
        nll = 0.0
        for f, pf in enumerate(fold_pos):
            for p in pf:
                tok_i = min(1 + tgt_start + p, lp20.shape[1] - 1)
                nll += max(0.0, float(-lp20[f, tok_i, AA20_INDEX[sequence[p]]]))
        return {"nll_sum": nll, "nll_per_residue": nll / max(len(sequence), 1)}

    def extract_representations_1024(
        self,
        sequences: Sequence[str],
        *,
        coords_list: Sequence[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        import torch

        if not sequences:
            return np.zeros((0, 1024), dtype=np.float32)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for seq in sequences:
                ids, mask = self._tokenize(seq)
                hs = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
                n_layers = len(hs) - 1
                mid_idx = max(1, n_layers // 2)
                L = min(len(seq), ids.shape[1] - 2)
                h_mid = hs[mid_idx][0, 1 : 1 + L, :].float().cpu().numpy()
                h_last = hs[-1][0, 1 : 1 + L, :].float().cpu().numpy()
                rows.append(pack_mid_and_last_hidden_states(h_mid, h_last))
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    def infill_local(
        self,
        template: str,
        mask_positions: Sequence[int],
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        rng = np.random.default_rng(int(seed))
        chars = list(template)
        remaining = list(dict.fromkeys(int(p) for p in mask_positions))
        while remaining:
            lp = self.site_conditional_logprobs("".join(chars), positions=remaining)
            probs = np.exp(lp)
            best_i = int(np.argmax(probs.max(axis=-1)))
            p = remaining.pop(best_i)
            prob_vec = probs[best_i] / probs[best_i].sum()
            chars[p] = AA20[int(rng.choice(20, p=prob_vec))]
        return "".join(chars)


# --------------------------------------------------------------------------- #
# 3. Dynamic External Plugin Loader ("path/to/file.py:Cls" or "pkg.mod:Cls")
# --------------------------------------------------------------------------- #


def load_custom_adapter(spec: str, **kwargs: Any) -> ProteinBenchmarkAdapter:
    """Dynamically load a user adapter from ``'path/to/file.py:Symbol'`` or ``'pkg.mod:Symbol'``."""
    if ":" not in spec:
        raise ValueError(f"Adapter spec {spec!r} must have format 'path/to/file.py:ClassName' or 'pkg.mod:ClassName'")
    target_path, symbol_name = spec.rsplit(":", 1)
    target_path = target_path.strip()
    symbol_name = symbol_name.strip()

    p = Path(target_path)
    if p.suffix == ".py" or p.is_file():
        if not p.is_file():
            raise FileNotFoundError(f"Custom adapter file not found: {p}")
        mod_spec = importlib.util.spec_from_file_location(f"_custom_bench_{p.stem}", str(p))
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Unable to load module spec from {p}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target_path)

    if not hasattr(module, symbol_name):
        raise AttributeError(f"Symbol {symbol_name!r} not found in {target_path!r}")
    obj = getattr(module, symbol_name)
    adapter = obj(**kwargs) if callable(obj) else obj
    if not isinstance(adapter, ProteinBenchmarkAdapter):
        raise TypeError(f"Loaded object from {spec!r} does not implement ProteinBenchmarkAdapter")
    return adapter


# --------------------------------------------------------------------------- #
# 4. First-Party Builtin Bridge (DiMA, DPLM-2, ProteinMPNN, general_plm)
# --------------------------------------------------------------------------- #


class _BuiltinBridgeAdapter(BaseProteinBenchmarkAdapter):
    """Wraps our first-party DiMA, DPLM-2, and ProteinMPNN adapters into ``ProteinBenchmarkAdapter``."""

    def __init__(self, inner: Any, kind: str, metadata: ModelMetadata, capabilities: CapabilityFlags) -> None:
        super().__init__(metadata, capabilities)
        self.inner = inner
        self.kind = kind

    def score_mutations(
        self,
        wildtype: str,
        mutations: Sequence[str],
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.kind == "dima":
            return self.inner.score_mutations(wildtype, mutations)
        if self.kind == "dplm2":
            return self.inner.score_variants(wildtype, mutations, coords=coords)
        if self.kind == "proteinmpnn":
            return self.inner.score_variants(wildtype, mutations, coords=coords)
        return self.inner.score_mutations(wildtype, mutations)

    def site_conditional_logprobs(
        self,
        sequence: str,
        positions: Sequence[int] | None = None,
        *,
        coords: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.kind == "dima":
            return self.inner.site_conditional_logprobs(sequence, positions=positions)
        if self.kind == "dplm2":
            return self.inner.site_conditional_logprobs(sequence, positions=positions, coords=coords)
        if self.kind == "proteinmpnn":
            return self.inner.site_conditional_logprobs(sequence, positions=positions, coords=coords)
        return self.inner.site_conditional_logprobs(sequence, positions=positions)

    def score_sequence_nll(
        self,
        sequence: str,
        *,
        context_items: Sequence[str] = (),
        coords: np.ndarray | None = None,
        context_coords: Sequence[np.ndarray] = (),
    ) -> dict[str, float]:
        if context_items:
            if self.kind == "dima":
                r = self.inner.score_r4_context(sequence, context_items)
                return {"nll_sum": float(r["nll_sum"]), "nll_per_residue": float(r["nll_per_residue"])}
            if self.kind == "dplm2":
                r = self.inner.score_r4_context(
                    sequence, context_items, target_coords=coords, context_coords=context_coords
                )
                return {"nll_sum": float(r["nll_sum"]), "nll_per_residue": float(r["nll_per_residue"])}
            if self.kind == "proteinmpnn":
                r = self.inner.score_r4_context(
                    sequence, context_items, target_coords=coords, context_coords=context_coords
                )
                return {"nll_sum": float(r["nll_sum"]), "nll_per_residue": float(r["nll_per_residue"])}
        pv = self.inner.pll_8fold(sequence) if self.kind == "dima" else (
            self.inner.pll_8fold(sequence, coords=coords) if self.kind == "dplm2" else self.inner.exact_pll(sequence, coords=coords)
        )
        return {"nll_sum": float(pv.nll_sum), "nll_per_residue": float(pv.nll_sum) / max(len(sequence), 1)}

    def extract_representations_1024(
        self,
        sequences: Sequence[str],
        *,
        coords_list: Sequence[np.ndarray | None] | None = None,
    ) -> np.ndarray:
        if self.kind == "dima":
            return self.inner.extract_readout_1024(sequences)
        if self.kind == "dplm2":
            return self.inner.extract_hidden_states(sequences, coords_list=coords_list, return_projected_1024=True)
        if self.kind == "proteinmpnn":
            return self.inner.extract_hidden_states(sequences, coords_list=coords_list, return_projected_1024=True)
        return self.inner.extract_representations_1024(sequences)

    def generate_unconditional(self, lengths: Sequence[int], *, seed: int = 20261124) -> list[str]:
        if self.kind == "dima":
            return [self.inner.generate_unconditional(int(L), seed=seed + i)[0]["sequence"] for i, L in enumerate(lengths)]
        if self.kind == "dplm2":
            return [self.inner.sample_diffusion_full(int(L), seed=seed + i)["sequence"] for i, L in enumerate(lengths)]
        return super().generate_unconditional(lengths, seed=seed)

    def infill_local(
        self,
        template: str,
        mask_positions: Sequence[int],
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        if not mask_positions:
            return template
        start = min(int(p) for p in mask_positions)
        extent = max(int(p) for p in mask_positions) - start + 1
        if self.kind == "dima":
            return self.inner.infill(template, start=start, extent=extent, seed=seed)["sequence"]
        if self.kind == "dplm2":
            return self.inner.infill_window(template, start=start, extent=extent, coords=coords, seed=seed)["sequence"]
        if self.kind == "proteinmpnn":
            return self.inner.infill_window(template, start=start, extent=extent, coords=coords, seed=seed)["sequence"]
        return super().infill_local(template, mask_positions, coords=coords, seed=seed)

    def sample_family_guided(
        self,
        length: int,
        *,
        n_samples: int,
        profile_log_bias: np.ndarray | None = None,
        guidance_scale: float = 0.0,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> list[str]:
        if self.kind == "dima":
            return [
                self.inner.sample_guided_family(
                    int(length),
                    profile_log_bias=profile_log_bias,
                    guidance_scale=guidance_scale,
                    seed=seed + i,
                )["sequence"]
                for i in range(int(n_samples))
            ]
        if self.kind == "dplm2":
            scaled = (profile_log_bias * guidance_scale) if (profile_log_bias is not None and guidance_scale != 0.0) else None
            return [
                self.inner.sample_diffusion_full(int(length), coords=coords, profile_log_bias=scaled, seed=seed + i)["sequence"]
                for i in range(int(n_samples))
            ]
        if self.kind == "proteinmpnn":
            scaled = (profile_log_bias * guidance_scale) if (profile_log_bias is not None and guidance_scale != 0.0) else None
            return [
                self.inner.generate(coords, profile_log_bias=scaled, seed=seed + i)["sequence"]
                for i in range(int(n_samples))
            ]
        return super().sample_family_guided(
            length,
            n_samples=n_samples,
            profile_log_bias=profile_log_bias,
            guidance_scale=guidance_scale,
            coords=coords,
            seed=seed,
        )

    def continue_from_prefix(
        self,
        context_items: Sequence[str],
        target_length: int,
        *,
        coords: np.ndarray | None = None,
        seed: int = 20261124,
    ) -> str:
        prefix = "".join(context_items)
        if self.kind == "dima":
            out = self.inner.continue_from_prefix(prefix, len(prefix) + int(target_length), seed=seed)["sequence"]
            return out[len(prefix) : len(prefix) + int(target_length)]
        if self.kind == "dplm2":
            out = self.inner.continue_suffix(prefix, len(prefix) + int(target_length), coords=coords, seed=seed)["sequence"]
            return out[len(prefix) : len(prefix) + int(target_length)]
        return super().continue_from_prefix(context_items, target_length, coords=coords, seed=seed)


def load_builtin_adapter(arm_id: str, *, device: str = "auto", dtype: str | None = None, **kwargs: Any) -> ProteinBenchmarkAdapter:
    """Load any first-party roster arm (``dima-*``, ``dplm2-*``, ``dplm2-*|struct``, ``proteinmpnn-*``, or HF MLM)."""
    from src.capability.unified.roster import (
        DIMA_4_ARMS,
        DPLM2_4_ARMS,
        PROTEINMPNN_11_ARMS,
        get_arm_metadata,
    )

    eff_dev = _resolve_device(device)
    struct_mode = arm_id.endswith("|struct")
    base_arm = arm_id[:-7] if struct_mode else arm_id
    arm_meta = get_arm_metadata(base_arm)

    if base_arm in DIMA_4_ARMS:
        from src.capability.unified.dima_adapter import load_dima_adapter

        inner = load_dima_adapter(base_arm, device=eff_dev, dtype=dtype, **kwargs)
        meta = ModelMetadata(
            model_id=arm_id,
            display_name=arm_meta.display_name,
            paradigm=arm_meta.paradigm,
            modality=arm_meta.modality,
            conditioning_regime="sequence_only",
            input_alignment_format="unaligned_concat",
            n_params_millions=arm_meta.params_m,
            max_positions=510,
        )
        caps = CapabilityFlags(
            score_mutations=True,
            site_conditional_logprobs=True,
            score_sequence_nll=True,
            extract_representations_1024=True,
            generate_unconditional=True,
            infill_local=True,
            sample_family_guided=True,
            continue_from_prefix=True,
        )
        return _BuiltinBridgeAdapter(inner, "dima", meta, caps)

    if base_arm in DPLM2_4_ARMS:
        from src.capability.unified.dplm2_adapter import load_dplm2_adapter

        mode = "struct_conditioned" if struct_mode else "seq_only"
        inner = load_dima = load_dplm2_adapter(base_arm, mode=mode, device=eff_dev, dtype=dtype, **kwargs)
        del load_dima
        meta = ModelMetadata(
            model_id=arm_id,
            display_name=f"{arm_meta.display_name} ({mode})",
            paradigm=arm_meta.paradigm,
            modality=arm_meta.modality,
            conditioning_regime="backbone_3d_conditioned_oracle" if struct_mode else "sequence_only",
            input_alignment_format="unaligned_concat",
            n_params_millions=arm_meta.params_m,
            max_positions=1022,
        )
        caps = CapabilityFlags(
            score_mutations=True,
            site_conditional_logprobs=True,
            score_sequence_nll=True,
            extract_representations_1024=True,
            generate_unconditional=not struct_mode,
            infill_local=True,
            sample_family_guided=True,
            continue_from_prefix=True,
            requires_backbone_3d=struct_mode,
        )
        return _BuiltinBridgeAdapter(inner, "dplm2", meta, caps)

    if base_arm in PROTEINMPNN_11_ARMS:
        from src.capability.unified.proteinmpnn_adapter import load_proteinmpnn_adapter

        inner = load_proteinmpnn_adapter(base_arm, device=eff_dev, dtype=dtype, **kwargs)
        meta = ModelMetadata(
            model_id=arm_id,
            display_name=arm_meta.display_name,
            paradigm=arm_meta.paradigm,
            modality=arm_meta.modality,
            conditioning_regime="backbone_3d_conditioned_oracle",
            input_alignment_format="two_chain_spatial_offset_3d",
            n_params_millions=arm_meta.params_m,
            max_positions=2048,
        )
        caps = CapabilityFlags(
            score_mutations=True,
            site_conditional_logprobs=True,
            score_sequence_nll=True,
            extract_representations_1024=True,
            infill_local=True,
            sample_family_guided=True,
            requires_backbone_3d=True,
        )
        return _BuiltinBridgeAdapter(inner, "proteinmpnn", meta, caps)

    raise ValueError(
        f"Builtin arm {arm_id!r} is not directly bridged here; use HuggingFaceCausalPLMAdapter, "
        "HuggingFaceMaskedPLMAdapter, or load_custom_adapter for external/HF checkpoints."
    )
