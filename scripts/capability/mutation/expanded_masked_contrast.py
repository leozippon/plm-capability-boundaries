#!/usr/bin/env python3
"""The masked arm of ``matched_discrimination_20261102`` support ``S2_expanded``.

ESM-2 is the manifest's ``different_scoring_rule_contrast``, and a masked
language model has no sequence log likelihood. It is therefore read under the
member this project already qualified and archived
(``scripts/capability/interactions/exploratory_denoising_arms.py``), at
corruption level ``1/L`` with the substituted position alone masked:

    D(w, j, a) = log p(a | w with j masked) - log p(w_j | w with j masked)

Wild type and mutant are read off the *same* forward, so the background is
matched by construction. One forward serves every variant of one position, which
is why the whole cohort costs one masked forward per distinct substituted site
rather than one per variant. The resulting quantity is in nats and is a
pseudo-likelihood contrast; the manifest reports this arm separately and never
pools it with the autoregressive arms.

The local checkpoint directory is ``esm2_t33_650M_UR50D`` while the repository's
masked roster and ``position_likelihood.MASKED_ARMS`` both expect ``esm2_650m``.
That is resolved by a symlink created inside this run's output directory and
recorded with its resolved target: nothing on disk is renamed and no loader is
edited.

Interface matches the stage contract of
``scripts/capability/decoding/run_local_gpu_manifest.py``, which injects
``--device`` and ``--out``.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.capability.core.amino_acids import AA20
from src.capability.core.io import sha256_file, write_json
from src.capability.mutation import expanded_discrimination as E


def resolve_model_root(out: Path) -> Path:
    """A directory holding the roster's expected name as a symlink to the real one."""

    checkpoint = Path(E.MASKED_CHECKPOINT)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"no masked checkpoint at {checkpoint}")
    root = out / "masked_model_root"
    root.mkdir(parents=True, exist_ok=True)
    link = root / E.MASKED_LINK_NAME
    if link.is_symlink() or link.exists():
        if link.resolve() != checkpoint.resolve():
            raise ValueError(f"{link} resolves to {link.resolve()}, not {checkpoint}")
    else:
        link.symlink_to(checkpoint)
    return root


def load_roster_module():
    """The prior declaration of where these checkpoints live and of the score member."""

    path = ROOT / "scripts/capability/interactions/exploratory_denoising_arms.py"
    spec = importlib.util.spec_from_file_location("exploratory_denoising_arms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.EXPLORATORY_ROSTER[E.MASKED_ARM]["directory"] != E.MASKED_LINK_NAME:
        raise ValueError("the masked roster no longer expects the resolved directory name")
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError("the masked batch holds at least one position")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out = Path(args.out)
    root = resolve_model_root(out)
    module = load_roster_module()
    cohort = json.loads(args.cohort.read_text())
    checkpoint = Path(E.MASKED_CHECKPOINT)

    began = time.monotonic()
    tokenizer, model, identity = module.load_arm(root, E.MASKED_ARM, args.device)
    scores: dict[str, dict[str, float]] = {}
    skipped: list[dict] = []
    forwards = 0
    for index, row in enumerate(cohort["assays"]):
        wildtype = row["wildtype"]
        if len(wildtype) > module.RESIDUE_CAPACITY:
            skipped.append(
                {
                    "assay": row["assay"],
                    "reason": (
                        f"{len(wildtype)} residues exceeds the declared "
                        f"{module.RESIDUE_CAPACITY}-position capacity of this architecture"
                    ),
                }
            )
            continue
        outside = sorted(set(wildtype) - set(AA20))
        if outside:
            skipped.append(
                {"assay": row["assay"], "reason": f"wild type carries {outside} outside AA20"}
            )
            continue
        # ``module.substitutions`` refuses a mutation string whose wild-type
        # residue disagrees with this wild type, so the frame is checked there.
        values = np.asarray(
            module.contrasts(model, tokenizer, wildtype, row["mutants"], args.device, args.batch),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{row['assay']}: a masked contrast is not finite")
        forwards += len({int(mutant[1:-1]) for mutant in row["mutants"]})
        scores[row["assay"]] = {
            mutant: float(value) for mutant, value in zip(row["mutants"], values)
        }
        print(f"{E.MASKED_ARM}: {index + 1}/{len(cohort['assays'])} {row['assay']}", flush=True)
    elapsed = time.monotonic() - began
    if not scores:
        raise RuntimeError("no assay was scored by the masked arm")
    write_json(
        out / f"masked_scores_{E.MASKED_ARM}.json",
        {
            "schema_version": E.SCHEMA,
            "analysis_id": E.ANALYSIS_ID,
            "declaration": E.DECLARATION_ID,
            "arm": E.MASKED_ARM,
            "manifest_arm": E.MASKED_MANIFEST_ARM,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": {
                **identity,
                "resolved_checkpoint": str(checkpoint.resolve()),
                "directory_resolution": (
                    "the local directory is esm2_t33_650M_UR50D; the masked roster and "
                    "position_likelihood.MASKED_ARMS expect esm2_650m, resolved by a "
                    "symlink inside this output directory. Nothing on disk was renamed"
                ),
                "checkpoint_metadata_sha256": {
                    path.name: sha256_file(path)
                    for path in sorted(checkpoint.iterdir())
                    if path.is_file() and (path.suffix in (".json", ".txt"))
                },
                "checkpoint_tensor_files": [
                    {"name": path.name, "bytes": path.stat().st_size}
                    for path in sorted(checkpoint.iterdir())
                    if path.is_file() and path.suffix in (".safetensors", ".bin", ".pt")
                ],
                "torch_version": torch.__version__,
            },
            "score_semantics": E.SCORING_CONVENTIONS["masked"],
            "cohort_sha256": sha256_file(args.cohort),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device) if args.device != "cpu" else None,
            "batch": args.batch,
            "masked_forwards": forwards,
            "assays": len(scores),
            "variants": sum(len(block) for block in scores.values()),
            "elapsed_seconds": elapsed,
            "skipped": skipped,
            "scores": scores,
        },
    )
    print(f"masked arm complete: {forwards} masked forwards in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
