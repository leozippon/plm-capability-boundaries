"""CPU contract for the tokenizer-only 13-text ProteinGym census.

No published weights are loaded. Real local tokenizers may be read. Tiny
synthetic ProteinGym CSVs stand in for LOOKUP replay. This is not a full
census and not a fitness result.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
import pytest
from transformers import AutoTokenizer, GPT2Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import arm_spec  # noqa: E402
from src.capability.models.fitness import load_assay  # noqa: E402
from src.capability.core.precision_policy import TEXT_AA_FP32_V1  # noqa: E402
from src.capability.models.text_aa_cohort import (  # noqa: E402
    ENCODE_FAIL,
    EXCEEDS_HARD_CONTEXT,
    IMPLEMENTED_PHASES,
    UNIMPLEMENTED_PHASES,
    _request_items,
    census_one_model,
    load_text_aa_boundary_table,
    source_fingerprints,
    support_sets,
    text_aa_model_names,
)
from src.capability.models.text_aa_fitness import (  # noqa: E402
    ENCODING_MERGED_BPE,
    ENCODING_ONE_TOKEN_PER_RESIDUE,
)
from src.capability.models.text_aa_fitness import (  # noqa: E402
    TextAAEncodingError,
    encode_text_aa,
    load_text_aa_scorer,
    load_text_aa_tokenizer,
    resolve_text_aa_boundary,
)

MODEL_BASE_DIR = Path(os.environ.get("MODEL_BASE_DIR", ""))
GPT2_DIR = MODEL_BASE_DIR / "gpt2"
BYGPT5_DIR = MODEL_BASE_DIR / "bygpt5-small-en"


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"local tokenizer directory is absent: {path}")
    return path


class _PerResidueTokenizer:
    """A GPT-2 tokenizer wrapper whose scored span is one token per residue.

    Wrapping the real tokenizer keeps the config, boundary and fail-fast paths
    under test while renumbering the residue ids, because GPT-2 merges amino
    acids and can no longer carry a legal multi-residue encoding. These
    accounting tests need a legal encoding *and* a declared hard context, and
    ByGPT5 -- the only declared checkpoint that reaches one token per residue --
    declares no hard context at all, so the over-window branch is otherwise
    unreachable.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.all_special_ids = list(getattr(inner, "all_special_ids", []))
        self._ids = {
            residue: 50000 + index for index, residue in enumerate(sorted(AA20))
        }
        self._reverse = {value: key for key, value in self._ids.items()}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        return {"input_ids": [self._ids[residue] for residue in sequence]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._reverse[int(token)] for token in ids)


def _load_cli():
    path = REPO_ROOT / "scripts/capability/mutation/text_aa_dms.py"
    spec = importlib.util.spec_from_file_location("text_aa_dms", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_native():
    path = REPO_ROOT / "scripts/capability/mutation/native_dms_extension.py"
    spec = importlib.util.spec_from_file_location("native_dms_extension_census_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_stage20():
    path = REPO_ROOT / "scripts/capability/stages/retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("stage20_census_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_assay_csv(
    directory: Path,
    name: str,
    *,
    wildtype: str,
    mutants: list[tuple[str, str]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"],
        )
        writer.writeheader()
        for mutant, sequence in mutants:
            writer.writerow(
                {
                    "mutant": mutant,
                    "mutated_sequence": sequence,
                    "DMS_score": 0.5,
                    "DMS_score_bin": "1",
                }
            )


def _write_wildtypes(path: Path, rows: list[tuple[str, str, str, int]]) -> Path:
    assay_to_wildtype = {name: ident for name, _sequence, ident, _cluster in rows}
    table: dict[str, dict] = {}
    for name, sequence, ident, cluster in rows:
        entry = table.setdefault(
            ident, {"length": len(sequence), "cluster": cluster, "assays": []}
        )
        entry["assays"].append(name)
    path.write_text(
        json.dumps(
            {
                "stage": "wildtypes",
                "assay_to_wildtype": assay_to_wildtype,
                "wildtypes": table,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fasta(path: Path, rows: list[tuple[str, str, str, int]]) -> Path:
    seen: dict[str, str] = {}
    chunks: list[str] = []
    for _name, sequence, ident, _cluster in rows:
        if ident in seen:
            continue
        seen[ident] = sequence
        chunks.append(f">{ident}\n{sequence}\n")
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def _digest(directory: Path, name: str, *, seed: int) -> str:
    native = _load_native()
    assay = load_assay(name, n=1000, seed=seed, directory=directory)
    return native.mutant_digest(assay.mutants)


def _lookup_row(name: str, *, index: int, digest: str, n_variants: int) -> dict:
    return {
        "assay": name,
        "cluster": index,
        "mutant_digest": digest,
        "n_variants": n_variants,
        "wildtype_id": f"q{index:05d}",
        "spearman": {"lookup": 0.1, "blosum62": 0.05},
    }


def _freeze(tmp_path: Path, gym: Path, rows: list[tuple[str, str, str, int]]):
    native = _load_native()
    seed = native.VARIANT_SEED
    assays = []
    for index, (name, _sequence, _ident, _cluster) in enumerate(rows):
        digest = _digest(gym, name, seed=seed + index)
        n_variants = len(load_assay(name, n=1000, seed=seed + index, directory=gym).sequences)
        assays.append(
            _lookup_row(name, index=index, digest=digest, n_variants=n_variants)
        )
    lookup = {"assays": assays}
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", rows)
    fasta_path = _write_fasta(tmp_path / "wildtypes.faa", rows)
    frozen = native.freeze_lookup_cohort(
        lookup,
        proteingym_dir=gym,
        wildtypes=json.loads(wildtypes_path.read_text(encoding="utf-8")),
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
    )
    return frozen, wildtypes_path, fasta_path, lookup


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    path = _require_dir(GPT2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def bygpt5_tokenizer():
    path = _require_dir(BYGPT5_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


def test_boundary_table_has_thirteen_and_no_path_source():
    table = load_text_aa_boundary_table()
    names = text_aa_model_names(table)
    assert names == (
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "dialogpt-small",
        "qwen2.5-0.5b",
        "qwen2.5-7b",
        "qwen2.5-32b",
        "llama-3.2-3b",
        "qwen3-8b-base",
        "bygpt5-small-en",
        "bygpt5-base-en",
        "bygpt5-medium-en",
    )
    qwen3 = resolve_text_aa_boundary("qwen3-8b-base", table)
    assert qwen3.model_type == "qwen3"
    for spec in table["text_aa_checkpoints"].values():
        assert "local_dir" not in spec
        assert "path" not in spec
    assert arm_spec("qwen3-8b-base").path.name == "Qwen3-8B-Base"
    assert arm_spec("qwen3-8b-base").path_variable == "TEXT_MODEL_BASE_DIR"


def test_load_text_aa_tokenizer_does_not_need_weights(gpt2_tokenizer, tmp_path):
    checkpoint = tmp_path / "tok-only"
    checkpoint.mkdir()
    for name in (
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = GPT2_DIR / name
        if src.is_file():
            (checkpoint / name).write_bytes(src.read_bytes())
    config = GPT2Config(
        vocab_size=len(gpt2_tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=4,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    config.save_pretrained(checkpoint)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    bundle = load_text_aa_tokenizer(checkpoint, boundary=boundary, name="gpt2")
    assert bundle.hard_context == 64
    assert bundle.tokenizer is not None
    assert not list(checkpoint.glob("*.safetensors"))
    assert not list(checkpoint.glob("pytorch_model.bin"))
    with pytest.raises(Exception):
        load_text_aa_scorer(
            checkpoint,
            boundary=boundary,
            application_window_tokens=64,
            device="cpu",
            name="gpt2",
        )


def test_over_window_excludes_whole_assay_and_keeps_digest(gpt2_tokenizer, tmp_path):
    """An over-window assay is excluded as a whole and keeps the freeze ledger.

    A substitution mutant has its wild type's length, so under a
    one-token-per-residue tokenizer the declared window admits or rejects every
    row of one assay together. The second assay stays inside the window so the
    census completes: what is pinned is that the over-window assay is excluded
    with the digest, seed, and n_variants it was frozen with, while the other
    assay remains admitted.
    """

    native = _load_native()
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym,
        "short_wt",
        wildtype="AAA",
        mutants=[("A1W", "WAA"), ("A3G", "AAG")],
    )
    _write_assay_csv(
        gym,
        "ok_wt",
        wildtype="AA",
        mutants=[("A1W", "WA"), ("A2G", "AG")],
    )
    rows = [("short_wt", "AAA", "q00000", 0), ("ok_wt", "AA", "q00001", 1)]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    tokenizer = _PerResidueTokenizer(gpt2_tokenizer)
    wt_ids = encode_text_aa(tokenizer, "AAA", boundary)
    mut_ids = encode_text_aa(tokenizer, "WAA", boundary)
    assert len(wt_ids) == len(mut_ids) == 4
    assert len(encode_text_aa(tokenizer, "AA", boundary)) == 3
    hard = 3
    payload = census_one_model(
        "gpt2",
        tokenizer=tokenizer,
        boundary=boundary,
        hard_context=hard,
        request=frozen,
        documented_context=hard,
    )
    row = payload["assays"][0]
    assert row["assay"] == "short_wt"
    assert row["admitted"] is False
    assert row["exclude_reason"] == EXCEEDS_HARD_CONTEXT
    assert row["n_variants"] == frozen["assays"][0]["n_variants"]
    assert row["mutant_digest"] == frozen["assays"][0]["mutant_digest"]
    assert row["seed"] == native.VARIANT_SEED
    assert payload["native_fixed"] == ["ok_wt"]
    assert frozen["sequences"]["short_wt"] == list(
        load_assay(
            "short_wt", n=1000, seed=native.VARIANT_SEED, directory=gym
        ).sequences
    )


def test_bygpt5_measures_length_then_window_and_wt_tie(
    bygpt5_tokenizer, tmp_path
):
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym,
        "first",
        wildtype="MKT",
        mutants=[("M1A", "AKT"), ("K2G", "MGT")],
    )
    _write_assay_csv(
        gym,
        "second",
        wildtype="MKT",
        mutants=[("M1A", "AKT"), ("K2G", "MGT")],
    )
    rows = [
        ("first", "MKT", "q00000", 0),
        ("second", "MKT", "q00001", 1),
    ]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("bygpt5-small-en", table)
    payload = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request=frozen,
        documented_context=None,
    )
    assert payload["window_rule"] == "max_legal_request_input"
    assert payload["hard_context"] is None
    # ByGPT5's boundary declares no conditioning token, so the longest legal
    # request input is the residue run itself; with a prefix it was one token
    # longer.
    expected = len("MKT")
    assert payload["application_window_tokens"] == expected
    probe = payload["longest_probe_identity"]
    assert probe["assay"] == "first"
    assert probe["role"] == "wildtype"
    assert probe["index"] == 0
    assert probe["n_input_tokens"] == expected
    assert "sequence" not in probe
    assert payload["native_fixed"] == ["first", "second"]


def test_illegal_encoding_excludes_and_file_errors_fail_fast(
    gpt2_tokenizer, tmp_path
):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    request = {
        "assays": [
            {
                "assay": "bad",
                "index": 0,
                "cluster": 0,
                "n_variants": 1,
                "mutant_digest": "d",
                "csv_sha256": "c",
                "seed": 1,
                "wildtype_id": "q00000",
                "wildtype_sequence": "A",
            }
        ],
        "sequences": {"bad": ["AAZ"]},
        "declared_assays": 1,
    }
    payload = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request=request,
        documented_context=1024,
    )
    assert payload["assays"][0]["exclude_reason"] == ENCODE_FAIL
    assert payload["assays"][0]["failed_sequences"][0]["role"] == "mutant"
    assert payload["assays"][0]["failed_sequences"][0]["mutant_index"] == 0

    def boom(*_args, **_kwargs):
        raise MemoryError("synthetic")

    monkey_request = dict(request)
    monkey_request["sequences"] = {"bad": ["AAA"]}
    import src.capability.models.text_aa_cohort as cohort

    original = cohort.encode_text_aa
    cohort.encode_text_aa = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(MemoryError):
            census_one_model(
                "gpt2",
                tokenizer=gpt2_tokenizer,
                boundary=boundary,
                hard_context=1024,
                request=monkey_request,
                documented_context=1024,
            )
    finally:
        cohort.encode_text_aa = original  # type: ignore[method-assign]

    cli = _load_cli()
    with pytest.raises(FileNotFoundError):
        cli.run_census(
            out=tmp_path / "out",
            lookup_path=tmp_path / "missing-lookup.json",
            wildtypes_path=tmp_path / "missing-wt.json",
            wildtypes_fasta_path=tmp_path / "missing.faa",
            proteingym_dir=tmp_path / "gym",
            models=["gpt2"],
            requested_device="cuda:0",
        )


def test_original_seed_index_is_kept(bygpt5_tokenizer, tmp_path):
    native = _load_native()
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym, "one", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    _write_assay_csv(
        gym, "two", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    rows = [
        ("one", "AAA", "q00000", 0),
        ("two", "AAA", "q00001", 1),
    ]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    assert [row["seed"] for row in frozen["assays"]] == [
        native.VARIANT_SEED,
        native.VARIANT_SEED + 1,
    ]
    table = load_text_aa_boundary_table()
    payload = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=resolve_text_aa_boundary("bygpt5-small-en", table),
        hard_context=None,
        request=frozen,
        documented_context=None,
    )
    assert [row["seed"] for row in payload["assays"]] == [
        native.VARIANT_SEED,
        native.VARIANT_SEED + 1,
    ]
    assert [row["index"] for row in payload["assays"]] == [0, 1]


def test_incomplete_models_cannot_claim_all13_or_mixed_families():
    table = load_text_aa_boundary_table()

    def _stub(name: str, admitted: list[str]) -> dict:
        return {
            "native_fixed": admitted,
            "assays": [
                {"assay": assay} for assay in ["a", "b", "c"]
            ],
        }

    partial = {
        "gpt2": _stub("gpt2", ["a", "b"]),
        "dialogpt-small": _stub("dialogpt-small", ["a", "c"]),
        "qwen3-8b-base": _stub("qwen3-8b-base", ["a", "b"]),
        "qwen2.5-0.5b": _stub("qwen2.5-0.5b", ["a"]),
    }
    sets = support_sets(partial, table=table)
    assert sets["common_all13"]["complete"] is False
    assert sets["common_all13"]["assays"] is None
    assert "gpt2-medium" in sets["common_all13"]["missing_models"]
    assert sets["families"]["gpt2"]["complete"] is False
    assert "dialogpt-small" not in table["families"]["gpt2"]
    assert "qwen3-8b-base" not in table["families"]["qwen2.5"]
    full_gpt2 = {
        name: _stub(name, ["a", "b"] if name != "gpt2-xl" else ["a"])
        for name in table["families"]["gpt2"]
    }
    family = support_sets(full_gpt2, table=table)["families"]["gpt2"]
    assert family["complete"] is True
    assert family["assays"] == ["a"]


def test_same_token_length_different_ids_change_digest(bygpt5_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("bygpt5-small-en", table)
    base = {
        "assays": [
            {
                "assay": "x",
                "index": 0,
                "cluster": 0,
                "n_variants": 1,
                "mutant_digest": "d",
                "csv_sha256": "c",
                "seed": 1,
                "wildtype_id": "q00000",
                "wildtype_sequence": "WWW",
            }
        ],
        "declared_assays": 1,
    }
    first = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request={**base, "sequences": {"x": ["KKK"]}},
        documented_context=None,
    )
    second = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request={**base, "sequences": {"x": ["MMM"]}},
        documented_context=None,
    )
    wt = encode_text_aa(bygpt5_tokenizer, "WWW", boundary)
    left = encode_text_aa(bygpt5_tokenizer, "KKK", boundary)
    right = encode_text_aa(bygpt5_tokenizer, "MMM", boundary)
    # One input token per byte and no conditioning prefix, so the three 3-residue
    # sequences have equal length and different ids.
    assert len(left) == len(right) == len(wt) == 3
    assert left != right
    assert first["assays"][0]["encoded_ids_digest"] != second["assays"][0]["encoded_ids_digest"]
    assert first["assays"][0]["n_input_tokens_max_legal"] == second["assays"][0][
        "n_input_tokens_max_legal"
    ]


def test_cli_accepts_injected_device_and_refuses_later_phases(
    gpt2_tokenizer, tmp_path, monkeypatch
):
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym, "one", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    rows = [("one", "AAA", "q00000", 0)]
    frozen, wildtypes_path, fasta_path, lookup = _freeze(tmp_path, gym, rows)
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
    frozen["lookup_sha256"] = "0" * 64
    frozen["lookup_path"] = str(lookup_path)
    frozen["proteingym_dir"] = str(gym)
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "freeze_request",
        lambda **_kwargs: frozen,
    )
    monkeypatch.setattr(
        "src.capability.models.text_aa_cohort.resolve_text_aa_checkpoint",
        lambda name: BYGPT5_DIR,
    )
    _require_dir(BYGPT5_DIR)
    out = tmp_path / "out"
    payload = cli.main(
        [
            "--device",
            "cuda:0",
            "--out",
            str(out),
            "census",
            "--lookup",
            str(lookup_path),
            "--wildtypes",
            str(wildtypes_path),
            "--wildtypes-fasta",
            str(fasta_path),
            "--proteingym-dir",
            str(gym),
            "--model",
            "bygpt5-small-en",
        ]
    )
    assert payload["execution"]["tokenizer_only"] is True
    assert payload["execution"]["weights_loaded"] is False
    assert payload["execution"]["cuda_invoked"] is False
    assert payload["execution"]["requested_device"] == "cuda:0"
    assert payload["execution"]["executed_on"] == "cpu"
    assert payload["support_sets"]["common_all13"]["complete"] is False
    assert payload["support_sets"]["common_all13"]["assays"] is None
    assert payload["experiment_admitted"] is False
    assert (out / "text_aa_census.json").is_file()
    with pytest.raises(ValueError, match="text-AA controller accepts only"):
        cli.main(
            [
                "--device",
                "cpu",
                "--out",
                str(out),
                "--protocol",
                "native-dms-v1",
                "score",
                "--model",
                "gpt2",
                "--census",
                str(out / "text_aa_census.json"),
                "--lookup",
                str(lookup_path),
                "--wildtypes",
                str(wildtypes_path),
                "--wildtypes-fasta",
                str(fasta_path),
            ]
        )
    assert IMPLEMENTED_PHASES == ("census", "probe", "score", "analyse")
    assert UNIMPLEMENTED_PHASES == ()


def test_analyse_refuses_a_missing_score(tmp_path):
    cli = _load_cli()
    census_path = tmp_path / "text_aa_census.json"
    census_path.write_text(
        json.dumps(
            {
                "protocol_id": TEXT_AA_FP32_V1,
                "models_requested": ["gpt2"],
                "support_sets": {
                    "native_fixed": {},
                    "common_all13": {"complete": False},
                    "families": {
                        "gpt2": {"complete": False, "assays": None},
                        "qwen2.5": {"complete": False, "assays": None},
                        "bygpt5": {"complete": False, "assays": None},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(json.dumps({"assays": []}), encoding="utf-8")
    out = tmp_path / "analyse"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="missing a score for gpt2"):
        cli.main(
            [
                "--out",
                str(out),
                "analyse",
                "--census",
                str(census_path),
                "--lookup",
                str(lookup_path),
            ]
        )


def test_old_default_doors_are_unchanged():
    stage20 = _load_stage20()
    from src.capability.core.arms import PANEL, STAGED_CANDIDATE_ARMS

    assert sorted(stage20.ARM_CORPUS) == ["progen2-medium", "progen3-112m", "protgpt2"]
    for name in text_aa_model_names():
        assert name not in stage20.SCOREABLE_ARMS
        assert name not in stage20.ARM_CORPUS
    assert "gpt2" in PANEL
    assert STAGED_CANDIDATE_ARMS == ("qwen3-8b-base", "protgpt3-1.3b")
    native = _load_native()
    with pytest.raises(ValueError, match="unknown native DMS protocol"):
        native.required_dtype("gpt2", TEXT_AA_FP32_V1)


def _toy_request(wt: str, mutants: list[str], *, assay: str = "toy") -> dict:
    return {
        "assays": [
            {
                "assay": assay,
                "index": 0,
                "cluster": 0,
                "n_variants": len(mutants),
                "mutant_digest": "d",
                "csv_sha256": "c",
                "seed": 20260807,
                "wildtype_id": "q00000",
                "wildtype_sequence": wt,
            }
        ],
        "sequences": {assay: mutants},
        "declared_assays": 1,
    }


class _CallBoomTokenizer:
    def __init__(self, inner, *, boom_on: str, error: Exception):
        self._inner = inner
        self._boom_on = boom_on
        self._error = error
        self.all_special_ids = getattr(inner, "all_special_ids", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        if sequence == self._boom_on:
            raise self._error
        return self._inner(sequence, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=False):
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)


class _DecodeBoomTokenizer:
    def __init__(self, inner, *, error: Exception):
        self._inner = inner
        self._error = error
        self.all_special_ids = getattr(inner, "all_special_ids", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        return self._inner(sequence, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=False):
        raise self._error


class _MemoryIds:
    def __iter__(self):
        raise MemoryError("synthetic ids conversion")


class _OutputTokenizer:
    def __init__(self, inner, *, boom_on: str, payload):
        self._inner = inner
        self._boom_on = boom_on
        self._payload = payload
        self.all_special_ids = getattr(inner, "all_special_ids", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, sequence, add_special_tokens=False):
        if sequence == self._boom_on:
            return self._payload
        return self._inner(sequence, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=False):
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (
            lambda tok: _CallBoomTokenizer(
                tok, boom_on="WWW", error=ValueError("internal call")
            ),
            ValueError,
        ),
        (
            lambda tok: _CallBoomTokenizer(
                tok, boom_on="WWW", error=TypeError("internal call")
            ),
            TypeError,
        ),
        (
            lambda tok: _DecodeBoomTokenizer(tok, error=ValueError("internal decode")),
            ValueError,
        ),
        (
            lambda tok: _DecodeBoomTokenizer(tok, error=TypeError("internal decode")),
            TypeError,
        ),
    ],
)
def test_tokenizer_internal_errors_fail_census_immediately(
    gpt2_tokenizer, factory, error
):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    with pytest.raises(error) as caught:
        census_one_model(
            "gpt2",
            tokenizer=factory(gpt2_tokenizer),
            boundary=boundary,
            hard_context=1024,
            request=_toy_request("AAA", ["WWW", "KKK"]),
            documented_context=1024,
        )
    assert not isinstance(caught.value, TextAAEncodingError)


def test_ids_memoryerror_and_missing_input_ids_fail_census(gpt2_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    request = _toy_request("AAA", ["WWW", "KKK"])
    with pytest.raises(MemoryError, match="synthetic ids conversion"):
        census_one_model(
            "gpt2",
            tokenizer=_OutputTokenizer(
                gpt2_tokenizer, boom_on="WWW", payload={"input_ids": _MemoryIds()}
            ),
            boundary=boundary,
            hard_context=1024,
            request=request,
            documented_context=1024,
        )
    with pytest.raises(ValueError, match="input_ids") as missing:
        census_one_model(
            "gpt2",
            tokenizer=_OutputTokenizer(gpt2_tokenizer, boom_on="WWW", payload={}),
            boundary=boundary,
            hard_context=1024,
            request=request,
            documented_context=1024,
        )
    assert not isinstance(missing.value, TextAAEncodingError)
    with pytest.raises(ValueError, match="batched tokenizer output"):
        census_one_model(
            "gpt2",
            tokenizer=_OutputTokenizer(
                gpt2_tokenizer, boom_on="WWW", payload={"input_ids": [[1, 2]]}
            ),
            boundary=boundary,
            hard_context=1024,
            request=request,
            documented_context=1024,
        )


def test_non_string_request_is_schema_failure_not_exclusion(gpt2_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    request = _toy_request("AAA", ["KKK"])
    request["sequences"]["toy"] = [123]
    with pytest.raises(TypeError, match="mutant 0 must be a str"):
        census_one_model(
            "gpt2",
            tokenizer=gpt2_tokenizer,
            boundary=boundary,
            hard_context=1024,
            request=request,
            documented_context=1024,
        )


def test_request_items_keep_none_wildtype_and_int_mutant_indices():
    items = _request_items({"wildtype_sequence": "AAA"}, ["KKK", "WWW"])
    assert items[0] == ("wildtype", None, "AAA")
    assert items[1] == ("mutant", 0, "KKK")
    assert items[2] == ("mutant", 1, "WWW")
    assert items[1][1] is not None and type(items[1][1]) is int
    assert items[2][1] is not None and type(items[2][1]) is int


def test_encode_fail_then_later_legal_max_and_digest(bygpt5_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("bygpt5-small-en", table)
    payload = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request=_toy_request("AAA", ["AAZ", "W" * 16]),
        documented_context=None,
    )
    row = payload["assays"][0]
    later_ids = encode_text_aa(bygpt5_tokenizer, "W" * 16, boundary)
    assert row["admitted"] is False
    assert row["exclude_reason"] == ENCODE_FAIL
    assert row["failed_sequences"][0]["error_class"] == "TextAAEncodingError"
    assert all(item.get("mutant_index") != 1 for item in row["failed_sequences"])
    assert row["n_input_tokens_max_legal"] == len(later_ids)
    assert payload["longest_probe_identity"]["mutant_index"] == 1
    assert payload["longest_probe_identity"]["n_input_tokens"] == len(later_ids)
    assert payload["native_fixed"] == []
    left = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request=_toy_request("AAA", ["AAZ", "W" * 8]),
        documented_context=None,
    )
    right = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request=_toy_request("AAA", ["AAZ", "M" * 8]),
        documented_context=None,
    )
    assert left["assays"][0]["encoded_ids_digest"] != right["assays"][0][
        "encoded_ids_digest"
    ]


def test_source_fingerprints_include_amino_acids_alphabet():
    fingerprints = source_fingerprints()
    assert fingerprints["src/capability/core/amino_acids.py"]
    assert fingerprints["boundaries"] == fingerprints[
        "src/capability/text_aa_boundaries.json"
    ]


def test_none_documented_context_means_no_hard_context(
    gpt2_tokenizer, bygpt5_tokenizer
):
    table = load_text_aa_boundary_table()
    request = _toy_request("AAA", ["KKK"])
    with pytest.raises(ValueError, match="documented context None disagrees"):
        census_one_model(
            "gpt2",
            tokenizer=gpt2_tokenizer,
            boundary=resolve_text_aa_boundary("gpt2", table),
            hard_context=1024,
            request=request,
            documented_context=None,
        )
    payload = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=resolve_text_aa_boundary("bygpt5-small-en", table),
        hard_context=None,
        request=request,
        documented_context=None,
    )
    assert payload["hard_context"] is None
    assert payload["documented_context"] is None
    assert payload["window_rule"] == "max_legal_request_input"
    assert payload["assays"][0]["encoding_class_wt"] == ENCODING_ONE_TOKEN_PER_RESIDUE


def test_over_window_counts_as_encoded_not_encode_fail(gpt2_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    tokenizer = _PerResidueTokenizer(gpt2_tokenizer)
    wt = "AAA"
    long_mut = "W" * 40
    short_mut = "GGG"
    wt_n = len(encode_text_aa(tokenizer, wt, boundary))
    long_n = len(encode_text_aa(tokenizer, long_mut, boundary))
    short_n = len(encode_text_aa(tokenizer, short_mut, boundary))
    assert long_n > wt_n + 1
    assert short_n <= wt_n + 1
    payload = census_one_model(
        "gpt2",
        tokenizer=tokenizer,
        boundary=boundary,
        hard_context=wt_n + 1,
        request=_toy_request(wt, [long_mut, short_mut]),
        documented_context=wt_n + 1,
    )
    row = payload["assays"][0]
    assert row["admitted"] is False
    assert row["exclude_reason"] == EXCEEDS_HARD_CONTEXT
    assert row["n_encode_ok"] == 3
    assert row["n_encode_fail"] == 0
    assert row["n_exceeds_hard_context"] == 1
    assert row["n_eligible_sequences"] == 2
    assert row["n_failed"] == 1
    assert payload["n_encode_ok"] == 3
    assert payload["n_encode_fail"] == 0
    assert payload["n_exceeds_hard_context"] == 1
    assert payload["n_sequences"] == payload["n_encode_ok"] + payload["n_encode_fail"]
    assert payload["n_encode_ok"] == sum(item["n_encode_ok"] for item in payload["assays"])
    assert row["n_input_tokens_max_legal"] == max(wt_n, short_n)


def test_merging_tokenizer_completes_a_census_and_tags_merged_bpe(gpt2_tokenizer):
    """Merged encodings are admitted. Illegal alphabet still excludes the assay."""

    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    payload = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request=_toy_request("AAA", ["WAA", "AAG"]),
        documented_context=1024,
    )
    assert payload["n_encode_ok"] == 3
    assert payload["n_encode_fail"] == 0
    assert payload["native_fixed"] == ["toy"]
    assert payload["assays"][0]["encoding_class_wt"] == ENCODING_MERGED_BPE
    assert payload["assays"][0]["n_merged_bpe"] == 3
    assert payload["longest_probe_identity"]["encoding_class"] == ENCODING_MERGED_BPE
    illegal = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request=_toy_request("AAA", ["AAZ"]),
        documented_context=1024,
    )
    assert illegal["assays"][0]["exclude_reason"] == ENCODE_FAIL
    assert illegal["assays"][0]["failed_sequences"][0]["error_class"] == "TextAAEncodingError"
