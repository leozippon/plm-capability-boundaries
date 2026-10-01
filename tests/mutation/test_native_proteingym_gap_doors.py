"""Stage-20 ProteinGym doors for remaining non-text main-table arms.

No weights, no GPU, no ρ. These tests pin format refusals, EC lookup, the
residues-2..L mask, and parser/default isolation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.core.amino_acids import AA20  # noqa: E402
from src.capability.core.arms import INPUT_FORMAT_GMASK_SOP_EOS, arm_spec  # noqa: E402
from src.capability.models.joint_modes import RESIDUE_UNIT, resolve  # noqa: E402
from src.capability.models.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from src.capability.core.scoring import sequence_target_mask, target_rule  # noqa: E402
from tests.core.test_joint_mode_qualification import instructprotein_stub  # noqa: E402


def _stage():
    path = REPO_ROOT / "scripts" / "capability" / "stages/retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("_stage_20_nontxt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_ARMS = ["progen2-medium", "progen3-112m", "protgpt2"]
NEW_DOORS = (
    "zymctrl",
    "proteinglm-7b-clm",
    "protgpt3-1.3b",
    "instructprotein",
)


def test_new_doors_are_explicit_and_do_not_widen_the_default_run():
    stage = _stage()
    assert sorted(stage.ARM_CORPUS) == DEFAULT_ARMS
    parser = stage.build_parser()
    assert parser.parse_args([]).arms == DEFAULT_ARMS
    for name in NEW_DOORS:
        assert name in stage.SCOREABLE_ARMS
        assert name not in stage.ARM_CORPUS
        assert parser.parse_args(["--arms", name]).arms == [name]


def test_text_and_unqualified_names_remain_refused():
    stage = _stage()
    parser = stage.build_parser()
    for name in ("qwen2.5-7b", "qwen3-8b-base", "gpt2", "galactica-125m-text"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--arms", name])
        with pytest.raises(KeyError):
            stage.corpus_record(name)
        assert name not in stage.SCOREABLE_ARMS


def test_corpus_records_are_not_retrieval_bounds():
    stage = _stage()
    zym = stage.corpus_record("zymctrl")
    assert zym["declared"] == arm_spec("zymctrl").pretraining_corpus
    assert "not a retrieval bound" in zym["identification"]
    glm = stage.corpus_record("proteinglm-7b-clm")
    assert "2..L" in glm["note"] or "residues 2" in glm["note"]
    gpt3 = stage.corpus_record("protgpt3-1.3b")
    assert gpt3["declared"] == arm_spec("protgpt3-1.3b").pretraining_corpus
    inst = stage.corpus_record("instructprotein")
    assert "protein mode only" in inst["note"]


def test_arm_scorer_refuses_ec_and_gmask_formats():
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=1)
    with pytest.raises(ValueError, match="ZymCTRL"):
        stage._ArmScorer("zymctrl", args)
    with pytest.raises(ValueError, match="residues 2..L"):
        stage._ArmScorer("proteinglm-7b-clm", args)


class _OneTokenPerResidueTokenizer:
    """The staged ProtGPT3 tokenizer's id layout, with no vocabulary file.

    Measured on the released checkpoint: pad/bos/eos/[UNK] are 0/1/2/3, the
    direction token ``"1"`` is an **ordinary** vocabulary entry rather than a
    special id, and each residue is exactly one token. What is stubbed is only
    that these ids are emitted without a ``tokenizer.json``, so the test runs on
    a host with no checkpoint staged.
    """

    pad_token_id = 0
    bos_token_id = 1
    bos_token = "<|bos|>"
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        self._table = {"1": 4}
        self._table.update({residue: 5 + i for i, residue in enumerate(AA20)})

    def convert_tokens_to_ids(self, token):
        return self._table.get(token, self.unk_token_id)

    def __call__(self, text, return_tensors=None):
        ids: list[int] = []
        if text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
            text = text[len(self.bos_token) :]
        ids.extend(self._table[character] for character in text)
        return {"input_ids": ids}


class _UniformLogits(nn.Module):
    """Constant logits, so every scored target is worth exactly ``-log(vocab)``.

    The scorer returns a sum over scored positions, so with this model the sum
    *is* the count of scored positions times ``-log(vocab)``, and the count can
    be read off a number without any weights.
    """

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = SimpleNamespace(max_position_embeddings=1025)

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        batch, tokens = input_ids.shape
        return SimpleNamespace(
            logits=torch.zeros(batch, tokens, self.vocab, dtype=torch.float32)
        )


def test_arm_scorer_scores_residues_not_the_bos_direction_prefix(monkeypatch):
    """Stage 20's ProteinGym cell for ProtGPT3-1.3B scores the residues only.

    The rendering the door declares is ``<|bos|>`` then the N-to-C direction
    token then the sequence, so the direction token is the *second* input token
    and is a target under any rule that means "every real token after the
    first". The model assigns it a high likelihood while a pooled unigram
    baseline prices it as rare, so scoring it inflates every
    context-information reading taken over the same positions; the exclusion has
    to reach the stage-20 scorer, not only ``sequence_target_mask``.
    """

    stage = _stage()
    tokenizer = _OneTokenPerResidueTokenizer()
    arm = SimpleNamespace(
        name="protgpt3-1.3b",
        modality="protein",
        spec=arm_spec("protgpt3-1.3b"),
        tokenizer=tokenizer,
        device="cpu",
        model=_UniformLogits(31),
        target_token_shuffle=None,
    )
    monkeypatch.setattr(stage, "load_arm_spec", lambda *a, **k: arm)
    scorer = stage._ArmScorer(
        "protgpt3-1.3b",
        argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=2),
    )

    assert scorer.context == 1025
    assert scorer.markers == (1, 4)
    assert scorer.target_rule == "all_valid"
    assert scorer.token_lengths(["MKT"]) == [5], "two prefix tokens plus three residues"

    rendered = scorer._render(["MKT"])
    ids, mask = stage.tokenize_batch(arm, rendered, scorer.context)
    assert ids.tolist()[0][:2] == [1, 4]
    assert int(ids.shape[1]) == 5
    # The difference is the direction token and nothing else: "every real token"
    # reads 4, the arm's own declaration reads 3.
    assert int(sequence_target_mask(ids, mask, rule="all_valid").sum()) == 4
    assert (
        int(
            sequence_target_mask(
                ids, mask, rule="all_valid", marker_token_ids=scorer.markers
            ).sum()
        )
        == 3
    )

    totals = scorer.log_likelihood(["MKT", "MKTAA"])
    assert np.allclose(totals, [-3 * np.log(31), -5 * np.log(31)])
    assert "declares as markers are context, not targets" in scorer.score_description


def test_ec_labels_by_sequence_refuses_conflict(tmp_path):
    stage = _stage()
    fasta = tmp_path / "ec.fasta"
    fasta.write_text(">P1|1.1.1.1\nACDE\n>P2|2.2.2.2\nACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maps to both"):
        stage.ec_labels_by_sequence(fasta)
    clean = tmp_path / "clean.fasta"
    clean.write_text(">P1|1.1.1.1\nACDE\n>P2|1.1.1.1\nACDE\n", encoding="utf-8")
    assert stage.ec_labels_by_sequence(clean) == {"ACDE": "1.1.1.1"}


def test_zymctrl_skips_missing_ec_and_refuses_unconditioned_render():
    stage = _stage()

    class _Tok:
        eos_token = None
        unk_token_id = 0
        pad_token_id = 1

        def convert_tokens_to_ids(self, token):
            return {"<start>": 10, "<end>": 11}.get(token, 0)

        def __call__(self, text, return_tensors=None):
            return {"input_ids": list(range(len(text)))}

    fake_arm = SimpleNamespace(
        name="zymctrl",
        modality="protein",
        spec=arm_spec("zymctrl"),
        tokenizer=_Tok(),
        device="cpu",
        model=SimpleNamespace(config=SimpleNamespace(n_positions=1024)),
    )
    scorer = object.__new__(stage._ZymCTRLScorer)
    scorer.torch = torch
    scorer.name = "zymctrl"
    scorer.batch_size = 1
    scorer.arm = fake_arm
    scorer.context = 1024
    scorer.start_id = 10
    scorer.end_id = 11
    scorer._ec_by_sequence = {"ACDE": "3.2.1.17"}
    scorer._bound_ec = None
    assert "absent from the EC-labelled" in scorer.bind_wildtype("MKTA")
    with pytest.raises(ValueError, match="unconditioned"):
        scorer._render(["ACDE"])
    assert scorer.bind_wildtype("ACDE") is None
    rendered = scorer._render(["ACDE", "ADDE"])
    assert rendered == [
        "3.2.1.17<sep><start>ACDE<end>",
        "3.2.1.17<sep><start>ADDE<end>",
    ]


def test_proteinglm_target_mask_drops_prefix_and_residue_one():
    ids = torch.tensor([[29, 32, 34, 5, 6, 7]], dtype=torch.long)
    mask = torch.ones_like(ids)
    keep = sequence_target_mask(
        ids, mask, rule=target_rule(INPUT_FORMAT_GMASK_SOP_EOS)
    )
    assert keep.shape == (1, 5)
    assert keep.tolist() == [[False, False, False, True, True]]
    bare = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    with pytest.raises(ValueError, match="gmask"):
        sequence_target_mask(
            bare, torch.ones_like(bare), rule=target_rule(INPUT_FORMAT_GMASK_SOP_EOS)
        )


def test_proteinglm_scorer_refuses_non_float32_and_wrong_name():
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=1)
    with pytest.raises(ValueError, match="FP32-only"):
        stage._ProteinGLMScorer("proteinglm-7b-clm", args)
    args.dtype = "float32"
    with pytest.raises(ValueError, match="proteinglm-7b-clm"):
        stage._ProteinGLMScorer("protgpt3-1.3b", args)


def test_instructprotein_scorer_uses_declared_residue_tokens_only():
    from src.capability.models import instructprotein_fitness as IP

    tokenizer = instructprotein_stub()
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<protein>")
    tokenisation = resolve(tokenizer, "instructprotein")
    record = tokenisation.render("MK")
    assert "<protein>" in record.text
    assert record.n_residues == record.n_scored_tokens == 2
    start_id = tokenizer.convert_tokens_to_ids("<protein>")
    end_id = tokenizer.convert_tokens_to_ids("</protein>")
    assert start_id in record.token_ids
    assert record.token_ids[-1] == end_id
    scored_ids = {record.token_ids[i] for i in record.scored_positions}
    assert start_id not in scored_ids
    assert end_id not in scored_ids

    class _Rule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            b, t = input_ids.shape
            logits = torch.zeros(b, t, 80, dtype=torch.float32)
            for row in range(b):
                for pos in range(t - 1):
                    logits[row, pos, int(input_ids[row, pos + 1])] = 1.0
            return SimpleNamespace(logits=logits)

    loaded = IP.LoadedInstructProtein(
        name="instructprotein",
        model=_Rule(),
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=64,
        facts={"vocab_size": 80, "checkpoint": "stub"},
    )
    scorer = IP.InstructProteinFitnessScorer(loaded, batch_size=2)
    assert scorer.scoring_stratum == STRATUM_N_TO_C
    assert scorer.symbol_unit == RESIDUE_UNIT
    totals = scorer.log_likelihood(["MK", "AC"])
    assert totals.shape == (2,)
    assert np.all(np.isfinite(totals))
    accounting = scorer.residue_accounting()
    assert accounting["residues"] == 4
    assert accounting["scored_tokens"] == 4


def _proteinglm_stage_scorer(stage, monkeypatch, *, log_likelihood_calls):
    """The real ``_ProteinGLMScorer`` with no weights and no tokenizer load.

    Only ``load_arm_spec`` is replaced, so ``__init__``'s own dtype, arm and
    serving-window checks still run against a config that declares the real
    1024. ``log_likelihood`` is replaced because a forward needs the 7B weights;
    ``token_lengths`` is the method under test and is left exactly as delivered.
    """

    config = SimpleNamespace(seq_length=stage.PGLM.CONTEXT_LENGTH)
    fake_arm = SimpleNamespace(
        model=SimpleNamespace(config=config), tokenizer=None, device="cpu"
    )
    monkeypatch.setattr(stage, "load_arm_spec", lambda *a, **k: fake_arm)

    args = argparse.Namespace(dtype="float32", device="cpu", batch_size=16)
    scorer = stage._ProteinGLMScorer("proteinglm-7b-clm", args)

    def fake_log_likelihood(sequences):
        log_likelihood_calls.append(list(sequences))
        return np.linspace(-1.0, -0.5, len(sequences))

    scorer.log_likelihood = fake_log_likelihood
    return scorer


def _install_gap_scorer(stage, monkeypatch, scorer):
    def fake_load(arm, args):
        assert arm == "proteinglm-7b-clm"
        return (
            scorer,
            scorer.context,
            {
                "checkpoint": "cpu-test-stub",
                "context": scorer.context,
                "input_format": INPUT_FORMAT_GMASK_SOP_EOS,
            },
        )

    monkeypatch.setattr(stage, "_load_scorer", fake_load)


def _install_two_assays(stage, monkeypatch, *, inside, over):
    """One assay inside the 1024 window and one past it, in the drawn order."""

    def fake_load_assay(name, *, n, seed, directory=None):
        sequence = inside if name == "assay_inside" else over
        return SimpleNamespace(
            name=name,
            mutants=["M1A", "M3T"],
            sequences=[sequence, "A" + sequence[1:]],
            scores=np.array([0.2, 0.8], dtype=np.float64),
            wildtype=sequence,
        )

    monkeypatch.setattr(stage, "load_assay", fake_load_assay)


def test_proteinglm_over_context_assay_is_skipped_not_a_crash(monkeypatch, tmp_path):
    """EXP-R2-240's `pgym_proteinglm-7b-clm` cell died in the length probe.

    ``_ProteinGLMScorer.token_lengths`` rendered through
    ``render_budget_sequence``, which refuses an over-context sequence, so the arm
    could never reach stage 20's own exclusion branch and lost every assay rather
    than the 16 the 1024 window excludes. The probe must therefore be total over
    the rendered length while the scoring path keeps refusing.
    """

    stage = _stage()
    pglm = stage.PGLM
    assert pglm.CONTEXT_LENGTH == 1024
    assert pglm.PREFIX_LENGTH == 3
    inside = "A" * 1021
    over = "A" * 1154
    assert pglm.budget_token_length(inside) == 1024
    assert pglm.budget_token_length(over) == 1157

    calls: list[list[str]] = []
    scorer = _proteinglm_stage_scorer(stage, monkeypatch, log_likelihood_calls=calls)

    # The probe returns; it does not raise. This is the line that used to kill the arm.
    assert scorer.token_lengths([inside, over]) == [1024, 1157]
    # The scoring path still refuses the same sequence, both directly and through
    # the render that `log_likelihood` calls first.
    with pytest.raises(ValueError, match="seq_length"):
        pglm.render_budget_sequence(over)
    with pytest.raises(ValueError, match="seq_length"):
        scorer._render([over])
    # Illegal input is still a defect on both paths rather than a window statement.
    with pytest.raises(ValueError, match="AA20"):
        pglm.budget_token_length("AX")
    with pytest.raises(ValueError, match="two residues"):
        pglm.budget_token_length("A")

    _install_gap_scorer(stage, monkeypatch, scorer)
    _install_two_assays(stage, monkeypatch, inside=inside, over=over)

    out = tmp_path / "out"
    out.mkdir()
    (out / "wildtypes.json").write_text(
        json.dumps(
            {"assay_to_wildtype": {"assay_inside": "q00000", "assay_over": "q00001"}}
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        arms=["proteinglm-7b-clm"],
        assays=["assay_inside", "assay_over"],
        variants=1000,
        seed=20260807,
        batch_size=16,
        dtype="float32",
        device="cpu",
        proteingym_dir=tmp_path,
        out=out,
    )

    payload = stage.stage_score(args)["proteinglm-7b-clm"]

    assert [row["assay"] for row in payload["assays"]] == ["assay_inside"]
    assert payload["assays"][0]["max_tokens"] == 1024
    skipped = payload["skipped"]
    assert [item["assay"] for item in skipped] == ["assay_over"]
    assert skipped[0]["max_tokens"] == 1157
    assert skipped[0]["context"] == 1024
    assert skipped[0]["reason"] == (
        "the rendered variant exceeds this arm's context; truncating would score a "
        "sequence that may not contain the mutated position"
    )
    # The excluded assay was never sent to the model.
    assert [len(batch) for batch in calls] == [2]


def test_load_scorer_routes_new_doors(monkeypatch):
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=16)

    class _Stub:
        context = 128
        score_description = "stub"
        scoring_stratum = STRATUM_N_TO_C
        facts = {
            "checkpoint": "stub",
            "scientific_role": "stub",
            "native_eos_token_id": 2,
            "config_eos_token_id": None,
            "config_eos_token_id_status": "ignored",
        }

        def __init__(self, *a, **k):
            return

    monkeypatch.setattr(stage, "_ZymCTRLScorer", type("Z", (_Stub,), {}))
    monkeypatch.setattr(stage, "_ProteinGLMScorer", type("G", (_Stub,), {}))
    monkeypatch.setattr(stage, "_ArmScorer", type("A", (_Stub,), {}))

    class _Loaded:
        facts = {
            "checkpoint": "stub",
            "scientific_role": "stub",
            "vocab_size": 80,
        }
        context = 128

    class _IPScorer(_Stub):
        def __init__(self, loaded, *, batch_size):
            self.context = 128
            self.facts = loaded.facts

    monkeypatch.setattr(stage.IP, "load_instructprotein", lambda *a, **k: _Loaded())
    monkeypatch.setattr(stage.IP, "InstructProteinFitnessScorer", _IPScorer)

    glm_args = argparse.Namespace(device="cpu", dtype="float32", batch_size=16)
    _, _, record = stage._load_scorer("zymctrl", args)
    assert record["input_format"] == "ec_conditioned"
    _, _, record = stage._load_scorer("proteinglm-7b-clm", glm_args)
    assert record["input_format"] == INPUT_FORMAT_GMASK_SOP_EOS
    scorer, _, record = stage._load_scorer("protgpt3-1.3b", args)
    assert type(scorer).__name__ == "A"
    _, _, record = stage._load_scorer("instructprotein", args)
    assert record["input_format"].startswith("instructprotein")
