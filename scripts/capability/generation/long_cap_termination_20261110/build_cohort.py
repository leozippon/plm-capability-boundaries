#!/usr/bin/env python3
"""Freeze the companion fold cohort and its population labels from the two cap-768 ledgers.

fold_cohort.jsonl: one row per distinct sequence (id = sha256 of the sequence, role generation), the
original cohort's row format. attempt_labels.tsv: one row per attempt with its population label, which
the folder never reads. No sequence is dropped or selected."""
import hashlib, json, sys
from pathlib import Path
H = Path(__file__).resolve().parents[4] / "results/long_cap_termination_20261110"
ARMS = {"base": "protgpt3-10b", "tuned": "protgpt3-10b-dpo"}
seqs, labels = {}, []
for tag, arm in ARMS.items():
    for i, line in enumerate(open(H / "out" / arm / "attempts.jsonl")):
        r = json.loads(line); s = r["sequence"]
        if not s:
            raise SystemExit("empty continuation in a ledger; none expected")
        d = hashlib.sha256(s.encode()).hexdigest()
        assert seqs.setdefault(d, s) == s
        pop = f"{tag}-{'natural' if r['stop_status'] == 'native_terminal' else 'capped'}"
        assert r["stop_status"] in ("native_terminal", "budget_censored")
        labels.append((tag, arm, i, d, len(s), r["stop_status"], pop))
(H / "fold").mkdir(exist_ok=True)
with open(H / "fold" / "fold_cohort.jsonl", "w") as f:
    for d in sorted(seqs):
        f.write(json.dumps({"id": d, "sequence": seqs[d], "role": "generation", "length": len(seqs[d])}) + "\n")
with open(H / "fold" / "query_cohort.jsonl", "w") as f:
    for d in sorted(seqs):
        f.write(json.dumps({"id": d, "sequence": seqs[d], "role": "generated", "length": len(seqs[d])}) + "\n")
with open(H / "fold" / "attempt_labels.tsv", "w") as f:
    f.write("arm_tag\tarm\tattempt_index\tsequence_sha256\tlength\tstop_status\tpopulation\n")
    for l in labels: f.write("\t".join(map(str, l)) + "\n")
from collections import Counter
print(len(labels), "attempts", len(seqs), "distinct sequences", Counter(l[-1] for l in labels))

# Length-ordered fold parts. A 768-residue structure costs about 100 s and 130 GB on a card against 6 s at 400,
# so one longest-first fold would put the primary 16-512 reading last. Each part is folded in its own tree under
# the same contract; the union is the cohort. Parts: main (<=512), mid (513-767), cap_tuned and cap_base (768).
parts = {"main": lambda L, tag: L <= 512, "mid": lambda L, tag: 513 <= L <= 767,
         "cap_tuned": lambda L, tag: L == 768 and tag == "tuned", "cap_base_full": lambda L, tag: L == 768 and tag == "base"}
tag_of = {}
for l in labels: tag_of.setdefault(l[3], set()).add(l[0])
assert all(len(v) == 1 for v in tag_of.values()), "a sequence is shared by both arms"
used = set()
for name, rule in parts.items():
    ids = [d for d in sorted(seqs) if rule(len(seqs[d]), next(iter(tag_of[d])))]
    used |= set(ids)
    with open(H / "fold" / f"fold_cohort_{name}.jsonl", "w") as f:
        for d in ids: f.write(json.dumps({"id": d, "sequence": seqs[d], "role": "generation", "length": len(seqs[d])}) + "\n")
    print(name, len(ids))
# Declared subsample of the 321 base-capped sequences: digest % 5 == 0, content-addressed and arm-blind like the
# original cohort's shards. Written under the part name the fold driver reads (cap_base); the full block is kept
# beside it as cap_base_full and is NOT folded. Anything reported from cap_base is this subsample, not the block.
SUBSAMPLE_MOD = 5
full = [json.loads(l) for l in open(H / "fold" / "fold_cohort_cap_base_full.jsonl")]
sub = [r for r in full if int(r["id"], 16) % SUBSAMPLE_MOD == 0]
with open(H / "fold" / "fold_cohort_cap_base.jsonl", "w") as f:
    for r in sub: f.write(json.dumps(r) + "\n")
print("cap_base subsample (digest %% %d == 0): %d of %d" % (SUBSAMPLE_MOD, len(sub), len(full)))
assert used == set(seqs)
