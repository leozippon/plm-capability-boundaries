"""Source mappings for the phenotype readings the rewritten R3 rests on.

R3 reports four phenotype cohorts beyond mutation ranking --- Venus activity,
SKEMPI binding ranking and quantitative binding, and saturation-genome-editing
cellular fitness --- the matched phenotype-pair panel, the three abilities the
refitted MegaScale stability grid separates (ranking, within-family calibrated
prediction and transfer to unseen families), and what the evolutionary profile
is worth against all of them. Every row here is read out of a retained artifact
of those campaigns, or computed from them by the arithmetic its claim states. No
fit is run and no interval is resampled: a simultaneous band is copied as the
bound the artifact records, and a reduction over a set of columns is the
smallest and largest member of that set.

The profile comparator is the section's organising contrast, and it is read one
way on each source rather than one way overall, because each source's baseline
is its own. What the profile is worth is the profile block's own held-out
contribution to that source's standing control set: on ProteinGym the
composition control, read from the local-context measurement's model-independent
designs, which are bit-identical across arms and checked to be; on Domainome and
on folding stability the ``prof`` and ``prof2`` entries of each gate's own
control ladder. What the profile absorbs is read with the other blocks held
fixed --- on ProteinGym the qualified seven-residue window is in both designs,
so the share is the profile's and not the window's --- and the share of the
profile's worth that the largest surviving likelihood increment reaches is taken
over each source's own profile-inclusive baseline. Those three shares sit on
three supports, which is why the span over them is emitted as ``range_bound``
rows carrying each end's own support and never as an estimate.

Three derived quantities carry arithmetic, stated on the row and verified here:

* the within-family calibrated reduction as a share of the error that remains
  after calibration, and the best model's reduction in kcal/mol rather than
  kcal^2/mol^2. Both need the matched baseline's own calibrated squared error,
  which the panel does not carry and every per-arm fit in ``fits/`` does, one
  per-family list per split seed. :func:`_calibrated_baseline` reads all 33,
  refuses unless they agree to 1e-12, and refuses unless each fit's own
  group-equal mean reproduces that arm's published band point: the numerator and
  the denominator then come from one reading of one cohort rather than from two
  files that merely sit next to each other. This is the repository's own
  ``family_calibrated`` reduction as the grid recorded it, not a refit.
* the agreement between the calibrated-error and ranking readings, as the
  Pearson correlation across the panel's 33 columns. The panel resamples each
  column and records no interval for the agreement between two readings, so the
  row carries the reason it has none rather than borrowing one.
* the activity band's half-width as a multiple of the beyond-profile stability
  gain, over the models that carry both readings. The two panels have different
  supports and the manuscript compares them deliberately --- a precision on one
  phenotype against an effect size on another --- so the ratio is emitted as a
  span of ``range_bound`` rows and never as an estimate.

Two rosters are declared because no artifact field records them.
``PROTEIN_SPECIALIZED`` and ``TEXT_AND_GALACTICA`` split the stability grid's 33
checkpoints into the protein-specialized or adapted models and the general-text
and Galactica models; the manuscript's "4 of the 14 such models" and "the text
and Galactica models on stability" select them. The two must be disjoint and
cover the panel's columns exactly, or the bound would be a bound over an
unstated subset.

The procedural constants these campaigns record are not emitted: bootstrap draw
counts and seeds, independence floors, control-column censuses, the critical
maximum absolute deviation of each band, and the baseline levels that only serve
as denominators. No claim quotes them, and each would dilute the attribution of
a number the manuscript does quote. Support sizes are emitted where a claim
quotes them --- 89 and 126 groups, 10 scored models, 30 matched contrasts, 33
and 14 stability checkpoints --- and the nine SGE genes are emitted with them
because the span over the three resolved models rests on that support and the
sentence cites it.
"""
from __future__ import annotations

from math import sqrt
from statistics import fmean
import glob
import os

from src.capability.core.evidence_ledger import LedgerError
# The emit contract and the recorded-span convention belong to the two sibling
# modules that already own them: a row's locator, digest and pointer to the
# follow-up module, and the ``<prefix>/min``--``<prefix>/max`` identifier pair
# the consistency checker matches a quoted range against to the propagation
# module. A second copy of either here would be a second place for it to drift.
from scripts.capability.reporting.followup_quantities import _emitter
from scripts.capability.reporting.propagation_quantities import _span

PHENOTYPE = "results/extensions/phenotype_breadth_20261008"
BREADTH = f"{PHENOTYPE}/fit-panel10/phenotype_breadth_panel.json"
#: One fitted panel per (cohort, endpoint) cell the breadth campaign admits.
#: The roll-up above states the cell roster and repeats each panel's own
#: per-arm results; the panels carry the support counts and the control
#: qualification, so a row is bound to the panel that computed it.
PANELS = {
    ("activity_venus", "ranking"):
        f"{PHENOTYPE}/fit-panel10/panels/activity_venus-ranking.json",
    ("binding_skempi", "ranking"):
        f"{PHENOTYPE}/fit-panel10/panels/binding_skempi-ranking.json",
    ("binding_skempi", "quantitative"):
        f"{PHENOTYPE}/fit-panel10/panels/binding_skempi-quantitative.json",
    ("cellular_fitness_sge", "ranking"):
        f"{PHENOTYPE}/fit-panel10/panels/cellular_fitness_sge-ranking.json",
}
MATCHED = f"{PHENOTYPE}/matched-panel10/matched_phenotype_analysis.json"
GRID_ROOT = ("results/external_baseline/20261008095614_7e8ae4b18a4d/"
             "stability_ranking_error_grid_cpu")
GRID = f"{GRID_ROOT}/panel.json"
GRID_FITS = f"{GRID_ROOT}/fits"
ABUNDANCE = "results/R1/external_confirmation_20260924/panel/panel.json"
#: The two gates' own control ladders, where each candidate block carries its
#: own held-out contribution over the standing set it was offered against. This
#: is what "the profile on its own is worth" reads: a contribution to a control
#: set, not a model result.
ABUNDANCE_LADDER = "results/R1/external_confirmation_20260924/controls_qualification.json"
STABILITY_LADDER = "results/R4/gate_stability_20260924/controls/controls_qualification.json"
#: The ProteinGym cohort R1 and R3 share, where the profile contrast is read
#: between control designs that differ in the profile block alone.
LOCAL_CONTEXT = ("results/R5/local_context_20260923/20260923233257_e429ce7f31e4/lcgp_admission/"
                 "local_context_measurement.json")

#: The stability grid's protein-specialized or adapted checkpoints. No artifact
#: field records this: a corpus is not a panel column, and ProLLaMA and
#: InstructProtein are protein-adapted text models rather than protein-only
#: ones. "4 of the 14 such models" is this set.
PROTEIN_SPECIALIZED = (
    "instructprotein", "progen2-base", "progen2-large", "progen2-medium", "progen2-small",
    "progen2-xlarge", "progen3-112m", "progen3-3b", "prollama", "prollama-stage-1",
    "proteinglm-7b-clm", "protgpt2", "protgpt3-1.3b", "rita-xl",
)
#: The rest of the grid: general-text checkpoints and the Galactica rungs, which
#: read protein without having been trained on it. "the text and Galactica
#: models on stability" is this set.
TEXT_AND_GALACTICA = (
    "bygpt5-base-en", "bygpt5-medium-en", "bygpt5-small-en", "dialogpt-small", "galactica-1.3b",
    "galactica-125m", "galactica-30b", "galactica-6.7b", "gpt2", "gpt2-large", "gpt2-medium",
    "gpt2-xl", "llama-2-7b", "llama-3.2-3b", "qwen2.5-0.5b", "qwen2.5-0.5b-instruct",
    "qwen2.5-32b", "qwen2.5-7b", "qwen3-8b-base",
)

#: How the breadth panels name a metric, and the unit a reduction in it carries.
#: The quantitative endpoint's unit is the square of its own traced label unit,
#: so the panel's declaration of that label is checked rather than assumed.
RANKING_UNIT = "dimensionless Spearman"
QUANTITATIVE_TARGET = "the oriented label in its own unit: dimensionless ln(Kd_mutant/Kd_wildtype)"
QUANTITATIVE_UNIT = "squared dimensionless ln Kd ratio"
SQUARED_KCAL = "kcal^2/mol^2"
#: The grid's six contrasts, by baseline and reading, with the unit each is
#: measured in. A rank correlation and a squared kcal/mol are not commensurable
#: and the panel says so, which is why the unit travels with the contrast.
GRID_UNITS = {
    "matched_ranking": RANKING_UNIT,
    "profile_ranking": RANKING_UNIT,
    "matched_calibrated_error": SQUARED_KCAL,
    "profile_calibrated_error": SQUARED_KCAL,
    "matched_transfer_error": SQUARED_KCAL,
    "profile_transfer_error": SQUARED_KCAL,
}
#: The grid's own verdict vocabulary, which is not the ledger's four outcomes.
RESOLVED_ABOVE = "above_zero"

#: The ProteinGym control designs the profile contrast is read between: the
#: composition control alone, which is what "on its own" adds the profile to,
#: and the pair that holds the qualified window block fixed, which is what the
#: absorbed share is read over.
PROFILE_WORTH_DESIGNS = ("C", "C_P")
WINDOW_HELD_DESIGNS = ("C_wall", "C_P_wall")
#: Which block each gate's ladder offers as the fitted independent-site profile.
#: Domainome offers ``prof`` and then ``prof2`` sequentially; the stability gate
#: offers both against the same standing set, and its profile-inclusive design
#: carries ``prof2``, so that is the entry its worth is read from.
LADDER_PROFILE_BLOCK = {"abundance": "prof", "stability": "prof2"}


def add_phenotype_quantities(ledger) -> None:
    emit = _emitter(ledger)
    cells = _breadth_cells(ledger)
    _breadth_bounds(ledger, emit, cells)
    _quantitative_share(ledger, emit, cells)
    _matched_components(ledger, emit)
    _stability_grid(ledger, emit)
    _abundance_ranking(ledger, emit)
    _activity_precision_ratio(ledger, emit, cells)
    _profile_comparator(ledger, emit)


# --------------------------------------------------------------------------- #
# The four phenotype cohorts, each on its own fitted panel.
# --------------------------------------------------------------------------- #

def _breadth_cells(ledger) -> dict[tuple[str, str], dict]:
    """The admitted (cohort, endpoint) cells, with panel and roll-up agreeing.

    The roll-up states which cells the campaign admits and repeats each panel's
    per-arm results. A cell the roll-up declares and no panel here reads, or a
    panel whose results are not the roll-up's, means the two files are two
    different readings: there is then no one reading to quote, so it refuses.
    """
    payload = ledger.artifacts.json(BREADTH)
    if payload["schema"] != "phenotype_breadth_v1" or payload["status"] != "complete":
        raise LedgerError(f"{BREADTH}: the breadth roll-up is not a complete "
                          "phenotype_breadth_v1 panel")
    declared = {(cell["cohort"], cell["endpoint"]): cell for cell in payload["results"]}
    if set(declared) != set(PANELS):
        raise LedgerError(
            f"{BREADTH}: the roll-up admits {sorted(declared)} and this module reads "
            f"{sorted(PANELS)}; a bound over one roster is not a bound over the other")
    cells: dict[tuple[str, str], dict] = {}
    for key, cell in sorted(declared.items()):
        path = PANELS[key]
        panel = ledger.artifacts.json(path)
        if cell["status"] != "admitted":
            raise LedgerError(f"{BREADTH}: the {key} cell is {cell['status']}, not admitted")
        if panel["arm_results"] != cell["arm_results"]:
            raise LedgerError(f"{path}: the panel's per-arm results are not the roll-up's, so the "
                              "two files are two readings of this cell")
        if sorted(panel["arms"]) != sorted(entry["arm"] for entry in panel["arm_results"]):
            raise LedgerError(f"{path}: the scored roster and the per-arm results differ")
        if len(panel["groups"]) != cell["groups"] or \
                panel["support"]["independent_groups"] != cell["groups"]:
            raise LedgerError(f"{path}: the panel's group roster, its support count and the "
                              "roll-up's disagree, so no group count can be stated")
        resolved = [entry["arm"] for entry in panel["arm_results"] if entry["resolved_positive"]]
        if len(resolved) != cell["resolved_positive"] or \
                panel["resolved_positive"] != cell["resolved_positive"]:
            raise LedgerError(f"{path}: the per-arm resolutions and the stated count disagree")
        cells[key] = {"path": path, "panel": panel, "resolved": resolved,
                      "support": _breadth_support(ledger, key, panel)}
    return cells


def _breadth_support(ledger, key: tuple[str, str], panel: dict) -> str:
    """One cohort--endpoint support, declared from the panel's own counts."""
    cohort, endpoint = key
    support, inference = panel["support"], panel["inference"]
    qualified = "+".join(panel["qualified_controls"])
    profile = panel["control_qualification"]["profile_blocks_qualified"]
    return ledger.declare_support(
        f"phenotype_breadth/{cohort}/{endpoint}",
        f"{len(panel['arms'])} scored checkpoints on the {panel['phenotype']} cohort, "
        f"{endpoint} endpoint: {support['independent_groups']} independent sequence-homology "
        f"groups over {support['backgrounds']} backgrounds and {support['units']} units, "
        f"{support['rows']} fitted rows from {support['source_rows']} source rows; the endpoint is "
        f"{panel['endpoint_definition']}; qualified controls {qualified} with "
        f"{'profile blocks ' + '+'.join(profile) if profile else 'no profile block'} qualified; "
        f"{panel['aggregation']}; the band is {inference['method']}")


def _upper_bound(panel: dict) -> tuple[str, float]:
    """The largest simultaneous upper bound over a cell's scored columns."""
    bounds = {}
    for entry in panel["arm_results"]:
        low, high = entry["simultaneous"]
        if not low < high:
            raise LedgerError(f"{entry['arm']}: the simultaneous band is not an interval")
        bounds[entry["arm"]] = float(high)
    arm = max(bounds, key=bounds.get)
    return arm, bounds[arm]


def _breadth_bounds(ledger, emit, cells: dict[tuple[str, str], dict]) -> None:
    """What each phenotype cell resolves, and what bounds it from above."""
    for (cohort, endpoint), cell in sorted(cells.items()):
        panel, path, support = cell["panel"], cell["path"], cell["support"]
        if panel["metric"] not in ("spearman", "error"):
            raise LedgerError(f"{path}: the cell is measured in {panel['metric']!r}, which carries "
                              "no stated unit here")
        unit = RANKING_UNIT if panel["metric"] == "spearman" else _quantitative_unit(path, panel)
        reading = "ranking" if endpoint == "ranking" else "quantitative prediction"
        name = f"{cohort}/{endpoint}"
        emit(path, ("arms",), identifier=f"phenotype_breadth/{name}/scored_models",
             claim=f"checkpoints scored on the {panel['phenotype']} {reading} cell",
             family="phenotype_breadth", support=support, unit="checkpoints",
             kind="support_count", value=len(panel["arms"]), level="likelihood")
        emit(path, ("support", "independent_groups"),
             identifier=f"phenotype_breadth/{name}/independent_groups",
             claim=f"independent sequence-homology groups the {panel['phenotype']} {reading} cell "
                   "resamples over",
             family="phenotype_breadth", support=support, unit="groups", kind="support_count",
             level="likelihood")
        resolved = cell["resolved"]
        emit(path, ("arm_results", "<resolved above zero>"),
             identifier=f"phenotype_breadth/{name}/models_resolving_above_zero",
             claim=f"checkpoints whose {panel['phenotype']} {reading} improvement resolves above "
                   "zero under the cell's simultaneous band",
             family="phenotype_breadth", support=support, unit="checkpoints", kind="count",
             value=len(resolved), level="likelihood",
             verdict="supported" if resolved else "unresolved")
        # A cell states one of two things, and the manuscript quotes whichever
        # it states: the range the models that resolve improve by, or --- where
        # nothing resolves --- the bound above which the band excludes an
        # improvement. The other number is not a result of the cell, so it is
        # not emitted.
        if resolved:
            _span(emit, path, ("arm_results", "<resolved above zero>", "point"),
                  prefix=f"phenotype_breadth/{name}/resolved_improvement",
                  claim=f"{panel['phenotype']} {reading} improvement across the {len(resolved)} "
                        "models that resolve above zero",
                  family="phenotype_breadth", support=support, unit=unit,
                  values={entry["arm"]: float(entry["point"]) for entry in panel["arm_results"]
                          if entry["arm"] in resolved})
            continue
        arm, bound = _upper_bound(panel)
        emit(path, ("arm_results", "<largest simultaneous upper bound>"),
             identifier=f"phenotype_breadth/{name}/simultaneous_upper_bound",
             claim=f"largest simultaneous upper bound on a {panel['phenotype']} {reading} "
                   f"improvement across the {len(panel['arms'])} scored models, which is what "
                   f"bounds the cell from above ({arm})",
             family="phenotype_breadth", support=support, unit=unit, kind="range_bound",
             value=bound, level="likelihood", verdict="unresolved")


def _quantitative_unit(path: str, panel: dict) -> str:
    """The squared unit of a quantitative cell's own traced label.

    A squared error is only readable in the square of the label's unit, and the
    breadth campaign reports a quantitative endpoint only where that unit is
    traced. The panel's own declaration of it is checked here, so a cohort whose
    label unit changed yields a refusal rather than a row in the wrong square.
    """
    target = panel["control_qualification"]["target"]
    if target != QUANTITATIVE_TARGET:
        raise LedgerError(f"{path}: the quantitative label is {target!r}, not the traced "
                          f"{QUANTITATIVE_TARGET!r}, so its squared error has no stated unit")
    return QUANTITATIVE_UNIT


def _quantitative_share(ledger, emit, cells: dict[tuple[str, str], dict]) -> None:
    """The quantitative binding bound as a share of the baseline's own error.

    A squared error in a label's own units says nothing on its own, so the
    manuscript quotes the bound against the error the qualified control set
    still makes. Both come from this panel: the bound from its simultaneous
    band and the baseline from its control qualification, averaged over the
    three split seeds the panel fits.
    """
    key = ("binding_skempi", "quantitative")
    cell = cells[key]
    panel, path = cell["panel"], cell["path"]
    if cell["resolved"]:
        raise LedgerError(f"{path}: {cell['resolved']} now resolve above zero, so this cell states "
                          "an improvement rather than a bound and a share of the baseline error is "
                          "no longer what it reports")
    qualification = panel["control_qualification"]
    by_seed = qualification["qualified_error_by_seed"]
    if sorted(by_seed) != sorted(str(seed) for seed in panel["split_seeds"]):
        raise LedgerError(f"{path}: the qualified baseline error does not cover the panel's split "
                          "seeds, so a share of it would be a share of another fit")
    baseline = fmean(float(value) for value in by_seed.values())
    if baseline <= 0.0:
        raise LedgerError(f"{path}: the qualified baseline error is not positive")
    arm, bound = _upper_bound(panel)
    emit(path, ("inference", "<largest simultaneous upper bound over the qualified baseline "
                "error>"),
         identifier=f"phenotype_breadth/{key[0]}/{key[1]}/upper_bound_share_of_baseline_error",
         claim="largest simultaneous upper bound on a quantitative binding improvement as a share "
               f"of the qualified control set's own held-out squared error ({baseline:.4f} squared "
               f"ln Kd ratio, the three-seed mean), which is the tightest bound the phenotype "
               f"panels carry ({arm})",
         family="phenotype_breadth", support=cell["support"], unit="percent", kind="ratio",
         value=100.0 * bound / baseline, level="likelihood")


def _matched_components(ledger, emit) -> None:
    """The matched phenotype-pair panel: every headline contrast, none resolved."""
    payload = ledger.artifacts.json(MATCHED)
    if payload["schema"] != "matched_phenotype_v1" or payload["status"] != "complete":
        raise LedgerError(f"{MATCHED}: the matched panel is not a complete matched_phenotype_v1 "
                          "analysis")
    pooled = payload["pooled_inference"]
    contrasts, arms = pooled["contrast_family"], payload["arms"]
    results = pooled["results"]
    if len(results) != len(arms) * len(contrasts):
        raise LedgerError(
            f"{MATCHED}: the pooled family carries {len(results)} cells against "
            f"{len(arms)} arms by {len(contrasts)} contrasts, so no count over the whole family "
            "can be stated")
    covered = {(entry["arm"], entry["contrast"]) for entry in results}
    missing = [cell for cell in ((arm, contrast) for arm in arms for contrast in contrasts)
               if cell not in covered]
    if missing:
        raise LedgerError(f"{MATCHED}: the pooled family does not cover {missing}")
    counts = payload["matched_pair_counts"]["pooled"]
    support = ledger.declare_support(
        "matched_phenotype_pairs",
        f"{len(arms)} checkpoints on {counts['assay_pairs']} matched assay pairs over "
        f"{counts['independent_groups']} proteins and {counts['matched_substitution_rows']} "
        f"matched substitution rows; {pooled['aggregation']}; "
        f"{payload['within_protein_partition']}")
    resolved = [entry for entry in results
                if entry["resolved_positive"] or entry["resolved_negative"]]
    emit(MATCHED, ("pooled_inference", "results"),
         identifier="phenotype_breadth/matched_proteingym/headline_contrasts",
         claim="phenotype-specific contrasts the matched pair panel measures: the mean ranking "
               "increment over the two phenotypes, the difference between them, and the change in "
               "the two phenotypes' shared residual, for every scored checkpoint",
         family="phenotype_breadth", support=support, unit="contrasts", kind="support_count",
         value=len(results), level="likelihood")
    emit(MATCHED, ("pooled_inference", "results", "<resolved either way>"),
         identifier="phenotype_breadth/matched_proteingym/contrasts_resolving",
         claim="phenotype-specific contrasts that resolve away from zero under the matched panel's "
               "simultaneous band",
         family="phenotype_breadth", support=support, unit="contrasts", kind="count",
         value=len(resolved), level="likelihood",
         verdict="supported" if resolved else "unresolved")


# --------------------------------------------------------------------------- #
# The refitted stability grid: three abilities on one cohort.
# --------------------------------------------------------------------------- #

def _grid(ledger) -> dict:
    """The grid panel, with both declared rosters covering its columns exactly."""
    payload = ledger.artifacts.json(GRID)
    if payload["schema"] != "stability_ranking_error_panel_v2":
        raise LedgerError(f"{GRID}: not a stability_ranking_error_panel_v2 grid")
    arms = payload["arms"]
    declared = PROTEIN_SPECIALIZED + TEXT_AND_GALACTICA
    if len(set(declared)) != len(declared) or set(declared) != set(arms):
        raise LedgerError(
            f"{GRID}: the declared rosters and the panel's {len(arms)} columns are not the same "
            "set, so a count over the protein-specialized models and a bound over the text and "
            "Galactica ones would each be over an unstated subset")
    for name in sorted(GRID_UNITS):
        family = payload["families"][name]
        if family["status"] != "complete" or family["columns"] != arms:
            raise LedgerError(f"{GRID}: the {name} family is incomplete or carries another roster")
        if family["undefined_families"]:
            raise LedgerError(f"{GRID}: the {name} family leaves "
                              f"{len(family['undefined_families'])} families undefined")
    return payload


def _contrast(payload: dict, name: str) -> dict:
    """One contrast's columns, keyed by arm, with its resolved set recomputed.

    The panel states which arms resolved and also carries the bands they were
    read from. Recomputing the set from the bands and refusing on disagreement
    is what keeps a count of resolved models a reading rather than a repetition.
    """
    family = payload["families"][name]
    bands = family["bands"]
    columns = {}
    for index, arm in enumerate(family["columns"]):
        low, high = bands["interval"][index]
        columns[arm] = {"point": float(bands["point"][index]), "low": float(low),
                        "high": float(high), "verdict": family["verdicts"][arm]}
    recomputed = sorted(arm for arm, column in columns.items() if column["low"] > 0.0)
    if recomputed != sorted(family["resolved_above_zero"]):
        raise LedgerError(f"{GRID}: the {name} family states {family['resolved_above_zero']} as "
                          f"resolved above zero and its bands give {recomputed}")
    if any(columns[arm]["verdict"] != RESOLVED_ABOVE for arm in recomputed):
        raise LedgerError(f"{GRID}: the {name} family's verdicts and bands disagree")
    return columns


def _calibrated_baseline(ledger, payload: dict) -> float:
    """The matched baseline's own within-family calibrated squared error.

    This is the error the calibrated reading still makes before any model
    column enters, and it is the denominator of the share the manuscript
    quotes. The panel does not carry it; every per-arm fit does, as one
    per-family list per split seed under the combined-channel contrast, and the
    baseline is a property of the baseline design rather than of the arm. So
    all 33 fits are read and three conditions are required: the per-family list
    covers the panel's own group roster, every fit states the same level, and
    each fit's own group-equal mean of the calibrated increment reproduces that
    arm's published band point. The third is what ties the denominator to the
    numerator: without it the share could pair a level from one reading with an
    increment from another.
    """
    columns = _contrast(payload, "matched_calibrated_error")
    groups, seeds = payload["support"]["groups"], payload["support"]["split_seeds"]
    paths = sorted(glob.glob(os.path.join(ledger.artifacts.root, GRID_FITS, "fit_*.json")))
    relative = [os.path.relpath(path, ledger.artifacts.root) for path in paths]
    levels: dict[str, float] = {}
    for path in relative:
        fit = ledger.artifacts.json(path)
        arm = fit["arm"]
        if arm not in columns:
            raise LedgerError(f"{path}: {arm} is not a column of the grid")
        if sorted(fit["seeds"]) != sorted(str(seed) for seed in seeds):
            raise LedgerError(f"{path}: the fit does not cover the panel's split seeds")
        baseline, increment = [], []
        for seed in sorted(fit["seeds"]):
            combined = fit["seeds"][seed]["contrasts"]["combined"]
            if len(combined["groups"]) != groups:
                raise LedgerError(f"{path}: the {seed} fit carries "
                                  f"{len(combined['groups'])} families against the panel's "
                                  f"{groups}")
            baseline.append(fmean(combined["baseline_calibrated_error"]["matched"]))
            increment.append(fmean(combined["matched_calibrated_error"]))
        levels[arm] = fmean(baseline)
        if abs(fmean(increment) - columns[arm]["point"]) > 1e-9:
            raise LedgerError(
                f"{path}: the fit's group-equal calibrated increment {fmean(increment)!r} is not "
                f"the panel's published point {columns[arm]['point']!r} for {arm}, so the share "
                "would pair two readings")
    if sorted(levels) != sorted(columns):
        raise LedgerError(f"{GRID_FITS}: the fits cover {len(levels)} arms against the panel's "
                          f"{len(columns)} columns")
    if max(levels.values()) - min(levels.values()) > 1e-12:
        raise LedgerError(
            f"{GRID_FITS}: the matched calibrated baseline is not one level across the arms "
            f"({min(levels.values())!r} to {max(levels.values())!r}); it is the baseline's own "
            "error and a share of it is only defined if the arms agree on it")
    return fmean(levels.values())


def _grid_support(ledger, payload: dict) -> str:
    """The grid's measured support, declared once for every reading on it."""
    return ledger.declare_support(
        "stability_ranking_error_grid",
        f"{len(payload['arms'])} checkpoints on the MegaScale single-substitution cohort: "
        f"{payload['support']['groups']} wild-type family groups over "
        f"{payload['support']['sites']} sites and {payload['support']['rows']} rows, three split "
        f"seeds averaged inside the family before resampling; the endpoint is "
        f"{payload['endpoint']}; {payload['multiplicity']}")


def _stability_grid(ledger, emit) -> None:
    payload = _grid(ledger)
    support = _grid_support(ledger, payload)
    emit(GRID, ("arms",), identifier="stability_grid/arms",
         claim="checkpoints the refitted stability grid scores on every contrast",
         family="stability_grid", support=support, unit="checkpoints", kind="support_count",
         value=len(payload["arms"]), level="likelihood")
    emit(GRID, ("arms", "<protein-specialized or adapted>"),
         identifier="stability_grid/protein_specialized_or_adapted_arms",
         claim="checkpoints of those that are protein-specialized or protein-adapted, which is the "
               "set the beyond-profile stability survivors are counted against",
         family="stability_grid", support=support, unit="checkpoints", kind="support_count",
         value=len(PROTEIN_SPECIALIZED), level="likelihood")

    rankings = {name: _contrast(payload, name) for name in GRID_UNITS}
    _grid_counts_and_spans(emit, support, rankings)
    _grid_text_bound(emit, support, rankings["matched_ranking"])
    _grid_calibrated(ledger, emit, support, payload, rankings)


def _grid_counts_and_spans(emit, support: str, rankings: dict[str, dict]) -> None:
    """How many models each contrast resolves, and over what range."""
    readings = (
        ("matched_ranking", "ranking beyond the matched non-evolutionary baseline"),
        ("profile_ranking", "ranking beyond the profile-inclusive baseline, which is the "
                            "evolutionary comparator"),
        ("matched_calibrated_error", "within-family calibrated squared-error reduction beyond the "
                                     "matched non-evolutionary baseline"),
        ("matched_transfer_error", "transferred global-slope squared-error reduction beyond the "
                                   "matched non-evolutionary baseline, which is the unseen-family "
                                   "reading"),
        ("profile_transfer_error", "transferred global-slope squared-error reduction once the "
                                   "profile enters the baseline"),
    )
    for name, description in readings:
        columns = rankings[name]
        resolved = sorted(arm for arm, column in columns.items() if column["low"] > 0.0)
        emit(GRID, ("families", name, "resolved_above_zero"),
             identifier=f"stability_grid/{name}/models_resolving_above_zero",
             claim=f"checkpoints whose {description} resolves above zero under the contrast's own "
                   "33-column simultaneous band",
             family="stability_grid", support=support, unit="checkpoints", kind="count",
             value=len(resolved), level="likelihood",
             verdict="supported" if resolved else "unresolved")
        if resolved and name in ("matched_ranking", "profile_ranking"):
            _span(emit, GRID, ("families", name, "bands", "point", "<resolved above zero>"),
                  prefix=f"stability_grid/{name}/resolved_improvement",
                  claim=f"{description}, across the {len(resolved)} checkpoints that resolve above "
                        "zero",
                  family="stability_grid", support=support, unit=GRID_UNITS[name],
                  values={arm: columns[arm]["point"] for arm in resolved})
    half_widths = {arm: (column["high"] - column["low"]) / 2.0
                   for arm, column in rankings["profile_transfer_error"].items()}
    widest = max(half_widths, key=half_widths.get)
    emit(GRID, ("families", "profile_transfer_error", "bands", "interval", "<widest half-width>"),
         identifier="stability_grid/profile_transfer_error/largest_half_width",
         claim="widest simultaneous half-width among the 33 transferred-slope columns once the "
               f"profile enters the baseline ({widest}); the reading is unresolved and unbounded "
               "rather than absent",
         family="stability_grid", support=support, unit=SQUARED_KCAL, kind="range_bound",
         value=half_widths[widest], level="likelihood", verdict="unresolved")


def _grid_text_bound(emit, support: str, columns: dict) -> None:
    """What bounds the text and Galactica models' stability ranking from above."""
    unresolved = [arm for arm in TEXT_AND_GALACTICA
                  if columns[arm]["verdict"] == RESOLVED_ABOVE]
    if unresolved:
        raise LedgerError(f"{GRID}: {unresolved} resolve above zero, so the text and Galactica "
                          "models are no longer a bounded negative on stability ranking")
    bounds = {arm: columns[arm]["high"] for arm in TEXT_AND_GALACTICA}
    arm = max(bounds, key=bounds.get)
    emit(GRID, ("families", "matched_ranking", "bands", "interval",
                "<largest text and Galactica upper bound>"),
         identifier="stability_grid/matched_ranking/text_and_galactica_upper_bound",
         claim=f"largest simultaneous upper bound on a stability ranking improvement across the "
               f"{len(TEXT_AND_GALACTICA)} text and Galactica checkpoints ({arm}), which is what "
               "makes this the one well-bounded negative rather than an underpowered one",
         family="stability_grid", support=support, unit=RANKING_UNIT, kind="range_bound",
         value=bounds[arm], level="likelihood", verdict="unresolved")


def _grid_calibrated(ledger, emit, support: str, payload: dict,
                     rankings: dict[str, dict]) -> None:
    """The within-family calibrated reading, in the units it is readable in.

    A calibrated squared-error reduction is quoted two ways: as the share of the
    error that remained after calibration, and as the kcal/mol the best model
    takes off the root of it. Both rest on the baseline's own calibrated error,
    and both are stated only over the checkpoints that resolve --- which have to
    be the same checkpoints the ranking reading resolves, or "the same 12
    models" is not a statement about this panel.
    """
    calibrated = rankings["matched_calibrated_error"]
    resolved = sorted(arm for arm, column in calibrated.items() if column["low"] > 0.0)
    ranking = sorted(arm for arm, column in rankings["matched_ranking"].items()
                     if column["low"] > 0.0)
    if resolved != ranking:
        raise LedgerError(
            f"{GRID}: the calibrated reading resolves for {resolved} and the ranking reading for "
            f"{ranking}; they are not the same models and the claim that they are cannot be made")
    baseline = _calibrated_baseline(ledger, payload)
    _span(emit, GRID, ("families", "matched_calibrated_error", "bands", "point",
                       "<over the baseline's own calibrated error>"),
          prefix="stability_grid/matched_calibrated_error/resolved_share_of_remaining_error",
          claim=f"within-family calibrated squared-error reduction as a share of the error the "
                f"matched baseline still makes after the same calibration ({baseline:.4f} "
                f"{SQUARED_KCAL}, group-equal over the same family groups and one level for every "
                f"arm), across the {len(resolved)} checkpoints that resolve",
          family="stability_grid", support=support, unit="percent",
          values={arm: 100.0 * calibrated[arm]["point"] / baseline for arm in resolved})
    best = max(resolved, key=lambda arm: calibrated[arm]["point"])
    emit(GRID, ("families", "matched_calibrated_error", "bands", "point",
                "<best model, as a root-mean-square difference>"),
         identifier="stability_grid/matched_calibrated_error/best_model_error_drop",
         claim="within-family calibrated error the best-resolving checkpoint removes, as the "
               f"difference between the roots of the baseline's own calibrated squared error and "
               f"what remains of it ({best}); a squared reduction is not readable as an error",
         family="stability_grid", support=support, unit="kcal/mol", kind="range_bound",
         value=sqrt(baseline) - sqrt(baseline - calibrated[best]["point"]), level="likelihood")
    emit(GRID, ("families", "<matched_calibrated_error against matched_ranking>", "bands",
                "point"),
         identifier="stability_grid/calibrated_error_tracks_ranking",
         claim="Pearson correlation between the within-family calibrated squared-error reduction "
               "and the ranking improvement across the panel's 33 columns, which is how closely "
               "the calibrated reading tracks ranking",
         family="stability_grid", support=support, unit="dimensionless", kind="estimate",
         value=_pearson([calibrated[arm]["point"] for arm in payload["arms"]],
                        [rankings["matched_ranking"][arm]["point"] for arm in payload["arms"]]),
         reason="the panel resamples each column of each contrast and retains no interval for the "
                "agreement between two readings, so none is copied and none is fitted here",
         level="likelihood")


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise LedgerError("a correlation across the panel's columns needs both readings on every "
                          "column")
    mean_left, mean_right = fmean(left), fmean(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    spread = sqrt(sum((a - mean_left) ** 2 for a in left) *
                  sum((b - mean_right) ** 2 for b in right))
    if spread <= 0.0:
        raise LedgerError("a correlation is undefined when a reading does not vary")
    return covariance / spread


# --------------------------------------------------------------------------- #
# Domain abundance: the independent source the phenotype paragraph opens on.
# --------------------------------------------------------------------------- #

def _abundance_ranking(ledger, emit) -> None:
    """The within-domain abundance ranking improvements, over the models that hold.

    The claim is a range over the models whose increment is above zero at every
    split, which is the panel's own verdict rule. The verdict is read rather
    than recomputed from the three intervals, and the two have to agree.
    """
    payload = ledger.artifacts.json(ABUNDANCE)
    if payload["schema"] != "nested_gate_panel_v1" or not payload["roster_complete"]:
        raise LedgerError(f"{ABUNDANCE}: the abundance panel is not a complete nested gate panel")
    units = payload["effective_units"]
    support = ledger.declare_support(
        "domain_abundance_primary_ranking",
        f"{payload['arms_fitted']} checkpoints on the Domainome fitted support: {units['variants']}"
        f" substitutions over {units['domains']} domains in {units['groups']} family groups, "
        f"ranked within domain against the primary control set "
        f"({'+'.join(payload['qualified_control_set'])}), which already carries the evolutionary "
        f"profile; {payload['verdict_rule']}")
    improvements, resolved = {}, []
    for arm, block in sorted(payload["arms"].items()):
        cell = block["cells"]["primary_likelihood_spearman"]
        if not cell["licensed_for_a_verdict"]:
            raise LedgerError(f"{ABUNDANCE}: {arm}'s ranking cell carries no licensed verdict")
        above = [seed for seed, fold in sorted(cell["per_seed"].items())
                 if fold["excludes_zero"] and fold["point"] > 0.0]
        if (cell["verdict"] == "above zero") != (len(above) == len(cell["per_seed"])):
            raise LedgerError(f"{ABUNDANCE}: {arm}'s verdict {cell['verdict']!r} and its "
                              f"{len(above)} resolved splits disagree")
        improvements[arm] = float(cell["three_seed_mean"])
        if cell["verdict"] == "above zero":
            resolved.append(arm)
    if not resolved:
        raise LedgerError(f"{ABUNDANCE}: no checkpoint improves abundance ranking at every split, "
                          "so there is no range to state")
    _span(emit, ABUNDANCE, ("arms", "<resolved at every split>", "cells",
                            "primary_likelihood_spearman", "three_seed_mean"),
          prefix="domainome/primary_ranking/resolved_improvement",
          claim=f"within-domain abundance ranking improvement over the primary control set, across "
                f"the {len(resolved)} checkpoints whose increment is above zero at every split",
          family="domainome", support=support, unit=RANKING_UNIT,
          values={arm: improvements[arm] for arm in resolved})


# --------------------------------------------------------------------------- #
# What the activity bound is worth: a precision, against an effect elsewhere.
# --------------------------------------------------------------------------- #

def _activity_precision_ratio(ledger, emit, cells: dict[tuple[str, str], dict]) -> None:
    """The activity band's half-width as a multiple of a beyond-profile gain.

    An upper bound says what a cell excludes, not what it could have found. The
    manuscript reads the activity bound against the only effect its own models
    are known to carry: their stability ranking gain beyond the evolutionary
    profile, from the grid above. The two panels are different supports, which
    is why this is a span of bounds and not an estimate, and the models have to
    be the ones that appear in both: an activity column and a resolved
    beyond-profile stability gain.
    """
    cell = cells[("activity_venus", "ranking")]
    panel, path = cell["panel"], cell["path"]
    widths = {entry["arm"]: (entry["simultaneous"][1] - entry["simultaneous"][0]) / 2.0
              for entry in panel["arm_results"]}
    if max(widths.values()) - min(widths.values()) > 1e-9:
        raise LedgerError(f"{path}: the simultaneous half-width differs across the scored columns "
                          f"({min(widths.values())!r} to {max(widths.values())!r}), so the cell "
                          "has no one half-width to quote")
    half_width = fmean(widths.values())
    gains = _contrast(_grid(ledger), "profile_ranking")
    shared = sorted(arm for arm in panel["arms"]
                    if arm in gains and gains[arm]["verdict"] == RESOLVED_ABOVE)
    if not shared:
        raise LedgerError(
            f"{GRID}: none of the activity cell's {len(panel['arms'])} scored models carries a "
            "resolved beyond-profile stability gain, so the activity bound cannot be read against "
            "one")
    _span(emit, path, ("inference", "simultaneous",
                       "<half-width over a beyond-profile stability gain>"),
          prefix="phenotype_breadth/activity_venus/ranking/half_width_over_stability_gain",
          claim="simultaneous half-width of the activity ranking cell as a multiple of the "
                "beyond-profile stability ranking gain the same checkpoint resolves in the "
                f"refitted grid, across the {len(shared)} checkpoints that carry both; the "
                "activity cell is underpowered against effects of that size rather than bounding "
                "them",
          family="phenotype_breadth", support=cell["support"], unit="dimensionless ratio",
          values={arm: half_width / gains[arm]["point"] for arm in shared})


# --------------------------------------------------------------------------- #
# The evolutionary profile: what it is worth, and what it absorbs.
# --------------------------------------------------------------------------- #

def _profile_comparator(ledger, emit) -> None:
    """What the profile is worth on each source, and how much it absorbs.

    Each source contributes one reading of the profile's own worth and one
    largest surviving likelihood increment over its own profile-inclusive
    baseline. The worths are emitted; the three shares are compared and only
    their smallest and largest member is emitted, because that pair is what the
    manuscript quotes and the middle one is quoted nowhere.
    """
    sources = {"ProteinGym": _proteingym_profile(ledger, emit),
               "Domainome abundance": _abundance_profile(ledger, emit),
               "folding stability": _stability_profile(ledger, emit)}
    _stability_absorbed_share(ledger, emit)
    _largest_increment_share(emit, sources)


def _local_context(ledger) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    """The ProteinGym measurement, with its roster and its fold seeds covered."""
    payload = ledger.artifacts.json(LOCAL_CONTEXT)
    if payload["schema_version"] != "d1_local_context_measurement_v1" or \
            payload["status"] != "admitted":
        raise LedgerError(f"{LOCAL_CONTEXT}: not an admitted local-context measurement")
    summaries = payload["summaries"]
    arms = tuple(sorted({key.split("/")[0] for key in summaries}))
    seeds = tuple(sorted({key.split("/")[1] for key in summaries}))
    declared = PROTEIN_SPECIALIZED + TEXT_AND_GALACTICA
    if set(arms) != set(declared):
        raise LedgerError(
            f"{LOCAL_CONTEXT}: the declared rosters and this cohort's {len(arms)} arms are not the "
            "same set, so an absorbed share over the protein-specialized models would be a share "
            "over an unstated subset")
    if seeds != tuple(sorted(str(seed) for seed in payload["fold_seeds"])) or \
            len(summaries) != len(arms) * len(seeds):
        raise LedgerError(f"{LOCAL_CONTEXT}: the measured cells are not every arm at every "
                          "declared fold seed")
    return payload, arms, seeds


def _proteingym_support(ledger, payload: dict) -> str:
    support = payload["support"]
    return ledger.declare_support(
        "proteingym_profile_contrast",
        f"the ProteinGym mutation cohort: {support['n_variants']} variants from "
        f"{support['n_assays']} assays in {support['n_families']} wild-type clusters at 50% "
        f"identity, held out by cluster at three split seeds; the profile contrast is read between "
        f"control designs differing in the mutation-local independent-site profile block alone, "
        f"and those designs carry no model quantity and are identical for every arm")


def _design_ranking(payload: dict, arms: tuple[str, ...], seeds: tuple[str, ...],
                    design: str) -> float:
    """One model-independent control design's held-out ranking, three-seed mean.

    The measurement states that every model-independent design is predicted
    bit-identically for every arm. That is what makes the design's ranking a
    property of the controls rather than of an arm, so it is checked here and a
    design that differs across arms is not read as one control reading.
    """
    if design not in payload["model_independent_designs"]:
        raise LedgerError(f"{LOCAL_CONTEXT}: {design} is not a model-independent design, so its "
                          "ranking is not the controls' own")
    field = f"{design}_spearman"
    per_arm = {arm: tuple(round(payload["summaries"][f"{arm}/{seed}"][field]["point"], 12)
                          for seed in seeds)
               for arm in arms}
    if len(set(per_arm.values())) != 1:
        raise LedgerError(f"{LOCAL_CONTEXT}: the {design} design's held-out ranking differs across "
                          "arms, so it is not one control reading")
    return fmean(payload["summaries"][f"{arms[0]}/{seed}"][field]["point"] for seed in seeds)


def _arm_increment(payload: dict, arm: str, seeds: tuple[str, ...],
                   design: str) -> tuple[float, bool]:
    """One arm's likelihood increment over a design, and whether it holds."""
    cells = [payload["summaries"][f"{arm}/{seed}"][f"increment_M_{design}"] for seed in seeds]
    point = fmean(float(cell["point"]) for cell in cells)
    return point, all(cell["excludes_zero"] and cell["point"] > 0.0 for cell in cells)


def _proteingym_profile(ledger, emit) -> dict:
    """The profile's worth on ProteinGym, and the share it absorbs there."""
    payload, arms, seeds = _local_context(ledger)
    support = _proteingym_support(ledger, payload)
    bare, with_profile = PROFILE_WORTH_DESIGNS
    worth = _design_ranking(payload, arms, seeds, with_profile) - \
        _design_ranking(payload, arms, seeds, bare)
    if worth <= 0.0:
        raise LedgerError(f"{LOCAL_CONTEXT}: the profile block does not raise the composition "
                          "control's held-out ranking, so it is worth nothing to divide by")
    emit(LOCAL_CONTEXT, ("summaries", "<model-independent designs>",
                         f"<{with_profile} minus {bare}>", "point"),
         identifier="evolutionary_profile/proteingym/own_worth",
         claim="what the fitted independent-site profile block is worth on its own on ProteinGym: "
               "the held-out mutation-ranking increment it adds to the composition control, as the "
               "three-seed mean of two model-independent control designs; no local window is in "
               "either",
         family="evolutionary_profile", support=support, unit=RANKING_UNIT, kind="estimate",
         value=worth, level="control",
         reason="the measurement carries a cluster-bootstrap interval for each design's own "
                "ranking and none for the difference between two designs, so none is copied and "
                "none is fitted here")

    held_bare, held_profile = WINDOW_HELD_DESIGNS
    shares = {}
    for arm in PROTEIN_SPECIALIZED:
        before, _ = _arm_increment(payload, arm, seeds, held_bare)
        after, _ = _arm_increment(payload, arm, seeds, held_profile)
        if before <= 0.0:
            raise LedgerError(f"{LOCAL_CONTEXT}: {arm}'s likelihood increment over "
                              f"{held_bare} is not positive, so no share of it is absorbed")
        shares[arm] = 100.0 * (1.0 - after / before)
    _span(emit, LOCAL_CONTEXT, ("summaries", "<protein-specialized or adapted>",
                                f"<increment_M_{held_profile} over increment_M_{held_bare}>"),
          prefix="evolutionary_profile/proteingym/absorbed_share_window_held",
          claim=f"share of the ProteinGym likelihood increment the profile block absorbs with the "
                f"qualified local window held fixed in both baselines, across the "
                f"{len(PROTEIN_SPECIALIZED)} protein-specialized or adapted checkpoints",
          family="evolutionary_profile", support=support, unit="percent", values=shares)

    best, increment = max(
        ((arm, _arm_increment(payload, arm, seeds, held_profile)) for arm in arms),
        key=lambda entry: entry[1][0])
    if not increment[1]:
        raise LedgerError(f"{LOCAL_CONTEXT}: the largest likelihood increment over "
                          f"{held_profile} is {best}'s and it does not hold at every split, so it "
                          "is not an increment this section reports")
    return {"worth": worth, "support": support, "increment": increment[0], "arm": best,
            "path": LOCAL_CONTEXT, "baseline": "profile- and window-inclusive",
            "pointer": ("summaries", f"<{best}>", f"<increment_M_{held_profile} over the "
                        "profile's own worth>", "point")}


def _ladder_entry(ledger, path: str, source: str, partner_digest: str) -> tuple[dict, dict, float]:
    """One gate ladder's profile entry, on the cohort its partner panel reads.

    The ladder and the panel it is quoted beside have to be the same cohort, or
    a contribution measured on one population would be divided into an
    increment measured on another.
    """
    payload = ledger.artifacts.json(path)
    if payload["cohort_sha256"] != partner_digest:
        raise LedgerError(f"{path}: the ladder's cohort is not the panel's, so the profile's worth "
                          "and the increment over it are two populations")
    block = LADDER_PROFILE_BLOCK[source]
    entries = {entry["candidate"]: entry for entry in payload["ladder"]}
    if block not in entries:
        raise LedgerError(f"{path}: the ladder offers {sorted(entries)} and not {block}, so the "
                          "profile block has no own contribution here")
    entry = entries[block]
    if block in entry["standing_set"]:
        raise LedgerError(f"{path}: {block} is already in the set it is offered against")
    seeds = sorted(str(seed) for seed in payload["split_seeds"])
    increments = entry["spearman_increment"]
    if sorted(increments) != seeds:
        raise LedgerError(f"{path}: {block}'s ranking contribution does not cover the gate's "
                          "split seeds")
    points = []
    for seed in seeds:
        cell = increments[seed]
        low, high = cell["interval"]
        if not low <= cell["point"] <= high:
            raise LedgerError(f"{path}: {block}'s interval at {seed} does not bracket its point")
        points.append(float(cell["point"]))
    worth = fmean(points)
    if worth <= 0.0:
        raise LedgerError(f"{path}: {block} does not raise the standing set's held-out ranking, so "
                          "it is worth nothing to divide by")
    return payload, entry, worth


def _ladder_worth_row(ledger, emit, *, path: str, source: str, name: str, reading: str,
                      support: str, payload: dict, entry: dict, worth: float) -> None:
    block = LADDER_PROFILE_BLOCK[source]
    emit(path, ("ladder", f"<{block}>", "spearman_increment", "<three-seed mean>", "point"),
         identifier=f"evolutionary_profile/{source}/own_worth",
         claim=f"what the fitted independent-site profile block ({block}) is worth on its own on "
               f"{name}: its held-out {reading} increment over the standing "
               f"{'+'.join(entry['standing_set'])} control set, as the three-seed mean of the gate "
               f"ladder's own contribution; the gate "
               f"{'kept' if entry['qualified'] else 'discarded'} it under its squared-error "
               "qualification rule",
         family="evolutionary_profile", support=support, unit=RANKING_UNIT, kind="estimate",
         value=worth, level="control",
         reason="the ladder carries a group-bootstrap interval for each split seed's own "
                "contribution and none for their mean; split seeds are a sensitivity and not "
                "replicates, so no interval is copied onto the mean")


def _abundance_profile(ledger, emit) -> dict:
    """The profile's worth on Domainome, and the increment that survives it."""
    panel = ledger.artifacts.json(ABUNDANCE)
    payload, entry, worth = _ladder_entry(ledger, ABUNDANCE_LADDER, "abundance",
                                          panel["cohort_sha256"])
    units = payload["effective_units"]
    support = ledger.declare_support(
        "domain_abundance_profile_contrast",
        f"the Domainome cohort's control ladder, on the cohort digest its fitted panel reads: "
        f"{units['variants']} substitutions over {units['domains']} domains in {units['groups']} "
        f"family groups, held out by group at three split seeds; a candidate block's own "
        f"within-domain ranking contribution over the standing set it is offered against, and the "
        f"likelihood increment over the full qualified set, on that one cohort")
    _ladder_worth_row(ledger, emit, path=ABUNDANCE_LADDER, source="abundance",
                      name="Domainome abundance", reading="within-domain abundance ranking",
                      support=support, payload=payload, entry=entry, worth=worth)
    best, increment = None, 0.0
    for arm, block in sorted(panel["arms"].items()):
        cell = block["cells"]["primary_likelihood_spearman"]
        if cell["three_seed_mean"] > increment:
            best, increment, verdict = arm, float(cell["three_seed_mean"]), cell["verdict"]
    if best is None or verdict != "above zero":
        raise LedgerError(f"{ABUNDANCE}: the largest abundance ranking increment is {best}'s and "
                          "its verdict is not above zero, so it is not an increment this section "
                          "reports")
    return {"worth": worth, "support": support, "increment": increment, "arm": best,
            "path": ABUNDANCE, "baseline": "full qualified control set",
            "pointer": ("arms", f"<{best}>", "cells", "primary_likelihood_spearman",
                        "<three_seed_mean over the profile's own worth>")}


def _stability_profile(ledger, emit) -> dict:
    """The profile's worth on folding stability, and what survives it there."""
    grid = _grid(ledger)
    payload, entry, worth = _ladder_entry(ledger, STABILITY_LADDER, "stability",
                                          grid["support"]["cohort_sha256"])
    support = ledger.declare_support(
        "stability_profile_contrast",
        f"the MegaScale single-substitution cohort's frozen control ladder, on the cohort digest "
        f"the refitted grid reads: {payload['rows']} rows over {payload['sites']} sites in "
        f"{payload['groups']} family groups, held out by group at three split seeds; a candidate "
        f"block's own within-background ranking contribution over the standing set it is offered "
        f"against, and the likelihood increment over the profile-inclusive design, on that one "
        f"cohort")
    _ladder_worth_row(ledger, emit, path=STABILITY_LADDER, source="stability",
                      name="folding stability", reading="within-background stability ranking",
                      support=support, payload=payload, entry=entry, worth=worth)
    columns = _contrast(grid, "profile_ranking")
    best = max(columns, key=lambda arm: columns[arm]["point"])
    if columns[best]["verdict"] != RESOLVED_ABOVE:
        raise LedgerError(f"{GRID}: the largest beyond-profile stability increment is {best}'s and "
                          "it does not resolve above zero, so it is not an increment this section "
                          "reports")
    return {"worth": worth, "support": support, "increment": columns[best]["point"], "arm": best,
            "path": GRID, "baseline": "profile-inclusive design",
            "pointer": ("families", "profile_ranking", "bands", "point",
                        f"<{best} over the profile's own worth>")}


def _stability_absorbed_share(ledger, emit) -> None:
    """How much of the stability ranking gain the profile-inclusive design takes.

    The grid's profile-inclusive design adds the independent-site profile block
    and composition over the matched baseline, so the claim names both: on this
    endpoint the frozen ladder measures composition's own ranking contribution
    at a fiftieth of the profile block's, but the grid carries no
    composition-only design to decompose the share with.
    """
    payload = _grid(ledger)
    support = _grid_support(ledger, payload)
    matched = _contrast(payload, "matched_ranking")
    profile = _contrast(payload, "profile_ranking")
    resolved = sorted(arm for arm, column in matched.items() if column["low"] > 0.0)
    total = sum(matched[arm]["point"] for arm in resolved)
    if not resolved or total <= 0.0:
        raise LedgerError(f"{GRID}: the resolved stability ranking gain is not positive, so no "
                          "share of it is absorbed")
    kept = sum(profile[arm]["point"] for arm in resolved)
    emit(GRID, ("families", "<profile_ranking over matched_ranking>", "bands", "point",
                "<resolved above zero>"),
         identifier="evolutionary_profile/stability/absorbed_share",
         claim=f"share of the stability ranking gain the profile-inclusive design absorbs, as one "
               f"minus the ratio of the two designs' summed increments over the {len(resolved)} "
               f"checkpoints that resolve beyond the matched baseline; the design adds the "
               f"independent-site profile block and composition, and the grid carries no "
               f"composition-only design to separate them",
         family="evolutionary_profile", support=support, unit="percent", kind="ratio",
         value=100.0 * (1.0 - kept / total), level="likelihood")


def _largest_increment_share(emit, sources: dict[str, dict]) -> None:
    """The largest surviving increment as a share of the profile's own worth.

    One share per source, each a quantity of that source alone: the largest
    likelihood increment any checkpoint holds over that source's own
    profile-inclusive baseline, divided by what the profile block is worth
    there. The span over the three is what the manuscript quotes, so each end
    carries the support and the artifact of the source it comes from.
    """
    shares = {name: 100.0 * info["increment"] / info["worth"]
              for name, info in sources.items()}
    if len(shares) < 3:
        raise LedgerError("the span over the phenotype sources needs all three of them")
    for bound, name in (("min", min(shares, key=shares.get)),
                        ("max", max(shares, key=shares.get))):
        info = sources[name]
        emit(info["path"], info["pointer"],
             identifier="evolutionary_profile/largest_increment_over_profile_worth/"
                        f"{bound}",
             claim=f"largest likelihood increment any checkpoint holds over the {info['baseline']} "
                   f"baseline on {name} ({info['arm']}), as a share of what the profile block is "
                   f"worth on that source; the {bound} of the three phenotype sources",
             family="evolutionary_profile", support=info["support"], unit="percent",
             kind="range_bound", value=shares[name], level="likelihood")
