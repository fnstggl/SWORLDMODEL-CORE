#!/usr/bin/env python3
"""Society bench: sample the compile stage N times and report the DISTRIBUTION.

Every prompt and schema change to the world compiler up to now has been judged by
eyeballing one run of a visibly stochastic process. The record of that method is in
`docs/LAUNCH_GAP_AUDIT.md`: eleven live compiles on one frozen OPEC+ store, entity
count swinging 0 to 11 across them, two prompt edits shipped and reverted on a single
read each. The agent that ran them said it plainly — *"my prompt edits are inside that
noise, which is itself the finding"*.

This harness exists so that stops. It runs the compile stage N times against a frozen
evidence store, holding question, store, cutoff, rules and schema fixed, and reports
what each society metric DID across those N runs — every observed value with its
count, never a mean. A mean of a bimodal scenery/over-compression split is a number
that describes neither mode, and this planner is visibly bimodal.

**The headline metric is `parties_holding_acts`** — how many distinct parties are named
as eligible actors on at least one compiled affordance. Across all eleven recorded runs
it has never once exceeded 1, while entity count swung wildly. Entity count is noise;
acting parties is the product.

WHAT IS MEASURED, AND WHAT IS CUT
---------------------------------
Scope is the COMPILE STAGE ONLY: the real planner and reviewer prompts, the real
parser, validator and lowerer, and every `compile_world` gate — the same functions
`scripts/semantic_slice.py` exercises, on the same frozen-store loader. What is cut is
research (the store is frozen and shared by every run, which is what makes the arms
comparable) and the simulation (`run_forecast`, whose 698-second / 17-call cost per
sample would make N=10 unaffordable and the instrument therefore unused).

The fidelity that costs: this bench cannot see anything the *runtime* does with a
world — whether the parties that hold affordances ever use them, whether a compiled
channel carries a message, whether the terminal resolves. It measures what was BUILT,
not what happened. `parties_holding_acts` is an upper bound on the parties that can
matter, never a claim that they did.

DETERMINISTIC WHERE IT CAN BE, HONEST WHERE IT CANNOT
-----------------------------------------------------
`semantic_compile.py` derives its planner seed from the question text, so every
production compile of one question asks with one seed. Repeating that N times samples
only whatever nondeterminism the provider has left over, and reports it as if it were
the planner's spread. So `--seed-mode vary` (the default) wraps the gateway and
re-derives each request's seed from `(bench_seed, run_index, task_kind, original
seed)`. The wrapper touches nothing inside the compiler; it substitutes a seed at the
boundary the architecture already provides. `--seed-mode production` reproduces the
shipped behaviour for comparison. Which mode ran is recorded and printed, because the
two measure different things.

Everything else is held fixed and hashed: question, store, cutoff, horizon, compiler
mode, gate setting, and a SHA-256 over the `sworldmodel` package actually imported. If
two runs in one configuration disagree on that hash, the compiler changed underneath
the bench and the report says so instead of averaging across two different systems.

USAGE
-----
    PYTHONPATH=src python3 scripts/society_bench.py \\
        --store artifacts/phase2/geopolitical2/evidence_store.json \\
        --question "Will OPEC+ announce an increase in crude oil production quotas ..." \\
        --as-of 2026-07-25T20:37:43+00:00 --horizon 2026-10-01T23:59:59+00:00 \\
        --runs 10 --label baseline --out /tmp/bench/baseline

    # compare two configurations that were each benched separately
    python3 scripts/society_bench.py --compare /tmp/bench/a/bench.json /tmp/bench/b/bench.json

To hold the compiler still while peers are editing `src/`, copy it first and point at
the copy — the bench records which package it imported either way:

    cp -r src /tmp/frozen_src && PYTHONPATH=/tmp/frozen_src python3 scripts/society_bench.py ...

READING A RESULT
----------------
The report ends with a NOISE section, and it is the point of the harness. It states,
for the N actually run, the smallest true rate the result rules out and the smallest
split a comparison arm would have to show before the difference is distinguishable
from chance. Both are exact — hypergeometric and binomial tails from `math.comb`, no
stats library, checkable by hand for small N. When N cannot separate two
configurations, the harness says that in those words rather than printing two numbers
and leaving a reader to infer a result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
# Only fall back to this working tree's `src/` when nothing else supplies the package.
# A bench run against a frozen copy (`PYTHONPATH=/tmp/frozen_src`) must measure THAT
# compiler; unconditionally inserting the repo at position 0 would silently measure the
# live tree instead, which is exactly the confusion this harness exists to end.
if importlib.util.find_spec("sworldmodel") is None and str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

# Bind `sworldmodel` BEFORE `_store_loader`, which re-inserts this repo's `src/` at the
# front of sys.path when it loads and imports `sworldmodel.evidence` on the way. Whoever
# imports the package first decides which copy every later import resolves against, so a
# frozen-src bench has to win that race here. Written as a call rather than an `import`
# statement on purpose: an import statement here gets sorted below `_store_loader`, and
# the ordering IS the mechanism.
importlib.import_module("sworldmodel")

from _store_loader import load_store, prepare_run_dir  # noqa: E402

from sworldmodel.errors import GatewayError, SWorldModelError  # noqa: E402
from sworldmodel.evidence import EvidenceStore  # noqa: E402
from sworldmodel.gateway import GatewayRequest, GatewayResponse, ModelGateway  # noqa: E402
from sworldmodel.models import ResolutionContract  # noqa: E402
from sworldmodel.research import assemble_bundle  # noqa: E402
from sworldmodel.semantic_compile import semantic_compile_live  # noqa: E402
from sworldmodel.world_compiler import compile_world  # noqa: E402

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# The headline. Every other number here is context for it.
HEADLINE = "parties_holding_acts"

# The threshold that separates a society from a switch. One party holding every
# affordance is `phase2/geopolitical2`: nine entities, one action, eight of them set
# dressing. Two is the least that can disagree.
SOCIETY_THRESHOLD = 2

# Metrics read off the LOWERED world spec — what was actually built.
SPEC_METRICS = (
    "entities",
    HEADLINE,
    "declared_actors",
    "affordances",
    "send_effects",
    "channels",
    "process_nodes",
)
# Metrics read off the SEMANTIC PLAN — what the planner said before lowering. Kept
# beside the lowered numbers because they are what a reader inspecting a plan by hand
# sees, and they do not always agree: a plan can declare seven deciders and lower to
# one party holding every act.
PLAN_METRICS = ("plan_entities", "plan_deciders", "plan_affordances", "plan_send_changes")

ALL_METRICS = SPEC_METRICS + PLAN_METRICS

# Outcomes. A gate refusal still leaves a compiled world to measure, and that is the
# most informative case there is — so membership in the distributions is decided by
# whether a world came out (`RunRecord.metrics` is non-empty), never by the outcome
# label. A provider outage during the gate stage loses only the verdict; the world it
# already built is real and still counts.
OUTCOMES = ("compiled", "refused_gates", "refused_plan", "harness_error")


# ---------------------------------------------------------------------------
# Configuration and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchSpec:
    """One configuration. Everything here is held fixed across the N runs in an arm,
    and everything that could make two arms incomparable is hashed into the report."""

    label: str
    question: str
    store_path: str
    as_of: datetime
    horizon: datetime
    mode: str = "semantic"
    gates: bool = True
    seed_mode: str = "vary"
    bench_seed: int = 0
    max_calls: int | None = 40

    def identity(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "question_sha256": _sha256_text(self.question),
            "store_sha256": _sha256_path(Path(self.store_path)),
            "store_path": self.store_path,
            "as_of": self.as_of.isoformat(),
            "horizon": self.horizon.isoformat(),
            "mode": self.mode,
            "gates": self.gates,
            "seed_mode": self.seed_mode,
            "bench_seed": self.bench_seed,
            "max_calls_per_run": self.max_calls,
        }


@dataclass
class RunRecord:
    """One compile. `metrics` is empty exactly when no world was produced."""

    run_index: int
    seed_salt: int
    outcome: str
    failure: str = ""
    stage: str = ""
    message: str = ""
    metrics: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    calls_by_kind: dict[str, int] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    compiler_sha256: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "seed_salt": self.seed_salt,
            "outcome": self.outcome,
            "failure": self.failure,
            "stage": self.stage,
            "message": self.message[:400],
            "metrics": dict(self.metrics),
            "llm_calls": self.llm_calls,
            "calls_by_kind": dict(self.calls_by_kind),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "seconds": round(self.seconds, 1),
            "compiler_sha256": self.compiler_sha256,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> RunRecord:
        return RunRecord(
            run_index=int(raw.get("run_index", 0)),
            seed_salt=int(raw.get("seed_salt", 0)),
            outcome=str(raw.get("outcome", "harness_error")),
            failure=str(raw.get("failure", "")),
            stage=str(raw.get("stage", "")),
            message=str(raw.get("message", "")),
            metrics={k: int(v) for k, v in (raw.get("metrics") or {}).items()},
            llm_calls=int(raw.get("llm_calls", 0)),
            calls_by_kind=dict(raw.get("calls_by_kind") or {}),
            tokens_in=int(raw.get("tokens_in", 0)),
            tokens_out=int(raw.get("tokens_out", 0)),
            seconds=float(raw.get("seconds", 0.0)),
            compiler_sha256=str(raw.get("compiler_sha256", "")),
        )


# ---------------------------------------------------------------------------
# Input hashing — so a reader can tell two configurations apart
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def compiler_fingerprint() -> tuple[str, str, int]:
    """SHA-256 over every ``.py`` in the ``sworldmodel`` package actually imported.

    Not the git commit: a working tree with uncommitted edits reports the same commit
    as a clean one, and this branch has ten agents editing it. The hash is over file
    contents, in sorted path order, so two arms compiled by different code cannot be
    silently compared.
    """

    import sworldmodel

    root = Path(sworldmodel.__file__).resolve().parent
    files = sorted(p for p in root.glob("*.py"))
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.name.encode("utf-8"))
        digest.update(p.read_bytes())
    return digest.hexdigest(), str(root), len(files)


# ---------------------------------------------------------------------------
# The seed-varying gateway wrapper
# ---------------------------------------------------------------------------


class SeedVaryingGateway(ModelGateway):
    """Delegates to an inner gateway, substituting a per-run seed on every request.

    The compiler pins its planner seed to the question text
    (``semantic_compile.py``: ``seed=int(prompt_hash(f"semantic{question}{attempt}")…)``),
    which is right for a production run — one question, one draw — and wrong for a
    bench, which needs N draws from the same configuration. This wrapper sits at the
    gateway boundary and changes nothing inside the compiler.

    The substituted seed is derived, not random: given the same ``salt`` the same run
    is requested again, so a bench invocation is reproducible to whatever extent the
    provider honours seeds at all. Accounting is read off THIS object; the inner
    gateway's counters are not consulted.
    """

    def __init__(self, inner: ModelGateway, salt: int) -> None:
        super().__init__()
        self._inner = inner
        self.salt = salt
        self.is_live = inner.is_live

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def derive_seed(self, request: GatewayRequest) -> int:
        material = f"{self.salt}:{request.task_kind}:{request.seed}"
        return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        return self._inner.generate(replace(request, seed=self.derive_seed(request)))


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def society_metrics(compilation: dict[str, Any]) -> dict[str, int]:
    """Read the society metrics off a compilation. Pure; no provider, no I/O.

    ``parties_holding_acts`` is the number of DISTINCT entity ids named in at least one
    action's ``eligible_actors``. That is deliberately not "entities marked is_actor"
    and not "len(actors)": `phase2/geopolitical2` declared one actor and nine entities,
    and the question is not who was labelled a decider but who was handed something to
    do. An action with no eligible actor is environment-driven and contributes nobody.
    """

    spec = compilation.get("world_spec") or {}
    entities = spec.get("entities") or []
    actions = spec.get("actions") or []

    holders: set[str] = set()
    send_effects = 0
    for action in actions:
        for actor_id in action.get("eligible_actors") or []:
            if str(actor_id).strip():
                holders.add(str(actor_id))
        for effect in action.get("effects") or []:
            if str(effect.get("op") or "") == "deliver_information":
                send_effects += 1

    process = spec.get("process") or {}
    plan = ((compilation.get("_semantic") or {}).get("plan")) or {}
    plan_entities = plan.get("entities") or []
    plan_affordances = plan.get("affordances") or []
    plan_sends = sum(
        1
        for a in plan_affordances
        for c in (a.get("changes") or [])
        if str(c.get("op") or "") == "send"
    )

    return {
        "entities": len(entities),
        HEADLINE: len(holders),
        "declared_actors": len(spec.get("actors") or []),
        "affordances": len(actions),
        "send_effects": send_effects,
        "channels": len(spec.get("channels") or []),
        "process_nodes": len(process.get("nodes") or []),
        "plan_entities": len(plan_entities),
        "plan_deciders": sum(1 for e in plan_entities if e.get("decides") is True),
        "plan_affordances": len(plan_affordances),
        "plan_send_changes": plan_sends,
    }


def _failure_code(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        code = details.get("failure")
        if code:
            return str(code)
    return type(exc).__name__


def _gateway_failure_code(exc: Exception) -> str:
    """Separate an exhausted bench ceiling from an unreachable provider.

    They need opposite responses — raise ``--max-calls`` versus wait for the endpoint —
    and both would otherwise read as one anonymous ``GatewayError`` in the report.
    """

    return "gateway_budget_exhausted" if "budget exhausted" in str(exc) else "gateway_error"


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def run_once(
    gateway: ModelGateway,
    spec: BenchSpec,
    store: EvidenceStore,
    *,
    run_index: int,
    seed_salt: int,
    out_dir: Path | None = None,
) -> RunRecord:
    """Compile once and record what came out. Never raises for a compile failure —
    a refusal is a measurement, not an error. Only a bench-side fault becomes
    ``harness_error``, and it is counted separately so it can never be read as a
    property of the compiler."""

    view = store.view(spec.as_of)
    if spec.max_calls is not None:
        gateway.set_budget(max_calls=spec.max_calls)
    started = time.monotonic()
    fingerprint, _root, _n = compiler_fingerprint()
    record = RunRecord(
        run_index=run_index,
        seed_salt=seed_salt,
        outcome="harness_error",
        compiler_sha256=fingerprint,
    )

    def finish(rec: RunRecord) -> RunRecord:
        rec.seconds = time.monotonic() - started
        rec.llm_calls = len(gateway.calls)
        by_kind: dict[str, int] = {}
        for call in gateway.calls:
            by_kind[call.task_kind] = by_kind.get(call.task_kind, 0) + 1
        rec.calls_by_kind = by_kind
        rec.tokens_in = gateway.total_tokens_in
        rec.tokens_out = gateway.total_tokens_out
        if out_dir is not None:
            (out_dir / "run_record.json").write_text(json.dumps(rec.to_json(), indent=1))
        return rec

    # --- stage 1: plan, review, validate, lower -----------------------------------
    try:
        if spec.mode == "direct":
            from sworldmodel.world_compiler import compile_world_spec_live

            compilation, _resp = compile_world_spec_live(
                gateway, spec.question, spec.as_of, spec.horizon, view
            )
        else:
            compilation, _resp = semantic_compile_live(
                gateway, spec.question, spec.as_of, spec.horizon, view
            )
    except GatewayError as exc:
        # BEFORE SWorldModelError, which GatewayError subclasses. A provider outage or
        # an exhausted call budget is OUR infrastructure stopping, not the compiler
        # judging a world — filing it as a refusal would put an unreachable endpoint
        # into the same column as `coverage_incomplete` and quietly shrink the
        # denominator the headline rate is computed over.
        record.outcome = "harness_error"
        record.stage = "plan"
        record.failure = _gateway_failure_code(exc)
        record.message = str(exc)
        return finish(record)
    except SWorldModelError as exc:
        record.outcome = "refused_plan"
        record.stage = "plan"
        record.failure = _failure_code(exc)
        record.message = str(exc)
        return finish(record)
    except Exception as exc:  # noqa: BLE001 — a bench fault is never a compiler result
        record.outcome = "harness_error"
        record.stage = "plan"
        record.failure = type(exc).__name__
        record.message = str(exc)
        return finish(record)

    record.metrics = society_metrics(compilation)
    if out_dir is not None:
        (out_dir / "semantic_plan.json").write_text(
            json.dumps((compilation.get("_semantic") or {}).get("plan"), indent=1, default=str)
        )
        (out_dir / "lowered_compilation.json").write_text(
            json.dumps(
                {k: v for k, v in compilation.items() if k != "_semantic"}, indent=1, default=str
            )
        )

    if not spec.gates:
        # --no-gates buys back the exclusion-challenge calls at the cost of the
        # refusal code: a world measured here may be one no gate would admit.
        record.outcome = "compiled"
        record.stage = "plan_only"
        return finish(record)

    # --- stage 2: assemble + every compile_world gate ------------------------------
    try:
        bundle = assemble_bundle(
            store,
            {
                "world_spec": compilation["world_spec"],
                "uncertainties": compilation["uncertainties"],
                "world_facts": compilation["world_facts"],
                "required_reality_facts": compilation["required_reality_facts"],
                "reality": {
                    "subject_entity": compilation.get("subject_entity"),
                    "resolution_units": compilation.get("resolution_units"),
                    "target_outcome": compilation.get("target_outcome"),
                    "expected_participants": compilation.get("expected_participants"),
                    "as_of": spec.as_of.isoformat(),
                    "horizon": spec.horizon.isoformat(),
                },
            },
        )
        contract = ResolutionContract(
            question=spec.question,
            as_of=spec.as_of,
            horizon=spec.horizon,
            subject_entity=bundle.subject_entity,
            resolution_units=bundle.resolution_units,
            terminal=bundle.spec.terminal,
            target_outcome=bundle.target_outcome,
            expected_participants=bundle.expected_participants,
        )
        compile_world(
            contract,
            view,
            bundle.spec,
            bundle.uncertainties,
            bundle.world_facts,
            gateway=gateway,
            seed=seed_salt if spec.seed_mode == "vary" else 0,
            max_branches=4,
        )
    except GatewayError as exc:
        # Same rule as stage 1, and it matters more here: the world WAS built, so its
        # metrics are already recorded and stay recorded. Only the gate verdict is
        # missing, and the reason it is missing is us.
        record.outcome = "harness_error"
        record.stage = "gates"
        record.failure = _gateway_failure_code(exc)
        record.message = str(exc)
        return finish(record)
    except SWorldModelError as exc:
        record.outcome = "refused_gates"
        record.stage = "gates"
        record.failure = _failure_code(exc)
        record.message = str(exc)
        return finish(record)
    except Exception as exc:  # noqa: BLE001
        record.outcome = "harness_error"
        record.stage = "gates"
        record.failure = type(exc).__name__
        record.message = str(exc)
        return finish(record)

    record.outcome = "compiled"
    record.stage = "gates"
    return finish(record)


# ---------------------------------------------------------------------------
# Exact statistics — stdlib only, hand-checkable
# ---------------------------------------------------------------------------


def binomial_upper_bound(k: int, n: int, conf: float = 0.95) -> float:
    """The largest true rate p for which seeing at most k of n has probability >= 1-conf.

    Clopper-Pearson's one-sided upper limit, found by bisection on the exact binomial
    tail, so no distribution table and no library is involved. For k=0 this is the
    number that matters: the closed form is 1 - (1-conf)^(1/n), and at n=10, conf=0.95
    it is 0.259 — a configuration that produced the event a quarter of the time is
    entirely consistent with ten straight failures to see it.
    """

    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - conf

    def at_most_k(p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1))

    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if at_most_k(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def binomial_lower_bound(k: int, n: int, conf: float = 0.95) -> float:
    """Mirror of :func:`binomial_upper_bound`: the smallest p consistent with >= k of n."""

    if n <= 0 or k <= 0:
        return 0.0
    return 1.0 - binomial_upper_bound(n - k, n, conf)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Arm A saw the event a of a+b times; arm B saw it c of c+d times. Under the null
    that both arms draw from one distribution, this sums the hypergeometric
    probability of every table at least as unlikely as the observed one. All integer
    arithmetic through :func:`math.comb`; for the small N a bench can afford, a reader
    can reproduce it with a calculator.
    """

    n1, n2, total = a + b, c + d, a + b + c + d
    successes = a + c
    if n1 == 0 or n2 == 0 or successes == 0 or successes == total:
        return 1.0
    denom = math.comb(total, successes)

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, successes - x) / denom

    observed = prob(a)
    lo = max(0, successes - n2)
    hi = min(n1, successes)
    # A tolerance is needed because floating point makes the mirrored table of a
    # symmetric split compare as very slightly larger and drop out of the sum.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def minimum_detectable_count(n_a: int, n_b: int, *, baseline: int = 0, alpha: float = 0.05) -> int:
    """How many of n_b a comparison arm must show before the difference is real.

    Given this arm saw the event ``baseline`` of ``n_a`` times, returns the smallest k
    such that k of ``n_b`` in another arm reaches ``alpha`` on the two-sided Fisher
    exact test. Returns ``n_b + 1`` when no split at this N can get there — which is
    the honest answer for a small bench and the one a reader most needs.
    """

    for k in range(baseline, n_b + 1):
        if fisher_exact_two_sided(baseline, n_a - baseline, k, n_b - k) <= alpha:
            return k
    return n_b + 1


def permutation_p(xs: list[int], ys: list[int], *, cap: int = 2_000_000) -> float | None:
    """Exact two-sided permutation p for a difference in means, or None if too big.

    Enumerates every way to split the pooled values into groups of the observed sizes
    and counts how many give a mean difference at least as large as the one observed.
    No distributional assumption, no library. Returns None rather than an approximation
    when the enumeration would exceed ``cap``: a p-value from a shortcut nobody named
    is worse than no p-value.
    """

    import itertools

    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return None
    if math.comb(n1 + n2, n1) > cap:
        return None
    pooled = xs + ys
    total = sum(pooled)
    observed = abs(sum(xs) / n1 - sum(ys) / n2)
    extreme = 0
    count = 0
    for combo in itertools.combinations(range(n1 + n2), n1):
        s = sum(pooled[i] for i in combo)
        count += 1
        if abs(s / n1 - (total - s) / n2) >= observed - 1e-12:
            extreme += 1
    return extreme / count


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def distribution(values: list[int]) -> dict[str, Any]:
    """Every observed value with its count, plus the order statistics. Never a bare
    mean: a mean of a bimodal split describes neither mode."""

    if not values:
        return {"n": 0, "counts": {}, "min": None, "median": None, "max": None, "mean": None}
    counts = Counter(values)
    return {
        "n": len(values),
        "counts": {str(k): counts[k] for k in sorted(counts)},
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
        "distinct": len(counts),
    }


def aggregate(spec: BenchSpec, records: list[RunRecord]) -> dict[str, Any]:
    """Everything the report needs, as plain JSON. Pure; no provider, no I/O."""

    produced = [r for r in records if r.metrics]
    outcomes = Counter(r.outcome for r in records)
    failures = Counter(r.failure for r in records if r.failure)
    fingerprints = sorted({r.compiler_sha256 for r in records if r.compiler_sha256})

    metrics: dict[str, Any] = {}
    for name in ALL_METRICS:
        metrics[name] = distribution([r.metrics[name] for r in produced if name in r.metrics])

    headline_values = [r.metrics[HEADLINE] for r in produced if HEADLINE in r.metrics]
    society_count = sum(1 for v in headline_values if v >= SOCIETY_THRESHOLD)
    n_headline = len(headline_values)

    return {
        "config": spec.identity(),
        "runs_attempted": len(records),
        "worlds_produced": len(produced),
        "outcomes": {name: outcomes.get(name, 0) for name in OUTCOMES},
        "failures": dict(sorted(failures.items(), key=lambda kv: (-kv[1], kv[0]))),
        "compiler_sha256": fingerprints[0] if len(fingerprints) == 1 else "",
        "compiler_sha256_disagreement": fingerprints if len(fingerprints) > 1 else [],
        "metrics": metrics,
        "headline": {
            "metric": HEADLINE,
            "threshold": SOCIETY_THRESHOLD,
            "n": n_headline,
            "at_or_above_threshold": society_count,
            "rate_upper_95": round(binomial_upper_bound(society_count, n_headline, 0.95), 4)
            if n_headline
            else None,
            "rate_lower_95": round(binomial_lower_bound(society_count, n_headline, 0.95), 4)
            if n_headline
            else None,
            "minimum_detectable_count": minimum_detectable_count(
                n_headline, n_headline, baseline=society_count
            )
            if n_headline
            else None,
        },
        "cost": {
            "llm_calls": sum(r.llm_calls for r in records),
            "tokens_in": sum(r.tokens_in for r in records),
            "tokens_out": sum(r.tokens_out for r in records),
            "wall_seconds_summed": round(sum(r.seconds for r in records), 1),
            "seconds_per_run_median": round(statistics.median([r.seconds for r in records]), 1)
            if records
            else 0.0,
            "calls_per_run_median": round(statistics.median([r.llm_calls for r in records]), 1)
            if records
            else 0.0,
        },
        "runs": [r.to_json() for r in records],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _bar(count: int, width: int = 28, scale: int = 1) -> str:
    return "#" * min(width, max(0, round(count * width / max(1, scale))))


def _histogram(name: str, dist: dict[str, Any], indent: str = "    ") -> list[str]:
    if not dist["n"]:
        return [f"{indent}{name:<24} no runs produced a world"]
    peak = max(dist["counts"].values())
    lines = [
        f"{indent}{name:<24} n={dist['n']}  min={dist['min']} median={dist['median']} "
        f"max={dist['max']}  (mean {dist['mean']} — do not quote this alone)"
    ]
    for value, count in dist["counts"].items():
        lines.append(f"{indent}  {value:>4} | {_bar(count, 28, peak):<28} {count}")
    return lines


def noise_statement(summary: dict[str, Any]) -> list[str]:
    """The paragraph the whole harness exists to print.

    States what this N can and cannot distinguish, in terms a reader can check, and
    errs toward saying less.
    """

    head = summary["headline"]
    n = int(head["n"] or 0)
    k = int(head["at_or_above_threshold"] or 0)
    metric, threshold = head["metric"], head["threshold"]
    if n == 0:
        return [
            "  No run produced a world, so this bench measured nothing about "
            f"{metric}. That is a result about the pipeline's ability to compile at "
            "all, not about the societies it compiles."
        ]

    upper = float(head["rate_upper_95"])
    mdc = int(head["minimum_detectable_count"])
    lines = [
        f"  N = {n} worlds produced. {metric} >= {threshold} in {k} of them.",
        f"  The true rate is not {k}/{n}. Exact binomial (Clopper-Pearson, one-sided "
        f"95%): any rate up to {upper:.2f} is consistent with this result, and only "
        f"rates above {upper:.2f} are ruled out.",
    ]
    if k == 0:
        lines.append(
            f"  So {k}/{n} does NOT mean impossible. A configuration that built a "
            f"{threshold}-party world {upper * 100:.0f}% of the time could show "
            f"{k} of {n} here by chance alone."
        )
    else:
        lower = float(head["rate_lower_95"])
        lines.append(
            f"  The symmetric reading of the lower end: rates below {lower:.2f} are ruled out."
        )
    if mdc > n:
        lines.append(
            f"  NOT ENOUGH RUNS TO COMPARE ANYTHING. At N={n} per arm, no split "
            "whatsoever reaches p<=0.05 on the two-sided Fisher exact test. Any "
            "difference this bench shows against another arm of the same size is "
            "indistinguishable from chance. Raise N before drawing a conclusion."
        )
    else:
        lines.append(
            f"  To beat this arm at p<=0.05 (Fisher exact, two-sided), a comparison "
            f"arm of N={n} must show {metric} >= {threshold} in at least {mdc} of its "
            f"{n} runs. {mdc - 1} of {n} or fewer is inside the noise at this N and "
            "must not be reported as an improvement."
        )
    lines.append(
        f"  This report shows {len(ALL_METRICS)} metrics. At p<=0.05 each, roughly one "
        "in twenty will look different by chance. Only "
        f"{metric} is the pre-registered result; every other number is descriptive."
    )
    return lines


def render_report(summary: dict[str, Any]) -> str:
    cfg = summary["config"]
    out: list[str] = []
    out.append("=" * 78)
    out.append(f"SOCIETY BENCH — configuration {cfg['label']!r}")
    out.append("=" * 78)
    out.append(f"  runs attempted        {summary['runs_attempted']}")
    out.append(f"  worlds produced       {summary['worlds_produced']}")
    out.append(f"  question sha256       {cfg['question_sha256'][:16]}")
    out.append(f"  store sha256          {cfg['store_sha256'][:16]}  ({cfg['store_path']})")
    out.append(f"  as_of / horizon       {cfg['as_of']}  ->  {cfg['horizon']}")
    out.append(
        f"  mode / gates / seeds  {cfg['mode']} / gates={cfg['gates']} / "
        f"seed-mode={cfg['seed_mode']} (bench-seed {cfg['bench_seed']})"
    )
    if summary["compiler_sha256_disagreement"]:
        out.append("  compiler sha256       *** DISAGREEMENT ACROSS RUNS ***")
        for h in summary["compiler_sha256_disagreement"]:
            out.append(f"                        {h[:16]}")
        out.append(
            "  THIS IS NOT ONE CONFIGURATION. src/sworldmodel changed while the bench "
            "was running; these runs measure two different compilers and must not be "
            "pooled. Re-run against a frozen copy of src."
        )
    else:
        out.append(f"  compiler sha256       {summary['compiler_sha256'][:16]}")
    out.append("")

    out.append("  OUTCOME")
    for name in OUTCOMES:
        count = summary["outcomes"][name]
        if count:
            out.append(f"    {name:<22} {_bar(count, 20, summary['runs_attempted']):<20} {count}")
    if summary["outcomes"]["harness_error"]:
        out.append(
            f"    ^ {summary['outcomes']['harness_error']} of "
            f"{summary['runs_attempted']} runs failed on OUR side (provider outage or "
            "an exhausted call ceiling). Those are not compiler results and are not "
            "counted as refusals; where one had already built a world, that world is "
            "still in the distributions below."
        )
    if summary["failures"]:
        out.append("  REFUSAL CODE")
        for code, count in summary["failures"].items():
            out.append(f"    {code:<22} {_bar(count, 20, summary['runs_attempted']):<20} {count}")
    out.append("")

    out.append(
        f"  DISTRIBUTIONS  (over the {summary['worlds_produced']} runs that produced a world)"
    )
    out.append(f"  --- headline: {HEADLINE} ---")
    out.extend(_histogram(HEADLINE, summary["metrics"][HEADLINE]))
    out.append("  --- lowered world spec ---")
    for name in SPEC_METRICS:
        if name != HEADLINE:
            out.extend(_histogram(name, summary["metrics"][name]))
    out.append("  --- semantic plan (before lowering) ---")
    for name in PLAN_METRICS:
        out.extend(_histogram(name, summary["metrics"][name]))
    out.append("")

    cost = summary["cost"]
    out.append("  COST")
    out.append(
        f"    {summary['runs_attempted']} runs, {cost['llm_calls']} provider calls, "
        f"{cost['tokens_in']} tokens in / {cost['tokens_out']} out"
    )
    out.append(
        f"    median {cost['seconds_per_run_median']}s and "
        f"{cost['calls_per_run_median']} calls per run; "
        f"{cost['wall_seconds_summed']}s of compile time summed across runs"
    )
    out.append("")

    out.append("  NOISE — what this N can and cannot distinguish")
    out.extend(_wrap(noise_statement(summary)))
    out.append("=" * 78)
    return "\n".join(out)


def _wrap(lines: list[str], width: int = 76) -> list[str]:
    """Wrap over-long prose for a terminal, preserving each line's own indent.

    Lines already inside the width pass through byte-for-byte: the histograms and the
    comparison table are column-aligned, and re-flowing them would break the alignment
    that makes a distribution readable at a glance.
    """

    import textwrap

    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()))
        wrapped.extend(
            textwrap.wrap(
                line.strip(), width=width, initial_indent=indent, subsequent_indent=indent + "  "
            )
            or [line]
        )
    return wrapped


def render_comparison(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Compare two benched configurations, and say plainly when N cannot separate them."""

    la, lb = a["config"]["label"], b["config"]["label"]
    out = ["=" * 78, f"COMPARISON — {la!r} vs {lb!r}", "=" * 78]

    # An A/B is only an A/B if exactly one thing differs. Say what differs first.
    same_keys = ("question_sha256", "store_sha256", "as_of", "horizon", "mode", "gates")
    differing = [k for k in same_keys if a["config"][k] != b["config"][k]]
    if differing:
        out.append("  *** THESE ARMS DO NOT SHARE THEIR INPUTS ***")
        for k in differing:
            out.append(f"    {k}: {str(a['config'][k])[:24]}  vs  {str(b['config'][k])[:24]}")
        out.append("  A difference in the results cannot be attributed to the compiler.")
    else:
        out.append("  Inputs match: same question, store, cutoff, horizon, mode and gate setting.")
    if a["compiler_sha256"] and a["compiler_sha256"] == b["compiler_sha256"]:
        out.append(
            f"  Compiler sha256 is IDENTICAL in both arms ({a['compiler_sha256'][:16]}) — "
            "these arms differ only in seeds, so any gap between them IS the noise."
        )
    else:
        out.append(
            f"  Compiler sha256 differs: {a['compiler_sha256'][:16]} vs "
            f"{b['compiler_sha256'][:16]} — that difference is the thing under test."
        )
    out.append("")

    ha, hb = a["headline"], b["headline"]
    na, ka = int(ha["n"] or 0), int(ha["at_or_above_threshold"] or 0)
    nb, kb = int(hb["n"] or 0), int(hb["at_or_above_threshold"] or 0)
    out.append(f"  HEADLINE  {HEADLINE} >= {SOCIETY_THRESHOLD}")
    out.append(f"    {la:<24} {ka} of {na}")
    out.append(f"    {lb:<24} {kb} of {nb}")
    if na and nb:
        p = fisher_exact_two_sided(ka, na - ka, kb, nb - kb)
        out.append(f"    Fisher exact, two-sided: p = {p:.3f}")
        if p <= 0.05:
            out.append("    DISTINGUISHABLE at p<=0.05. The two arms are not the same draw.")
        else:
            need = minimum_detectable_count(na, nb, baseline=ka)
            out.append(
                "    NOT DISTINGUISHABLE at this N. These two numbers do not establish "
                "a difference."
            )
            if need > nb:
                out.append(
                    f"    At N={na}/{nb} NO split at all could have reached p<=0.05. "
                    "This comparison was underpowered before it was run."
                )
            else:
                out.append(
                    f"    {lb!r} would have needed at least {need} of {nb} to separate "
                    f"from {ka} of {na}."
                )
    out.append("")

    out.append("  COUNT METRICS  (exact two-sided permutation test on the difference of means)")
    for name in ALL_METRICS:
        xs = _values(a, name)
        ys = _values(b, name)
        if not xs or not ys:
            continue
        ma, mb = sum(xs) / len(xs), sum(ys) / len(ys)
        p = permutation_p(xs, ys)
        if p is None:
            verdict = "N too large for exact enumeration — read the distributions"
        elif p <= 0.05:
            verdict = f"p={p:.3f} distinguishable"
        else:
            verdict = f"p={p:.3f} NOT distinguishable"
        out.append(
            f"    {name:<24} {ma:>6.2f} vs {mb:>6.2f}   ranges "
            f"[{min(xs)}..{max(xs)}] vs [{min(ys)}..{max(ys)}]   {verdict}"
        )
    out.append("")
    out.append(
        "  Read only the headline as a result. With "
        f"{len(ALL_METRICS)} metrics compared at p<=0.05, about one apparent "
        "difference per twenty is expected from chance alone, and nothing here "
        "corrects for that."
    )
    out.append("=" * 78)
    return "\n".join(_wrap(out, 78))


def _values(summary: dict[str, Any], name: str) -> list[int]:
    """The raw per-run values behind a metric, rebuilt from the stored run records."""

    return [
        int(r["metrics"][name]) for r in summary.get("runs", []) if name in (r.get("metrics") or {})
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_gateway(spec: BenchSpec, seed_salt: int) -> ModelGateway:
    from sworldmodel.deepseek_gateway import DeepSeekGateway
    from sworldmodel.http import UrllibTransport

    inner: ModelGateway = DeepSeekGateway(UrllibTransport())
    if spec.seed_mode == "vary":
        return SeedVaryingGateway(inner, seed_salt)
    return inner


def run_bench(
    spec: BenchSpec,
    runs: int,
    out: Path,
    *,
    concurrency: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    store = load_store(Path(spec.store_path))
    if verbose:
        print(
            f"[store]  {len(store.all())} claims, "
            f"{len(store.view(spec.as_of).available())} admissible at {spec.as_of.isoformat()}"
        )
        fingerprint, root, count = compiler_fingerprint()
        print(f"[code]   {root}  ({count} files)  sha256 {fingerprint[:16]}")

    def one(index: int) -> RunRecord:
        salt = int(
            hashlib.sha256(f"{spec.bench_seed}:{index}".encode()).hexdigest()[:8],
            16,
        )
        run_dir = out / f"run_{index:02d}"
        prepare_run_dir(
            run_dir,
            question=spec.question,
            as_of=spec.as_of,
            horizon=spec.horizon,
            mode=spec.mode,
        )
        gateway = build_gateway(spec, salt)
        record = run_once(gateway, spec, store, run_index=index, seed_salt=salt, out_dir=run_dir)
        if verbose:
            summary = (
                f"{HEADLINE}={record.metrics.get(HEADLINE)} "
                f"entities={record.metrics.get('entities')} "
                f"affordances={record.metrics.get('affordances')}"
                if record.metrics
                else "no world"
            )
            print(
                f"[run {index:02d}] {record.outcome:<14} {record.failure or '-':<28} "
                f"{summary}  ({record.seconds:.0f}s, {record.llm_calls} calls)"
            )
        return record

    started = time.monotonic()
    records: list[RunRecord] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for record in pool.map(one, range(runs)):
            records.append(record)
    records.sort(key=lambda r: r.run_index)

    summary = aggregate(spec, records)
    summary["cost"]["wall_seconds_elapsed"] = round(time.monotonic() - started, 1)
    summary["cost"]["concurrency"] = concurrency
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(summary, indent=1, default=str))
    report = render_report(summary)
    (out / "report.txt").write_text(report + "\n")
    if verbose:
        print()
        print(report)
        print(f"\n[done]   {out}/bench.json   {out}/report.txt")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two bench.json files")
    ap.add_argument("--store")
    ap.add_argument("--question")
    ap.add_argument("--as-of")
    ap.add_argument("--horizon")
    ap.add_argument("--out")
    ap.add_argument("--label", default="unnamed")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--mode", choices=("semantic", "direct"), default="semantic")
    ap.add_argument(
        "--no-gates",
        action="store_true",
        help="stop after lowering — cheaper, but no refusal code and the world may be "
        "one no gate would admit",
    )
    ap.add_argument(
        "--seed-mode",
        choices=("vary", "production"),
        default="vary",
        help="vary: re-derive each request seed per run so the planner is sampled. "
        "production: leave the compiler's question-derived seeds alone.",
    )
    ap.add_argument("--bench-seed", type=int, default=0)
    ap.add_argument(
        "--max-calls",
        type=int,
        default=40,
        help="per-run provider call ceiling; a runaway repair loop stops instead of "
        "spending the budget for the whole bench",
    )
    args = ap.parse_args(argv)

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        print(render_comparison(a, b))
        return 0

    missing = [
        name for name in ("store", "question", "as_of", "horizon", "out") if not getattr(args, name)
    ]
    if missing:
        ap.error(
            "missing required arguments: " + ", ".join("--" + m.replace("_", "-") for m in missing)
        )

    spec = BenchSpec(
        label=args.label,
        question=args.question,
        store_path=str(Path(args.store).resolve()),
        as_of=datetime.fromisoformat(args.as_of),
        horizon=datetime.fromisoformat(args.horizon),
        mode=args.mode,
        gates=not args.no_gates,
        seed_mode=args.seed_mode,
        bench_seed=args.bench_seed,
        max_calls=args.max_calls if args.max_calls > 0 else None,
    )
    run_bench(spec, args.runs, Path(args.out), concurrency=args.concurrency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
