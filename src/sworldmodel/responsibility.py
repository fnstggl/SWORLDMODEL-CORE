"""What produced the number — computed in-run, as a publication gate (D6 / FI-3 / FI-4).

A published probability is a claim about the world. This module decides, before the
forecast is written, whether the run is entitled to make it: it deletes parts of the
run's own recorded ledger, re-evaluates each branch's terminal on what is left, replaces
the branch weights with equal ones, and perturbs the ungrounded numeric alternatives
inside the range the plan itself called plausible. Whatever survives every deletion was
not produced by the thing deleted.

The defect this exists for (FD-4): the classification existed *only* in the offline
forensic script, so publication was ungated — a Bank of England run whose number was
one YES cell out of four equally weighted symmetric-ignorance cells classified
BRANCH_WEIGHTS_DOMINATED after the fact and published 0.25 as a forecast anyway. The
classification now runs before the artifact is written, and four of its eight values
forbid an answer entirely.

Every replay here goes through :mod:`sworldmodel.replaycore` — the one ledger-replay
implementation (D7). Nothing in this module re-simulates, and nothing in it calls a
model: the tests are deterministic replays of events that already happened.

The §14 test set, all mandatory before publication:

* remove every actor-produced event;
* remove one actor's events at a time;
* remove one non-actor process's events at a time;
* remove every recorded event, preserving initialization, and re-evaluate the terminal;
* replace the branch weights with equal weights and re-aggregate;
* perturb the ungrounded numeric alternatives within plausible ranges;
* remove the terminal-relevant actions.

A gate that could not run every test publishes nothing. "The check did not run" and
"the check passed" are different facts, and only one of them licenses an answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import (
    ACTOR_AND_PROCESS_CAUSED,
    ACTOR_CAUSED,
    BRANCH_WEIGHTS_DOMINATED,
    FACTUALLY_RESOLVED,
    INITIAL_ASSUMPTIONS_DOMINATED,
    PROCESS_CAUSED,
    PUBLISHING_CLASSIFICATIONS,
    REQUIRED_RESPONSIBILITY_TESTS,
    RESPONSIBILITY_INVALID,
    RESPONSIBILITY_UNRESOLVED,
    BranchCounterfactual,
    BranchOutcome,
    ResponsibilityReport,
)
from .replaycore import (
    EventLike,
    KeepPredicate,
    Unreconstructable,
    branch_initial_state,
    branch_key,
    counterfactual_outcome,
    event_actor_id,
    event_kind,
    group_events_by_branch,
    initial_fields_from_world,
    reevaluate_terminal,
    replay_fields,
    terminal_ast_from_world,
    terminal_field_reads,
)
from .worldspec import Expr, parse_expr

__all__ = [
    "classify_responsibility",
    "equal_weight_probability",
    "numeric_condition_ranges",
    "threshold_straddling_variables",
    "weighted_probability",
]

# The event kinds that *do* something to the world a terminal can read, as opposed to
# the engine's own bookkeeping. Same set the shared replay core uses for its
# terminal-relevant deletion, named here so the gate's own vocabulary is visible.
_TERMINAL_RELEVANT_KINDS = ("create_event", "append_record", "set_field")

_UNRECONSTRUCTABLE = "UNRECONSTRUCTABLE"


# --------------------------------------------------------------------------- #
# Small numeric helpers (universal; no question, domain or family appears here)
# --------------------------------------------------------------------------- #


def _numeric(value: Any) -> float | None:
    """The value as a number, or None. Booleans are not numbers here: True/False are
    categorical alternatives, and perturbing them "within a plausible range" is
    meaningless."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def weighted_probability(branch_outcomes: Sequence[BranchOutcome]) -> float | None:
    """P(YES) conditional on resolved mass, at the branches' own weights."""

    yes = sum(b.weight for b in branch_outcomes if b.resolved and b.outcome == "YES")
    resolved = sum(b.weight for b in branch_outcomes if b.resolved and b.outcome in ("YES", "NO"))
    return (yes / resolved) if resolved > 1e-12 else None


def equal_weight_probability(branch_outcomes: Sequence[BranchOutcome]) -> float | None:
    """The same aggregation with every branch weighted equally (§14 weight sensitivity).

    Equal weights are not a better estimate — they are a *second arbitrary* one. The
    point of substituting them is the distance between the two: if the number moves,
    the number was the weights.
    """

    resolved = [b for b in branch_outcomes if b.resolved and b.outcome in ("YES", "NO")]
    if not resolved:
        return None
    return sum(1.0 for b in resolved if b.outcome == "YES") / len(resolved)


def numeric_condition_ranges(
    branch_outcomes: Sequence[BranchOutcome],
) -> dict[str, tuple[float, float]]:
    """Per ungrounded uncertainty variable, the numeric range the run itself enumerated.

    The plan asserted every one of these draws as a possible value of that variable, so
    the interval they span is the range it called plausible — a band derived from the
    run's own claims rather than an invented tolerance.
    """

    draws: dict[str, list[float]] = {}
    for b in branch_outcomes:
        if b.weight_grounded:
            continue
        for name, value in b.key_conditions:
            number = _numeric(value)
            if number is not None:
                draws.setdefault(name, []).append(number)
    return {
        name: (min(values), max(values)) for name, values in draws.items() if len(set(values)) > 1
    }


def threshold_straddling_variables(
    branch_outcomes: Sequence[BranchOutcome],
) -> tuple[str, ...]:
    """D3/FI-5 at aggregation time: ungrounded numeric draws on opposite sides.

    For each variable, branches are grouped by *the rest* of their condition tuple, so
    only branches that differ in this one variable are compared. If two such branches
    drew different numbers for it, both weights are ungrounded, and they resolved
    differently, then the answer is the choice of those numbers — the semantic
    validator's static gate (:data:`THRESHOLD_STRADDLING_UNGROUNDED_SCENARIOS`) read the
    plan's arithmetic; this reads the branch table the run actually produced, and
    catches the same defect in a world the static gate never saw.
    """

    resolved = [
        b
        for b in branch_outcomes
        if b.resolved and b.outcome in ("YES", "NO") and not b.weight_grounded
    ]
    variables = {name for b in resolved for name, _v in b.key_conditions}
    straddling: set[str] = set()
    for variable in variables:
        groups: dict[tuple[tuple[str, str], ...], list[tuple[float, str]]] = {}
        for b in resolved:
            conditions = dict(b.key_conditions)
            number = _numeric(conditions.get(variable))
            if number is None or b.outcome is None:
                continue
            rest = tuple(sorted((k, v) for k, v in conditions.items() if k != variable))
            groups.setdefault(rest, []).append((number, b.outcome))
        for entries in groups.values():
            if len({n for n, _o in entries}) > 1 and len({o for _n, o in entries}) > 1:
                straddling.add(variable)
                break
    return tuple(sorted(straddling))


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def classify_responsibility(
    branch_outcomes: Sequence[BranchOutcome],
    *,
    events: Sequence[EventLike],
    actor_decisions: Sequence[Any] = (),
    world: Any = None,
    rendered_terminal: Mapping[str, Any] | None = None,
) -> ResponsibilityReport:
    """Run the §14 test set over a finished run and classify what produced its number.

    ``world`` is the executable world — a :class:`~sworldmodel.compiled.CompiledWorld`,
    a live :class:`~sworldmodel.worldspec.WorldSpec`, the persisted
    ``compiled_world.json`` dict, or ``None``. Without one (and without a recorded
    rendered terminal) the deletion replays cannot be evaluated, and the gate fails
    **closed**: it returns :data:`INVALID` naming the missing input rather than a
    classification that would let an unchecked run publish.

    Never raises. This gate describes a run that already finished; a fault in the
    description must not be able to destroy the thing described. A fault does, however,
    withhold publication — which is the opposite default from the advisory audits.
    """

    try:
        return _classify(
            branch_outcomes,
            events=events,
            actor_decisions=actor_decisions,
            world=world,
            rendered_terminal=rendered_terminal,
        )
    except Exception as exc:  # noqa: BLE001 — a gate that crashes must not publish
        return _refuse(
            branch_outcomes,
            reason=(
                "the responsibility gate could not run, so nothing verified what "
                f"produced this number: {type(exc).__name__}: {exc}"
            ),
            error=f"{type(exc).__name__}: {exc}",
        )


def _refuse(
    branch_outcomes: Sequence[BranchOutcome], *, reason: str, error: str = ""
) -> ResponsibilityReport:
    return ResponsibilityReport(
        classification=RESPONSIBILITY_INVALID,
        may_publish_answer=False,
        point_estimate_permitted=False,
        reason=reason,
        weights_grounded=all(b.weight_grounded for b in branch_outcomes),
        trace_reproducible=False,
        trace_reproducible_basis=(
            "the recorded ledger was not replayed, so nothing confirms the published "
            "outcomes reproduce"
        ),
        tests_completed=(),
        tests_missing=REQUIRED_RESPONSIBILITY_TESTS,
        error=error,
    )


def _without_actor(actor: str) -> KeepPredicate:
    """Keep everything except one named actor's output (§14, one actor at a time)."""

    def keep(event: EventLike) -> bool:
        return bool(event_actor_id(event) != actor)

    return keep


def _without_process(kind: str) -> KeepPredicate:
    """Keep everything except one non-actor process's output (§14, one at a time)."""

    def keep(event: EventLike) -> bool:
        return bool(event_actor_id(event) is not None or event_kind(event) != kind)

    return keep


def _spec_of(world: Any) -> Any:
    """The executable spec, whether a CompiledWorld, a spec, or an artifact dict."""

    if world is None:
        return None
    spec = getattr(world, "spec", None)
    return world if spec is None else spec


# Collection/event aggregates that read a record's CONTENT rather than how many records
# there are. ``sum`` and ``values`` always read ``record['value']``; ``count``,
# ``exists`` and ``event_count`` do so only when they carry a where-predicate.
_CONTENT_READING_ALWAYS = ("sum", "values")
_CONTENT_READING_WITH_PREDICATE = ("count", "exists", "event_count")


def _content_reading_terminal_ops(terminal: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Aggregates in the terminal that read record content, which the replay cannot supply.

    :class:`~sworldmodel.replaycore.ReplayWorld` reconstructs collection CARDINALITY from
    the ledger and presents each record as an empty placeholder, deliberately carrying no
    fabricated content. A cardinality terminal (``count('positions') >= 5``) replays
    exactly. A terminal that predicates on content (``count('positions',
    equals(item('value'), 'hold')) >= 5``) evaluates over blank placeholders and returns
    a confident zero — the branch's real YES replays as NO, and the gate would report a
    reproduction failure that is an artifact of the replay, not of the run.

    So it is detected and named instead. The recorded ``append_record`` payloads DO carry
    ``key``/``value``/``extra``, so this is reconstructable in principle; until the shared
    replay core reconstructs them, the honest statement is that the mandatory
    counterfactuals could not be evaluated for this terminal, and the gate refuses rather
    than guessing in either direction.
    """

    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            node = parse_expr(node)
        if not isinstance(node, Expr):
            return
        arity = len(node.args)
        if node.op in _CONTENT_READING_ALWAYS or (
            node.op in _CONTENT_READING_WITH_PREDICATE and arity > 1
        ):
            found.add(node.op)
        for arg in node.args:
            walk(arg)

    for key in ("yes_when", "unresolved_when"):
        node = (terminal or {}).get(key)
        if node is None:
            continue
        try:
            walk(node)
        except ValueError:  # an expression this build cannot parse is not a content read
            continue
    return tuple(sorted(found))


def _classify(
    branch_outcomes: Sequence[BranchOutcome],
    *,
    events: Sequence[EventLike],
    actor_decisions: Sequence[Any],
    world: Any,
    rendered_terminal: Mapping[str, Any] | None,
) -> ResponsibilityReport:
    spec = _spec_of(world)
    terminal = terminal_ast_from_world(spec)
    rendered: dict[str, Any] = dict(rendered_terminal or {})
    if terminal is None and not rendered:
        return _refuse(
            branch_outcomes,
            reason=(
                "no executable terminal was available to re-evaluate, so the deletion "
                "counterfactuals could not be computed and nothing verified what "
                "produced this number"
            ),
        )

    content_ops = _content_reading_terminal_ops(terminal)
    if content_ops:
        return _refuse(
            branch_outcomes,
            reason=(
                "the recorded terminal reads record content through "
                + ", ".join(f"{op}(...)" for op in content_ops)
                + ", which the shared replay core does not reconstruct — it replays "
                "collection cardinality and presents each record as an empty "
                "placeholder. Re-evaluating this terminal over those placeholders would "
                "answer confidently and wrongly, so the mandatory counterfactuals could "
                "not be run and no answer is published"
            ),
            error="replay_cannot_reconstruct_record_content",
        )

    weights_grounded = all(b.weight_grounded for b in branch_outcomes)
    world_initials = initial_fields_from_world(spec)
    by_branch = group_events_by_branch(events)
    terminal_fields = terminal_field_reads(terminal, rendered)
    ranges = numeric_condition_ranges(branch_outcomes)

    counterfactuals: list[BranchCounterfactual] = []
    perturbation_findings: list[str] = []
    reproduction_failures: list[str] = []
    unreconstructable: list[str] = []

    for branch in branch_outcomes:
        evs = by_branch.get(branch_key(branch.branch_id), [])
        initial = branch_initial_state(world_initials, evs)

        def answer(
            keep: KeepPredicate,
            evs: list[EventLike] = evs,
            initial: dict[str, Any] = initial,
        ) -> str | None:
            # `evs` and `initial` bound as defaults: a closure capturing the loop
            # variables by reference would evaluate every branch against the last one.
            return counterfactual_outcome(
                evs, keep, initial=initial, terminal=terminal, rendered=rendered
            )

        recomputed = answer(lambda _e: True)
        published = branch.outcome if branch.resolved else "UNRESOLVED"
        if recomputed == _UNRECONSTRUCTABLE:
            unreconstructable.append(branch.branch_id)
        elif recomputed != published:
            reproduction_failures.append(
                f"{branch.branch_id}: published {published}, ledger replay gives {recomputed}"
            )

        actors = sorted({a for e in evs if (a := event_actor_id(e)) is not None})
        process_kinds = sorted({event_kind(e) for e in evs if event_actor_id(e) is None})
        perturbations = _perturbations(
            evs,
            branch=branch,
            initial=initial,
            terminal=terminal,
            rendered=rendered,
            terminal_fields=terminal_fields,
            ranges=ranges,
        )
        for field_name, perturbed in perturbations:
            if perturbed not in (recomputed, _UNRECONSTRUCTABLE):
                perturbation_findings.append(
                    f"{branch.branch_id}: {field_name} moved inside the range this run's own "
                    f"ungrounded alternatives span, and the terminal answer changed "
                    f"{recomputed} -> {perturbed}"
                )

        counterfactuals.append(
            BranchCounterfactual(
                branch_id=branch.branch_id,
                outcome=published,
                recomputed_outcome=recomputed,
                all_actor_output_removed=answer(lambda e: event_actor_id(e) is None),
                all_process_output_removed=answer(lambda e: event_actor_id(e) is not None),
                initialization_preserved=answer(lambda _e: False),
                terminal_relevant_actions_removed=answer(
                    lambda e: event_kind(e) not in _TERMINAL_RELEVANT_KINDS
                ),
                per_actor_removed=tuple((a, answer(_without_actor(a))) for a in actors),
                per_process_removed=tuple((k, answer(_without_process(k))) for k in process_kinds),
                numeric_perturbations=tuple(perturbations),
            )
        )

    p_weighted = weighted_probability(branch_outcomes)
    p_equal = equal_weight_probability(branch_outcomes)
    span = abs(p_weighted - p_equal) if p_weighted is not None and p_equal is not None else None
    straddling = threshold_straddling_variables(branch_outcomes)

    trace_reproducible = not reproduction_failures and not unreconstructable
    if reproduction_failures:
        basis = "the ledger replay disagrees with the published outcome: " + "; ".join(
            sorted(reproduction_failures)
        )
    elif unreconstructable:
        basis = (
            "the recorded terminal could not be re-evaluated for "
            + ", ".join(sorted(unreconstructable))
            + ", so the published outcome was not confirmed against the ledger"
        )
    else:
        basis = (
            f"every one of {len(branch_outcomes)} branch outcome(s) reproduces exactly by "
            "replaying the recorded ledger and re-evaluating the terminal, with no model "
            "call. This confirms the arithmetic, nothing more."
        )

    classification, reason = _decide(
        branch_outcomes,
        counterfactuals=counterfactuals,
        trace_reproducible=trace_reproducible,
        trace_basis=basis,
        weights_grounded=weights_grounded,
        straddling=straddling,
    )
    may_publish = classification in PUBLISHING_CLASSIFICATIONS
    # D6, second sentence: an actor- or process-caused result still needs grounded
    # weights before its magnitude is a calibrated point estimate. A factual resolution
    # is exempt — the cited record decided it, and no weight enters the answer.
    point_estimate_permitted = may_publish and (
        classification == FACTUALLY_RESOLVED or (weights_grounded and not straddling)
    )
    return ResponsibilityReport(
        classification=classification,
        may_publish_answer=may_publish,
        point_estimate_permitted=point_estimate_permitted,
        reason=reason,
        weights_grounded=weights_grounded,
        trace_reproducible=trace_reproducible,
        trace_reproducible_basis=basis,
        branch_counterfactuals=tuple(counterfactuals),
        weighted_probability=p_weighted,
        equal_weight_probability=p_equal,
        weight_sensitivity_span=span,
        threshold_straddling_variables=straddling,
        numeric_perturbation_findings=tuple(perturbation_findings),
        tests_completed=REQUIRED_RESPONSIBILITY_TESTS,
        tests_missing=(),
    )


def _perturbations(
    events: Sequence[EventLike],
    *,
    branch: BranchOutcome,
    initial: Mapping[str, Any],
    terminal: Mapping[str, Any] | None,
    rendered: Mapping[str, Any],
    terminal_fields: Sequence[str],
    ranges: Mapping[str, tuple[float, float]],
) -> list[tuple[str, str | None]]:
    """§14: perturb this branch's ungrounded numeric alternatives, within plausible range.

    The band is taken from the run's own enumeration: for each ungrounded numeric
    variable, the interval spanned by every draw the run made for it. The plan asserted
    each of those values as possible, so moving inside that interval is moving inside a
    range the run itself called plausible — never an invented tolerance.

    The perturbation is applied to the numeric world fields the terminal actually reads,
    one field at a time, scaled by the extreme ratios the band implies for this branch's
    own draw. Perturbing one field at a time is deliberate: scaling every terminal input
    together can cancel exactly (a quantity and the threshold it is compared against),
    which would hide the sensitivity the test exists to find.
    """

    if not terminal_fields:
        return []
    conditions = dict(branch.key_conditions)
    ratios: set[float] = set()
    for name, (low, high) in ranges.items():
        drawn = _numeric(conditions.get(name))
        if drawn is None or abs(drawn) < 1e-12:
            continue
        for edge in (low, high):
            ratio = edge / drawn
            if abs(ratio - 1.0) > 1e-9:
                ratios.add(ratio)
    if not ratios:
        return []

    replayed, counts = replay_fields(events, lambda _e: True)
    state = {**dict(initial), **replayed}
    out: list[tuple[str, str | None]] = []
    for field_name in terminal_fields:
        base = _numeric(state.get(field_name))
        if base is None or abs(base) < 1e-12:
            continue
        for ratio in sorted(ratios):
            probe = {**state, field_name: base * ratio}
            try:
                resolved, outcome, _how = reevaluate_terminal(terminal, rendered, probe, counts)
            except Unreconstructable:
                out.append((f"{field_name} x{ratio:.6g}", _UNRECONSTRUCTABLE))
                continue
            out.append((f"{field_name} x{ratio:.6g}", outcome if resolved else "UNRESOLVED"))
    return out


def _decide(
    branch_outcomes: Sequence[BranchOutcome],
    *,
    counterfactuals: Sequence[BranchCounterfactual],
    trace_reproducible: bool,
    trace_basis: str,
    weights_grounded: bool,
    straddling: Sequence[str],
) -> tuple[str, str]:
    """The D6 classification, from the counterfactual table alone. Never asserted."""

    if not trace_reproducible:
        return RESPONSIBILITY_INVALID, trace_basis

    resolved = [b for b in branch_outcomes if b.resolved and b.outcome in ("YES", "NO")]
    if not resolved:
        return (
            RESPONSIBILITY_UNRESOLVED,
            "no branch reached a terminal answer, so there is nothing whose cause could "
            "be attributed and nothing to publish",
        )

    by_id = {c.branch_id: c for c in counterfactuals}
    resolved_cf = [by_id[b.branch_id] for b in resolved if b.branch_id in by_id]

    # Everything removed, initialization preserved: does the terminal still say YES?
    # Restricted to YES deliberately, matching the aggregation's own framing — a
    # terminal that said NO at t0 and still says NO is the ordinary open world in which
    # nobody produced the outcome, which is a simulated result, not a citation.
    if resolved_cf and all(c.initialization_preserved == c.outcome == "YES" for c in resolved_cf):
        return (
            FACTUALLY_RESOLVED,
            "every resolved branch still answers YES with every recorded event deleted: "
            "the cited record answered the question before the window opened, and the "
            "simulation did not produce this answer",
        )

    actor_changed = [c.branch_id for c in resolved_cf if c.all_actor_output_removed != c.outcome]
    process_changed = [
        c.branch_id for c in resolved_cf if c.all_process_output_removed != c.outcome
    ]

    # BRANCH_WEIGHTS_DOMINATED outranks the caused classes on purpose. The trajectory
    # may well have been necessary to produce any YES at all — the deletion table records
    # that separately — but when every branch's answer is fixed by its own condition
    # tuple and every weight is ungrounded, the MAGNITUDE is an enumeration artifact:
    # re-weight the same cells to any evidence-plausible asymmetry and the probability
    # moves with nothing else changing. That is a Bank of England 0.25 and a Tesla 0.50,
    # both of which published as forecasts (FD-2, FD-4, D9).
    answers_by_conditions: dict[tuple[tuple[str, str], ...], set[str]] = {}
    for b in resolved:
        answers_by_conditions.setdefault(tuple(sorted(b.key_conditions)), set()).add(str(b.outcome))
    answer_is_a_function_of_the_conditions = (
        len(answers_by_conditions) == len(resolved)
        and all(len(v) == 1 for v in answers_by_conditions.values())
        and len({next(iter(v)) for v in answers_by_conditions.values()}) > 1
    )
    if not weights_grounded and answer_is_a_function_of_the_conditions:
        return (
            BRANCH_WEIGHTS_DOMINATED,
            f"each of the {len(resolved)} resolved branches has its own condition tuple and "
            "its answer is fixed once that tuple is chosen, while no branch weight is "
            "grounded in an identified distribution — so the probability is the weight of "
            "the winning cells, and re-weighting the same cells to any evidence-plausible "
            "asymmetry moves it with nothing else changing",
        )
    if straddling:
        return (
            BRANCH_WEIGHTS_DOMINATED,
            "ungrounded numeric alternatives for "
            + ", ".join(f"{v!r}" for v in straddling)
            + " sit on opposite sides of the terminal threshold, so the answer is the "
            "choice of those numbers and the fact that there are two of them",
        )

    if actor_changed and process_changed:
        return (
            ACTOR_AND_PROCESS_CAUSED,
            f"deleting actor output changes {len(actor_changed)} branch outcome(s) and "
            f"deleting non-actor process output changes {len(process_changed)}: both "
            "produced the answer",
        )
    if actor_changed:
        return (
            ACTOR_CAUSED,
            f"deleting every actor-produced event changes {len(actor_changed)} of "
            f"{len(resolved_cf)} resolved branch outcome(s): the actors produced the answer",
        )
    if process_changed:
        return (
            PROCESS_CAUSED,
            f"deleting every non-actor process event changes {len(process_changed)} of "
            f"{len(resolved_cf)} resolved branch outcome(s): the modeled process produced "
            "the answer",
        )
    return (
        INITIAL_ASSUMPTIONS_DOMINATED,
        "deleting every actor output and every process output leaves every resolved "
        "branch answering exactly what it already answered: nothing the run did changed "
        "any terminal, so the answer is the initialization read back",
    )
