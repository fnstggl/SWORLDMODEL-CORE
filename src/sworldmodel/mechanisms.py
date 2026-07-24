"""Deterministic institutional mechanisms.

After human/institutional *choices* occur (produced by simulated actors), the exact
consequences are computed here by code: counting votes, applying seat weights,
comparing a tally to a verified threshold, and evaluating the terminal predicate.

No LLM ever runs in this module. These functions are pure and total.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import DecisionRule, TerminalSpec


def tally_votes(votes: dict[str, str], vote_powers: dict[str, int]) -> dict[str, int]:
    """Return option -> summed seat power. Missing power defaults to 1."""

    tally: dict[str, int] = {}
    for actor_id, option in votes.items():
        power = vote_powers.get(actor_id, 1)
        tally[option] = tally.get(option, 0) + power
    return tally


def carried_option(tally: dict[str, int], rule: DecisionRule) -> str | None:
    """Which option the body *decided on* under its own rule (not the YES/NO
    predicate). Returns ``None`` if the rule's threshold is not met by any option."""

    if not tally:
        return None
    best_option = max(tally, key=lambda opt: (tally[opt], opt))
    best = tally[best_option]
    if rule.kind == "plurality":
        # A strict plurality winner; ties => undecided.
        top = [opt for opt, n in tally.items() if n == best]
        return best_option if len(top) == 1 else None
    # majority / supermajority / unanimous all encode a numeric threshold.
    if best >= rule.threshold:
        return best_option
    return None


@dataclass(frozen=True)
class TerminalEvaluation:
    resolved: bool
    outcome: str | None  # "YES" | "NO" | None
    reason: str
    tally: tuple[tuple[str, int], ...]
    carried: str | None


def evaluate_terminal(
    votes: dict[str, str],
    vote_powers: dict[str, int],
    expected_seats: int,
    rule: DecisionRule,
    spec: TerminalSpec,
) -> TerminalEvaluation:
    """Evaluate the question predicate from the *complete* set of final votes.

    If not every expected seat has cast a vote, the branch is *unresolved* — never
    forced. This is the one place that decides YES/NO, and it is deterministic.
    """

    seats_voted = len(votes)
    tally = tally_votes(votes, vote_powers)
    tally_t = tuple(sorted(tally.items()))
    carried = carried_option(tally, rule)

    if seats_voted < expected_seats:
        missing = expected_seats - seats_voted
        return TerminalEvaluation(
            resolved=False,
            outcome=None,
            reason=f"only {seats_voted}/{expected_seats} seats voted ({missing} missing)",
            tally=tally_t,
            carried=carried,
        )

    target = spec.target_option
    target_power = tally.get(target, 0)
    total_power = sum(vote_powers.get(a, 1) for a in votes)

    if spec.yes_condition == "unanimous_for_option":
        outcome = "YES" if target_power == total_power and target_power > 0 else "NO"
        reason = (
            f"all seats chose {target!r}" if outcome == "YES" else f"not unanimous for {target!r}"
        )
    elif spec.yes_condition == "at_least_k_for_option":
        k = spec.k if spec.k is not None else rule.threshold
        outcome = "YES" if target_power >= k else "NO"
        reason = (
            f"{target_power}>={k} chose {target!r}"
            if outcome == "YES"
            else f"{target_power}<{k} chose {target!r}"
        )
    elif spec.yes_condition == "majority_for_option":
        outcome = "YES" if carried == target else "NO"
        reason = f"body carried {carried!r}"
    else:  # pragma: no cover - guarded by contract construction
        raise ValueError(f"Unknown yes_condition {spec.yes_condition!r}")

    return TerminalEvaluation(
        resolved=True,
        outcome=outcome,
        reason=reason,
        tally=tally_t,
        carried=carried,
    )
