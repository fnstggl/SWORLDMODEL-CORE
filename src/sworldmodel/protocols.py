"""Protocol graphs: generic institutional/social procedure, compiled per scenario.

Institutional coordination is *not* a numeric convergence. It is a declarative
sequence of generic primitives (distribute briefing, introduce proposal, request
statements, deliver them, open the decision, cast votes, tally, publish). The
runtime interprets this graph, driving actors through it. Members receive
substantive colleague statements before being asked to react to them.

Nothing here is scenario-specific: the compiler builds the applicable graph from
verified rules and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class StepKind:
    DISTRIBUTE_BRIEFING = "distribute_briefing"
    INTRODUCE_PROPOSAL = "introduce_proposal"
    REQUEST_STATEMENTS = "request_statements"
    DELIVER_STATEMENTS = "deliver_statements"
    PERMIT_QUESTIONS = "permit_questions"
    REVISE_PROPOSAL = "revise_proposal"
    OPEN_DECISION = "open_decision"
    CAST_VOTES = "cast_votes"
    TALLY = "tally"
    PUBLISH = "publish"


@dataclass(frozen=True)
class ProtocolStep:
    kind: str
    params: tuple[tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)


@dataclass(frozen=True)
class ProtocolGraph:
    steps: tuple[ProtocolStep, ...] = ()

    def kinds(self) -> tuple[str, ...]:
        return tuple(s.kind for s in self.steps)


def _step(kind: str, **params: Any) -> ProtocolStep:
    return ProtocolStep(kind=kind, params=tuple(sorted(params.items())))


def committee_protocol(
    *,
    chair_actor_id: str,
    proposal_option: str,
    proposal_text: str,
    briefing_text: str,
    voting_actor_ids: tuple[str, ...],
    allow_revision: bool = True,
) -> ProtocolGraph:
    """Assemble the canonical committee decision procedure.

    This is the real event sequence: the meeting opens with a briefing, a proposal is
    introduced, members state positions, those statements are delivered, the decision
    is opened, every seat votes, and code tallies the result.
    """

    steps: list[ProtocolStep] = [
        _step(StepKind.DISTRIBUTE_BRIEFING, text=briefing_text),
        _step(
            StepKind.INTRODUCE_PROPOSAL,
            chair=chair_actor_id,
            option=proposal_option,
            text=proposal_text,
        ),
        _step(StepKind.REQUEST_STATEMENTS, members=voting_actor_ids),
        _step(StepKind.DELIVER_STATEMENTS),
    ]
    if allow_revision:
        steps.append(_step(StepKind.REVISE_PROPOSAL, chair=chair_actor_id))
    steps.extend(
        [
            _step(StepKind.OPEN_DECISION),
            _step(StepKind.CAST_VOTES, members=voting_actor_ids),
            _step(StepKind.TALLY),
            _step(StepKind.PUBLISH),
        ]
    )
    return ProtocolGraph(steps=tuple(steps))
