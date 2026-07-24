"""LLM research planning: work backward from the outcome.

The planner turns a natural-language question into a targeted research plan — the
resolution event, deadline, authoritative source, decision-makers, rules, prior
actions, scheduled events, causal drivers, the facts that must be known before
simulation, and concrete search queries and official domains to pursue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash

# How many already-collected propositions are summarized back to the planner. A prompt
# size bound on one call; the canonical store keeps every claim regardless.
_SUMMARY_ITEMS = 40


@dataclass(frozen=True)
class ResearchPlan:
    process_summary: str
    resolution_event: str
    deadline: str
    authoritative_sources: tuple[str, ...]
    decision_makers: tuple[str, ...]
    rules: tuple[str, ...]
    prior_actions: tuple[str, ...]
    scheduled_events: tuple[str, ...]
    causal_drivers: tuple[str, ...]
    required_facts: tuple[str, ...]
    initial_queries: tuple[str, ...]
    official_domains: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the plan into the research trace.

        Follow-up research must continue the *same* plan rather than re-plan from
        scratch, or a second pass can chase a differently-worded resolution event than
        the evidence already gathered was gathered for.
        """

        return {
            "process_summary": self.process_summary,
            "resolution_event": self.resolution_event,
            "deadline": self.deadline,
            "authoritative_sources": list(self.authoritative_sources),
            "decision_makers": list(self.decision_makers),
            "rules": list(self.rules),
            "prior_actions": list(self.prior_actions),
            "scheduled_events": list(self.scheduled_events),
            "causal_drivers": list(self.causal_drivers),
            "required_facts": list(self.required_facts),
            "initial_queries": list(self.initial_queries),
            "official_domains": list(self.official_domains),
        }

    @classmethod
    def from_dict(cls, data: object) -> ResearchPlan | None:
        """Rebuild a plan written by :meth:`to_dict`; ``None`` if it is not one."""

        if not isinstance(data, dict) or "resolution_event" not in data:
            return None
        return cls(
            process_summary=str(data.get("process_summary", "")),
            resolution_event=str(data.get("resolution_event", "")),
            deadline=str(data.get("deadline", "")),
            authoritative_sources=_strs(data.get("authoritative_sources")),
            decision_makers=_strs(data.get("decision_makers")),
            rules=_strs(data.get("rules")),
            prior_actions=_strs(data.get("prior_actions")),
            scheduled_events=_strs(data.get("scheduled_events")),
            causal_drivers=_strs(data.get("causal_drivers")),
            required_facts=_strs(data.get("required_facts")),
            initial_queries=_strs(data.get("initial_queries")),
            official_domains=_strs(data.get("official_domains")),
        )


_PLAN_KEYS = (
    "process_summary",
    "resolution_event",
    "deadline",
    "authoritative_sources",
    "decision_makers",
    "rules",
    "prior_actions",
    "scheduled_events",
    "causal_drivers",
    "required_facts",
    "initial_queries",
    "official_domains",
)


def plan_research(
    gateway: ModelGateway, question: str, as_of: datetime, horizon: datetime
) -> ResearchPlan:
    prompt = f"""You are planning targeted evidence research for a forecasting question.
Work BACKWARD from the outcome: outcome <- terminal decision/action <- actor choices
<- proposal/alternatives <- interaction <- prior positions <- incoming information.

QUESTION: {question}
INFORMATION CUTOFF (as_of): {as_of.isoformat()}
HORIZON (resolution deadline is on/before this): {horizon.isoformat()}

Do NOT force the question into any predefined category. Return a JSON object with these
keys (all arrays are arrays of short strings):
- process_summary: a one-line, free-text description of the REAL process that actually
  resolves this question (whatever it genuinely is), discovered from the question itself.
- resolution_event: the exact event that resolves the question.
- deadline: ISO date/datetime the outcome is determined by.
- authoritative_sources: the official sources that would confirm the outcome.
- decision_makers: the people/roles/entities whose choices produce the outcome.
- rules: governing rules, thresholds, or procedures.
- prior_actions: relevant prior decisions/votes/statements to find.
- scheduled_events: known upcoming dated events between as_of and the horizon.
- causal_drivers: information releases or shocks that could move the outcome.
- required_facts: facts that MUST be verified before simulating (roster, rule, dates).
- initial_queries: 4-8 specific web/news search queries to run now.
- official_domains: official institution/government domains to search directly (host only)."""
    resp = gateway.generate(
        GatewayRequest(
            task_kind="research_plan",
            prompt=prompt,
            context={"question": question},
            seed=_seed(question),
            expected_keys=("initial_queries",),
        )
    )
    d = resp.data
    return ResearchPlan(
        process_summary=str(d.get("process_summary", "")),
        resolution_event=str(d.get("resolution_event", "")),
        deadline=str(d.get("deadline", horizon.isoformat())),
        authoritative_sources=_strs(d.get("authoritative_sources")),
        decision_makers=_strs(d.get("decision_makers")),
        rules=_strs(d.get("rules")),
        prior_actions=_strs(d.get("prior_actions")),
        scheduled_events=_strs(d.get("scheduled_events")),
        causal_drivers=_strs(d.get("causal_drivers")),
        required_facts=_strs(d.get("required_facts")),
        initial_queries=_strs(d.get("initial_queries")),
        official_domains=_strs(d.get("official_domains")),
    )


def followup_queries(
    gateway: ModelGateway,
    question: str,
    *,
    missing_facts: list[str],
    contradictions: list[str],
    have_summary: list[str],
    limit: int = 5,
) -> list[str]:
    """Ask for more specific queries.

    ``missing_facts`` names planned facts that lexical retrieval turned up nothing for.
    That is a statement about retrieval, not about whether the fact is established —
    this function only steers the next round's queries.
    """

    prompt = f"""Continuing evidence research for: {question}

Planned facts for which nothing relevant has been retrieved yet: {missing_facts or "none"}
Unresolved contradictions to break: {contradictions or "none"}
Evidence already collected (short): {have_summary[:_SUMMARY_ITEMS]}

Return JSON {{"queries": [ ... ]}} with up to {limit} NEW, more specific search queries
that would fetch stronger or primary sources for the outstanding/contradicted items. If
no useful new query exists, return an empty list."""
    resp = gateway.generate(
        GatewayRequest(
            task_kind="followup_queries",
            prompt=prompt,
            context={"question": question},
            seed=_seed(question + "".join(missing_facts)),
            expected_keys=("queries",),
        )
    )
    return list(_strs(resp.data.get("queries")))[:limit]


def _strs(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(x).strip() for x in value if str(x).strip())
    return ()


def _seed(text: str) -> int:
    return int(prompt_hash(text)[:8], 16)
