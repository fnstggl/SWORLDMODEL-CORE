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


@dataclass(frozen=True)
class ResearchPlan:
    process_type: str
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


_PLAN_KEYS = (
    "process_type",
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

Return a JSON object with these keys (all arrays are arrays of short strings):
- process_type: one of "committee_vote", "individual_response", "individual_decision",
  "multiparty", "negotiation", "organizational", "geopolitical", "population",
  "deadline_event", "quantitative", "other" — your best classification.
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
        process_type=str(d.get("process_type", "other")),
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
    prompt = f"""Continuing evidence research for: {question}

Facts still MISSING authoritative support: {missing_facts or "none"}
Unresolved contradictions to break: {contradictions or "none"}
Evidence already collected (short): {have_summary[:40]}

Return JSON {{"queries": [ ... ]}} with up to {limit} NEW, more specific search queries
that would fetch stronger or primary sources for the missing/contradicted items. If no
useful new query exists, return an empty list."""
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
