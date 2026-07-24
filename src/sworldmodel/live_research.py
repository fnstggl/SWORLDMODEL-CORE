"""The live, question-only research backend.

Accepts only ``research(question, as_of, horizon)`` and automatically builds the
complete evidence store from live sources: it plans research from the question,
issues Google News RSS and official-domain queries, fetches the underlying pages,
extracts and verifies claims, groups them by lineage, enforces the cutoff, and loops
with follow-up queries until evidence saturates or the budget is exhausted. It then
LLM-compiles the reality and uncertainty frame into a :class:`ResearchBundle`.

No ``corpus.json`` is read. This is the production research path.
"""

from __future__ import annotations

import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .evidence import EvidenceClaim, EvidenceStore, EvidenceView
from .gateway import ModelGateway
from .http import HttpError, HttpTransport, UrllibTransport
from .ids import content_id
from .models import AuthorityLevel, EpistemicType, SourceType
from .research import ResearchBundle
from .research_planner import ResearchPlan, followup_queries, plan_research
from .rss import google_news_rss_url, parse_rss
from .search import duckduckgo_search, site_query
from .source_extract import extract_claims
from .source_fetch import FetchedSource, fetch_source
from .universal_compiler import build_live_bundle

_WORD = re.compile(r"[a-z0-9]{4,}")


@dataclass
class ResearchBudget:
    max_rounds: int = 3
    max_queries: int = 12
    max_queries_per_round: int = 5
    max_pages_per_query: int = 4
    max_fetches: int = 20
    max_extract_calls: int = 28
    max_seconds: float = 240.0
    fetch_concurrency: int = 6
    extract_concurrency: int = 4
    low_info_rounds_to_stop: int = 2


@dataclass
class ResearchTrace:
    queries: list[str] = field(default_factory=list)
    rss_requests: list[dict[str, Any]] = field(default_factory=list)
    fetched: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    claim_count: int = 0
    contradictions: list[str] = field(default_factory=list)
    stop_reason: str = ""
    extract_calls: int = 0

    def to_dict(self, plan: ResearchPlan) -> dict[str, Any]:
        return {
            "process_type": plan.process_type,
            "resolution_event": plan.resolution_event,
            "required_facts": list(plan.required_facts),
            "queries": self.queries,
            "rss_requests": self.rss_requests,
            "official_domains": list(plan.official_domains),
            "sources_fetched": self.fetched,
            "sources_rejected": self.rejected,
            "claim_count": self.claim_count,
            "contradictions": self.contradictions,
            "extract_calls": self.extract_calls,
            "stop_reason": self.stop_reason,
        }


class LiveResearchBackend:
    is_live = True

    def __init__(
        self,
        gateway: ModelGateway,
        transport: HttpTransport | None = None,
        *,
        budget: ResearchBudget | None = None,
        now: datetime | None = None,
    ) -> None:
        self.gateway = gateway
        self.transport = transport or UrllibTransport()
        self.budget = budget or ResearchBudget()
        self._now = now  # injectable for tests; else datetime.now(tz) at call time

    def research(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        *,
        extra_queries: tuple[str, ...] = (),
    ) -> ResearchBundle:
        now = self._now or datetime.now(as_of.tzinfo)
        t0 = time.monotonic()
        plan = plan_research(self.gateway, question, as_of, horizon)
        trace = ResearchTrace()
        store = EvidenceStore()
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()

        queries: deque[str] = deque(plan.initial_queries[: self.budget.max_queries])
        # Targeted queries (coverage repair) go first so they run even under a budget.
        for q in extra_queries:
            queries.appendleft(q[:120])
        # Per-decision-maker queries surface articles that name each individual actor.
        for maker in plan.decision_makers[:6]:
            queries.append(f"{maker} {plan.resolution_event or question}"[:120])
        # Official-domain queries reach primary sources directly.
        terms = plan.resolution_event or question
        for dom in plan.official_domains[:4]:
            queries.append(site_query(dom, terms))

        low_info = 0
        rounds = 0
        while queries and rounds < self.budget.max_rounds and not self._time_up(t0):
            rounds += 1
            candidates = self._collect_candidates(queries, trace)
            sources = self._fetch_all(candidates, seen_urls, now, trace)
            added = self._extract_all(question, as_of, sources, seen_hashes, store, trace)

            trace.claim_count = len(store.all())
            trace.contradictions = [f"{a}<>{b}" for a, b in store.contradictions()]
            missing = _missing_facts(plan, store.view(as_of))

            if added == 0:
                low_info += 1
            else:
                low_info = 0
            if not missing:
                trace.stop_reason = "all required facts have supporting evidence"
                break
            if low_info >= self.budget.low_info_rounds_to_stop:
                trace.stop_reason = "evidence saturation (consecutive low-information rounds)"
                break
            if trace.extract_calls >= self.budget.max_extract_calls:
                trace.stop_reason = "extract budget exhausted"
                break
            fq = followup_queries(
                self.gateway,
                question,
                missing_facts=missing,
                contradictions=trace.contradictions,
                have_summary=[c.proposition for c in store.view(as_of).available()],
            )
            for q in fq:
                if q not in trace.queries and len(trace.queries) < self.budget.max_queries:
                    queries.append(q)
        if not trace.stop_reason:
            trace.stop_reason = "round/time/query budget reached"

        bundle = build_live_bundle(self.gateway, question, as_of, horizon, store, plan)
        return replace(bundle, live_trace=trace.to_dict(plan))

    def augment_for_coverage(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        missing: list[str],
        prior: ResearchBundle,
    ) -> ResearchBundle | None:
        """Targeted follow-up research when the coverage gate finds a material item
        absent. Each missing candidate becomes an explicit query so the follow-up
        research goes looking for exactly what the world was missing."""

        queries = tuple(_missing_query(m) for m in missing if _missing_query(m))
        if not queries:
            return None
        return self.research(question, as_of, horizon, extra_queries=queries)

    # -- research loop helpers --------------------------------------------------

    def _collect_candidates(self, queries: deque[str], trace: ResearchTrace) -> list[str]:
        urls: list[str] = []
        for _ in range(min(len(queries), self.budget.max_queries_per_round)):
            if len(trace.queries) >= self.budget.max_queries:
                break
            q = queries.popleft()
            trace.queries.append(q)
            # Channel 1: Google News RSS (recorded as a discovery signal). Its links are
            # obfuscated redirects, so we do not fetch them directly.
            rss_url = google_news_rss_url(q)
            try:
                resp = self.transport.get(rss_url, timeout=15)
                items = parse_rss(resp.text)
            except HttpError:
                items = []
            trace.rss_requests.append({"query": q, "url": rss_url, "items": len(items)})
            # Channel 2: DuckDuckGo — real, fetchable publisher/official URLs.
            for u in duckduckgo_search(self.transport, q, limit=self.budget.max_pages_per_query):
                urls.append(u)
        return urls

    def _fetch_all(
        self, urls: list[str], seen: set[str], now: datetime, trace: ResearchTrace
    ) -> list[FetchedSource]:
        todo = [u for u in dict.fromkeys(urls) if u not in seen][: self.budget.max_fetches]
        for u in todo:
            seen.add(u)
        results: list[FetchedSource] = []
        if not todo:
            return results
        with ThreadPoolExecutor(max_workers=self.budget.fetch_concurrency) as pool:
            for src in pool.map(lambda u: fetch_source(self.transport, u, now=now), todo):
                if src.ok:
                    results.append(src)
                    trace.fetched.append(
                        {
                            "url": src.final_url,
                            "publisher": src.publisher,
                            "status": src.status,
                            "published_at": src.published_at.isoformat()
                            if src.published_at
                            else None,
                        }
                    )
                else:
                    trace.rejected.append(
                        {"url": src.url, "reason": f"unreachable/empty (status {src.status})"}
                    )
        return results

    def _extract_all(
        self,
        question: str,
        as_of: datetime,
        sources: list[FetchedSource],
        seen_hashes: set[str],
        store: EvidenceStore,
        trace: ResearchTrace,
    ) -> int:
        # For a pastcast, a source published after the cutoff can only yield claims that
        # are then excluded — so it must not consume the (bounded) extract budget. This
        # concentrates the budget on usable pre-cutoff sources, which is what makes
        # roster/vote verification reliable rather than variance-dependent.
        fresh: list[FetchedSource] = []
        for s in sources:
            if s.published_at is not None and s.published_at > as_of:
                trace.rejected.append(
                    {
                        "url": s.final_url,
                        "reason": f"published after cutoff ({s.published_at.date()})",
                    }
                )
            elif s.content_hash not in seen_hashes:
                fresh.append(s)
        budget_left = self.budget.max_extract_calls - trace.extract_calls
        sources = fresh[: max(0, budget_left)]
        for s in sources:
            seen_hashes.add(s.content_hash)
        if not sources:
            return 0
        added = 0
        with ThreadPoolExecutor(max_workers=self.budget.extract_concurrency) as pool:
            batches = list(
                pool.map(lambda s: (s, extract_claims(self.gateway, question, s, as_of)), sources)
            )
        trace.extract_calls += len(sources)
        for source, claims in batches:
            for c in claims:
                if not c.verified_in_text:
                    trace.rejected.append(
                        {
                            "url": source.final_url,
                            "reason": "claim not supported by fetched text",
                            "proposition": c.proposition[:120],
                        }
                    )
                    continue
                if self._add_claim(store, source, c, as_of):
                    added += 1
        return added

    def _add_claim(
        self, store: EvidenceStore, source: FetchedSource, claim: Any, as_of: datetime
    ) -> bool:
        available_at = source.published_at or source.fetched_at
        published_at = source.published_at or source.fetched_at
        cid = content_id("c", source.final_url, claim.proposition, claim.normalized_value)
        if cid in store.claims:
            return False
        entity = claim.entities[0] if claim.entities else source.publisher
        lineage = content_id(
            "evt", _topic(claim.proposition), entity.lower(), claim.normalized_value.lower()
        )
        store.add(
            EvidenceClaim(
                id=cid,
                proposition=claim.proposition,
                normalized_value=claim.normalized_value,
                entities=claim.entities,
                valid_from=None,
                valid_until=None,
                published_at=published_at,
                available_at=available_at,
                source_id=source.publisher or source.final_url,
                source_url=source.final_url,
                source_title=source.title,
                source_type=_source_type(claim.authority_hint),
                authority_level=AuthorityLevel(claim.authority_hint),
                supporting_excerpt=claim.supporting_excerpt,
                lineage_event_id=lineage,
                epistemic_type=EpistemicType(claim.epistemic_type),
                confidence=0.6 + 0.1 * claim.authority_hint,
                retrieved_at=source.fetched_at,
            )
        )
        return True

    def _time_up(self, t0: float) -> bool:
        return (time.monotonic() - t0) > self.budget.max_seconds


def _missing_query(label: str) -> str:
    """Extract the identity from a coverage 'missing' label ('[kind] identity — why')
    so it can be researched directly."""

    text = label.split("]", 1)[1] if "]" in label else label
    return text.split("—", 1)[0].strip()


def _topic(proposition: str) -> str:
    return proposition.split(":", 1)[0].strip().lower() if ":" in proposition else "fact"


def _source_type(hint: int) -> SourceType:
    return {
        4: SourceType.OFFICIAL_INSTITUTIONAL,
        3: SourceType.PRIMARY_RECORD,
        2: SourceType.CONTEMPORANEOUS_REPORTING,
        1: SourceType.LOW_QUALITY,
    }.get(hint, SourceType.CONTEMPORANEOUS_REPORTING)


def _missing_facts(plan: ResearchPlan, view: EvidenceView) -> list[str]:
    """A required fact is 'covered' if some available claim shares vocabulary with it."""

    available = view.available()
    corpus_tokens = set()
    for c in available:
        corpus_tokens |= set(_WORD.findall((c.proposition + " " + " ".join(c.entities)).lower()))
    missing = []
    for fact in plan.required_facts:
        tokens = set(_WORD.findall(fact.lower()))
        if tokens and len(tokens & corpus_tokens) / len(tokens) < 0.5:
            missing.append(fact)
    return missing
