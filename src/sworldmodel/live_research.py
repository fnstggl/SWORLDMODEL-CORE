"""The live, question-only research backend.

Accepts only ``research(question, as_of, horizon)`` and builds the complete evidence
store from live sources: it plans research from the question, issues authoritative,
Google News RSS and general web queries, fetches the underlying pages *as they stood at
the cutoff*, extracts and verifies claims, groups them by lineage, detects
contradictions, and loops with follow-up queries until evidence saturates or the budget
is exhausted. It then LLM-compiles the reality and uncertainty frame into a
:class:`ResearchBundle`.

Three properties of the loop are load-bearing:

* **Authoritative discovery runs first and cannot be starved.** Official-domain and
  planned-authoritative-source queries are issued ahead of general discovery and hold a
  reserved share of the query budget, so a broad question can never spend the whole cap
  on general search before reaching a primary source.
* **Only admissible evidence counts as progress.** A source that cannot be shown to
  predate the cutoff never reaches the extractor, and only claims that are reachable
  through the cutoff view count toward the claim total or reset the saturation counter.
* **Follow-up research extends the store it was given.** Coverage repair appends to the
  existing store — same claim ids, lineage preserved — so nothing already verified is
  discarded by researching more.

No ``corpus.json`` is read. This is the production research path.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from .errors import GatewayError, WorldIntegrityError
from .evidence import EvidenceClaim, EvidenceStore, EvidenceView
from .gateway import GatewayRequest, ModelGateway
from .gnews_decode import GoogleNewsDecoder
from .gnews_decode import article_id_of as article_id_of_link
from .http import (
    DEFAULT_POLICY,
    FetchPolicy,
    HttpError,
    HttpTransport,
    UrllibTransport,
    UrlRejected,
    check_url_shape,
)
from .ids import content_id, sha256_hex
from .models import AuthorityLevel, EpistemicType, SourceType
from .providers import (
    ProviderHealth,
    jina_reader,
    jina_title_search,
    serper_search,
)
from .research import ResearchBundle, assemble_bundle
from .research_planner import ResearchPlan, followup_queries, plan_research
from .rss import (
    RssItem,
    discover_feed_links,
    feed_urls_for,
    google_news_rss_url,
    is_google_redirect,
    parse_rss,
    resolve_item_url,
    site_roots,
)
from .search import duckduckgo_search, site_query
from .source_extract import ExtractedClaim, ExtractionResult, distinctive_terms, extract_claims
from .source_fetch import (
    FetchedSource,
    RetrievalMode,
    extract_text,
    fetch_source,
    requires_archived_copy,
)

# A search engine's URL length limit; a query longer than this is truncated by the
# engine anyway, so it is trimmed here where the truncation is visible in the trace.
_MAX_QUERY_CHARS = 120


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
    max_contradiction_checks: int = 8
    # Ceilings for the WHOLE question — the opening pass plus every repair round —
    # rather than per pass. Repair used to receive a fresh reduced budget each time it
    # ran, so a question that repaired six times could spend six times the opening
    # allowance and still be inside "the budget"; one live OPEC+ run was killed from
    # outside at forty minutes with 26 logical queries issued. Research stops when a
    # question has cost what a question may cost, wherever in the run that happens.
    total_queries: int = 40
    total_fetches: int = 120
    total_extract_calls: int = 90
    total_seconds: float = 900.0

    @property
    def authoritative_query_reserve(self) -> int:
        """The share of the query budget general discovery may not consume.

        Discovery has two channels — authoritative/official and general — and this is an
        even split between them while both have work queued, with the odd query going to
        the authoritative side. It is a division of a budget between channels, not a
        tuned preference: whenever either queue empties, the other may use the whole
        remaining cap.

        Both directions of starvation have actually happened here. Authoritative queries
        were once enqueued last and never ran; then the fix gave them unconditional
        priority, and a live run spent all twenty queries on official domains — most of
        which blocked or returned 403 — while eleven queued news queries never issued.
        A share is a floor for one channel and a ceiling for it at the same time.
        """

        return (self.max_queries + 1) // 2


@dataclass
class ResearchTrace:
    """The replayable record of one research session (possibly extended by repair)."""

    queries: list[dict[str, Any]] = field(default_factory=list)
    rss_requests: list[dict[str, Any]] = field(default_factory=list)
    search_failures: list[dict[str, Any]] = field(default_factory=list)
    attempted_urls: list[str] = field(default_factory=list)
    seen_content_hashes: list[str] = field(default_factory=list)
    fetched: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    extraction_calls: list[dict[str, Any]] = field(default_factory=list)
    claim_count: int = 0
    admissible_claim_count: int = 0
    contradictions: list[str] = field(default_factory=list)
    contradiction_checks: int = 0
    stop_reason: str = ""
    extract_calls: int = 0
    rounds: int = 0
    fact_retrieval: dict[str, list[str]] = field(default_factory=dict)
    retrieval_mode: dict[str, Any] = field(default_factory=dict)
    # Provider accounting, carried across passes so a run's retrieval cost is one number.
    provider_requests: dict[str, int] = field(default_factory=dict)
    provider_health: dict[str, Any] = field(default_factory=dict)

    def bump(self, provider: str, n: int = 1) -> None:
        self.provider_requests[provider] = self.provider_requests.get(provider, 0) + n

    def query_texts(self) -> list[str]:
        return [q["query"] for q in self.queries]

    def to_dict(self, plan: ResearchPlan, store: EvidenceStore) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "plan": plan.to_dict(),
            "process_summary": plan.process_summary,
            "resolution_event": plan.resolution_event,
            "required_facts": list(plan.required_facts),
            "official_domains": list(plan.official_domains),
            "queries": self.queries,
            "rss_requests": self.rss_requests,
            "search_failures": self.search_failures,
            "attempted_urls": self.attempted_urls,
            "seen_content_hashes": self.seen_content_hashes,
            "sources_fetched": self.fetched,
            "sources_rejected": self.rejected,
            "extraction_calls": self.extraction_calls,
            "claim_count": self.claim_count,
            "admissible_claim_count": self.admissible_claim_count,
            "contradictions": self.contradictions,
            "contradiction_checks": self.contradiction_checks,
            "extract_calls": self.extract_calls,
            "rounds": self.rounds,
            "provider_requests": self.provider_requests,
            "provider_health": self.provider_health,
            # Labeled for what it is: lexical retrieval over claim text, NOT a check
            # that a required fact is established. Nothing here asserts coverage.
            "required_fact_lexical_retrieval": {
                "method": "distinctive-term overlap between the planned fact and claim text",
                "warning": "a listed claim is a retrieval candidate, not verified support",
                "candidates": self.fact_retrieval,
            },
            # Everything needed to re-open the exact document behind every claim.
            "claim_provenance": store.provenance_records(),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def resume(cls, prior: dict[str, Any] | None) -> ResearchTrace:
        """Rebuild the session state of a previous research run so a follow-up round
        continues it (same seen-URL and seen-content sets, cumulative counters)."""

        if not prior:
            return cls()
        return cls(
            retrieval_mode=dict(prior.get("retrieval_mode") or {}),
            queries=list(prior.get("queries", [])),
            rss_requests=list(prior.get("rss_requests", [])),
            search_failures=list(prior.get("search_failures", [])),
            attempted_urls=list(prior.get("attempted_urls", [])),
            seen_content_hashes=list(prior.get("seen_content_hashes", [])),
            fetched=list(prior.get("sources_fetched", [])),
            rejected=list(prior.get("sources_rejected", [])),
            extraction_calls=list(prior.get("extraction_calls", [])),
            claim_count=int(prior.get("claim_count", 0)),
            admissible_claim_count=int(prior.get("admissible_claim_count", 0)),
            contradictions=list(prior.get("contradictions", [])),
            contradiction_checks=int(prior.get("contradiction_checks", 0)),
            extract_calls=int(prior.get("extract_calls", 0)),
            rounds=int(prior.get("rounds", 0)),
            provider_requests=dict(prior.get("provider_requests") or {}),
            provider_health=dict(prior.get("provider_health") or {}),
        )


@dataclass
class _QueryQueues:
    """Discovery queries split by channel, drained authoritative-first."""

    targeted: deque[str] = field(default_factory=deque)
    authoritative: deque[str] = field(default_factory=deque)
    general: deque[str] = field(default_factory=deque)
    authoritative_used: int = 0
    general_used: int = 0

    def empty(self) -> bool:
        return not (self.targeted or self.authoritative or self.general)


@dataclass
class _Session:
    """One pass of the research loop, with its own budget.

    The trace accumulates across passes, but the *budget* counters do not: a follow-up
    pass that inherited an exhausted query or extract count could never issue the very
    query it was created to issue, which would make coverage repair a no-op.
    """

    queues: _QueryQueues
    seen_urls: set[str]
    seen_hashes: set[str]
    queries_used: int = 0
    extract_calls: int = 0


@dataclass(frozen=True)
class _ModeDispatchConfig:
    """The minimal config slice ``api.compile_for_mode`` reads (its
    ``CompileModeConfig`` protocol), so the initial live compilation goes through the
    one shared mode dispatch without this module owning a second copy of it."""

    gateway: ModelGateway
    compiler_mode: str


class LiveResearchBackend:
    is_live = True

    def __init__(
        self,
        gateway: ModelGateway,
        transport: HttpTransport | None = None,
        *,
        budget: ResearchBudget | None = None,
        now: datetime | None = None,
        compiler_mode: str = "direct",
    ) -> None:
        self.gateway = gateway
        self.transport = transport or UrllibTransport()
        self.budget = budget or ResearchBudget()
        self.compiler_mode = compiler_mode
        self._now = now  # injectable for tests; else datetime.now(tz) at call time
        # Per-question retrieval state, shared across the opening pass and every repair
        # round: the health ledger circuit-breaks a blocked provider once for the whole
        # run, and the decoder caches each article id so a headline seen twice is
        # resolved once. Created on the first public call.
        self._health = ProviderHealth()
        self._decoder = GoogleNewsDecoder(self.transport)
        # Cumulative spend for this question, across the opening pass and every repair.
        self._spent_queries = 0
        self._spent_fetches = 0
        self._spent_extracts = 0
        self._started = time.monotonic()

    def _cumulative_room(self) -> str:
        """Which whole-question ceiling is exhausted, if any."""

        if self._spent_queries >= self.budget.total_queries:
            return f"question query budget exhausted ({self._spent_queries} logical queries)"
        if self._spent_fetches >= self.budget.total_fetches:
            return f"question fetch budget exhausted ({self._spent_fetches} pages)"
        if self._spent_extracts >= self.budget.total_extract_calls:
            return f"question extraction budget exhausted ({self._spent_extracts} calls)"
        if time.monotonic() - self._started >= self.budget.total_seconds:
            return f"question research clock exhausted ({self.budget.total_seconds:.0f}s)"
        return ""

    # -- public API -------------------------------------------------------------

    def retrieval_mode(self, as_of: datetime) -> RetrievalMode:
        """Decide nowcast-vs-pastcast once, against the moment this run started.

        Pinning ``self._now`` here is the point: every later fetch compares the cutoff
        against the same instant, so a run cannot begin as a nowcast and become a
        pastcast because the clock moved past its own cutoff mid-session.
        """

        if self._now is None:
            self._now = datetime.now(as_of.tzinfo)
        return RetrievalMode.decide(as_of, self._now)

    def research(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        *,
        extra_queries: tuple[str, ...] = (),
    ) -> ResearchBundle:
        mode = self.retrieval_mode(as_of)
        # Printed before a single request is spent. The previous acceptance run passed a
        # cutoff hours in the past, silently got archive-only retrieval, and the
        # resulting refusals read as a research problem rather than a mode error.
        print(mode.describe(), file=sys.stderr, flush=True)
        plan = plan_research(self.gateway, question, as_of, horizon)
        store = EvidenceStore()
        trace = ResearchTrace(retrieval_mode=mode.as_dict())
        self._run_rounds(question, as_of, plan, store, trace, extra_queries)
        return self._compile(question, as_of, horizon, plan, store, trace)

    def augment_for_coverage(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        missing: list[str],
        prior: ResearchBundle,
    ) -> ResearchBundle | None:
        """Targeted follow-up research when the coverage gate finds a material item
        absent. Each missing candidate becomes an explicit query so the follow-up goes
        looking for exactly what the world was missing.

        The follow-up **extends** ``prior``: the same :class:`EvidenceStore` is carried
        forward and new claims are appended to it, so every previously verified claim
        keeps its id and lineage and nothing already established is thrown away by the
        act of researching more. The trace continues too, so the audit record covers
        both passes.
        """

        queries = [q for q in (_missing_query(m) for m in missing) if q]
        return self.augment_targeted(question, as_of, horizon, queries, prior)

    def augment_targeted(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        queries: list[str],
        prior: ResearchBundle,
    ) -> ResearchBundle | None:
        """Run a specific set of follow-up queries and rebuild the bundle.

        This is the general form of targeted repair: the caller has diagnosed exactly
        what is missing — an office-holder, a procedure, a production pathway — and hands
        over the queries that would establish it.

        The store is carried forward and appended to, never replaced, so previously
        verified claims keep their ids, their lineage and their epistemic labels. If the
        pass adds nothing, ``None`` says so: that is the signal that this line of enquiry
        is exhausted, and the caller must not read it as a reason to try again.
        """

        wanted = tuple(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not wanted:
            return None
        mode = self.retrieval_mode(as_of)
        store = prior.evidence_store
        trace = ResearchTrace.resume(prior.live_trace)
        trace.retrieval_mode = mode.as_dict()
        plan = ResearchPlan.from_dict((prior.live_trace or {}).get("plan"))
        if plan is None:
            plan = plan_research(self.gateway, question, as_of, horizon)
        before = len(store.claims)
        # A follow-up looks for one named thing, so it gets a follow-up's budget rather
        # than a fresh survey's. Repair may now run several rounds, and giving each of
        # them the full opening budget is how a question that was answerable ran out of
        # wall clock before it ever reached the simulation.
        with self._followup_budget():
            self._run_rounds(question, as_of, plan, store, trace, wanted)
        if len(store.claims) == before:
            return None
        return self._compile(question, as_of, horizon, plan, store, trace)

    @contextmanager
    def _followup_budget(self) -> Iterator[None]:
        """Run one targeted follow-up under a reduced budget, then restore the original."""

        original = self.budget
        self.budget = replace(
            original,
            max_rounds=1,
            max_queries=max(4, original.max_queries // 3),
            max_seconds=max(60.0, original.max_seconds / 3),
            max_fetches=max(6, original.max_fetches // 2),
            max_extract_calls=max(8, original.max_extract_calls // 2),
        )
        try:
            yield
        finally:
            self.budget = original

    # -- research loop ----------------------------------------------------------

    def _run_rounds(
        self,
        question: str,
        as_of: datetime,
        plan: ResearchPlan,
        store: EvidenceStore,
        trace: ResearchTrace,
        extra_queries: tuple[str, ...],
    ) -> None:
        now = self._now or datetime.now(as_of.tzinfo)
        t0 = time.monotonic()
        session = _Session(
            queues=self._build_queues(plan, question, extra_queries, trace.query_texts()),
            seen_urls=set(trace.attempted_urls),
            seen_hashes=set(trace.seen_content_hashes),
        )

        low_info = 0
        rounds = 0
        while (
            not session.queues.empty() and rounds < self.budget.max_rounds and not self._time_up(t0)
        ):
            exhausted = self._cumulative_room()
            if exhausted:
                trace.stop_reason = exhausted
                break
            rounds += 1
            trace.rounds += 1
            candidates = self._collect_candidates(session, as_of, trace)
            if rounds == 1 and plan.official_domains:
                # Once, at the start: go straight to the institutions the plan named.
                candidates = (
                    self._policy_filtered(
                        self._official_feed_candidates(
                            tuple(plan.official_domains), question, as_of, trace
                        ),
                        trace,
                    )
                    + candidates
                )
            sources = self._fetch_all(candidates, session, now, as_of, trace)
            added = self._extract_all(question, as_of, sources, session, store, trace)

            view = store.view(as_of)
            trace.claim_count = len(store.all())
            trace.admissible_claim_count = len(view.available())
            self._detect_contradictions(store, as_of, trace, question=question, plan=plan)
            trace.contradictions = [f"{a}<>{b}" for a, b in store.contradictions()]
            unsupported = self._record_fact_retrieval(plan, view, trace)

            low_info = low_info + 1 if added == 0 else 0
            if low_info >= self.budget.low_info_rounds_to_stop:
                trace.stop_reason = "evidence saturation (consecutive rounds added no usable claim)"
                break
            if session.extract_calls >= self.budget.max_extract_calls:
                trace.stop_reason = "extract budget exhausted"
                break
            if session.queries_used >= self.budget.max_queries:
                trace.stop_reason = "query budget exhausted"
                break
            for q in followup_queries(
                self.gateway,
                question,
                missing_facts=unsupported,
                contradictions=trace.contradictions,
                have_summary=[c.proposition for c in view.available()],
            ):
                if q not in trace.query_texts():
                    session.queues.general.append(q[:_MAX_QUERY_CHARS])
        if not trace.stop_reason:
            trace.stop_reason = "round/time/query budget reached"
        trace.attempted_urls = sorted(session.seen_urls)
        trace.seen_content_hashes = sorted(session.seen_hashes)

    def _build_queues(
        self,
        plan: ResearchPlan,
        question: str,
        extra_queries: tuple[str, ...],
        already_run: list[str],
    ) -> _QueryQueues:
        terms = plan.resolution_event or question
        queues = _QueryQueues()
        seen: set[str] = set(already_run)  # never re-issue a query an earlier pass ran
        for q in extra_queries:
            text = q[:_MAX_QUERY_CHARS].strip()
            if text and text not in seen:
                seen.add(text)
                queues.targeted.append(text)

        def push(queue: deque[str], text: str) -> None:
            q = text[:_MAX_QUERY_CHARS].strip()
            if q and q not in seen:
                seen.add(q)
                queue.append(q)

        # Authoritative channel: official domains first, then the sources the plan says
        # would actually confirm the outcome — which were previously never queried.
        for domain in plan.official_domains:
            push(queues.authoritative, site_query(domain, terms))
        for source in plan.authoritative_sources:
            if _looks_like_domain(source):
                push(queues.authoritative, site_query(source, terms))
            else:
                push(queues.authoritative, f"{source} {terms}")
        # General discovery.
        for q in plan.initial_queries:
            push(queues.general, q)
        for maker in plan.decision_makers:
            push(queues.general, f"{maker} {terms}")
        return queues

    def _next_queries(self, session: _Session) -> list[tuple[str, str]]:
        """Pop this round's queries as ``(channel, query)``, authoritative first.

        General discovery is held to ``max_queries - authoritative_query_reserve``
        while authoritative work remains queued, which is what stops a broad question
        from spending the whole cap before it reaches a primary source.
        """

        queues = session.queues
        picked: list[tuple[str, str]] = []
        reserve = self.budget.authoritative_query_reserve
        general_cap = self.budget.max_queries - reserve
        while (
            len(picked) < self.budget.max_queries_per_round
            and session.queries_used + len(picked) < self.budget.max_queries
        ):
            # Targeted repair queries are the caller naming exactly what is missing, so
            # they always go first.
            if queues.targeted:
                picked.append(("targeted", queues.targeted.popleft()))
                continue
            # Then the two channels *share* the budget, each bounded by its own half.
            #
            # This used to be an elif chain that drained the authoritative queue
            # completely before general discovery was reached, so the general branch's
            # cap was unreachable and the general channel was the one starved — the
            # opposite of the stated design. A live run spent all ten of its queries on
            # official domains, was blocked or 403'd on most of them, and never issued
            # any of its eleven queued news queries.
            want_authoritative = queues.authoritative and queues.authoritative_used < reserve
            want_general = queues.general and queues.general_used < general_cap
            if want_authoritative:
                queues.authoritative_used += 1
                picked.append(("authoritative", queues.authoritative.popleft()))
            elif want_general:
                queues.general_used += 1
                picked.append(("general", queues.general.popleft()))
            elif queues.authoritative:
                # The other channel is exhausted or over its share; the whole remaining
                # cap belongs to whichever still has work.
                queues.authoritative_used += 1
                picked.append(("authoritative", queues.authoritative.popleft()))
            elif queues.general:
                queues.general_used += 1
                picked.append(("general", queues.general.popleft()))
            else:
                break
        return picked

    def _feed_urls(self, domain: str) -> list[str]:
        """Ask the site where its feed is, then fall back to conventional paths."""

        declared: list[str] = []
        for root in site_roots(domain):
            try:
                resp = self.transport.get(root, timeout=10)
            except HttpError:
                continue
            if resp.ok:
                declared.extend(discover_feed_links(resp.text, root))
                if declared:
                    break
        return list(dict.fromkeys(declared + feed_urls_for(domain)))

    def _official_feed_candidates(
        self,
        domains: tuple[str, ...],
        question: str,
        as_of: datetime,
        trace: ResearchTrace,
    ) -> list[str]:
        """Article URLs straight from the institutions' own feeds.

        Search engines block, and the news aggregator no longer yields URLs at all. An
        institution that publishes decisions, minutes or press releases almost always
        publishes a feed of them, and those items carry real article URLs. This channel
        depends on neither a search engine nor a redirect that has to be inverted, and it
        lands on exactly the sources the question needs.
        """

        question_terms = distinctive_terms(question)
        found: list[str] = []
        for domain in domains[:6]:
            hit = False
            for feed_url in self._feed_urls(domain):
                if hit:
                    break
                try:
                    resp = self.transport.get(feed_url, timeout=12)
                except HttpError:
                    continue
                if not resp.ok or ("<item" not in resp.text and "<entry" not in resp.text):
                    continue
                items = [
                    i
                    for i in parse_rss(resp.text)
                    if i.link
                    and not is_google_redirect(i.link)
                    and (i.published is None or i.published <= as_of)
                ]
                if not items:
                    continue
                hit = True
                # An institution's feed is its whole output — fines, consultations,
                # appointments — in reverse date order. Taking the newest few would
                # spend the fetch budget on whatever happened yesterday. Rank by what
                # the question is actually about, and keep one recent item so a feed
                # with no lexical overlap still contributes something.
                ranked = sorted(
                    items,
                    key=lambda i: (
                        -len(question_terms & distinctive_terms(f"{i.title} {i.description}")),
                        -(i.published.timestamp() if i.published else 0.0),
                    ),
                )
                found.extend(i.link for i in ranked[: self.budget.max_pages_per_query])
                trace.rss_requests.append(
                    {
                        "channel": "official_feed",
                        "domain": domain,
                        "url": feed_url,
                        "items": len(items),
                        "resolved": len(items),
                    }
                )
            if not hit:
                trace.rss_requests.append(
                    {"channel": "official_feed", "domain": domain, "resolved": 0}
                )
        return found

    def _collect_candidates(
        self, session: _Session, as_of: datetime, trace: ResearchTrace
    ) -> list[str]:
        """Candidate publisher URLs for this round's queries, cheapest route first.

        Per logical query the order is: Google News RSS (headlines for exactly this
        query) with each opaque link resolved by the batchexecute decoder; then the
        optional low-priority DuckDuckGo channel; and only when this round has produced
        nothing does Serper act as the paid last resort. Every provider is behind the
        run's health ledger, so one that is blocking is rested rather than re-asked.
        """

        urls: list[str] = []
        queries = self._next_queries(session)
        for channel, q in queries:
            session.queries_used += 1
            self._spent_queries += 1
            trace.queries.append({"query": q, "channel": channel})
            urls.extend(self._news_candidates(q, as_of, trace))
            if self._health.usable("duckduckgo"):
                outcome = duckduckgo_search(
                    self.transport, q, limit=self.budget.max_pages_per_query
                )
                trace.bump("duckduckgo")
                if outcome.ok:
                    self._health.note_success("duckduckgo")
                    urls.extend(outcome.urls)
                else:
                    self._health.note_block("duckduckgo")
                    trace.search_failures.append(
                        {"query": q, "channel": channel, "error": outcome.error}
                    )
        # Serper is the paid last resort: only when the free channels yielded nothing
        # for this round's queries, and only while it is healthy.
        if not urls and queries and self._health.usable("serper"):
            for _channel, q in queries:
                found = serper_search(
                    self.transport, q, self._health, limit=self.budget.max_pages_per_query
                )
                trace.bump("serper")
                if found:
                    trace.queries.append({"query": q, "channel": "serper", "urls": len(found)})
                    urls.extend(found)
        trace.provider_health = self._health.as_dict()
        return self._policy_filtered(urls, trace)

    def _news_candidates(self, query: str, as_of: datetime, trace: ResearchTrace) -> list[str]:
        """Publisher URLs behind a Google News query.

        RSS gives the headlines and opaque links for exactly this query. Each link is
        resolved to its publisher URL by the batchexecute decoder — the old
        ``resolve_item_url`` recovers nothing from a live feed now that Google encodes a
        server-side reference in every id, which is why the whole news channel was
        returning a hundred items and zero candidates. A link the decoder cannot resolve
        falls back to exact-title Jina Search on the headline and publisher the item
        already carries.
        """

        rss_url = google_news_rss_url(query)
        trace.bump("google_news_rss")
        try:
            resp = self.transport.get(rss_url, timeout=15)
            items = parse_rss(resp.text)
        except HttpError as exc:
            trace.rss_requests.append({"query": query, "url": rss_url, "error": str(exc)})
            return []
        # The feed is fetched live, so it lists today's articles. An item published after
        # the cutoff cannot inform a pastcast and is dropped before any request is spent.
        within = [i for i in items if i.published is None or i.published <= as_of]
        resolved: list[str] = []
        decoded = direct = via_search = 0
        for item in within[: self.budget.max_pages_per_query]:
            url = self._resolve_item(item, trace)
            if url is None:
                continue
            resolved.append(url)
            if url == item.link:
                direct += 1
            elif article_id_of_link(item.link):
                decoded += 1
            else:
                via_search += 1
        trace.rss_requests.append(
            {
                "query": query,
                "url": rss_url,
                "items": len(items),
                "within_cutoff": len(within),
                "resolved_to_publisher": len(resolved),
                "decoded": decoded,
                "recovered_via_title_search": via_search,
                "urls_queued": resolved,
            }
        )
        return resolved

    def _resolve_item(self, item: RssItem, trace: ResearchTrace) -> str | None:
        """One RSS item to a publisher URL: decoder first, then exact-title recovery."""

        # A non-Google link (rare, from a plain feed) is already the publisher URL.
        plain = resolve_item_url(item)
        if plain and not is_google_redirect(plain):
            return plain
        if self._health.usable("google_news_decoder"):
            result = self._decoder.decode(item.link)
            trace.bump("google_news_decoder")
            if result.ok:
                self._health.note_success("google_news_decoder")
                return result.publisher_url
            if result.failure in ("challenge", "throttled"):
                self._health.note_block("google_news_decoder")
            else:
                self._health.note_error("google_news_decoder")
        # Decode failed: recover from the headline and publisher the item already names.
        if item.title and item.source and self._health.usable("jina_search"):
            found = jina_title_search(self.transport, item.title, item.source, self._health)
            trace.bump("jina_search")
            for url in found:
                if not is_google_redirect(url):
                    return url
        return None

    def _policy_filtered(self, urls: list[str], trace: ResearchTrace) -> list[str]:
        """Drop URLs that must not be requested at all. These come off third-party HTML,
        so a non-http(s) scheme or a private/loopback literal is refused here, before a
        request exists. The transport re-checks every hop, including after redirects."""

        policy: FetchPolicy = getattr(self.transport, "policy", DEFAULT_POLICY)
        out: list[str] = []
        for url in urls:
            try:
                check_url_shape(url, policy)
            except UrlRejected as exc:
                trace.rejected.append({"url": url, "reason": f"refused by fetch policy: {exc}"})
            else:
                out.append(url)
        return out

    def _fetch_all(
        self,
        urls: list[str],
        session: _Session,
        now: datetime,
        as_of: datetime,
        trace: ResearchTrace,
    ) -> list[FetchedSource]:
        seen = session.seen_urls
        room = max(0, self.budget.total_fetches - self._spent_fetches)
        todo = [u for u in dict.fromkeys(urls) if u not in seen][
            : min(self.budget.max_fetches, room)
        ]
        for u in todo:
            seen.add(u)
        self._spent_fetches += len(todo)
        results: list[FetchedSource] = []
        if not todo:
            return results
        with ThreadPoolExecutor(max_workers=self.budget.fetch_concurrency) as pool:
            fetched = list(
                pool.map(lambda u: fetch_source(self.transport, u, now=now, as_of=as_of), todo)
            )
        pastcast = requires_archived_copy(as_of, now)
        for src in fetched:
            if not src.ok and not pastcast:
                # A known URL that direct fetch could not read — blocked, empty,
                # JavaScript-only or badly extracted — is exactly what Jina Reader is
                # for. Only on a nowcast: Reader reads the live page, which cannot stand
                # in for the archived pre-cutoff content a pastcast requires.
                recovered = self._reader_fallback(src, now, trace)
                if recovered is not None:
                    src = recovered
            if src.ok:
                results.append(src)
                trace.fetched.append(
                    {
                        "url": src.url,
                        "fetched_url": src.fetched_url,
                        "publisher": src.publisher,
                        "status": src.status,
                        "published_at": _iso(src.published_at),
                        "archived_at": _iso(src.archived_at),
                        "observed_at": _iso(src.observed_at),
                        "content_sha256": src.content_hash,
                        "via": "jina_reader" if src.fetched_url.startswith("jina:") else "direct",
                    }
                )
            else:
                trace.rejected.append(
                    {
                        "url": src.url,
                        "reason": src.rejection_reason
                        or f"unreachable/empty (status {src.status})",
                    }
                )
        return results

    def _reader_fallback(
        self, failed: FetchedSource, now: datetime, trace: ResearchTrace
    ) -> FetchedSource | None:
        """Read a known publisher URL through Jina Reader when direct fetch failed."""

        if not self._health.usable("jina_reader"):
            return None
        result = jina_reader(self.transport, failed.url, self._health)
        trace.bump("jina_reader")
        if not result.ok:
            return None
        text = extract_text(result.text) if "<" in result.text[:2000] else result.text
        if not text.strip():
            return None
        return replace(
            failed,
            fetched_url=f"jina:{failed.url}",
            status=200,
            reachable=True,
            text=text,
            content_hash=sha256_hex(text),
            rejection_reason="",
        )

    def _extract_all(
        self,
        question: str,
        as_of: datetime,
        sources: list[FetchedSource],
        session: _Session,
        store: EvidenceStore,
        trace: ResearchTrace,
    ) -> int:
        """Extract from admissible, unseen sources. Returns the number of claims that
        actually entered the store *and* are reachable through the cutoff view — the
        only kind of progress that justifies another round."""

        fresh: list[FetchedSource] = []
        for s in sources:
            # A source whose content cannot be dated at or before the cutoff can only
            # produce claims that are invisible through the cutoff view. Such a source
            # must not consume the bounded extract budget, and must not be able to
            # report progress by producing claims nobody can use.
            if s.evidence_time() > as_of:
                trace.rejected.append(
                    {
                        "url": s.url,
                        "reason": (
                            f"content not datable at or before the cutoff "
                            f"(usable from {s.evidence_time().isoformat()})"
                        ),
                    }
                )
            elif s.content_hash in session.seen_hashes:
                trace.rejected.append(
                    {"url": s.url, "reason": "duplicate of a source already read"}
                )
            else:
                fresh.append(s)
        budget_left = min(
            max(0, self.budget.max_extract_calls - session.extract_calls),
            max(0, self.budget.total_extract_calls - self._spent_extracts),
        )
        chosen = fresh[:budget_left]
        self._spent_extracts += len(chosen)
        for s in chosen:
            session.seen_hashes.add(s.content_hash)
        if not chosen:
            return 0

        def _extract(s: FetchedSource) -> tuple[FetchedSource, ExtractionResult | None, str]:
            """Read one source. A provider failure here costs us *that source*.

            Research is a best-effort gathering pass over many documents. One page whose
            extraction call will not come back is a page we did not read, and the honest
            response is to record it as unread and carry on with the rest — not to
            abandon the question. This is the opposite of an *actor* call failing, where
            a decision genuinely did not happen and the branch's mass must stay
            unresolved. The distinction is between "we could not read a source" and "we
            do not know what someone decided".
            """

            try:
                return s, extract_claims(self.gateway, question, s, as_of), ""
            except GatewayError as exc:
                return s, None, str(exc)

        with ThreadPoolExecutor(max_workers=self.budget.extract_concurrency) as pool:
            batches = list(pool.map(_extract, chosen))
        session.extract_calls += len(chosen)
        trace.extract_calls += len(chosen)
        added = 0
        for source, result, failure in batches:
            if result is None:
                trace.rejected.append(
                    {"url": source.fetched_url, "reason": f"extraction failed: {failure}"}
                )
                continue
            trace.extraction_calls.append(
                {
                    "fetched_url": source.fetched_url,
                    "content_sha256": source.content_hash,
                    "prompt_sha256": result.prompt_sha256,
                    "prompt": result.prompt,
                    "window_chars": result.window_chars,
                    "document_truncated": result.truncated,
                    "claims_returned": len(result.claims),
                }
            )
            for c in result.claims:
                if not c.verified_in_text:
                    trace.rejected.append(
                        {
                            "url": source.fetched_url,
                            "reason": f"claim not verified: {c.rejection_reason}",
                            "proposition": c.proposition[:_MAX_QUERY_CHARS],
                        }
                    )
                    continue
                if self._add_claim(store, source, c, result, as_of):
                    added += 1
        return added

    def _add_claim(
        self,
        store: EvidenceStore,
        source: FetchedSource,
        claim: ExtractedClaim,
        result: ExtractionResult,
        as_of: datetime,
    ) -> bool:
        available_at = source.evidence_time()
        cid = content_id("c", source.url, claim.proposition, claim.normalized_value)
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
                published_at=available_at,
                available_at=available_at,
                source_id=source.publisher or source.url,
                source_url=source.url,
                source_title=source.title,
                source_type=_source_type(claim.authority_hint),
                authority_level=AuthorityLevel(claim.authority_hint),
                supporting_excerpt=claim.supporting_excerpt,
                lineage_event_id=lineage,
                epistemic_type=EpistemicType(claim.epistemic_type),
                # The source's authority level normalized onto [0,1]. It is a rank of
                # source authority, NOT a probability that the claim is true, and no
                # invented arithmetic is applied to it.
                confidence=int(claim.authority_hint) / int(max(AuthorityLevel)),
                retrieved_at=source.fetched_at,
                retrieved_url=source.fetched_url,
                archived_at=source.archived_at,
                content_sha256=source.content_hash,
                extraction_prompt_sha256=result.prompt_sha256,
            )
        )
        # Defence in depth: the source gate above should already guarantee this, and a
        # claim invisible through the cutoff view is not progress even when it is stored.
        return store.claims[cid].is_available_at(as_of)

    # -- contradiction detection ------------------------------------------------

    def _detect_contradictions(
        self,
        store: EvidenceStore,
        as_of: datetime,
        trace: ResearchTrace,
        *,
        question: str = "",
        plan: ResearchPlan | None = None,
    ) -> None:
        """Find claims that disagree about the same fact and record the decisive ones.

        Candidates are claims available at the cutoff that describe the same subject
        (same proposition topic and same entities) but carry different canonical values.
        Sharing a subject is not yet a contradiction — "is Chair" and "is a member" are
        compatible — so each candidate pair is adjudicated by the model's
        ``contradiction`` task, and only an explicitly decisive verdict is recorded.

        Restricting candidates to the cutoff view keeps a post-cutoff claim from
        manufacturing a conflict that blocks a pastcast.
        """

        already = {tuple(sorted(p)) for p in store.contradictions()}
        groups: dict[str, list[EvidenceClaim]] = {}
        for claim in store.view(as_of).available():
            key = _subject_key(claim)
            if key:
                groups.setdefault(key, []).append(claim)
        for _key, claims in sorted(groups.items()):
            by_value: dict[str, EvidenceClaim] = {}
            for c in sorted(claims, key=lambda c: c.id):
                by_value.setdefault(c.normalized_value.strip().lower(), c)
            values = sorted(by_value)
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    if trace.contradiction_checks >= self.budget.max_contradiction_checks:
                        return
                    a, b = by_value[values[i]], by_value[values[j]]
                    if tuple(sorted((a.id, b.id))) in already:
                        continue
                    if not (_is_observation(a) and _is_observation(b)):
                        # A contradiction of *fact* needs two claims of fact. A live
                        # Banxico run was refused because "Banxico is prioritizing
                        # credibility over speed, implying gradual easing" was set
                        # against "Banxico signaled an end to rate cuts": two readings of
                        # the same posture, neither of them an observation, adjudicated
                        # as something the world cannot have both ways.
                        continue
                    trace.contradiction_checks += 1
                    if self._is_decisive_conflict(a, b, question=question, plan=plan):
                        store.record_contradiction(a.id, b.id)
                        already.add(tuple(sorted((a.id, b.id))))

    def _is_decisive_conflict(
        self,
        a: EvidenceClaim,
        b: EvidenceClaim,
        *,
        question: str = "",
        plan: ResearchPlan | None = None,
    ) -> bool:
        asked = (
            f"""
THE QUESTION THIS EVIDENCE IS FOR: {question}
The resolution event: {plan.resolution_event if plan else ""}
The deadline: {plan.deadline if plan else ""}

A difference that leaves the answer to that question unchanged is not decisive, whatever
else it is. Two outlets reporting a signing on 17 January and on 18 January disagree, and
a question asking whether the signing happens before 1 October is answered the same way by
both, so blocking on it refuses a question the evidence has already settled. Decisive
means the two claims imply different answers.
"""
            if question
            else ""
        )
        prompt = f"""Two evidence claims describe the same subject but carry different values.
Decide whether they are DECISIVELY contradictory: whether both cannot be true of the
same subject at the same time. Complementary facts, different aspects, different points
in time, or differing levels of detail are NOT contradictions.

Neither is a disagreement about what WILL happen. A decisive contradiction is about a
matter of present or past fact that the world cannot have both ways — a body with five
members and nine members, a rate held and cut on the same date, a person in office and
not in office. Two sources differing about a future decision, a forecast, an intention or
a direction of travel are not contradicting each other about reality; they are the
uncertainty the simulation exists to resolve, and calling that decisive refuses a
question that is merely genuinely open. Answer false for those.

Two claims about DIFFERENT quantities are not contradictory even when their values look
opposed: a group-wide baseline quota held steady and a subset's voluntary cut being
unwound are two true facts about different instruments, not a contradiction; a headline
figure and one of its components can both be true. Answer false when the claims measure
different things, cover different scopes, or apply to different sub-populations.

{asked}
CLAIM A: {a.proposition}
  value: {a.normalized_value}
  source: {a.source_id} ({a.published_at.date()})
  excerpt: {a.supporting_excerpt}

CLAIM B: {b.proposition}
  value: {b.normalized_value}
  source: {b.source_id} ({b.published_at.date()})
  excerpt: {b.supporting_excerpt}

Return JSON {{"decisive": true|false, "reason": "<one line>"}}."""
        try:
            resp = self.gateway.generate(
                GatewayRequest(
                    task_kind="contradiction",
                    prompt=prompt,
                    context={"a": a.id, "b": b.id},
                    seed=int(content_id("x", a.id, b.id)[-8:], 16),
                    expected_keys=("decisive",),
                )
            )
        except GatewayError:
            # Unadjudicated means unrecorded: we do not assert a conflict we could not
            # confirm, and we do not silently claim the pair is consistent either — the
            # check count in the trace shows the pair was examined.
            return False
        if resp.data.get("decisive") is not True:
            return False
        # A decisive verdict blocks the whole run and cannot be cleared by recompilation —
        # the contradiction is recorded on the store, so a false positive is fatal. Before
        # recording it, make one adversarial attempt to RECONCILE the pair: a live OPEC+
        # run was blocked because "group-wide quotas held for 2026" was set against "the
        # July increase unwinds the 2023 voluntary cuts" — two true facts about different
        # instruments. Only a contradiction that survives an honest reconciliation attempt
        # is real enough to end a run.
        return not self._can_reconcile(a, b, question=question)

    def _can_reconcile(self, a: EvidenceClaim, b: EvidenceClaim, *, question: str = "") -> bool:
        """Whether the two claims can BOTH be true under some reading — the skeptic's pass."""

        prompt = f"""Two evidence claims about the same subject carry different values and a
first reviewer called them decisively contradictory. Your job is the opposite: find the
reading, if one exists, under which BOTH claims are simply true at once.

They are reconcilable (both true) when they measure different quantities or instruments,
cover different scopes or sub-populations, describe different points in time, sit at
different levels of detail, or when one is a forecast/intention and the other a present
fact. They are genuinely contradictory ONLY when the same quantity, at the same time, is
asserted to be two values the world cannot hold at once.

Consider the question being answered: {question or "(none given)"}

CLAIM A: {a.proposition} — value: {a.normalized_value}
  excerpt: {a.supporting_excerpt}
CLAIM B: {b.proposition} — value: {b.normalized_value}
  excerpt: {b.supporting_excerpt}

Return JSON {{"reconcilable": true|false, "reading": "<one line: how both are true, or why they cannot both be>"}}."""
        try:
            resp = self.gateway.generate(
                GatewayRequest(
                    task_kind="contradiction",
                    prompt=prompt,
                    context={"a": a.id, "b": b.id, "pass": "reconcile"},
                    seed=int(content_id("reconcile", a.id, b.id)[-8:], 16),
                    expected_keys=("reconcilable",),
                )
            )
        except GatewayError:
            # If we cannot run the reconciliation pass, do not upgrade to a blocking
            # contradiction on the strength of one vote — a false block is unrecoverable.
            return True
        return resp.data.get("reconcilable") is True

    # -- required facts (retrieval only) ----------------------------------------

    def _record_fact_retrieval(
        self, plan: ResearchPlan, view: EvidenceView, trace: ResearchTrace
    ) -> list[str]:
        """Record, per planned required fact, the claims a lexical search retrieves, and
        return the facts nothing was retrieved for.

        This is retrieval, not verification: sharing a name or a number with a fact does
        not establish it. Nothing here is reported as coverage and it never stops
        research — its only job is to steer the next round's queries. The mechanical
        check that a required fact is actually supported is the compiled world's
        ``required_reality_facts``, where each fact must cite claim ids that exist and
        are available at the cutoff.
        """

        available = view.available()
        unsupported: list[str] = []
        trace.fact_retrieval = {}
        for fact in plan.required_facts:
            terms = distinctive_terms(fact)
            hits = (
                [c.id for c in available if distinctive_terms(_claim_text(c)) & terms]
                if terms
                else []
            )
            trace.fact_retrieval[fact] = sorted(hits)
            if not hits:
                unsupported.append(fact)
        return unsupported

    # -- compilation ------------------------------------------------------------

    def _compile(
        self,
        question: str,
        as_of: datetime,
        horizon: datetime,
        plan: ResearchPlan,
        store: EvidenceStore,
        trace: ResearchTrace,
    ) -> ResearchBundle:
        trace.claim_count = len(store.all())
        trace.admissible_claim_count = len(store.view(as_of).available())
        trace.contradictions = [f"{a}<>{b}" for a, b in store.contradictions()]
        try:
            # The initial live compilation dispatches through the SAME entry point as
            # the repair recompile, the structural alternatives and the frozen replay
            # (api.compile_for_mode), so the mode switch lives in exactly one place and
            # the compilation carries its compiler_mode stamp from the start. Imported
            # locally: api must not be imported at live_research module level.
            from .api import compile_for_mode

            compilation = compile_for_mode(
                _ModeDispatchConfig(gateway=self.gateway, compiler_mode=self.compiler_mode),
                question,
                as_of,
                horizon,
                store.view(as_of),
            )
        except WorldIntegrityError as exc:
            # The research preceding this refusal is COMPLETE — queries, sources,
            # claims, the whole record. It rides on the exception so the caller can
            # checkpoint it and name the true stage; without this, a compile-stage
            # refusal erased the run's entire research record and its diagnosis read
            # the zeros as a discovery failure.
            exc.partial_live_trace = trace.to_dict(plan, store)  # type: ignore[attr-defined]
            exc.partial_evidence_store = store  # type: ignore[attr-defined]
            raise
        data = {
            "world_spec": compilation["world_spec"],
            "uncertainties": compilation.get("uncertainties", []),
            "world_facts": compilation.get("world_facts", []),
            "required_reality_facts": compilation.get("required_reality_facts", []),
            "reality": {
                "subject_entity": compilation.get("subject_entity"),
                "resolution_units": compilation.get("resolution_units"),
                "target_outcome": compilation.get("target_outcome"),
                "expected_participants": compilation.get("expected_participants"),
                "as_of": as_of.isoformat(),
                "horizon": horizon.isoformat(),
                "authoritative_sources": list(plan.authoritative_sources),
            },
            "_compile_responses": list(compilation.get("_compile_responses") or []),
        }
        try:
            bundle = assemble_bundle(store, data)
        except (TypeError, ValueError, KeyError) as exc:
            # The compiler emitted a shape the parser cannot read. That is a defect in
            # one compilation, not a fact about the world, and it must not surface as a
            # raw TypeError from inside a parser — which is how a live run died before
            # it could write any diagnosis at all. Raised as recompilable so the repair
            # loop asks again, naming exactly what failed to parse.
            raise WorldIntegrityError(
                "the compiled world could not be parsed — the model emitted a shape "
                f"the schema does not allow: {type(exc).__name__}: {exc}",
                details={
                    "failure": "malformed_compilation",
                    "recompilable": True,
                    "parser_error": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        live_trace = trace.to_dict(plan, store)
        # The mode stamp is part of the run's record: which compiler produced this
        # world must be readable off research_trace.json, not inferred.
        live_trace["compiler_mode"] = str(compilation.get("compiler_mode") or self.compiler_mode)
        if "_semantic" in compilation:
            # The semantic plan, its independent review and the semantic→runtime mapping
            # are part of this run's record: they land in research_trace.json beside the
            # queries and sources, so a lowered world is always auditable back to the
            # meaning it lowered from.
            live_trace["semantic_compilation"] = compilation["_semantic"]
        return replace(bundle, live_trace=live_trace)

    def _time_up(self, t0: float) -> bool:
        return (time.monotonic() - t0) > self.budget.max_seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _missing_query(label: str) -> str:
    """Extract the identity from a coverage 'missing' label ('[kind] identity — why')
    so it can be researched directly."""

    text = label.split("]", 1)[1] if "]" in label else label
    return text.split("—", 1)[0].strip()


def _topic(proposition: str) -> str:
    return proposition.split(":", 1)[0].strip().lower() if ":" in proposition else "fact"


def _is_observation(claim: EvidenceClaim) -> bool:
    """Whether this claim states a fact rather than reading one.

    Only two statements of fact can contradict each other about reality. An inference
    and a hypothesis are the system's own reasoning, and two of them disagreeing is the
    uncertainty a simulation exists to resolve, not a reason to refuse one.
    """

    return claim.epistemic_type is EpistemicType.OBSERVATION


def _subject_key(claim: EvidenceClaim) -> str:
    """A claim's subject: its proposition topic plus the entities it is about.

    Two claims sharing a subject are talking about the same thing, which is the
    precondition for their values being able to disagree. A claim naming no entity has
    no identifiable subject and is not paired with anything.
    """

    if not claim.entities:
        return ""
    entities = ",".join(sorted(e.strip().lower() for e in claim.entities))
    return f"{_topic(claim.proposition)}|{entities}"


def _looks_like_domain(text: str) -> bool:
    candidate = text.strip().lower()
    return " " not in candidate and "." in candidate and not candidate.endswith(".")


def _claim_text(claim: EvidenceClaim) -> str:
    return f"{claim.proposition} {' '.join(claim.entities)} {claim.supporting_excerpt}"


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# The source category implied by each declared authority level. Keyed by the enum rather
# than by bare integers so the mapping stays total as the enum changes.
_SOURCE_TYPE_BY_AUTHORITY: dict[AuthorityLevel, SourceType] = {
    AuthorityLevel.AUTHORITATIVE: SourceType.OFFICIAL_INSTITUTIONAL,
    AuthorityLevel.HIGH: SourceType.PRIMARY_RECORD,
    AuthorityLevel.MEDIUM: SourceType.CONTEMPORANEOUS_REPORTING,
    AuthorityLevel.LOW: SourceType.LOW_QUALITY,
}


def _source_type(hint: int) -> SourceType:
    return _SOURCE_TYPE_BY_AUTHORITY[AuthorityLevel(hint)]
