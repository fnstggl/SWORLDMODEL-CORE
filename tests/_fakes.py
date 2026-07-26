"""Test-only stand-ins.

These live under ``tests/`` on purpose. Nothing in ``src/sworldmodel`` imports them and
nothing in the production package can reach them: a deterministic reasoner or a
prepared corpus sitting in the shipped package is how a simulator ends up quietly
producing forecasts from hand-written rules while reporting that actors decided.

``ProgrammableGateway`` is deliberately *not* a "generic behavior model". It is a
script: a test states exactly what each actor returns at each invocation, so a test
that passes tells you what the runtime did with those answers, and never flatters the
runtime by having a stand-in reason plausibly on its behalf.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sworldmodel.errors import GatewayError
from sworldmodel.evidence import EvidenceClaim, EvidenceStore
from sworldmodel.gateway import GatewayRequest, GatewayResponse, ModelGateway
from sworldmodel.http import HttpCall, HttpError, HttpResponse
from sworldmodel.ids import canonical_json, prompt_hash
from sworldmodel.models import (
    AuthorityLevel,
    EpistemicType,
    RequiredRealityFact,
    SourceType,
)
from sworldmodel.research import ResearchBundle
from sworldmodel.world import WorldFact
from sworldmodel.world_compiler import (
    parse_required_facts,
    parse_uncertainties,
    parse_world_facts,
)
from sworldmodel.worldspec import parse_world_spec


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ProgrammableGateway(ModelGateway):
    """Returns whatever the test says, per task kind, with full call accounting.

    ``actor_decision`` may be a callable taking the decision context, so a test can
    make an actor's answer depend on what it actually perceived — which is how you
    prove the runtime delivered the right local view.
    """

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        fail_tasks: frozenset[str] = frozenset(),
        model: str = "programmable",
    ) -> None:
        super().__init__()
        self._responses = responses or {}
        self._fail = fail_tasks
        self._model = model
        self.seen: list[GatewayRequest] = []

    @property
    def model_id(self) -> str:
        return self._model

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        self.seen.append(request)
        if request.task_kind in self._fail:
            raise GatewayError(f"scripted failure for task {request.task_kind!r}")
        entry = self._responses.get(request.task_kind, {})
        data = entry(request.context) if callable(entry) else entry
        if not isinstance(data, dict):
            raise GatewayError(f"programmed response for {request.task_kind!r} is not an object")
        raw = canonical_json(data)
        return GatewayResponse(
            task_kind=request.task_kind,
            data=data,
            raw_text=raw,
            model=self._model,
            params={"temperature": request.temperature},
            seed=request.seed,
            prompt_hash=prompt_hash(request.prompt),
            tokens_in=_tokens(request.prompt),
            tokens_out=_tokens(raw),
        )


def wait_decision(reason: str = "nothing to do") -> dict[str, Any]:
    return {"plan_disposition": "continue", "action_mode": "wait", "reasoning": reason}


def act(action_id: str, params: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "plan_disposition": "continue",
        "action_mode": "compiled_action",
        "compiled_action_id": action_id,
        "params": params or {},
        "reasoning": f"taking {action_id}",
    }
    out.update(extra)
    return out


def propose(description: str, effects: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "plan_disposition": "continue",
        "action_mode": "novel_action",
        "novel_action": {
            "description": description,
            "target": extra.pop("target", ""),
            "parameters": {"effects": effects},
            "intended_effect": extra.pop("intended_effect", description),
        },
        "reasoning": description,
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# In-memory research bundles
# ---------------------------------------------------------------------------


@dataclass
class FixtureResearchBackend:
    """Wraps a pre-built bundle. ``is_live`` is False, so ``ForecastConfig.is_live`` is
    False and the CLI's live gate refuses it — which is the point."""

    bundle: ResearchBundle
    is_live: bool = False

    def research(self, question: str, as_of: datetime, horizon: datetime) -> ResearchBundle:
        return self.bundle


def build_bundle(data: dict[str, Any]) -> ResearchBundle:
    """Materialize a bundle from an authored dict (tests only)."""

    store = EvidenceStore()
    reality = data.get("reality", {})
    as_of = datetime.fromisoformat(reality["as_of"])
    horizon = datetime.fromisoformat(reality["horizon"])
    for raw in data.get("claims", []):
        at = datetime.fromisoformat(raw.get("published_at", reality["as_of"]))
        cid = str(raw["id"])
        store.add(
            EvidenceClaim(
                id=cid,
                proposition=str(raw["proposition"]),
                normalized_value=str(raw.get("value", "")),
                entities=tuple(raw.get("entities", []) or []),
                valid_from=at,
                valid_until=None,
                published_at=at,
                available_at=at,
                source_id=str(raw.get("source_id", f"src_{cid}")),
                source_url=str(raw.get("source_url", "https://example.test/doc")),
                source_title=str(raw.get("source_title", "fixture source")),
                source_type=SourceType(raw.get("source_type", "official_institutional")),
                authority_level=AuthorityLevel[str(raw.get("authority", "AUTHORITATIVE"))],
                supporting_excerpt=str(raw.get("supporting_excerpt", "excerpt")),
                lineage_event_id=str(raw.get("lineage_event_id", f"ev_{cid}")),
                epistemic_type=EpistemicType(raw.get("epistemic_type", "observation")),
                confidence=float(raw.get("confidence", 0.9)),
                retrieved_at=at,
            )
        )
    spec = parse_world_spec(data["world_spec"])
    available = {c.id for c in store.all()}
    return ResearchBundle(
        evidence_store=store,
        spec=spec,
        uncertainties=parse_uncertainties(data.get("uncertainties"), available),
        world_facts=parse_world_facts(data.get("world_facts"), as_of),
        required_reality_facts=parse_required_facts(data.get("required_reality_facts")),
        subject_entity=str(reality.get("subject_entity", spec.subject_entity or "subject")),
        resolution_units=str(reality.get("resolution_units", spec.resolution_units or "outcome")),
        target_outcome=str(reality.get("target_outcome", "")),
        expected_participants=reality.get("expected_participants"),
        authoritative_sources=tuple(reality.get("authoritative_sources", []) or []),
        horizon=horizon,
        research_plan=("fixture",),
        as_of=as_of,
    )


def fact(text: str, *, at: datetime, claims: tuple[str, ...] = ()) -> WorldFact:
    return WorldFact(
        fact_id=f"f_{abs(hash(text)) % 10**8}",
        text=text,
        evidence_claim_ids=claims,
        available_at=at,
    )


def required(key: str, description: str, claims: tuple[str, ...]) -> RequiredRealityFact:
    return RequiredRealityFact(key=key, description=description, evidence_claim_ids=claims)


DecisionFn = Callable[[dict[str, Any]], dict[str, Any]]


# ---------------------------------------------------------------------------
# HTTP transport double (relocated from sworldmodel.http — M-8: a canned-response
# transport must not live in the production package)
# ---------------------------------------------------------------------------

Route = tuple[Callable[[str], bool], object]


class FakeTransport:
    """Deterministic transport for tests. Routes are matched in order; each route maps
    a URL predicate to an ``HttpResponse`` or a callable ``(method, url, body)`` ->
    ``HttpResponse``. Unmatched requests raise ``HttpError`` (a network failure).

    It performs no DNS and enforces no fetch policy: policy belongs at the socket, and a
    test double that resolved hostnames could not serve fixture domains.
    """

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.calls: list[HttpCall] = []
        # Branches are simulated concurrently, so the call log is written from several
        # threads. It is an audit record: losing an entry would understate what the run
        # actually did on the wire.
        self._lock = threading.Lock()

    def add(self, predicate: Callable[[str], bool], response: object) -> FakeTransport:
        self.routes.append((predicate, response))
        return self

    def add_url(self, url: str, response: HttpResponse) -> FakeTransport:
        return self.add(lambda u: u == url, response)

    def _dispatch(self, method: str, url: str, body: dict[str, object] | None) -> HttpResponse:
        for predicate, response in self.routes:
            if predicate(url):
                resp = response(method, url, body) if callable(response) else response
                assert isinstance(resp, HttpResponse)
                self.calls.append(
                    HttpCall(
                        method, url, resp.final_url, resp.status, resp.elapsed_ms, len(resp.text)
                    )
                )
                return resp
        self.calls.append(HttpCall(method, url, url, 0, 0, 0, error="no route"))
        raise HttpError(f"FakeTransport: no route for {method} {url}")

    def get(
        self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0
    ) -> HttpResponse:
        return self._dispatch("GET", url, None)

    def post_json(
        self,
        url: str,
        body: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        return self._dispatch("POST", url, body)

    def post_form(
        self,
        url: str,
        form: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        return self._dispatch("POST", url, {"form": form})
