"""The model gateway: the single, swappable boundary to a reasoning model.

The kernel only talks to a model through :class:`ModelGateway`. Two implementations
ship here:

* :class:`DeterministicGateway` — a transparent, generic behavior model used for
  offline runs and all unit tests. It contains **no scenario facts** (no names,
  dates, rates); it reasons purely over the *structure* of its typed input. This is
  what makes runs reproducible and tests deterministic, and it is honestly labeled
  as a ``calibrated_behavior_model`` wherever its outputs feed a branch weight.
* :class:`ScriptedGateway` — replays a fixed mapping (used to test provider-failure
  handling and malformed-output repair).

A live provider gateway would implement the same interface (sending
``request.prompt``); because the prompt is rendered from the same context the
deterministic reasoner reads, "the view used in the trace is the view sent to the
model" holds for both.

The gateway is *not* used to count votes, apply thresholds, invent authoritative
facts, change the contract, or write terminal outcomes — those are code.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from typing import Any

from .errors import GatewayError
from .ids import canonical_json, prompt_hash
from .models import IntentKind


@dataclass(frozen=True)
class GatewayRequest:
    task_kind: str
    prompt: str
    context: dict[str, Any]
    seed: int
    temperature: float = 0.0
    expected_keys: tuple[str, ...] = ()


@dataclass
class GatewayResponse:
    task_kind: str
    data: dict[str, Any]
    raw_text: str
    model: str
    params: dict[str, Any]
    seed: int
    prompt_hash: str
    tokens_in: int
    tokens_out: int
    retries: int = 0
    validation_failures: tuple[str, ...] = ()
    latency_ms: int = 0


class ModelGateway(abc.ABC):
    """Base gateway with thread-safe call/token/latency accounting.

    ``is_live`` distinguishes real provider gateways (that make network calls) from
    the deterministic/scripted gateways used only in tests. The production CLI
    refuses to label a run "live" unless the gateway is live.
    """

    is_live: bool = False

    def __init__(self) -> None:
        self.call_count: int = 0
        self.total_tokens: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.retries: int = 0
        self.failed_calls: int = 0
        self.calls: list[GatewayResponse] = []
        self.stage_calls: dict[str, int] = {}
        self.latencies_ms: list[int] = []
        self._lock = threading.Lock()

    @property
    @abc.abstractmethod
    def model_id(self) -> str: ...

    @abc.abstractmethod
    def _generate(self, request: GatewayRequest) -> GatewayResponse: ...

    def generate(self, request: GatewayRequest) -> GatewayResponse:
        response = self._generate(request)
        with self._lock:
            self.call_count += 1
            self.total_tokens += response.tokens_in + response.tokens_out
            self.total_tokens_in += response.tokens_in
            self.total_tokens_out += response.tokens_out
            self.retries += response.retries
            self.calls.append(response)
            self.stage_calls[response.task_kind] = self.stage_calls.get(response.task_kind, 0) + 1
            if response.latency_ms:
                self.latencies_ms.append(response.latency_ms)
        return response

    def note_failure(self) -> None:
        with self._lock:
            self.failed_calls += 1

    def stage_call_counts(self) -> dict[str, int]:
        return dict(self.stage_calls)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ScriptedGateway(ModelGateway):
    """Returns a pre-set response per task kind, or raises to simulate a provider
    failure. Used to prove that a gateway failure leaves *unresolved* mass rather
    than becoming a prior or a default action."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        *,
        fail_tasks: frozenset[str] = frozenset(),
        model: str = "scripted",
    ) -> None:
        super().__init__()
        self._responses = responses or {}
        self._fail = fail_tasks
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        if request.task_kind in self._fail:
            raise GatewayError(f"scripted failure for task {request.task_kind!r}")
        data = self._responses.get(request.task_kind, {})
        raw = canonical_json(data)
        return GatewayResponse(
            task_kind=request.task_kind,
            data=data,
            raw_text=raw,
            model=self._model,
            params={"temperature": request.temperature},
            seed=request.seed,
            prompt_hash=prompt_hash(request.prompt),
            tokens_in=_estimate_tokens(request.prompt),
            tokens_out=_estimate_tokens(raw),
        )


class DeterministicGateway(ModelGateway):
    """A generic, transparent reasoning stand-in.

    Every method reasons only over the structure of ``request.context``. It never
    references a scenario by name. Swap it for a live LLM gateway and the pipeline is
    unchanged.
    """

    @property
    def model_id(self) -> str:
        return "deterministic-reasoner-v1"

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        dispatch = {
            "compile_world": self._compile_world,
            "actor_decision": self._actor_decision,
            "reflect": self._reflect,
        }
        handler = dispatch.get(request.task_kind)
        if handler is None:
            raise GatewayError(f"DeterministicGateway has no handler for {request.task_kind!r}")
        data = handler(request.context)
        raw = canonical_json(data)
        return GatewayResponse(
            task_kind=request.task_kind,
            data=data,
            raw_text=raw,
            model=self.model_id,
            params={"temperature": 0.0, "deterministic": True},
            seed=request.seed,
            prompt_hash=prompt_hash(request.prompt),
            tokens_in=_estimate_tokens(request.prompt),
            tokens_out=_estimate_tokens(raw),
        )

    # -- compile ----------------------------------------------------------------

    def _compile_world(self, ctx: dict[str, Any]) -> dict[str, Any]:
        frame = ctx["frame"]
        guidance = frame.get("guidance_option")
        options: list[str] = list(ctx["options"])
        shared_rules = list(frame.get("reaction_rules", []))
        actors_out = []
        for member in ctx["members"]:
            prior = member.get("prior_action")
            # Members share the same evidence-grounded reaction rules. Heterogeneity
            # comes from their dissent thresholds and focal-proposal acceptance
            # (below), which are grounded in each member's revealed prior position —
            # not from personality labels or a hand-tuned per-member reaction map.
            member_rules = [dict(r) for r in shared_rules]
            # Current inclination is the newly-established common position where one
            # exists; otherwise the actor's own most recent revealed position. This is
            # derived from evidence, never assigned as a trait.
            if guidance is not None:
                inclination = guidance
            elif prior is not None:
                inclination = prior
            else:
                inclination = options[0]
            # A seat that just held a *different* position than the new common one is
            # structurally more willing to break from it again.
            if prior is not None and guidance is not None and prior != guidance:
                dissent_threshold = 0.65
                reasoning = (
                    f"Prior revealed position was {prior!r}; now accepts the common "
                    f"position {inclination!r} but remains readier to diverge."
                )
            elif prior is not None and prior == inclination:
                dissent_threshold = 0.4
                reasoning = f"Prior position {prior!r} already matches the common position."
            else:
                dissent_threshold = 0.5
                reasoning = f"Adopts the common position {inclination!r}; no divergent prior."
            actors_out.append(
                {
                    "actor_id": member["actor_id"],
                    "current_inclination": inclination,
                    "reaction_rules": member_rules,
                    "dissent_threshold": dissent_threshold,
                    "reasoning": reasoning,
                    "evidence_claim_ids": member.get("evidence_claim_ids", []),
                }
            )
        return {"actors": actors_out}

    # -- actor decision ---------------------------------------------------------

    def _actor_decision(self, ctx: dict[str, Any]) -> dict[str, Any]:
        stage: str = ctx["stage"]
        feasible: list[str] = ctx["feasible_actions"]
        inclination: str = ctx["current_inclination"]
        options: list[str] = list(ctx["options"])
        tolerance: float = float(ctx.get("acceptance_tolerance", 0.5))
        dissent: float = float(ctx.get("dissent_threshold", 0.5))
        rules: list[dict[str, Any]] = ctx.get("reaction_rules", [])
        active_proposal: dict[str, Any] | None = ctx.get("active_proposal")
        memories: list[dict[str, Any]] = ctx.get("retrieved_memories", [])
        observations: list[dict[str, Any]] = ctx.get("observations", [])

        # Assemble every signal the actor can currently observe: released data plus
        # any substantive information carried inside colleague statements.
        signals: dict[str, float] = dict(ctx.get("observed_signals", {}))
        reacted_obs: list[str] = []
        for obs in observations:
            info = obs.get("info_signals") or {}
            for name, level in info.items():
                signals[name] = float(level)
                reacted_obs.append(obs["obs_id"])

        preferred, triggered = _apply_reaction_rules(inclination, rules, signals)

        # A durable commitment memory narrows what this actor will accept.
        committed = _committed_option(memories)
        used_memory_ids = tuple(m["id"] for m in memories if _is_commitment(m))

        # ---- stage: initial -> the actor rationally waits for the proposal -------
        if stage == "initial" and active_proposal is None:
            return {
                "kind": IntentKind.WAIT,
                "vote_option": "",
                "rationale": (
                    "No proposal or colleague positions available yet; waiting to hear "
                    "them before committing a vote."
                ),
                "expected_effect": "Await briefing, proposal, and colleagues' positions.",
                "pending_need": "await_proposal_and_positions",
                "referenced_memory_ids": [],
                "referenced_observation_ids": [],
            }

        # ---- stage: act (non-committee) -> take the resolving action or wait ------
        if stage == "act":
            # options[0] is the "no action" baseline; any other preferred option is a
            # substantive action (respond / commit). This maps a response/negotiation
            # question onto the same evidence-grounded reaction machinery.
            if preferred != options[0]:
                return {
                    "kind": IntentKind.MAKE_COMMITMENT,
                    "vote_option": "",
                    "text": _statement_text(ctx["actor_id"], preferred, triggered, committed),
                    "rationale": f"I take the action: {preferred}.",
                    "expected_effect": f"Perform action {preferred!r}.",
                    "referenced_memory_ids": list(used_memory_ids),
                    "referenced_observation_ids": sorted(set(reacted_obs)),
                }
            return {
                "kind": IntentKind.WAIT,
                "vote_option": "",
                "rationale": "No trigger to act; I take no action for now.",
                "expected_effect": "No action taken.",
                "pending_need": "await_trigger",
                "referenced_memory_ids": [],
                "referenced_observation_ids": sorted(set(reacted_obs)),
            }

        # ---- stage: positions -> the actor states a substantive position ---------
        if stage == "positions" and IntentKind.MAKE_STATEMENT in feasible:
            share = {
                r["trigger_signal"]: signals[r["trigger_signal"]]
                for r in rules
                if r["trigger_signal"] in signals
            }
            text = _statement_text(ctx["actor_id"], preferred, triggered, committed)
            return {
                "kind": IntentKind.MAKE_STATEMENT,
                "vote_option": "",
                "favored_option": preferred,
                "statement_text": text,
                "info_signals": share,
                "rationale": f"Stating current position: {preferred!r}.",
                "expected_effect": "Colleagues learn my position and reasoning.",
                "referenced_memory_ids": list(used_memory_ids),
                "referenced_observation_ids": sorted(set(reacted_obs)),
            }

        # ---- stage: decision -> cast a final vote --------------------------------
        if IntentKind.CAST_VOTE in feasible:
            vote = preferred
            accepted_focal = False
            if active_proposal is not None:
                prop = active_proposal["option"]
                if _accepts_focal_proposal(prop, preferred, options, tolerance, dissent, committed):
                    vote = prop
                    accepted_focal = _bool_flag(prop != preferred)
            rationale = _vote_rationale(
                preferred, triggered, active_proposal, accepted_focal, committed
            )
            return {
                "kind": IntentKind.CAST_VOTE,
                "vote_option": vote,
                "rationale": rationale,
                "expected_effect": f"Record a vote for {vote!r}.",
                "referenced_memory_ids": list(used_memory_ids),
                "referenced_observation_ids": sorted(set(reacted_obs)),
            }

        # No feasible terminal action available: wait rather than fabricate one.
        return {
            "kind": IntentKind.WAIT,
            "vote_option": "",
            "rationale": "No feasible decision action available at this stage.",
            "expected_effect": "Wait for the decision to open.",
            "referenced_memory_ids": [],
            "referenced_observation_ids": [],
        }

    # -- reflect ----------------------------------------------------------------

    def _reflect(self, ctx: dict[str, Any]) -> dict[str, Any]:
        obs = ctx.get("observations", [])
        insights = []
        for o in obs:
            if o.get("kind") in {
                "external_data_released",
                "statement_made",
                "briefing_distributed",
            }:
                insights.append(
                    {
                        "content": f"Noted: {o.get('summary', o.get('obs_id'))}",
                        "kind": "semantic",
                        "importance": 0.5,
                        "evidence_claim_ids": o.get("evidence_claim_ids", []),
                    }
                )
        return {"beliefs_update": [], "new_memories": insights, "plan_note": ""}


# ---------------------------------------------------------------------------
# Pure reasoning helpers (generic; no scenario knowledge)
# ---------------------------------------------------------------------------


def _bool_flag(value: bool) -> bool:
    return bool(value)


def _apply_reaction_rules(
    inclination: str, rules: list[dict[str, Any]], signals: dict[str, float]
) -> tuple[str, dict[str, Any] | None]:
    """Return ``(preferred_option, triggered_rule_or_None)``.

    Among all rules whose signal crosses its threshold, the one with the largest
    margin wins (deterministic tie-break by list order). No numeric averaging.
    """

    best_margin = 0.0
    best: dict[str, Any] | None = None
    for rule in rules:
        level = signals.get(rule["trigger_signal"])
        if level is None:
            continue
        threshold = float(rule["threshold"])
        crossed = (rule["direction"] == "above" and level >= threshold) or (
            rule["direction"] == "below" and level <= threshold
        )
        if not crossed:
            continue
        margin = abs(level - threshold)
        if best is None or margin > best_margin:
            best_margin = margin
            best = rule
    if best is None:
        return inclination, None
    return best["moves_to_option"], best


def _ordinal_distance(a: str, b: str, options: list[str]) -> int:
    try:
        return abs(options.index(a) - options.index(b))
    except ValueError:
        return 0 if a == b else 99


def _accepts_focal_proposal(
    proposal: str,
    preferred: str,
    options: list[str],
    tolerance: float,
    dissent: float,
    committed: str | None,
) -> bool:
    """Discrete focal-proposal acceptance within an evidence-grounded tolerance band.

    This is *not* scalar consensus smoothing: there is no moving mean and no
    per-round convergence. A seat accepts the single focal proposal on the table
    only when it is at most one notch from its own preference and its openness
    exceeds its willingness to hold a divergent line. A commitment overrides.
    """

    if committed is not None and proposal != committed:
        return False
    dist = _ordinal_distance(proposal, preferred, options)
    if dist == 0:
        return True
    # Accept only a one-notch move to the focal proposal, and only if openness beats
    # the seat's willingness to hold a divergent line. No moving mean, no per-round pull.
    return dist == 1 and tolerance >= dissent


def _is_commitment(memory: dict[str, Any]) -> bool:
    return any(str(t).startswith("commitment:") for t in memory.get("tags", []))


def _committed_option(memories: list[dict[str, Any]]) -> str | None:
    for m in memories:
        for tag in m.get("tags", []):
            if str(tag).startswith("commitment:"):
                return str(tag).split(":", 1)[1]
    return None


def _statement_text(
    actor_id: str, preferred: str, triggered: dict[str, Any] | None, committed: str | None
) -> str:
    parts = [f"My position is to {preferred}."]
    if triggered is not None:
        parts.append(
            f"The {triggered['trigger_signal']} reading moves me toward {triggered['moves_to_option']}."
        )
    if committed is not None:
        parts.append(f"I remain committed to {committed} from my earlier position.")
    else:
        parts.append("Absent a decisive new development I expect to keep this position.")
    return " ".join(parts)


def _vote_rationale(
    preferred: str,
    triggered: dict[str, Any] | None,
    proposal: dict[str, Any] | None,
    accepted_focal: bool,
    committed: str | None,
) -> str:
    if accepted_focal and proposal is not None:
        return (
            f"My own lean is {preferred!r}, but the proposal {proposal['option']!r} is within "
            f"my tolerance, so I support the focal proposal."
        )
    reason = f"I vote my considered preference, {preferred!r}."
    if triggered is not None:
        reason += f" The {triggered['trigger_signal']} signal crossed my reaction threshold."
    if committed is not None:
        reason += f" I hold to my prior commitment to {committed!r}."
    return reason
