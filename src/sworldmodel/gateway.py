"""The model gateway: the single, swappable boundary to a reasoning model.

The kernel only talks to a model through :class:`ModelGateway`. Two implementations
ship here:

* :class:`DeterministicGateway` — a transparent, generic behavior model used for
  offline runs and all unit tests. It contains **no scenario facts** (no names,
  dates, rates, and no question-family knowledge); it reasons purely over the
  *structure* of its typed input — the compiled action menu, the actor's compiled
  policy, and the observed field levels. This is what makes runs reproducible and
  tests deterministic, and it is honestly labeled as a ``calibrated_behavior_model``
  wherever its outputs feed a branch weight.
* :class:`ScriptedGateway` — replays a fixed mapping (used to test provider-failure
  handling and malformed-output repair).

A live provider gateway would implement the same interface. Because the prompt is
rendered from the same context the deterministic reasoner reads, "the view used in the
trace is the view sent to the model" holds for both.

The gateway is *not* used to count records, apply thresholds, invent authoritative
facts, change the contract, or write terminal outcomes — those are code.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass
from typing import Any

from .errors import GatewayError
from .ids import canonical_json, prompt_hash


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
    """Base gateway with thread-safe call/token/latency accounting."""

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
    failure. Used to prove that a gateway failure leaves *unresolved* mass rather than
    becoming a prior or a default action, and to feed a fixed compiled world for
    live-path plumbing tests."""

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

    Every method reasons only over the *structure* of ``request.context``: the actor's
    compiled policy, the field levels it currently observes, and the feasible action
    menu. It never references a scenario by name and knows nothing of votes, committees,
    negotiations, or any question family. Swap it for a live LLM gateway and the pipeline
    is unchanged."""

    @property
    def model_id(self) -> str:
        return "deterministic-reasoner-v1"

    def _generate(self, request: GatewayRequest) -> GatewayResponse:
        dispatch = {
            "actor_decision": self._actor_decision,
            "reflect": self._reflect,
            "interpret_novel": self._interpret_novel,
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

    # -- actor decision ---------------------------------------------------------

    def _actor_decision(self, ctx: dict[str, Any]) -> dict[str, Any]:
        policy = ctx.get("policy", {}) or {}
        observed = ctx.get("observed_fields", {}) or {}
        allow_novel = bool(ctx.get("allow_novel", True))
        feasible_ids = {c.get("action_id") for c in ctx.get("feasible_actions", [])}

        # First matching policy rule wins; otherwise the default disposition applies.
        chosen: dict[str, Any] | None = None
        for rule in policy.get("rules", []):
            if _rule_matches(rule, observed):
                chosen = {
                    "action_id": rule.get("action_id", ""),
                    "params": rule.get("params", {}) or {},
                    "novel": rule.get("novel", {}) or {},
                }
                break
        if chosen is None:
            chosen = {
                "action_id": policy.get("default_action_id", ""),
                "params": policy.get("default_params", {}) or {},
                "novel": policy.get("default_novel", {}) or {},
            }

        novel = chosen.get("novel") or {}
        params = dict(chosen.get("params") or {})
        target = str(params.pop("target", ""))

        base = {
            "reasoning": "acting on compiled disposition given observed conditions",
            "referenced_memory_ids": [m.get("id") for m in ctx.get("retrieved_memories", [])][:3],
            "referenced_observation_ids": [o.get("obs_id") for o in ctx.get("observations", [])],
        }
        if novel and allow_novel:
            return {
                **base,
                "action_mode": "novel_action",
                "novel_action": {
                    "description": str(novel.get("description", "")),
                    "target": str(novel.get("target", target)),
                    "parameters": novel.get("parameters", novel.get("params", {})) or {},
                    "intended_effect": str(novel.get("intended_effect", "")),
                },
            }
        action_id = str(chosen.get("action_id", ""))
        if action_id and action_id in feasible_ids:
            return {
                **base,
                "action_mode": "compiled_action",
                "compiled_action_id": action_id,
                "params": params,
                "target": target,
            }
        return {
            **base,
            "action_mode": "wait",
            "reasoning": "no feasible compiled action fits the current situation",
        }

    # -- novel-action interpretation --------------------------------------------

    def _interpret_novel(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Offline, an authored novel proposal carries its own safe effect mapping in
        its parameters; the interpreter simply returns it (or declares it unrepresentable
        when no mapping is provided — never approximating it into a known action)."""

        params = ctx.get("parameters", {}) or {}
        effects = params.get("effects")
        if isinstance(effects, list) and effects:
            return {
                "representable": True,
                "reason": "authored safe-effect mapping",
                "required_authority": list(params.get("required_authority", []) or []),
                "effects": effects,
            }
        return {"representable": False, "reason": "no safe universal representation available"}

    # -- reflect ----------------------------------------------------------------

    def _reflect(self, ctx: dict[str, Any]) -> dict[str, Any]:
        insights = []
        for o in ctx.get("observations", []):
            if o.get("kind") in {"release_data", "create_event", "deliver_information"}:
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


def _rule_matches(rule: dict[str, Any], observed: dict[str, Any]) -> bool:
    fieldname = str(rule.get("when_field", ""))
    op = str(rule.get("op", "present"))
    if op == "present":
        return fieldname in observed
    if fieldname not in observed:
        return False
    cur = observed[fieldname]
    target = rule.get("value")
    if op == "above":
        return _num(cur) >= _num(target)
    if op == "below":
        return _num(cur) <= _num(target)
    if op == "equals":
        if _is_num(cur) and _is_num(target):
            return _num(cur) == _num(target)
        return str(cur) == str(target)
    return False


def _is_num(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
