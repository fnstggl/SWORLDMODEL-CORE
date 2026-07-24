"""The mandatory acceptance tests for the universal simulator.

These lock in the central architectural correction: SWORLDMODEL is ONE universal world
simulator that compiles the actual causal world for each arbitrary question. It is not a
committee simulator, a router between fixed scenario types, or a collection of hardcoded
action families. Every test below runs the *same* runtime and evaluator on *data-only*
worlds.
"""

from __future__ import annotations

import copy
import pathlib
import re
from typing import Any

import pytest

import _worlds as W
from sworldmodel.engine import evaluate_terminal
from sworldmodel.worldspec import parse_world_spec

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "sworldmodel"
# The production runtime path — the files that must contain no question-family routing.
RUNTIME_FILES = [
    SRC / n
    for n in (
        "engine.py",
        "executor.py",
        "effects.py",
        "expressions.py",
        "world.py",
        "actors.py",
        "novel.py",
        "world_compiler.py",
        "reality.py",
        "worldspec.py",
        "models.py",
        "outcomes.py",
        "uncertainty.py",
    )
]


def _decisions(ctx: Any) -> list[Any]:
    return ctx.run_result.actor_decisions


# 1. Unknown-action test ------------------------------------------------------


def test_scenario_executes_action_absent_from_source_code() -> None:
    action_name = "zorptcast_the_glyph"
    # The action name appears nowhere in the source tree.
    for path in SRC.glob("*.py"):
        assert action_name not in path.read_text(), f"{action_name} leaked into {path.name}"

    result, ctx = W.run_corpus(W.unknown_action_world(action_name))
    assert result.status.value == "resolved"
    assert result.simulation_probability == 1.0  # the invented action executed
    executed = [d for d in _decisions(ctx) if d.status == "executed"]
    assert any(d.choice["action_id"] == action_name for d in executed)


# 2. Novel-action accepted ----------------------------------------------------


def test_novel_action_accepted_because_authorized_and_feasible() -> None:
    result, ctx = W.run_corpus(W.novel_action_world(mode="accept"))
    novel = [d for d in _decisions(ctx) if d.choice["mode"] == "novel_action"]
    assert novel, "expected a novel action proposal"
    assert all(d.status == "executed" for d in novel)
    # The novel action id is NOT in the compiled menu, yet it resolved the world YES.
    assert result.simulation_probability == 1.0


# 3. Novel-action rejected for lack of authority ------------------------------


def test_novel_action_rejected_for_missing_authority() -> None:
    result, ctx = W.run_corpus(W.novel_action_world(mode="reject_authority"))
    novel = [d for d in _decisions(ctx) if d.choice["mode"] == "novel_action"]
    assert novel and all(d.status == "rejected" for d in novel)
    assert any("authority" in d.reason for d in novel)
    assert result.simulation_probability == 0.0  # nothing decisive was recorded


# 4. Action-name independence -------------------------------------------------


def _rename_actions(corpus: dict[str, Any], suffix: str) -> dict[str, Any]:
    c = copy.deepcopy(corpus)
    ws = c["world_spec"]
    remap = {a["action_id"]: a["action_id"] + suffix for a in ws["actions"]}
    for a in ws["actions"]:
        a["action_id"] = remap[a["action_id"]]
    for node in ws["process"]["nodes"]:
        node["action_ids"] = [remap.get(x, x) for x in node.get("action_ids", [])]
    for actor in ws["actors"]:
        pol = actor.get("policy", {})
        if pol.get("default_action_id") in remap:
            pol["default_action_id"] = remap[pol["default_action_id"]]
        for rule in pol.get("rules", []):
            if rule.get("action_id") in remap:
                rule["action_id"] = remap[rule["action_id"]]
    return c


def test_renaming_every_action_does_not_change_execution() -> None:
    base = W.committee_world({"a": "hold", "b": "hold", "c": "cut"})
    r_base, _ = W.run_corpus(base)
    r_ren, _ = W.run_corpus(_rename_actions(base, "_xyz"))
    assert r_base.simulation_probability == r_ren.simulation_probability
    base_dist = sorted(sorted(v for _, v in b.records) for b in r_base.branch_outcomes)
    ren_dist = sorted(sorted(v for _, v in b.records) for b in r_ren.branch_outcomes)
    assert base_dist == ren_dist


# 5. No scenario-family router in the production path -------------------------


FORBIDDEN_ROUTING = [
    "committee_vote",
    "actor_action",
    "weighted_majority",
    "committee_protocol",
    "general_protocol",
    "evaluate_action_terminal",
    "terminal.mechanism",
    ".mechanism ==",
]


def test_production_path_has_no_mechanism_family_router() -> None:
    blob = "\n".join(p.read_text() for p in RUNTIME_FILES)
    for token in FORBIDDEN_ROUTING:
        assert token not in blob, f"runtime still routes on {token!r}"
    # No branch keys on a question family anywhere in the runtime.
    family = re.compile(
        r'==\s*[\'"](committee|negotiation|election|geopolitical|population|response)[\'"]'
    )
    assert not family.search(blob), "runtime branches on a question-family literal"


# 6. Dynamic terminal: different expressions, one evaluator -------------------


def test_two_questions_compile_different_terminals_same_evaluator() -> None:
    committee = parse_world_spec(W.committee_world({"a": "hold", "b": "hold"})["world_spec"])
    response = parse_world_spec(W.individual_response_world()["world_spec"])
    # Different declarative terminal predicates...
    assert committee.terminal.yes_when != response.terminal.yes_when
    assert committee.terminal.yes_when.op != response.terminal.yes_when.op
    # ...evaluated by the SAME universal function (imported once, used for both).
    assert evaluate_terminal.__module__ == "sworldmodel.engine"
    r1, _ = W.run_corpus(W.committee_world({"a": "hold", "b": "hold"}))
    r2, _ = W.run_corpus(W.individual_response_world())
    assert r1.status.value == "resolved" and r2.status.value == "resolved"


# 7. Dynamic protocol: different event graphs, no source change ---------------


def test_two_scenarios_compile_different_process_graphs() -> None:
    committee = parse_world_spec(W.committee_world({"a": "hold"})["world_spec"])
    negotiation = parse_world_spec(W.negotiation_world()["world_spec"])
    geo = parse_world_spec(W.geopolitical_world()["world_spec"])
    graphs = {
        "committee": tuple((n.node_id, n.stage) for n in committee.process.nodes),
        "negotiation": tuple((n.node_id, n.stage) for n in negotiation.process.nodes),
        "geopolitical": tuple((n.node_id, n.stage) for n in geo.process.nodes),
    }
    assert len({graphs["committee"], graphs["negotiation"], graphs["geopolitical"]}) == 3


# 8. Novel causal pathway changes a trajectory --------------------------------


def test_novel_action_opens_a_pathway_that_changes_the_trajectory() -> None:
    with_novel, _ = W.run_corpus(W.novel_action_world(mode="accept"))

    # The same world, but the node forbids novel actions: the only compiled action is a
    # no-op, so the decisive record is never created and the trajectory flips.
    suppressed = W.dup(W.novel_action_world(mode="accept"))
    for node in suppressed["world_spec"]["process"]["nodes"]:
        node["allow_novel"] = False
    without_novel, ctx = W.run_corpus(suppressed)

    assert with_novel.simulation_probability == 1.0
    assert without_novel.simulation_probability == 0.0
    assert all(d.choice["mode"] != "novel_action" for d in _decisions(ctx))


# 9. Unrepresentable-action honesty -------------------------------------------


def test_unrepresentable_novel_action_is_rejected_not_coerced() -> None:
    result, ctx = W.run_corpus(W.novel_action_world(mode="unrepresentable"))
    novel = [d for d in _decisions(ctx) if d.choice["mode"] == "novel_action"]
    assert novel and all(d.status == "rejected" for d in novel)
    assert any("unrepresentable" in d.reason or "no safe" in d.reason for d in novel)
    # It was NOT silently converted into the nearest known action (noop):
    assert all(d.choice.get("action_id") != "noop" for d in novel)
    # No decisive record was fabricated.
    assert result.simulation_probability == 0.0


# 10. Cross-domain, question-only, one code path ------------------------------


def test_materially_different_questions_share_one_runtime() -> None:
    worlds = {
        "individual_response": W.individual_response_world(),
        "committee_decision": W.committee_world({"a": "hold", "b": "hold", "c": "hold"}),
        "negotiation": W.negotiation_world(),
        "population_behavior": W.population_world(),
        "geopolitical_process": W.geopolitical_world(),
    }
    probs = {}
    for name, corpus in worlds.items():
        result, _ = W.run_corpus(corpus)
        assert result.status.value == "resolved", name
        assert result.probability_source == "weighted_simulated_trajectories", name
        probs[name] = result.simulation_probability
    # The worlds are genuinely different (not a single template): distinct probabilities.
    assert len(set(probs.values())) >= 4, probs


@pytest.mark.parametrize(
    "mode,expected", [("accept", 1.0), ("reject_authority", 0.0), ("unrepresentable", 0.0)]
)
def test_novel_action_matrix(mode: str, expected: float) -> None:
    result, _ = W.run_corpus(W.novel_action_world(mode=mode))
    assert result.simulation_probability == expected
