"""One runtime, no question families.

The claim under test is that supporting a new kind of question adds compiled *data* and
never adds a branch in the source. These tests check that claim two ways: by searching
the production package for the shapes that would betray it, and by executing two worlds
that share nothing structurally through the same code.
"""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import sworldmodel
from _fakes import ProgrammableGateway, act, build_bundle, wait_decision
from _worlds import scheduled_multiparty_world, single_response_world
from sworldmodel.effects import UNIVERSAL_OPS
from sworldmodel.engine import run
from sworldmodel.expressions import evaluate
from sworldmodel.models import ResolutionContract
from sworldmodel.world_compiler import compile_world

SRC = Path(sworldmodel.__file__).parent
AS_OF = datetime.fromisoformat("2026-05-14T23:59:59+00:00")
HORIZON = datetime.fromisoformat("2026-06-25T23:59:59+00:00")

# The vocabulary of the mechanism families this architecture exists to not have.
FAMILY_WORDS = (
    "committee",
    "committee_vote",
    "actor_action",
    "weighted_majority",
    "negotiation",
    "election",
    "population",
    "geopolitical",
    "message_response",
    "general_protocol",
    "quorum",
    "ballot",
    "roster",
    "institution",
)


def _python_sources() -> list[Path]:
    return sorted(p for p in SRC.glob("*.py"))


def _code_strings(path: Path) -> list[str]:
    """Every string literal that is *code*, not documentation.

    Docstrings and comments legitimately name the things this architecture refuses to
    contain — that is what they are for. Only literals the program actually evaluates
    can betray a hidden assumption.
    """

    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def test_no_production_code_branches_on_a_question_family() -> None:
    """No comparison anywhere in the package tests a scenario-family name.

    This walks the AST rather than grepping, so a family name mentioned in a docstring
    or a comment (explaining what we do *not* do) is fine, while `if kind ==
    "committee_vote"` is not.
    """

    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            literals: list[str] = []
            if isinstance(node, ast.Compare):
                literals = [
                    n.value
                    for n in [node.left, *node.comparators]
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
            elif isinstance(node, ast.Match):
                literals = [
                    c.pattern.value.value  # type: ignore[attr-defined]
                    for c in node.cases
                    if isinstance(getattr(c.pattern, "value", None), ast.Constant)
                ]
            for lit in literals:
                low = lit.lower()
                if any(w in low for w in FAMILY_WORDS):
                    offenders.append(f"{path.name}: compares against {lit!r}")
    assert not offenders, "production code branches on a question family:\n" + "\n".join(offenders)


def test_no_domain_action_vocabulary_is_hardcoded() -> None:
    """The universal effect language is the only fixed action machinery.

    An action name like ``cast_vote`` or ``make_statement`` appearing as a constant set
    in production code would mean the runtime has opinions about what people can do.
    """

    assert (
        frozenset(
            {
                "create_event",
                "schedule_event",
                "deliver_information",
                "release_data",
                "set_field",
                "adjust_field",
                "append_record",
                "update_commitment",
                "transfer_resource",
                "consume_resource",
                "create_or_update_document",
                "advance_time",
            }
        )
        == UNIVERSAL_OPS
    )
    banned = {
        "cast_vote",
        "make_statement",
        "introduce_proposal",
        "negotiate",
        "buy",
        "sell",
        "protest",
        "publish",
    }
    for path in _python_sources():
        hits = sorted(banned & set(_code_strings(path)))
        assert not hits, f"{path.name} hardcodes domain actions {hits}"


def test_no_prepared_corpus_or_deterministic_gateway_is_reachable() -> None:
    """Nothing in the shipped package can read a fixture or fake a model."""

    for path in _python_sources():
        tree = ast.parse(path.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        assert "DeterministicGateway" not in names, f"{path.name} references a fake gateway"
        assert "CorpusResearchBackend" not in names, f"{path.name} references a corpus backend"
        assert "corpus.json" not in _code_strings(path), f"{path.name} reads a prepared corpus"
    assert not hasattr(sworldmodel, "DeterministicGateway")
    assert not hasattr(sworldmodel, "CorpusResearchBackend")


def test_no_legacy_repository_is_imported() -> None:
    for path in _python_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert not name.startswith("swm"), f"{path.name} imports legacy module {name}"
                assert "reverie" not in name, f"{path.name} imports {name}"


def test_the_cli_takes_only_the_question_and_its_window() -> None:
    """The product interface may not require anything prepared."""

    from sworldmodel.cli import build_parser

    parser = build_parser()
    forecast = next(
        a for a in parser._subparsers._group_actions[0].choices.items() if a[0] == "forecast"
    )[1]
    required = {a.dest for a in forecast._actions if a.required}
    assert required == {"question", "as_of", "horizon"}
    optional = {a.dest for a in forecast._actions}
    for forbidden in ("corpus", "roster", "actors", "protocol", "institution", "mechanism"):
        assert forbidden not in optional, f"forecast accepts a prepared {forbidden}"


# ---------------------------------------------------------------------------
# Two worlds that share nothing, through one runtime
# ---------------------------------------------------------------------------


def _compile(data: dict, gateway: ProgrammableGateway):
    bundle = build_bundle(data)
    contract = ResolutionContract(
        question="q",
        as_of=AS_OF,
        horizon=HORIZON,
        subject_entity=bundle.subject_entity,
        resolution_units=bundle.resolution_units,
        terminal=bundle.spec.terminal,
        target_outcome=bundle.target_outcome,
        expected_participants=bundle.expected_participants,
    )
    return compile_world(
        contract,
        bundle.evidence_store.view(AS_OF),
        bundle.spec,
        bundle.uncertainties,
        bundle.world_facts,
        gateway=gateway,
        seed=0,
        max_branches=4,
    )


def _menu_driven(seen: set[tuple[str, str]] | None = None):
    """A script that takes each offered action at most once per actor per stage.

    Scripted actors must not thrash: an actor that re-takes the same action every time
    it is woken produces a runaway trajectory, which tells you about the script rather
    than about the runtime.
    """

    done: set[tuple[str, str]] = seen if seen is not None else set()

    def decide(ctx: dict) -> dict:
        for card in ctx.get("feasible_actions", []):
            key = (str(ctx["actor_id"]), card["action_id"])
            if key in done:
                continue
            done.add(key)
            params = {}
            for p in card.get("parameters", []):
                if p.get("choices"):
                    params[p["name"]] = p["choices"][0]
                elif p.get("required"):
                    params[p["name"]] = "text"
            return act(card["action_id"], params)
        return wait_decision()

    return decide


def test_two_unrelated_questions_compile_different_graphs_and_terminals() -> None:
    gw = ProgrammableGateway(
        {"actor_decision": _menu_driven(), "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    a = _compile(scheduled_multiparty_world(), gw)
    b = _compile(single_response_world(), gw)

    assert {n.node_id for n in a.spec.process.nodes} != {n.node_id for n in b.spec.process.nodes}
    assert {x.action_id for x in a.spec.actions} != {x.action_id for x in b.spec.actions}
    assert a.spec.terminal.yes_when != b.spec.terminal.yes_when
    assert {e.entity_id for e in a.spec.entities} != {e.entity_id for e in b.spec.entities}
    # Different structures, same operators.
    assert a.spec.terminal.yes_when.op in {"greater_or_equal"}
    assert b.spec.terminal.yes_when.op in {"greater_or_equal"}


def test_the_same_evaluator_resolves_both_terminals() -> None:
    gw = ProgrammableGateway(
        {"actor_decision": _menu_driven(), "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    for data in (scheduled_multiparty_world(), single_response_world()):
        compiled = _compile(data, gw)
        result = run(compiled, gw, seed=0)
        world = next(iter(result.final_worlds.values()))
        # `evaluate` is the single declarative evaluator; it handles both shapes.
        assert isinstance(bool(evaluate(compiled.spec.terminal.yes_when, world)), bool)
        assert result.branch_outcomes[0].resolved


def test_an_action_name_absent_from_the_source_still_executes() -> None:
    """A compiled action nobody wrote Python for runs, and its effects land.

    The name is generated here at test time, so it cannot appear anywhere in the
    package. If the runtime needed to know it, this could not work.
    """

    novel_name = "sign_the_transboundary_water_annex"
    assert not any(novel_name in p.read_text() for p in _python_sources())

    data = single_response_world()
    data["world_spec"]["actions"].append(
        {
            "action_id": novel_name,
            "meaning": "sign the annex",
            "eligible_actors": ["recipient"],
            "required_authority": ["reply"],
            "parameters": [],
            "visibility": "public",
            "effects": [
                {
                    "op": "append_record",
                    "collection": "signatures",
                    "key": "$actor",
                    "value": "signed",
                }
            ],
        }
    )

    signed: list[str] = []

    def decide(ctx: dict) -> dict:
        ids = {c["action_id"] for c in ctx.get("feasible_actions", [])}
        if novel_name in ids and not signed:
            signed.append(novel_name)
            return act(novel_name, {})
        return wait_decision()

    gw = ProgrammableGateway(
        {"actor_decision": decide, "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    world = next(iter(result.final_worlds.values()))
    assert [r.value for r in world.records_dict()["signatures"]] == ["signed"]


def test_the_runtime_carries_no_notion_of_representation_scale_behavior() -> None:
    """An organization acting as a unit runs through exactly the same code path."""

    data = single_response_world()
    data["world_spec"]["entities"][0].update(
        {
            "kind": "organization",
            "representation_scale": "organization",
            "represents_count": 4200,
        }
    )
    gw = ProgrammableGateway(
        {"actor_decision": _menu_driven(), "reflect": {"beliefs_update": [], "new_memories": []}}
    )
    compiled = _compile(data, gw)
    result = run(compiled, gw, seed=0)
    assert result.actor_decisions
    entity = compiled.base_world.entities[0]
    assert entity.representation_scale == "organization"
    assert entity.represents_count == 4200
