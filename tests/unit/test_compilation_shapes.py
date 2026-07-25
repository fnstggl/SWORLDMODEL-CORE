"""A compiler's punctuation must not be able to kill a run.

Two live runs died in parsers, not in reasoning. One wrote `{"op": "const", "args":
false}` — the single argument bare rather than in a list — and iterating a bool raised
TypeError from inside `parse_expr`, during research, before the repair loop exists, so
the run left a stack trace and no diagnosis at all.

Sweeping the parsers with the shapes a real model actually drifts into found ten more.
Each one is a live run that would have ended the same way, so they are all pinned here.

The rule these encode: **coerce shape, never invent content.** A lone object becomes a
one-element list; an absent field becomes an empty one; anything that is not an object
is dropped rather than guessed at. Nothing unsaid becomes said, and the integrity gates
still see exactly what the compiler produced — including its omissions.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from sworldmodel.evidence import EvidenceStore
from sworldmodel.research import assemble_bundle
from sworldmodel.worldspec import Expr, as_objects, parse_expr

BASE: dict[str, Any] = {
    "world_spec": {
        "title": "w",
        "entities": [
            {
                "entity_id": "a",
                "name": "A",
                "kind": "person",
                "is_actor": True,
                "role": "member",
                "authority": ["decide"],
                "evidence_claim_ids": [],
            }
        ],
        "actors": [{"entity_id": "a", "reasoning": "r", "memory_seeds": []}],
        "fields": [{"field_id": "f", "value_type": "number", "initial": 0}],
        "resources": [],
        "channels": [],
        "documents": [],
        "actions": [
            {
                "action_id": "act",
                "meaning": "m",
                "eligible_actors": ["*"],
                "effects": [{"op": "set_field", "field": "f", "value": 1}],
                "evidence_claim_ids": [],
            }
        ],
        "process": {
            "nodes": [
                {
                    "node_id": "n",
                    "stage": "s",
                    "description": "d",
                    "at": "2026-06-25T00:00:00+00:00",
                    "participants": ["*"],
                    "action_ids": ["*"],
                }
            ]
        },
        "terminal": {
            "yes_when": {"op": "equals", "args": [{"op": "field", "args": ["f"]}, 1]},
            "unresolved_when": {"op": "const", "args": [False]},
            "description": "t",
        },
    },
    "uncertainties": [],
    "world_facts": [],
    "required_reality_facts": [],
    "reality": {
        "subject_entity": "s",
        "resolution_units": "u",
        "target_outcome": "o",
        "as_of": "2026-05-14T00:00:00+00:00",
        "horizon": "2026-06-25T00:00:00+00:00",
    },
}

WS = ["world_spec"]

# (name, path into the compilation, the drifted value a model actually writes)
DRIFT: list[tuple[str, list[str], Any]] = [
    (
        "terminal args written bare",
        [*WS, "terminal", "unresolved_when"],
        {"op": "const", "args": False},
    ),
    ("terminal as a bare string", [*WS, "terminal", "yes_when"], "f == 1"),
    ("entities as one bare object", [*WS, "entities"], {"entity_id": "a", "name": "A"}),
    ("actors null", [*WS, "actors"], None),
    (
        "authority as a string",
        [*WS, "entities"],
        [{"entity_id": "a", "name": "A", "authority": "decide"}],
    ),
    (
        "claim ids as a string",
        [*WS, "entities"],
        [{"entity_id": "a", "name": "A", "evidence_claim_ids": "c1"}],
    ),
    (
        "action effects null",
        [*WS, "actions"],
        [{"action_id": "act", "meaning": "m", "effects": None}],
    ),
    (
        "effect with no op",
        [*WS, "actions"],
        [{"action_id": "act", "meaning": "m", "effects": [{"field": "f", "value": 1}]}],
    ),
    ("process as a bare node list", [*WS, "process"], [{"node_id": "n", "participants": ["*"]}]),
    ("process nodes null", [*WS, "process"], {"nodes": None}),
    ("uncertainties as one object", ["uncertainties"], {"variable": "f", "outcomes": []}),
    ("uncertainty outcomes null", ["uncertainties"], [{"variable": "f", "outcomes": None}]),
    (
        "field_effects written flat",
        ["uncertainties"],
        [{"variable": "f", "outcomes": [{"value": "v", "weight": 1, "field_effects": ["f", 1]}]}],
    ),
    ("world_facts as a string", ["world_facts"], "nothing known"),
    ("required facts as one object", ["required_reality_facts"], {"key": "k"}),
    ("wake_rules as a string", [*WS], dict(BASE["world_spec"], wake_rules="none")),
]


def _mutate(path: list[str], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE)
    node: Any = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return data


@pytest.mark.parametrize(("name", "path", "value"), DRIFT, ids=[d[0] for d in DRIFT])
def test_compiler_drift_never_crashes_the_parser(name: str, path: list[str], value: Any) -> None:
    """Every one of these raised TypeError, AttributeError, KeyError or ValueError from
    inside a parser. A run may be refused; it may not die of a comma."""

    assemble_bundle(EvidenceStore(), _mutate(path, value))


def test_a_lone_argument_is_a_one_argument_list() -> None:
    assert parse_expr({"op": "const", "args": False}) == Expr("const", (False,))
    assert parse_expr({"op": "field", "args": "rate"}) == Expr("field", ("rate",))
    assert parse_expr({"op": "const", "args": None}) == Expr("const", ())
    nested = parse_expr({"op": "equals", "args": [{"op": "field", "args": ["x"]}, 3]})
    assert nested.op == "equals" and nested.args[1] == 3


def test_the_runtime_survives_compiler_drift_too() -> None:
    """The parsers were not the only place a near-miss could kill a run.

    Sweeping the *runtime* with the same kind of drift — an effect op outside the
    universal set, an action writing an undeclared field, a node dated "soon", a
    wake rule naming an entity that does not exist, a cost written as a one-element
    list — every case must either run or be refused. Only the malformed cost crashed,
    with IndexError from a tuple unpack inside the parser.
    """

    from sworldmodel.worldspec import parse_action

    action = parse_action(
        {
            "action_id": "act",
            "meaning": "m",
            "resource_costs": [["only_one"], ["votes", 2], ["bad", "amount"], "junk"],
        }
    )
    # Only the entries that are genuinely (resource, amount) pairs survive.
    assert action.resource_costs == (("votes", 2.0),)


def test_coercion_never_invents_content() -> None:
    """The line this robustness must not cross."""

    assert as_objects({"a": 1}) == [{"a": 1}]  # a lone object is a one-element list
    assert as_objects(None) == []  # absent is empty, not a placeholder
    assert as_objects("text") == []  # a non-object is dropped, never guessed at
    assert as_objects([{"a": 1}, "junk", None]) == [{"a": 1}]  # partial input keeps what is real


def test_a_null_quantity_is_read_as_none_written_not_as_a_crash() -> None:
    """A live OPEC+ run died on `float(None)` in the resource parser, during a repair
    recompile, past every gate that would have turned it into a diagnosis — out through
    the entry point, leaving a traceback and no artifacts at all. `d.get(k, 0.0)` returns
    None when the key is present and null, which is a shape a model produces routinely."""

    from sworldmodel.worldspec import parse_world_spec

    spec = parse_world_spec(
        {
            "title": "t",
            "entities": [{"entity_id": "a", "name": "A", "kind": "organization"}],
            "actors": [],
            "fields": [],
            "resources": [
                {"resource_id": "quota", "holder_entity_id": "a", "quantity": None},
                {"resource_id": "spare", "holder_entity_id": "a", "quantity": "1.5"},
                {"resource_id": "junk", "holder_entity_id": "a", "quantity": "n/a"},
            ],
            "channels": [],
            "documents": [],
            "actions": [],
            "process": {"nodes": []},
            "terminal": {"yes_when": {"op": "const", "args": [True]}},
        }
    )
    assert [r.quantity for r in spec.resources] == [0.0, 1.5, 0.0]
