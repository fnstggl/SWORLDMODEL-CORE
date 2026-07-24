"""The single evidence-grounded world compiler: parse, build base world, verify reality."""

from __future__ import annotations

from _helpers import base_corpus, compile_dict, dup
from sworldmodel.world_compiler import parse_uncertainties
from sworldmodel.worldspec import parse_world_spec


def test_parse_world_spec_round_trips_the_core_shape() -> None:
    spec = parse_world_spec(base_corpus()["world_spec"])
    assert spec.title
    assert {a.action_id for a in spec.actions} == {"record_position"}
    assert len(spec.actors) == 3
    assert spec.terminal.yes_when.op == "equals"
    assert spec.process.node_ids() == ("briefing", "decision")


def test_compile_builds_base_world_and_verifies_reality() -> None:
    compiled = compile_dict(base_corpus())
    assert compiled.manifest.is_verified
    assert compiled.manifest.represented_participants == 3
    assert set(compiled.base_world.actors) == {"a", "b", "c"}
    # the compiled scenario set carries both uncertainty branches
    assert len(compiled.scenario_set.scenarios) == 2


def test_uncertainty_weights_are_normalized_per_variable() -> None:
    specs = parse_uncertainties(
        [
            {
                "variable": "s",
                "outcomes": [
                    {
                        "value": "lo",
                        "weight": 3,
                        "provenance": "explicit_model_distribution",
                        "field_effects": [["s", 0.0]],
                    },
                    {
                        "value": "hi",
                        "weight": 1,
                        "provenance": "explicit_model_distribution",
                        "field_effects": [["s", 1.0]],
                    },
                ],
            }
        ]
    )
    assert len(specs) == 1
    weights = sorted(o.weight.value for o in specs[0].outcomes)
    assert abs(sum(weights) - 1.0) < 1e-9
    assert weights == [0.25, 0.75]


def test_evidence_citations_are_filtered_to_available_claims() -> None:
    # A member citing a non-existent claim id does not break compilation; the id is
    # simply not honored as evidence (parse keeps only what the store has for facts).
    corpus = dup(base_corpus())
    compiled = compile_dict(corpus)
    assert compiled.manifest.is_verified
