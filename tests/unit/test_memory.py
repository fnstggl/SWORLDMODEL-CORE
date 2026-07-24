from __future__ import annotations

from datetime import datetime, timedelta

from sworldmodel.memory import MemoryStream
from sworldmodel.models import MemorySeed

T0 = datetime.fromisoformat("2024-01-01T00:00:00+00:00")


def test_seeding_gives_non_empty_persistent_memory() -> None:
    mem = MemoryStream()
    mem.seed(
        (
            MemorySeed(
                content="I favored hold last time", kind="episodic", importance=0.8, valid_time=T0
            ),
            MemorySeed(
                content="Inflation is the key risk", kind="semantic", importance=0.6, valid_time=T0
            ),
        ),
        default_time=T0,
    )
    assert len(mem) == 2


def test_retrieve_returns_relevant_and_refreshes_access() -> None:
    mem = MemoryStream()
    mem.add_memory("inflation risk is elevated", kind="semantic", importance=0.6, created=T0)
    mem.add_memory("the weather is nice today", kind="episodic", importance=0.2, created=T0)
    now = T0 + timedelta(days=1)
    out = mem.retrieve("what about inflation", now=now, top_k=1)
    assert len(out) == 1
    assert "inflation" in out[0].content
    # retrieval refreshes last_accessed
    assert mem.get(out[0].node_id).last_accessed == now


def test_changing_a_memory_changes_what_is_retrieved() -> None:
    def top(content: str) -> str:
        mem = MemoryStream()
        mem.add_memory(content, kind="semantic", importance=0.5, created=T0)
        mem.add_memory("unrelated council logistics", kind="episodic", importance=0.5, created=T0)
        got = mem.retrieve("growth is slowing sharply", now=T0, top_k=1)
        return got[0].content

    assert top("growth is slowing sharply this quarter") != top("prices are stable and calm")
