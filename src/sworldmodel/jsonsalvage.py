"""Recover the usable prefix of a JSON reply the provider cut off mid-object.

Adapted from the legacy repository's ``_salvage_json`` (``swm/world_model_v2/compiler.py``),
whose own experiment forensics identified truncation as the cause of exactly the symptom
this system kept hitting: a world compilation that returns *nothing*, and therefore no
actors, no actions and no process — a hollow world blamed on the evidence.

The current gateway responded to truncation by doubling ``max_tokens`` and retrying from
scratch, discarding a prefix that was often almost the whole world. A compiled world is
the largest JSON object this system ever asks for; if the retry truncates too, the run
gets nothing at all from two expensive calls.

Salvage does not invent. It closes brackets and strings the provider left open, and if
that will not parse, trims the trailing partial member and tries again. Everything it
returns was produced by the model. What was cut off stays missing — and missing is
visible to the schema check, which is the caller's cue to ask for the rest.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["salvage_json"]

_FENCE = re.compile(r"```(?:json)?|```")
# How many trailing members to trim before giving up. A truncated tail is one partial
# member; more than a handful means the reply was not a JSON object at all.
_MAX_TRIMS = 40


def salvage_json(text: str) -> dict[str, Any] | None:
    """The largest valid object that is a prefix of ``text``, or ``None``.

    Returns ``None`` rather than ``{}`` for "nothing usable", so a caller cannot mistake
    an unsalvageable reply for a successfully parsed empty world.
    """

    if not isinstance(text, str):
        return None
    body = _FENCE.sub("", text).strip()
    start = body.find("{")
    if start < 0:
        return None
    body = body[start:]

    closed = _close(body)
    obj = _load(closed)
    if obj is not None:
        return obj

    # Drop the trailing partial member and re-close, repeatedly. Each cut removes one
    # incomplete element from the end while keeping everything the model finished.
    for _ in range(_MAX_TRIMS):
        cut = max(body.rfind(","), body.rfind("}"), body.rfind("]"))
        if cut <= 0:
            return None
        body = body[:cut]
        obj = _load(_close(body))
        if obj:
            return obj
    return None


def _close(body: str) -> str:
    """Close every string and bracket the text left open, in the right order."""

    in_string = False
    escaped = False
    stack: list[str] = []
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    return body + ('"' if in_string else "") + "".join(reversed(stack))


def _load(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None
