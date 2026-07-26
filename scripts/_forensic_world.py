#!/usr/bin/env python3
"""Compatibility shim: the forensic replay world now lives in the shared replay core.

The read-only :class:`~sworldmodel.expressions.ExprContext` adapter that presents
replayed ledger state to the engine's own evaluator was promoted into
:mod:`sworldmodel.replaycore` (decision D7: one ledger-replay implementation, consumed
by the production trace writer, the publication gate, and the forensic tool alike).
This module only re-exports it under its historical name so anything that imported
``ForensicWorld`` from here keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sworldmodel.replaycore import ReplayWorld as ForensicWorld  # noqa: E402

__all__ = ["ForensicWorld"]
