"""Canonical, explicit exceptions for the world model.

Every failure mode that must *stop* a run (rather than be silently repaired) has a
dedicated exception. The kernel never swallows these into a prior or a default;
they propagate so a caller can see exactly why the world could not be simulated.
"""

from __future__ import annotations


class SWorldModelError(Exception):
    """Base class for all world-model errors."""


class WorldIntegrityError(SWorldModelError):
    """Raised when the represented world does not faithfully match verified reality.

    This is the reality-integrity invariant. It is intentionally *not* recoverable:
    a structurally false world (missing seats, rescaled threshold, wrong roster) must
    stop the run, not be repaired into a different question.
    """

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        base = super().__str__()
        if not self.details:
            return base
        lines = [base]
        for key, value in self.details.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


class CutoffViolationError(SWorldModelError):
    """Raised when a fact available only after ``as_of`` is used in a pastcast."""


class ContractMutationError(SWorldModelError):
    """Raised when a downstream component tries to change a verified contract field."""


class IntentValidationError(SWorldModelError):
    """Raised when an actor intent is infeasible, unauthorized, or ill-formed."""


class MassConservationError(SWorldModelError):
    """Raised when branch weights fail to conserve total probability mass."""


class EvidenceError(SWorldModelError):
    """Raised for malformed evidence claims or unresolved decisive contradictions."""


class GatewayError(SWorldModelError):
    """Raised when a model gateway fails to produce a usable structured response.

    A gateway failure must leave explicitly *unresolved* mass — it must never be
    converted into a prior or a default action.
    """


class UndeterminedExpressionError(SWorldModelError):
    """Raised when a declarative expression reads a value the world never determined.

    This is not a defect in the world: a compiled terminal may legitimately reference a
    date or field that a given branch never sets. The honest outcome is an *unresolved*
    branch — never a crash, and never a fabricated YES/NO.
    """
