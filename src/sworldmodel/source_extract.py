"""LLM claim extraction with verification.

The LLM reads a *fetched* page and extracts claims relevant to the question. A claim
is kept only if its supporting excerpt actually appears in the fetched text — a page
that does not contain supporting material is rejected. This is the verification step
that stops fabricated citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .gateway import GatewayRequest, ModelGateway
from .ids import prompt_hash
from .source_fetch import FetchedSource

_PROMPT_WINDOW = 9000  # per-source prompt window; the canonical store keeps ALL claims


@dataclass(frozen=True)
class ExtractedClaim:
    proposition: str
    normalized_value: str
    entities: tuple[str, ...]
    epistemic_type: str
    supporting_excerpt: str
    authority_hint: int
    verified_in_text: bool


def extract_claims(
    gateway: ModelGateway, question: str, source: FetchedSource, as_of: datetime
) -> list[ExtractedClaim]:
    if not source.ok:
        return []
    window = source.text[:_PROMPT_WINDOW]
    prompt = f"""Extract factual claims from this source that bear on the question.
Only extract claims the TEXT actually supports. For each claim provide a short
supporting_excerpt copied VERBATIM from the text. Do not invent anything.

QUESTION: {question}
SOURCE URL: {source.final_url}
PUBLISHER: {source.publisher}

TEXT:
\"\"\"
{window}
\"\"\"

Return JSON {{"claims": [ {{
  "proposition": "<short factual statement, prefixed with a topic namespace like 'roster:' or 'vote:' or 'rule:' or 'date:' or 'context:'>",
  "normalized_value": "<canonical value>",
  "entities": ["..."],
  "epistemic_type": "observation" | "inference" | "hypothesis",
  "supporting_excerpt": "<verbatim snippet from the text>",
  "authority_hint": 1-4
}} ] }}. If the page supports nothing relevant, return {{"claims": []}}."""
    resp = gateway.generate(
        GatewayRequest(
            task_kind="extract_claims",
            prompt=prompt,
            context={"url": source.final_url},
            seed=int(prompt_hash(source.final_url)[:8], 16),
            expected_keys=("claims",),
        )
    )
    out: list[ExtractedClaim] = []
    haystack = _norm(source.text)
    for c in resp.data.get("claims", []):
        if not isinstance(c, dict):
            continue
        excerpt = str(c.get("supporting_excerpt", "")).strip()
        verified = bool(excerpt) and _contains(haystack, excerpt)
        out.append(
            ExtractedClaim(
                proposition=str(c.get("proposition", "")).strip(),
                normalized_value=str(c.get("normalized_value", "")).strip(),
                entities=tuple(str(e).strip() for e in c.get("entities", []) if str(e).strip()),
                epistemic_type=_epi(str(c.get("epistemic_type", "observation"))),
                supporting_excerpt=excerpt,
                authority_hint=_auth(c.get("authority_hint")),
                verified_in_text=verified,
            )
        )
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _contains(haystack: str, excerpt: str) -> bool:
    needle = _norm(excerpt)
    if len(needle) < 8:
        return needle in haystack
    # Tolerant containment: require a strong contiguous overlap of the excerpt.
    if needle in haystack:
        return True
    words = needle.split()
    if len(words) >= 6:
        # allow a 6-word contiguous span match (handles minor whitespace/entity diffs)
        for i in range(len(words) - 5):
            span = " ".join(words[i : i + 6])
            if span in haystack:
                return True
    return False


def _epi(value: str) -> str:
    v = value.lower().strip()
    return v if v in {"observation", "inference", "hypothesis"} else "observation"


def _auth(value: object) -> int:
    try:
        return max(1, min(4, int(str(value))))
    except (TypeError, ValueError):
        return 2
