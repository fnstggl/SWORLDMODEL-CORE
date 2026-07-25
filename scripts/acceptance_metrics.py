#!/usr/bin/env python3
"""The PART 18 machine-readable metrics, per case and averaged across all five.

Reads only what the runs wrote. Every field traces to a specific artifact, so the
numbers cannot drift from the traces they claim to summarize. Writes
``artifacts/acceptance/metrics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CASES = ("geopolitical", "individual", "negotiation", "population", "committee")
ROOT = Path(__file__).resolve().parent.parent / "artifacts" / "acceptance"


def load(case: str, name: str) -> Any:
    for base in (ROOT / case / "run_trace", ROOT / case):
        p = base / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
    return None


def _int(x: Any) -> int:
    return int(x) if isinstance(x, (int, float)) else 0


def case_metrics(case: str) -> dict[str, Any]:
    d = load(case, "diagnosis.json")
    if not isinstance(d, dict) or "root_cause_vocabulary" not in d:
        return {"case": case, "status": "no artifact"}

    rt = load(case, "research_trace.json") or {}
    provider = rt.get("provider_requests") or {}
    disc, fetch = d["discovery"], d["fetching"]
    epi = (d["epistemic_classification"] or {}).get("counts") or {}
    audit = load(case, "run_audit.json") or {}
    stages = audit.get("model_calls_by_stage") or {}
    runtime = d.get("runtime") or {}
    integ = d.get("forecast_integrity") or {}

    rss = rt.get("rss_requests") or []
    decoded = sum(_int(r.get("decoded")) for r in rss)

    def stage_calls(*kinds: str) -> int:
        return sum(_int(stages.get(k)) for k in kinds)

    return {
        "case": case,
        "outcome": d["outcome"],
        "logical_search_queries": len(d["research_planning"]["queries_issued"]),
        "google_news_rss_requests": _int(provider.get("google_news_rss")),
        "google_news_urls_decoded": decoded,
        "official_discovery_requests": len(disc.get("official_domain_urls_found") or []),
        "jina_search_requests": _int(provider.get("jina_search")),
        "jina_reader_requests": _int(provider.get("jina_reader")),
        "serper_requests": _int(provider.get("serper")),
        "optional_duckduckgo_requests": _int(provider.get("duckduckgo")),
        "urls_discovered": disc.get("urls_considered_count", 0),
        "pages_fetched": fetch.get("fetched_count", 0),
        "pages_accepted": fetch.get("fetched_count", 0),
        "pages_rejected": fetch.get("rejected_count", 0),
        "claims_verified": _int(epi.get("verified")),
        "claims_inferred": _int(epi.get("inferred")),
        "claims_hypothetical": _int(epi.get("hypothetical")),
        "claims_unsupported": _int(epi.get("unsupported")),
        "research_llm_calls": stage_calls(
            "research_plan", "followup_queries", "extract_claims", "contradiction"
        ),
        "compilation_llm_calls": stage_calls("compile_world_spec", "exclusion_challenge"),
        "actor_llm_calls": stage_calls("actor_decision", "interpret_novel", "reflect"),
        "audit_llm_calls": stage_calls("world_review", "trajectory_audit", "assess_structure"),
        "total_llm_calls": d.get("model_calls", 0),
        "provider_failures": sum(
            v.get("blocks", 0) + v.get("errors", 0) + v.get("timeouts", 0)
            for v in (rt.get("provider_health") or {}).values()
        ),
        "retries": audit.get("retries", 0),
        "wall_seconds": d.get("wall_seconds", 0),
        "event_count": runtime.get("event_count", 0),
        "actor_invocations": runtime.get("actor_invocations", 0),
        "resolved_yes_mass": _forecast(case, "resolved_yes_mass"),
        "resolved_no_mass": _forecast(case, "resolved_no_mass"),
        "unresolved_mass": runtime.get("unresolved_mass"),
        "probability_before_simulation": integ.get("probability_before_simulation"),
        "probability_after_simulation": integ.get("probability_after_simulation"),
        "point_estimate_is_calibrated": integ.get("point_estimate_is_calibrated"),
        "trajectory_classification": (d.get("trajectory_audit") or {}).get("classification"),
        "total_tokens": _int(audit.get("tokens_in")) + _int(audit.get("tokens_out")),
    }


def _forecast(case: str, key: str) -> Any:
    f = load(case, "forecast.json") or {}
    return f.get(key)


def main() -> int:
    per_case = [case_metrics(c) for c in CASES]
    numeric_keys = [
        k
        for k, v in (per_case[0] if per_case else {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    completed = [m for m in per_case if m.get("outcome") == "completed"]
    averages = {
        k: round(sum(m.get(k, 0) or 0 for m in per_case) / len(per_case), 2) for k in numeric_keys
    }
    out = {
        "commit": (ROOT / "status.log").read_text().splitlines()[0].split(":", 1)[1].strip()
        if (ROOT / "status.log").exists()
        else "?",
        "cases": per_case,
        "averages_across_all_five": averages,
        "completed_count": len(completed),
    }
    (ROOT / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
