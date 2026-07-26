#!/bin/bash
# Final A/B benchmark: every case in artifacts/ab_matrix.json (acceptance + holdouts),
# both compiler modes, identical frozen stores, ONE unchanged commit. Per case the two
# modes run concurrently; cases are sequential to bound provider load. Every run's
# summary line lands in artifacts/ab/ab_results.jsonl.
set -u
cd /home/user/SWORLDMODEL-CORE
COMMIT=$(git rev-parse --short HEAD)
OUT=artifacts/ab
RESULTS=$OUT/ab_results.jsonl
: > "$RESULTS"
echo "A/B at commit $COMMIT"

run_case() {
  local case="$1" store="$2" question="$3" as_of="$4" horizon="$5"
  for mode in semantic direct; do
    (
      python3 scripts/frozen_forecast.py --mode "$mode" --store "$store" \
        --question "$question" --as-of "$as_of" --horizon "$horizon" \
        --out "$OUT/${case}_${mode}" > "$OUT/${case}_${mode}.log" 2>&1
      code=$?
      python3 - "$case" "$mode" "$code" "$COMMIT" <<'PY' >> /home/user/SWORLDMODEL-CORE/artifacts/ab/ab_results.jsonl
import json, sys
from pathlib import Path
case, mode, code, commit = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = Path(f"/home/user/SWORLDMODEL-CORE/artifacts/ab/{case}_{mode}")
row = {"case": case, "mode": mode, "exit": code, "commit": commit}
f = d / "forecast.json"
g = d / "diagnosis.json"
m = d / "metrics.json"
if f.exists():
    fc = json.loads(f.read_text())
    row.update(status=fc.get("status"), probability=fc.get("simulation_probability"),
               source=fc.get("probability_source"), yes=fc.get("resolved_yes_mass"),
               no=fc.get("resolved_no_mass"), unresolved=fc.get("unresolved_mass"),
               calls=fc.get("model_call_count"), tokens=fc.get("token_usage"))
elif g.exists():
    dg = json.loads(g.read_text())
    row.update(status="refused", failure_stage=dg.get("failure_stage"),
               failure=str(dg.get("failure"))[:200], calls=dg.get("model_calls"),
               wall=dg.get("wall_seconds"))
if m.exists():
    mt = json.loads(m.read_text())
    row.update(wall=mt.get("wall_seconds"), calls=mt.get("calls"),
               tokens_in=mt.get("tokens_in"), tokens_out=mt.get("tokens_out"),
               tokens_cached=mt.get("tokens_cached_prompt"),
               cache_hit_rate=mt.get("prompt_cache_hit_rate"),
               memo_reuses=mt.get("memo_reuses"),
               calls_by_stage=mt.get("calls_by_stage"),
               provider_seconds=mt.get("provider_seconds_total"),
               max_call_seconds=mt.get("max_call_seconds"))
print(json.dumps(row, sort_keys=True))
PY
    ) &
  done
  wait
  echo "=== $case done $(date -u +%H:%M:%S)"
}

python3 - <<'PY' > /tmp/ab_cases.tsv
import json
m = json.load(open('/home/user/SWORLDMODEL-CORE/artifacts/ab_matrix.json'))
for c in m['cases'] + m['holdouts']:
    print('\t'.join([c['case'], c['store'], c['question'], c['as_of'], c['horizon']]))
PY
while IFS=$'\t' read -r case store question as_of horizon; do
  echo "=== $case start $(date -u +%H:%M:%S)"
  run_case "$case" "$store" "$question" "$as_of" "$horizon"
done < /tmp/ab_cases.tsv
echo "=== A/B COMPLETE at $COMMIT"
