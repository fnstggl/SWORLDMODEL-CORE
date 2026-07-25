#!/bin/bash
# Run the five cross-domain acceptance questions against ONE frozen commit.
#
# The questions, cutoffs and horizons are fixed in artifacts/acceptance/questions.json
# and must not be simplified. Four are nowcasts and generate their cutoff at launch, so
# retrieval is live; the fifth is a sealed pastcast whose cutoff is historical and whose
# sources must therefore be archived captures.
#
# No source file may change between these runs. The commit is recorded with the results.
set -u
cd /home/user/SWORLDMODEL-CORE

COMMIT=$(git rev-parse HEAD)
STATUS=artifacts/acceptance/status.log
mkdir -p artifacts/acceptance
: > "$STATUS"
echo "commit: $COMMIT" >> "$STATUS"
echo "dirty:  $(git status --porcelain | wc -l) file(s)" >> "$STATUS"

run() {
  local name="$1" asof="$2" horizon="$3" q="$4"
  mkdir -p "artifacts/acceptance/$name"
  local t0=$SECONDS
  timeout 2400 python3 -u -m sworldmodel forecast \
    --question "$q" --as-of "$asof" --horizon "$horizon" \
    --max-branches 4 --max-queries 20 --research-rounds 3 --research-seconds 420 \
    --max-structures 2 --trace "artifacts/acceptance/$name/run_trace" \
    > "artifacts/acceptance/$name/run.log" 2>&1
  echo "=== $name exit=$? wall=$((SECONDS - t0))s ===" >> "$STATUS"
}

now() { date -u -d '+1 minute' +%Y-%m-%dT%H:%M:%S+00:00; }

run geopolitical "$(now)" "2026-10-01T23:59:59+00:00" \
  "Will OPEC+ announce an increase in crude oil production quotas at or before its next scheduled ministerial meeting in 2026?" &
run individual "$(now)" "2026-09-15T23:59:59+00:00" \
  "Will Andrew Bailey, Governor of the Bank of England, publicly signal support for a further Bank Rate cut before September 15, 2026?" &
wait

run negotiation "$(now)" "2026-10-01T23:59:59+00:00" \
  "Will the European Union and Mercosur formally sign their trade agreement before October 1, 2026?" &
run population "$(now)" "2026-10-05T23:59:59+00:00" \
  "Will Tesla report more than 400,000 vehicle deliveries for the third quarter of 2026?" &
wait

# The pastcast. Its cutoff is sealed at 14 May 2026 and the June decision is after it.
run committee "2026-05-14T23:59:59-06:00" "2026-06-25T23:59:59-06:00" \
  "Will Banco de Mexico's Governing Board vote unanimously to hold its policy interest rate unchanged at its June 25, 2026 monetary policy decision?"

echo "ACCEPTANCE COMPLETE" >> "$STATUS"
