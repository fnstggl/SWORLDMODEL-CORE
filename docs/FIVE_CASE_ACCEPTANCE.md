# The five-case frozen acceptance run

Five questions, five shapes, one commit. The questions, cutoffs and horizons are fixed in
`artifacts/acceptance/questions.json` and are not simplified between runs. Four are
nowcasts whose cutoff is generated at launch; the fifth is a sealed pastcast whose cutoff
is historical and whose sources must be archived captures.

`scripts/acceptance.sh` runs them from clean processes and records the commit and
working-tree state. `scripts/acceptance_verify.py` judges each case against its own
artifacts; `scripts/acceptance_metrics.py` emits the machine-readable metrics.

## The bar

A case may honestly end as: a completed forecast; a factual resolution because the record
already proves the outcome; a bounded forecast with unresolved mass; a justified
abstention because a critical component cannot be grounded; or an externally-blocked
provider failure with all work preserved.

These are failures: a crash; a timeout with no diagnosis; a hollow world; an arbitrary
50/50; an answer inherited from branch initialization; a missing causal producer; a false
refusal from a gate defect; fabricated evidence; an ungrounded precise operating model; a
forced outcome; an unresolved value silently converted to YES or NO.

A zero exit code is not a pass. The trajectory auditor's classification and the
forecast-integrity record are what distinguish a genuine simulation from a hollow one.

## Case-specific requirements

| case | passes only when |
| --- | --- |
| Bank of England | Bailey is the terminal actor; material economic and institutional influences reach his context; the result is not an arbitrary 50/50; each call has a real wake reason; the trace explains who else was included or excluded |
| OPEC+ | represented membership matches reality; cardinality, votes and thresholds preserved; a valid aggregate is accepted; internal disagreement exposed where material; terminal logic complete |
| Tesla | the world models the production of deliveries, not their publication; demand is segmented at an evidence-supported level; totals accumulate through real processes; no invented multiplier decides it; honest if operational evidence is thin |
| EU–Mercosur | no single fetch failure crashes the run; research artifacts survive network failures; institutions and process represented; already-resolved facts identified as factual resolution; cutoff obeyed |
| Banxico | provider failures retried and preserved; correct roster and decision rule verified; each seat exists where votes matter; no outdated member; no uncertainty supplies the vote; unanimity evaluated mechanically |

<!-- RESULTS: filled from the frozen run -->
