# The frozen acceptance run

Five questions, five shapes, one commit. The questions, cutoffs and horizons are fixed
in `artifacts/acceptance/questions.json` and are not simplified between runs. Four are
nowcasts whose cutoff is generated at launch, so retrieval is live; the fifth is a
sealed pastcast whose cutoff is historical and whose sources must therefore be archived
captures.

`scripts/acceptance.sh` runs them and records the commit and the working-tree state with
the results. `scripts/acceptance_verify.py` judges each case against eleven criteria,
every one of them reading a file the run wrote. `scripts/acceptance_report.py` prints the
per-stage summary from the same artifacts.

The bar: **a refusal, a crash, a timeout, a hollow answer and an unproduced terminal are
each a failure.** A zero exit code is not a pass on its own — the Tesla case exited zero
with every branch unresolved, and that is counted here as the failure it was.

## What each case is for

| case | shape | what it has to get right |
| --- | --- | --- |
| geopolitical | multi-organization | many bodies, one of which acts as a unit; the others are represented without being made to deliberate |
| individual | one person's own act | the outcome is a single person's statement; the people around them are context, not co-producers |
| negotiation | two parties, a signature | a question whose answer may already be in the record before the window opens |
| population | aggregate throughput | a quantity produced by operations, with no individual who decides it |
| committee | sealed pastcast | archived captures only, and a body whose full membership may not be in the record |

<!-- RESULTS -->

## Reading a case's artifacts

Each case writes to `artifacts/acceptance/<case>/`:

* `run.log` — what the process printed, including the retrieval mode, before research
* `diagnosis.json` — the stage-by-stage record: planning, discovery, fetching,
  extraction, epistemic classification, world compilation, integrity and grounding,
  runtime, root cause
* `compiled_world.json` — the world that was simulated, or the world that was refused
* `research_trace.json` — every query, URL, fetch and rejection
* `evidence_store.json` — every stored claim with its excerpt and availability
* `run_trace/` — the full trace: event ledger, actor decisions with byte-exact prompts,
  branch schedule, coverage report, forecast, model calls

A refusal writes the same records as a completion. That is deliberate: the runs that
refuse are the ones a reader most needs to be able to diagnose, and an earlier pass of
this suite left four refusals with nothing but a stack trace.
