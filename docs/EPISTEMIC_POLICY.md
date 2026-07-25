# The epistemic policy

A social world model cannot directly verify a private belief, an intention, an
interpretation, a future action, an organizational response or a population's behavior.
A system that demands it refuses every real question. A system that skips the
distinction presents a guess as a fact. The way out is a taxonomy carried end to end.

## The four classes

| class | what it is | where it may appear |
|---|---|---|
| `VERIFIED` | a directly supported real-world fact | evidence store, world spec, actor prompt as fact |
| `INFERRED` | a defensible conclusion from verified facts | actor prompt, marked as reasoning, carrying its evidence |
| `HYPOTHETICAL` | a plausible unresolved alternative | the branch structure — the thing the simulation resolves |
| `UNSUPPORTED` | no defensible evidentiary or causal basis | nowhere |

Defined in `epistemics.py`. They are kinds of statement, not confidence bands, and the
rules differ: `VERIFIED` may be stated as fact, `INFERRED` may be shown but always marked
and always with its citations, `HYPOTHETICAL` belongs to the branches rather than to
anyone's knowledge, `UNSUPPORTED` never enters the world.

## What must be verified

Structural facts, because getting them wrong means simulating a different world: the
resolution contract, the focal institution or population, real identities where named
individuals are required, cutoff-correct office holders, formal membership, authority and
procedural control, the decision rule, seat or unit counts, known deadlines and scheduled
events, known past actions, observed starting conditions, units and thresholds,
authoritative resolution sources.

## What may be inferred

Interpretation of new information; current but unstated priorities; which obligation is
most salient; a reasonable next step implied by a role or an existing plan;
organization-level behavior implied by established policy; likely communication
pathways; openness to reconsideration; which production or logistical pathway is active.
Every inference keeps its supporting claim ids and is never displayed to an actor as
established.

## What must be uncertainty

Private preferences, willingness to compromise, interpretation of ambiguous evidence,
future data, future demand, implementation success, message attention, response timing,
organizational delay, operational throughput, population response. A simulation is not
refused because these are unverified — resolving them is what it is *for*.

## The grounding hierarchy

An actor does not need a quotation to exist. It needs to be the real occupant of a real
role inside the causal boundary. `GroundingLevel` ranks how strongly the surviving
citations attach an actor to the world:

1. `DIRECT_RECORD` — its own action or first-person statement
2. `OFFICIAL_ROLE` — verified office, membership and authority
3. `DOCUMENTED_PRIOR_BEHAVIOR` — recorded past conduct attributed to it
4. `INSTITUTIONAL_POLICY` — official organizational or institutional policy
5. `CONTEMPORANEOUS_REPORTING` — dated reporting that names it
6. `ROLE_LEVEL_BEHAVIOR` — behavior evidenced at the role or institution level
7. `LABELED_UNCERTAIN` — nothing but explicitly labeled alternatives

Levels 1–6 admit an actor. The level decides what may be *said* about it, not whether it
exists: below a direct record its disposition is `INFERRED` or `HYPOTHETICAL`, marked as
such in its own prompt. Level 7 and below is a name with no referent, and stays refused.

This replaced a rule requiring every actor to carry a cited, first-person record of its
own. Faced with a trade negotiation or a quarterly production figure — where office and
authority are public record and no retrieved source quotes anyone in the first person — a
compiler obeying that rule had one safe move: emit no actors. The runtime then refused
the world for having nobody in it. Four of five live cases died there, and the rule never
prevented a single invention.

## Causal producers need not be people

The reality gate asks for a causal pathway, not a person. A producer may be an
individual, an organization, an institutional body, a subunit, a coalition, a population
stratum, a network, a market, an administrative process, a production system, a logistical
system, or an external physical or economic process. A quarterly delivery total is
produced by production, inventory, logistics and demand; requiring an individual there
means inventing an executive who decides how many cars get built, which is a *less*
faithful world than one with no individuals in it.

A person the evidence names must be present in the world, but not necessarily as a
deliberating actor — and a person is present through the body they act within when a
verified claim names both (`reality._covered_by_an_organization`).

## The outcome must still be produced

Relaxing who may exist is only safe alongside a stricter account of what produces the
answer. Three rules, all in `world_compiler.enforce_outcome_is_produced`:

* every term the terminal reads must be writable by an action, a process node or an
  external process — an uncertainty is never a producer;
* a world that compiles actors must let some action of theirs move a terminal term;
* in a world with actors, no scheduled process may set a terminal term to a literal
  unless it is gated on a field an action writes — otherwise the calendar announces the
  answer before anyone acts.

`engine.terminal_lineage` is the runtime half: it walks the branch's own ledger and names
the event, actor and causal parents behind every terminal term.
