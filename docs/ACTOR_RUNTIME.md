# Actor runtime

Actors are persistent and live inside an explicit external world. They do **not**
generate both their decision and the reality resulting from it in one response: they
emit a typed *intention*, and the environment produces the *consequence*.

## The cognitive loop

Adapted conceptually from Generative Agents (perceive → retrieve → plan/react →
reflect → execute), with no grid, movement, filesystem, or Smallville prompt code.

`ActorRuntime.step(actor_state, local_view)`:

1. **Perceive.** Turn newly delivered/visible events from the local view into
   observations and durable episodic memories. A sent message is not automatically
   noticed — only a delivered one is visible.
2. **Retrieve.** Pull relevant memories from the actor's own `MemoryStream`, scored by
   `w_recency·decay^rank + w_relevance·lexical + w_importance·poignancy` (each
   min-max normalized). Retrieval refreshes recency. The retrieved ids are recorded.
   No heuristic encodes a social outcome (there is no "consensus pull").
3. **Plan / react.** Ask the gateway for a typed intention, given the actor's
   conditional behavior, the active proposal, observations, retrieved memories, and
   feasible actions. The actor may wait — but waiting records a pending information
   need, which the protocol can later satisfy.
4. **Reflect** (when accumulated importance crosses a threshold). Updates beliefs and
   durable memories only; it never changes external reality.
5. **Emit intent.** The gateway output is validated into a typed `Intent`; free-form
   prose is never accepted as the state transition. The natural-language reasoning and
   message content are preserved in the trace.

The actor holds `ActorState` (memory, beliefs, plan, pending questions, last-observed
ids) and is cloned per branch so memories stay isolated. It never touches
`WorldState`.

## The local view

`world.view_for(actor_id, trigger)` derives a read-only `LocalView` from a single
authoritative world: role, authority, stage, options, visible proposals, public
facts, and the delivered/visible observations this actor could have received by the
branch time. It cannot expose another actor's private state, undelivered messages,
future events, post-cutoff publications, the eventual outcome, or the aggregate
forecast. The view sent to the model is exactly the view recorded in the trace.

## Intent vs. consequence

`Intent` can only carry one of the allowed kinds (send message, make statement,
request information, introduce/revise/support/oppose a proposal, make a commitment,
act, cast a vote, wait, preserve plan). There is deliberately **no** kind for
asserting a consequence — an actor has no vocabulary to say "another member was
persuaded", "a coalition formed", or "the vote passed". A gateway that returns such a
kind is rejected (`test_actor_output_cannot_mark_another_actor_persuaded`).

The `Environment` validates authority/feasibility/timing/availability and then
executes the intent into events. Statements keep their substantive content — a
communication is never flattened to "actor emitted statement" — and are delivered to
other members as observable content they can react to
(`test_recipient_can_react_to_a_statement_and_removing_it_changes_the_vote`).

## Institution processes

Coordination emerges from events and reactions, not numeric smoothing. There is no
scalar support convergence, no movement toward a weighted mean, no consensus pull,
leader floor, coalition-discipline constant, or visible-tally multiplier. A
`ProtocolGraph` of generic primitives (distribute briefing, introduce proposal,
request statements, deliver them, open decision, cast votes, tally, publish) is
compiled per scenario; the runtime interprets it, delivering colleague statements to
members before asking them to react.

## The gateway boundary

The kernel talks to a reasoning model only through `ModelGateway`. Offline runs use a
transparent `DeterministicGateway` that reasons purely over the *structure* of its
typed input (no scenario facts), which makes runs reproducible and tests
deterministic; it is honestly labeled as a calibrated-behavior model wherever its
output feeds a branch weight. A live provider gateway implements the same interface.
The gateway is never used to count votes, apply thresholds, invent authoritative
facts, change the contract, or write terminal outcomes — those are code.
