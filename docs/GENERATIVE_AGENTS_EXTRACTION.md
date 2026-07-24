# Generative Agents: component-level extraction record

Read from the actual implementation at `../generative_agents-reference`
(upstream: joonspk-research/generative_agents), not from the paper. Line references are
to that checkout.

The reason this document exists is that adopting Generative Agents is easy to fake: any
runtime can grow methods named `perceive`, `retrieve`, `plan` and `reflect` while
implementing none of the persistence that makes those methods mean anything. What
follows records, per component, what state it owns, what behavioral realism it actually
contributes, what is Smallville-specific, and what happened to it here.

---

## `persona/persona.py` — `Persona.move`

**State owned** the persona object itself, which survives every step.

**Real contribution** the cognitive sequence runs against the *current* world and the
*current* simulation time on a persona that was not reconstructed. Continuity is the
product; the five verbs are the mechanism.

**Smallville-specific** takes `maze` and `curr_tile`; the sequence is driven once per
tick for every persona from `reverie.py`'s step loop.

**Decision: ADAPT.** `src/sworldmodel/actors.py ActorRuntime.step` runs the same shape —
perceive → retrieve → (reflect) → decide → emit — on a persistent `ActorState`. What is
*not* adapted is who calls it: here nothing iterates over actors. An actor is invoked
because the event queue produced a cause naming it.

---

## `persona/memory_structures/scratch.py` — `Scratch`

**State owned** (the fields that matter, all persistent across steps):
`curr_time`, `currently`, `daily_plan_req`, `daily_req`, `f_daily_schedule`,
`f_daily_schedule_hourly_org`, `act_address`, `act_start_time`, `act_duration`,
`act_description`, `act_event`, `chatting_with`, `chatting_with_buffer`,
`chatting_end_time`, `planned_path`.

**Real contribution** this is the heart of the reference implementation and the thing
SWORLDMODEL-CORE was missing. An agent is not a function of the world; it is a thing
with a current activity that has a start time and a duration, an interaction it is
currently inside, and a schedule it is working through.

**Arbitrary constants** `recency_w = 1`, `relevance_w = 1`, `importance_w = 1`,
`recency_decay = 0.99`, `importance_trigger_max = 150`, `daily_reflection_time = 180`,
`daily_reflection_size = 5`.

**Decision: PORT (selectively).**

| GA field | here | note |
| --- | --- | --- |
| `act_start_time` / `act_duration` / `act_description` | `actors.OngoingAction` | plus `expected_completion` and `world_version`, which GA has no need for |
| `currently` | `actors.Plan.goal` | |
| `f_daily_schedule` | `actors.Plan.steps` | **not** a day grid — a sparse list of grounded steps, each optionally dated |
| `chatting_with` / `chatting_end_time` | `ActorState.pending_needs` + `Delivery` | an exchange here is real messages with real delivery times, not a paired chat state |
| `chatting_with_buffer` | *rejected* | a cooldown counter is an invented number about how soon people re-engage |
| `planned_path`, `curr_tile`, `act_address` | *rejected* | no spatial world |
| the retrieval/reflection constants | *rejected* | see below |

---

## `memory_structures/associative_memory.py`

**State owned** `ConceptNode` sequences (`seq_event`, `seq_thought`, `seq_chat`) with
keyword indices, `poignancy`, `created`, `expiration`, `filling` (the memories a
reflection was drawn from).

**Real contribution** reflections that point back at the memories that produced them —
the trace of *why* an agent believes something.

**Decision: KEEP CORE, adopt the back-pointer.** `src/sworldmodel/memory.py` already
implements a memory stream with kinds, importance, recency and a `depth` marker for
reflections, and it additionally carries `evidence_claim_ids` — which GA has no concept
of and which this product requires. Reflections here record their supporting claim ids.
Porting GA's structure would have been a lateral move.

---

## `memory_structures/spatial_memory.py`

**Decision: REJECT.** A tree of world/sector/arena/object addresses. There is no
physical space in this product; what an actor can reach is expressed by compiled
`authority` and action `preconditions`.

---

## `cognitive_modules/perceive.py`

**Real contribution** an agent perceives a *subset*, and perception writes to memory
independently of action.

**Smallville-specific / arbitrary** `vision_r` (a tile radius), `att_bandwidth` (how many
events per step), `retention` (how long an event is suppressed after being perceived).

**Decision: REIMPLEMENT.** `world.view_for` returns only what the actor has *noticed*,
where noticing is the third of three separately-timestamped transitions
(visible → available → noticed). The filter is causal rather than numeric. Nothing in
this runtime has an attention bandwidth.

---

## `cognitive_modules/retrieve.py`

**Real contribution** ranking memories by recency, relevance and importance together.

**Arbitrary** the weights are the literal list `[0.5, 3, 2]`, and the file's own comment
reads: *"We currently use hard coded weights… we could learn optimal weights"*.

**Decision: KEEP CORE, REJECT the weights.** `memory.py`'s retrieval already combines
recency and relevance; adopting a triple the reference implementation itself flags as
unjustified would have imported a defect.

---

## `cognitive_modules/plan.py`

The largest and most valuable module, and the most Smallville-entangled.

| function | contribution | decision |
| --- | --- | --- |
| `generate_first_daily_plan`, `_long_term_planning` | a plan exists before events arrive | **ADAPT** → `ActorSpec.initial_plan`, but it must carry `basis` (a verified schedule, role obligation or existing commitment) and is expected to be sparse. A compiler that invents a day for someone is producing fiction. |
| `_determine_action` + `determine_decomp` | a new action is chosen only when the current one is finished | **PORT** → `actor.is_busy` gates interruption; `executor` schedules completion |
| `scratch.act_check_finished()` | the actual persistence mechanism | **PORT** (see manifest §5) |
| `_should_react` (`lets_talk`, `lets_react`) | new information can override the plan | **REIMPLEMENT** — the *idea* is right, the implementation is a pile of Smallville conditions: refuses when `curr_time.hour == 23`, when either description contains `"sleeping"`, when `act_address` contains `"<waiting>"`, when a buffer counter is positive. Replaced by `engine._relevance` plus the actor's own `plan_disposition`. |
| `_create_react` | reschedules around an interruption, keeping the rest | **ADAPT** → `interrupt` pauses a plan without destroying it; `revise` keeps its identity and increments `revision_count` |
| `_chat_react` / `_wait_react` | two hardcoded reaction categories | **REJECT** — a closed set of social reactions is exactly what this architecture forbids; the action menu is compiled per world and open to novel proposals |
| `revise_identity` | the agent updates its self-description | **REJECT for now** — without evidence to ground it this is invented biography |

---

## `cognitive_modules/reflect.py`

**Real contribution** synthesizing higher-level thoughts from accumulated experience,
linked to their supporting memories.

**Arbitrary** fires when `importance_trigger_curr` crosses a threshold seeded at 150.

**Decision: ADAPT, REJECT the threshold.** Reflection here happens when the actor's own
response sets `reflection_needed`. The core previously had the same defect in smaller
numbers (`REFLECTION_TRIGGER = 1.5`); it was removed in this change, not preserved.

---

## `cognitive_modules/converse.py`

**Real contribution** multi-turn exchange with memory of what was said.

**Smallville-specific** co-location, a paired `chatting_with` state, and a summarized
transcript.

**Decision: REIMPLEMENT.** Conversation here is not a special mode. It is ordinary
compiled actions producing `deliver_information` events with real delivery and notice
times, which reach an actor, may wake it, and are answered by another action. The
`pending_needs` mechanism gives the asymmetry GA's paired chat state cannot express:
*asking* someone something and never getting an answer is a real, traceable outcome that
wakes the asker at its deadline.

---

## `cognitive_modules/execute.py`

**Decision: REJECT.** Translates an action description into a tile path. No analogue
here; `executor.py` translates an intention into validated universal effects.

---

## `reverie.py` — the step loop

```python
for persona in self.personas.values():
    ...
self.curr_time += datetime.timedelta(seconds=self.sec_per_step)
```

**Decision: REJECT — emphatically.** This is the exact structure this consolidation
exists to remove: every agent invoked every tick, time advancing by a fixed constant.
`engine._event_loop` advances to the next scheduled thing on the branch calendar and
invokes only the actors that thing actually affects.

---

## `maze.py`

**Decision: REJECT.** A tile maze with collision blocks and address tuples.

---

## What "we adopted Generative Agents" is allowed to mean here

Not the vocabulary. The following behaviors, each demonstrable in
`tests/invariants/test_scheduling.py`:

- an actor with an unfinished action is not asked to decide again
  (`test_an_actor_busy_with_an_action_is_not_interrupted_by_a_mere_opportunity`);
- a plan survives events and keeps its identity through revision
  (`test_plan_identity_survives_a_revision`);
- an actor's own planned step creates a real future opportunity in the world
  (`test_a_planned_step_with_a_time_schedules_a_real_future_opportunity`);
- perception is a filter: something visible and irrelevant is remembered without
  provoking a decision
  (`test_visible_but_irrelevant_event_does_not_force_a_decision_call`);
- waiting is a state with a future consequence
  (`test_an_unmet_information_need_wakes_the_actor_at_its_deadline`).

None of these can pass under a fixed per-stage actor loop, which is the point.
