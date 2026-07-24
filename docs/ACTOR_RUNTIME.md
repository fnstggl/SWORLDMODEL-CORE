# The actor runtime

An actor is a persistent being, not a function the runtime calls at protocol stages.

## What an actor carries between invocations

| state | why it exists |
| --- | --- |
| `plan` | a goal, sparse dated steps, a status, a revision count, and a `basis` saying which verified schedule, role obligation or existing commitment makes it admissible |
| `current_action` | what it is doing now, with a start time and an expected completion |
| `commitments` | undertakings with due times, each a real future cause |
| `pending_needs` | what it asked for and has not received, with an optional deadline |
| `revisit_conditions` | conditions it named itself for coming back |
| `memory` | episodic and semantic, with the evidence claim ids behind each |
| `beliefs`, `goals`, `relationships`, `unresolved_questions` | its own view, which nothing else may write |

A plan keeps its identity through revision and interruption, so the trace shows one plan
changing rather than a new plan appearing.

## Why an actor is invoked

Only with a stated cause. The complete set:

| reason | meaning |
| --- | --- |
| `process_opportunity` | a compiled process node opened actions for it |
| `directed_information` | something was addressed to it personally |
| `compiled_wake_rule` | a rule compiled *for this world* says this matters to it |
| `own_revisit_condition` | a condition it set itself came true |
| `pending_need_answered` | an answer arrived from whom it asked |
| `pending_need_unanswered_at_deadline` | the answer definitively did not come |
| `commitment_due` | something it undertook fell due |
| `own_action_resolved` | its own attempt failed or was refused |
| `own_plan_step_due` | its own planned step's moment arrived |
| `deadline_reached` | a real deadline in the process |

Anything else it notices enters memory and costs no model call. An actor busy with its
own unfinished action is not interrupted by a mere opportunity — that is what makes a
plan persist. Simultaneous reasons merge into one invocation carrying all of them,
because a person facing two things at once has one moment of attention.

## What one invocation looks like

```
perceive   newly noticed observations enter episodic memory
retrieve   ranked by the actor's actual situation: its plan, its current action,
           what it is waiting for, what just reached it
decide     one call covering BOTH what happens to the plan and what to do
reflect    only when the actor's own response says its understanding changed
```

The actor is told *why it is being asked now* and *what it was already doing*. It is not
asked to reconsider the world from scratch.

There is no reflection threshold, no attention bandwidth, no importance score that
decides whether it acts, and no retrieval weight triple. Those numbers existed in both
donor implementations and in this one; they are gone. A number that decides when a
person changes their mind is an assertion about people that nothing in the evidence
supports.

## What an actor cannot do

It has no reference to `WorldState`. It cannot mutate reality, see another actor's
private state, or assert that its action succeeded. It emits an intention; the
environment decides the consequence.

If its response cannot be read, that is a provider failure and the branch's mass stays
unresolved. It never becomes a decision to wait.
