# Control Discipline

`control/` owns durable Goal, Plan, Task, and Frontier state.

```text
control/
├── AGENTS.md
├── frontier.md
└── <goal-id>/
    ├── goal.md
    ├── plan.md
    └── tasks/
        └── TNNN-<slug>.md
```

1. One control session MAY manage multiple Goals.

2. Each Goal MUST use one stable `<goal-id>` directory. Goal IDs and Task IDs MUST NOT be reused.

3. `goal.md` defines the authorized Goal, scope, constraints, exclusions, and completion condition. It is not a progress log.

4. Create a new Goal when the intended outcome, completion condition, object, or authority materially changes. Do not stretch an existing Goal merely because later work is related.

5. A completed Goal MUST NOT be reopened for ordinary follow-on work; create a new Goal and reference the prior result. Reopen only to correct that Goal itself.

6. `plan.md` is the current state of the Goal: established results, decisive frontier, active/blocking Tasks, and next action. It is not history.

7. Obsolete assumptions, plans, and unnecessary future Tasks MUST be removed rather than accumulated.

8. Each Task represents one coherent professional problem and uses `TNNN-<slug>.md`.

9. Every Task MUST state:

```text
Status: READY | ACTIVE | DONE | BLOCKED
Work area
Objective
Inputs
Completion
Result
Remaining
```

10. The Task file is both instruction and handoff. The working session writes the result into the same file; do not create duplicate control reports.

11. Tasks MUST be coarse enough for one professional session to own end-to-end. Do not split reconnaissance, implementation, ordinary verification, and reporting into procedural stages without a real professional boundary.

12. Create Tasks only for work necessary to advance the Goal. Do not create Tasks for reassurance, status, generic review, or additional evidence without a concrete need.

13. Professional work belongs in its applicable work area. The control session owns Goal, Plan, Task boundaries, integration, and frontier; it does not perform substantial design, implementation, experiment, research, review, or operations work.

14. `DONE` means the Task's stated Completion is satisfied. `BLOCKED` means a concrete condition prevents further progress and that condition is recorded.

15. Completed Task results MUST be integrated into `plan.md` by conclusion, not copied as history.

16. `frontier.md` is the authoritative cross-Goal view of current unfinished work.

17. `frontier.md` MUST contain only active or intentionally deferred work. DONE, cancelled, superseded, obsolete, or inactive entries MUST be deleted immediately.

18. Each frontier entry uses:

```text
<goal-id>
- <task-id> <title> — READY | BLOCKED | DEFERRED
  - frontier: <current decisive issue or next action>
  - task: <task-file>
  - unblock/reopen: <condition, when applicable>
```

19. Every change to the decisive frontier MUST update `frontier.md` in the same control transition.

20. `frontier.md` is not history or backlog. Task detail, evidence, results, and past state belong in Goal/Plan/Task files; keep the frontier short by deleting obsolete state.

21. Before acting on a Goal, read its `goal.md`, `plan.md`, relevant Tasks, and current frontier entry. Do not rely on conversational memory from another Goal.

22. Stop managing a Goal only when it is complete, genuinely blocked, or requires a material user decision.

