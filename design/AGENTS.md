# Design Discipline

`design/` contains durable architecture and design knowledge. It is not a task-control directory and not an implementation workspace.

```text
design/
├── AGENTS.md
├── architecture/
│   └── <area>.md
├── proposals/
│   └── <topic>.md
└── decisions/
    └── ADR-NNNN-<slug>.md
```

1. Work only on the assigned design question. Do not expand into unrelated redesign, implementation, testing, operations, or documentation cleanup.

2. `architecture/` describes the currently accepted design. Keep one authoritative document per coherent design area; update it instead of creating competing versions.

3. `proposals/` contains unresolved design choices. A proposal is not authority and MUST NOT be presented as an accepted design.

4. `decisions/` contains confirmed architectural decisions only. An ADR records an already-authorized decision; it does not approve its own proposal.

5. Use a new document when the design object or decision is materially independent. Amend the existing document when the work refines or corrects the same design object. Do not create versioned copies merely to preserve history.

6. Design reasoning MUST distinguish:

```text
OBSERVED
ASSUMED
PROPOSED
UNRESOLVED
```

7. Material factual premises MUST be traceable to project evidence. Current implementation is evidence of current behavior, not architectural authority unless explicitly established as such.

8. Preserve established requirements, invariants, ownership, and interface boundaries. Do not silently redefine them to make a proposal easier.

9. Prefer the smallest design that satisfies established requirements. Do not introduce abstraction, extensibility, infrastructure, gates, or generality without a concrete need.

10. Consider alternatives, migration, reversibility, compatibility, and failure consequences only where they can materially change the design decision.

11. Do not silently resolve conflicting authority or genuine design alternatives. Record the conflict or decision required.

12. Recommendations are not self-approving. When authority is required, state the decision and material alternatives; leave the choice to the applicable authority.

13. After a design is accepted, update the applicable `architecture/` document and record any material decision in `decisions/`. Do not leave accepted architecture represented only by a proposal.

14. Superseded decisions remain recorded with their status and successor reference. Do not rewrite historical ADRs to make them appear current.

15. Keep design documents concise and structural. Record current design and decisive rationale, not chronological discussion, session history, implementation logs, or control state.
