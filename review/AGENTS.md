# Review Discipline

`review/` contains independent review work. Review evaluates an explicitly assigned claim, deliverable, or stage; it does not own the underlying work.

```text
review/
├── AGENTS.md
├── R001-<slug>/
│   ├── review.md
│   └── recheck-01.md
├── R002-<slug>/
│   └── review.md
└── ...
```

1. Each `RNNN-<slug>/` represents one coherent review question. Review IDs are stable and MUST NOT be reused.

2. Organize reviews by review question, not by artifact type, repository layer, or specialist role.

3. Every review MUST identify:

   * the exact question or claim under review;
   * the reviewed targets;
   * the applicable requirements or acceptance criteria.

4. Review the actual target and primary evidence. Do not substitute plans, summaries, status reports, or intended state.

5. Do not modify reviewed work, create new requirements, or expand into adjacent audit, redesign, testing, or investigation without explicit scope.

6. Findings are:

```text
BLOCKING
NON_BLOCKING
UNRESOLVED
```

7. A `BLOCKING` finding MUST identify the violated requirement or invariant, affected claim, concrete evidence, and acceptance consequence.

8. `NON_BLOCKING` findings do not prevent acceptance. `UNRESOLVED` uncertainty is not blocking unless applicable authority requires it to be resolved.

9. High-risk review MAY additionally distinguish:

```text
CONFIRMED DEFECT
PLAUSIBLE RISK
UNSUPPORTED CONCERN
```

Do not promote risk or concern into a defect without evidence and applicability.

10. A defect invalidates only dependent claims. Preserve unaffected established work.

11. Do not block on preference, hypothetical improvement, stylistic disagreement, or requirements that were never authoritative.

12. `review.md` MUST remain concise and contain:

```text
Question
Targets / criteria
Verdict: ACCEPT | BLOCKING_FINDINGS
Findings
Decisive evidence
Remaining uncertainty
```

13. A review artifact is evidence, not authority. `ACCEPT` does not itself approve a design, complete a Goal, or release a stage.

14. If correction requires re-review, preserve the original review and add `recheck-NN.md`. Do not rewrite historical findings away.

15. Stop when the assigned review question is answered. Do not continue searching for additional defects merely to increase assurance.

