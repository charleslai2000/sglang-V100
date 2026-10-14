# Operations Discipline

`ops/` contains durable operational knowledge, repeatable procedures, and material operational records.

```text
ops/
├── AGENTS.md
├── environments/
│   └── <environment>.md
├── runbooks/
│   └── <operation>.md
└── records/
    └── O001-<slug>.md
```

1. Work only on the assigned operational problem: runtime, host, service, deployment, build environment, GPU/resource state, or infrastructure configuration.

2. `environments/` records durable current facts about an environment. Keep one authoritative document per coherent environment; do not use it as a chronological log.

3. `runbooks/` contains confirmed repeatable procedures. A runbook MUST describe an operation that is already understood; unresolved diagnosis or experimental procedure does not belong there.

4. `records/` contains material operational changes, incidents, recoveries, or one-off interventions worth preserving. Do not record routine commands merely because they were executed.

5. Diagnose from actual environment state. Distinguish observed facts from hypotheses and intended configuration.

6. Before consequential execution, verify the actual target, required authority, preconditions, affected resources, and rollback or recovery path where materially necessary.

7. When an applicable runbook exists, follow it. Do not silently substitute targets, commands, parameters, environments, or procedures.

8. If the procedure is not determined, resolve the operational question before executing consequential changes. Do not improvise through ambiguity.

9. Prefer the smallest reversible action and minimum blast radius. Preserve unrelated services, processes, workloads, files, and infrastructure.

10. Do not introduce monitoring, tracing, logging, interception, or other observation mechanisms merely to increase confidence.

11. Verify consequential operations from resulting state, not command success or intent. Preserve unsuccessful and partial outcomes as real results.

12. Do not turn an operational problem into unrelated hardening, migration, cleanup, redesign, or infrastructure expansion.

13. Keep credentials, private infrastructure details, internal addresses, logs, and other sensitive operational data within their authorized boundary.

14. Large builds, caches, binaries, traces, logs, and generated artifacts MUST follow the environment's designated storage rules; do not place them in the source tree merely for convenience.

15. When work materially changes durable environment state or an established procedure, update the applicable `environments/` or `runbooks/` document. Do not leave the authoritative operational description stale.

16. Keep operational artifacts concise. Preserve durable state, decisive procedure, material incident findings, and recovery information; omit routine shell history and transient status.

