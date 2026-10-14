# Experiment Discipline

`experiments/` contains isolated experiments, benchmarks, reproductions, and prototypes used to answer bounded empirical questions. Experimental artifacts are evidence, not production authority.

```text
experiments/
├── AGENTS.md
├── E001-<slug>/
│   ├── experiment.md
│   ├── results.md
│   ├── scripts/        # optional, experiment-specific only
│   └── artifacts/      # optional small retained outputs
└── E002-<slug>/
```

1. Each experiment MUST occupy one `ENNN-<slug>/` directory. Experiment IDs are stable and MUST NOT be reused.

2. `experiment.md` MUST define before execution:

   * Question;
   * hypothesis or prediction, when applicable;
   * experimental object and identity;
   * controlled and varied inputs;
   * procedure;
   * measurements needed to answer the Question.

3. Run the smallest experiment capable of resolving the stated uncertainty. Do not expand into sweeps, longer runs, additional metrics, or adjacent questions without a concrete need.

4. Freeze the declared experimental conditions before the decisive run. A material change to model, inputs, procedure, numerical semantics, runtime path, or measured question creates a new experiment or an explicitly identified new run condition.

5. When the claim concerns production behavior, use the real production execution path and identity. A mock, fake, prototype, or reference path MUST NOT be presented as evidence for a different production path.

6. Keep experimental scripts, temporary configuration, and generated state isolated from production source and authoritative configuration unless the experiment explicitly requires an authorized production change.

7. Record what actually ran. Preserve material execution identity, parameters, failures, and deviations necessary to interpret the result.

8. `results.md` MUST distinguish:

   * `OBSERVED`;
   * `INFERRED`;
   * `UNRESOLVED`.

   A prediction, expectation, or prior result is not an observation.

9. Negative, contradictory, and failed results MUST be preserved. Do not rerun, tune, discard, or reinterpret a result merely because it is uninteresting or conflicts with the hypothesis.

10. Reuse established experiments when their relevant identity and conditions match. Do not repeat work merely because rerunning is easier than understanding existing evidence.

11. Large raw outputs, logs, binaries, traces, and generated datasets belong in the project-designated artifact storage. Reference them from the experiment directory rather than copying them into the repository.

12. `results.md` MUST end with the bounded answer to the experiment Question, the evidence supporting it, and any material limitation. Do not generalize beyond the conditions actually tested.

13. An experiment does not authorize production adoption, parameter selection, architecture change, or a new project requirement. Those decisions belong to the applicable control or design authority.

14. Do not use credentials, private data, paid external services, destructive operations, or new instrumentation unless explicitly authorized for that experiment.

