# T011 fixed-input Prefill diagnostic artifacts

The controlled diagnostic script at `/data/t011/artifacts/t011-prefill-first-error.py` runs one arm per fresh process to avoid reinitializing distributed/model-parallel state in-process. It uses fixed `input_ids=positions=0..511`, resets seeds, loads the same pinned checkpoint and frozen T011 QPN2 binary, and captures every linear boundary's finite status plus the actual L43 down input activation. Native and candidate logs:

- `/data/t011/artifacts/commonops128/t011-prefill-first-error.log`
- `/data/t011/artifacts/commonops128/t011-prefill-first-error-candidate.log`

Both logs record IDs/positions digest `5738153ec97595b1c1e4dc027f7b7fb4534f19ed2ce9f9ee712e6d34a384cde7` and the checkpoint/model parameters are same. Native first nonfinite output is L43/down `[229,175]` Inf; candidate has no nonfinite observed layer boundary and final hidden/logits are finite. Native and candidate down inputs differ (`62ffb168...` vs `1b63e53a...`) because preceding execution arithmetic diverges, so each FP64 dot is computed from its own finite captured activation; they are not a same-input numerical comparison. Detailed exact values and explanation of the two historical FP64 values are in `prefill-first-error-closure.md`.

These are unqualified diagnostic logs, not strict all-coordinate oracle, timing, or memory-peak evidence. The previous attempt to run both arms in one worker process hit an expected existing TP group initialization assertion; the one-arm-per-process logs supersede that failed attempt.
