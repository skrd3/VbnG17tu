# Text tournament trainer

This repository keeps the official standalone text trainer contract:

- Dockerfile: `dockerfiles/standalone-text-trainer.dockerfile`.
- Entrypoint: `scripts/entrypoint.sh` → `start_training.py` → `text_trainer.py`.
- Tasks: InstructTextTask, ChatTask, DpoTask and GrpoTask.
- Cache: read-only `/cache`; final model: `/app/checkpoints/{task_id}/{expected_repo_name}`.
- LICENSE.md and NOTICE remain unchanged.

## Optimization 002

1. Resource plans use actual visible GPUs; batch resolution follows length selection.
   LR probes use the real instruct/KL objective and matching optimizer. Search
   caches include task/data/resource identity; online callbacks measure all backends.
2. Instruct selection uses unpacked per-example CE plus token-weighted completion
   KL. Optional reclaim training needs an independent acceptance gate. Disk soup
   is disabled until a backend-safe implementation is validated.
3. A deadline established before ML imports is shared by all attempts. Checkpoints
   are validated, staged on the output filesystem, and exchanged atomically using
   Linux `renameat2(RENAME_EXCHANGE)`. Unsupported exchange preserves the old result.
   There is no random-noise success fallback and no delete-before-copy publication.
4. GRPO selection records raw reward components and base-to-policy prompt KL.
   The first scored policy supplies fixed reward scales and an explicitly local
   anchor. Multi-seed monotone reward utility selects checkpoints; normalized
   anchor margins remain diagnostics, never an assumed actual Boss score.

## Review fixes (2026-11-02)

Non-KL training preserves native/fused loss; auxiliary prompt labels cannot
enter completion KL. Runtime timing no longer restarts warmup/schedulers.
DPO probes use the configured optimizer; selection evaluates beta=0.1 sigmoid
DPO independently of training loss modifiers and rejects unstable paired mean
gains on dev sets with at least 32 rows. This internal 90%/512-resample gate is
distinct from the final official 99%/10000-resample matchup gate.

Publication scans finite weights, records weight hashes/selected step and probes
the actual Linux output mount before preparation. Soup evaluations cannot
change selection incumbents or trigger checkpoint callbacks. Control decisions
and nonfinite-loss handling are synchronized even without token counters.

Detailed implementation, limits and experiment plan:

## Controls

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `LR_SEARCH_FRACTION` | `0.08` | Share of remaining time for probes; `0` disables; capped at 0.20. |
| `ENABLE_DEV_PASS` | unset | `1` enables a non-sharded reclaim candidate only with an independent gate and sufficient time. |
| `GRPO_SELECTION_PROMPTS` | `64` | Deterministic cap on selection prompts. |
| `GRPO_SELECTION_SEEDS` | `42,314159` | One to four distinct seeds; full configured evaluation must fit the budget. |
| `GRPO_EVAL_INITIAL` | unset | `1` opts into a pre-training audit; default prioritizes a trained artifact. |
| `SUBMISSION_IO_MIBPS` | `512` | Initial effective I/O estimate for save/scan/hash reserve; calibrate on target storage. |
| `SN56_VERBOSE` | unset | `1` enables diagnostic logs for local development. |
| `TRAINING_STARTED_AT`, `TRAINING_DEADLINE` | set by launcher | Shared timestamps; do not extend these during retries. |

The gpt-oss dependency overlay is built in the Dockerfile; no packages are
downloaded/uninstalled during training. Hub access is disabled at runtime. Eager
attention avoids a runtime download of a community attention kernel.

## CPU regression checks

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-test.txt
.venv/bin/python -m compileall -q scripts tests
.venv/bin/ruff check scripts tests --select E9,F821,F823
.venv/bin/python -m pytest tests -q
.venv/bin/python tests/distributed_control_probe.py
.venv/bin/python -m pytest scripts/test_helpers -q
```

On Windows use `.venv\Scripts\python.exe`. Run the legacy helper suite in a
separate interpreter: some original tests replace global import modules with stubs.
GitHub Actions also exercises actual atomic directory exchange on Linux.

## Required target-environment verification

Before submitting a Git SHA, build this exact Dockerfile, then run all four task
types with real cached model/data paths, no public network, read-only `/cache`,
assigned GPUs and the official timeout. Check output with `submission.validate_submission`,
then reload the full model/adapter and run a finite-loss forward. Exercise the
official uploader against the same mount. Include 4-GPU Chat/KL and a real
30B–72B ZeRO-3 job, and kill a container during candidate writing/exchange.

CPU tests do not establish CUDA/vLLM/ZeRO compatibility or model-quality gains.
All backends retain their established LR scheduler; measured timing and shared
deadline still apply. Unsupported directory exchange fails the Linux startup
probe; an exchange failure during publication preserves the previous model.

Inside the built image (or equivalent dependency environment), preflight an
existing output without modifying it:

```bash
python -m check_submission --output /app/checkpoints/TASK/REPO --base /cache/models/BASE --reload
```

Omit `--reload` for header/index/config and finite-weight checks. Reloading a large model needs
sufficient GPU/host memory and is not a substitute for the official uploader test.

Compare same-split legal experiment exports offline (the reference must actually
come from the Boss run; the tool does not obtain or train it):

```bash
python scripts/compare_runs.py --reference boss_eval.json --candidate candidate_eval.json --kind paired --output comparison.json
python scripts/compare_runs.py --reference boss_grpo.json --candidate candidate_grpo.json --kind grpo --output comparison_grpo.json
```

Fingerprint/order and protocol mismatches are rejected. These are verdicts on
provided experiment exports, not official hidden-test results. Runtime DPO dev
selection still uses the training truncation; audit long pairs separately with
the official evaluator before choosing a competition configuration.
