"""Collective-safe wall-clock control and measured plan updates."""
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import TrainerCallback

from submission import publish_checkpoint, validate_submission
from training_budget import TrainingBudget


class TrainingControlCallback(TrainerCallback):
    def __init__(self, trainer, request):
        self.trainer, self.request = trainer, request
        self.budget = TrainingBudget(request["end_time"])
        self.last_step = None
        self.steps = []
        self.eval_seconds = 0.0
        self.save_seconds = 60.0
        self.token_count = 0
        self.completion_tokens = 0
        self._save_start = None
        self._scheduled = False

    def _sync(self, value, op=torch.distributed.ReduceOp.MAX):
        if torch.distributed.is_initialized():
            tensor = torch.tensor([float(value)], device=self.trainer.args.device)
            torch.distributed.all_reduce(tensor, op=op)
            return tensor.item()
        return float(value)

    def on_step_begin(self, args, state, control, **kwargs):
        self.last_step = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._token_counts = None
        self.trainer._last_finite_loss = None
        model = getattr(self.trainer, "model", None)
        if model is not None:
            adapter = bool(getattr(model, "peft_config", None))
            size = sum(getattr(p, "ds_numel", p.numel()) * p.element_size()
                       for p in model.parameters() if not adapter or p.requires_grad)
            bandwidth = float(os.environ.get("SUBMISSION_IO_MIBPS", "512"))
            if not math.isfinite(bandwidth) or bandwidth <= 0:
                raise ValueError("SUBMISSION_IO_MIBPS must be finite and positive")
            # Model save + stage read/write + finite scan + hash. This is an
            # estimate, updated with actual save timings; no claimed guarantee.
            self.save_seconds = max(self.save_seconds, 30 + 5 * size / (bandwidth * 1024**2))

    def _write_observations(self, state):
        if self.trainer.is_world_process_zero():
            path = Path(self.request["output_dir"]) / "resolved_training_plan.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            plan = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            plan.update(total_observed_tokens=self.token_count, total_completion_tokens=self.completion_tokens,
                        measured_tokens_per_update=self.token_count / max(1, state.global_step),
                        measured_completion_tokens_per_update=self.completion_tokens / max(1, state.global_step),
                        training_token_seconds=sum(self.steps),
                        completion_tokens_per_second=self.completion_tokens / max(.001, sum(self.steps)))
            path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.eval_seconds = max(self.eval_seconds, float(metrics.get("eval_runtime", 0)))

    def on_step_end(self, args, state, control, **kwargs):
        counts = getattr(self.trainer, "_token_counts", None)
        # Always execute the same collective, even when a backend exposes no
        # recognized token fields on a rank. Loss validity is independent of
        # whether token accounting is available.
        if counts is None:
            counts = torch.zeros(2, dtype=torch.int64, device=args.device)
        finite = getattr(self.trainer, "_last_finite_loss", None)
        flag = finite.to(counts.device, dtype=torch.int64).reshape(1) if finite is not None else counts.new_ones(1)
        payload = torch.cat((counts, flag))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(payload)
        total, completion, valid_ranks = payload.cpu().tolist()
        if valid_ranks != args.world_size:
            raise RuntimeError("Nonfinite training loss; preserving previous committed model")
        self.token_count += total
        self.completion_tokens += completion
        self.trainer._token_counts = None
        self.trainer._last_finite_loss = None
        # Earlier callbacks use local clocks. Resolve their control flags on
        # all ranks before HF branches into evaluation or backend-aware saves.
        control.should_training_stop = bool(self._sync(control.should_training_stop))
        control.should_evaluate = bool(self._sync(control.should_evaluate))
        control.should_save = bool(self._sync(control.should_save))
        if self.last_step is not None:
            self.steps.append(self._sync(time.monotonic() - self.last_step))
        # End-time decision uses the slowest rank's reserve and least remaining
        # time. Every rank executes precisely the same collectives per step.
        remaining = self._sync(self.budget.remaining(), torch.distributed.ReduceOp.MIN)
        step_seconds = max(self.steps[-8:] or [10.0])
        reserve = self._sync(max(120.0, self.save_seconds * 1.5) + step_seconds * 2 + 30)
        if remaining <= reserve:
            control.should_training_stop = True
            control.should_evaluate = False
            # Existing best stays committed; a last unevaluated checkpoint must
            # not overwrite it. If there has been no save, request the first.
            control.should_save = state.global_step > 0
        elif state.global_step >= state.max_steps:
            control.should_save = True
        elif state.global_step == 2:
            control.should_save = True
            control.should_evaluate = False
        elif control.should_evaluate and remaining <= reserve + max(30, self.eval_seconds * 1.5):
            control.should_evaluate = False
        if control.should_save:
            self._save_start = time.monotonic()
        if len(self.steps) >= 8 and not self._scheduled:
            self._scheduled = True
            steady = sorted(self.steps[2:])
            estimate = steady[len(steady) // 2]
            remaining_updates = max(1, int((remaining - reserve - self.eval_seconds) / max(estimate, .001)))
            remaining_updates = min(remaining_updates, max(1, state.max_steps - state.global_step))
            # Keep the established schedule on every backend. Reconstructing
            # LambdaLR here restarts at peak LR and discards unfinished warmup.
            # Timing informs stop/eval reserves, not a mid-run optimizer reset.
            if self.trainer.is_world_process_zero():
                plan = Path(self.request["output_dir"]) / "resolved_training_plan.json"
                values = json.loads(plan.read_text(encoding="utf-8")) if plan.exists() else {}
                values.update(step_seconds=estimate, remaining_updates=remaining_updates,
                              scheduler_strategy="preserve_original_warmup_and_schedule",
                              effective_blocks_per_update=args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)
                plan.write_text(json.dumps(values, indent=2), encoding="utf-8")
            self._write_observations(state)

    def on_save(self, args, state, control, **kwargs):
        if self._save_start is not None:
            self.save_seconds = max(self.save_seconds, time.monotonic() - self._save_start)
            self._save_start = None

    def on_train_end(self, args, state, control, **kwargs):
        self._write_observations(state)
        valid = False
        if self.trainer.is_world_process_zero():
            try:
                validate_submission(self.request["submission_dir"], self.request["model_path"])
                valid = True
            except (OSError, ValueError):
                pass
        valid = bool(self._sync(valid))
        if valid:
            return
        if state.global_step <= 0:
            raise RuntimeError("No training update completed; refusing to submit base weights")
        # HF's backend-aware save must run on every rank (ZeRO gather included).
        destination = str(Path(args.output_dir) / "final-trained-model")
        self.trainer.save_model(destination)
        error = None
        if self.trainer.is_world_process_zero():
            try:
                publish_checkpoint(destination, self.request["submission_dir"], self.request["model_path"],
                                   state.global_step, "final_without_eval")
            except Exception as exc:
                error = str(exc)
        failed = self._sync(error is not None)
        if failed:
            raise RuntimeError(error or "Rank 0 could not commit final model")
