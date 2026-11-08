"""GRPO selection proxy: fixed prompts, raw rewards and reverse prompt KL.

The anchor is the first successfully scored local policy, never an unavailable
opponent. Raw-utility selection is monotone; official matchup normalization is
logged separately and must be recomputed against a real opponent offline.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from trl import GRPOTrainer

from reward_scoring import normalize_scores, reward_utility
from training_budget import TrainingBudget
from paired_selection import dataset_fingerprint


@contextmanager
def selection_randomness(trainer, seed):
    """Restore training RNG and seed colocated vLLM SamplingParams explicitly."""
    devices = [trainer.accelerator.device.index] if trainer.accelerator.device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    llm = getattr(trainer, "llm", None)
    original_generate = getattr(llm, "generate", None)
    had_override = llm is not None and "generate" in getattr(llm, "__dict__", {})
    old_override = getattr(llm, "__dict__", {}).get("generate")
    if original_generate is not None:
        def generate(*args, **kwargs):
            params = kwargs.get("sampling_params")
            previous_seed = getattr(params, "seed", None)
            if params is not None:
                params.seed = seed
            try:
                return original_generate(*args, **kwargs)
            finally:
                if params is not None:
                    params.seed = previous_seed
        llm.generate = generate
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if original_generate is not None:
            if had_override:
                llm.generate = old_override
            else:
                del llm.generate


@contextmanager
def reference_policy(trainer):
    policy = trainer.accelerator.unwrap_model(trainer.model)
    ref = getattr(trainer, "ref_model", None)
    if ref is not None:
        ref.eval()
        yield ref
    elif hasattr(policy, "disable_adapter"):
        mode = policy.training
        try:
            policy.eval()
            with policy.disable_adapter():
                yield policy
        finally:
            policy.train(mode)
    else:
        raise ValueError("Prompt KL requires a frozen base model or a LoRA base")


@torch.no_grad()
def prompt_reverse_kl(trainer, dataset):
    tokenizer = trainer.processing_class
    policy = trainer.model_wrapped if trainer.is_deepspeed_enabled or getattr(trainer, "is_fsdp_enabled", False) else trainer.accelerator.unwrap_model(trainer.model)
    was_training = policy.training
    policy.eval()
    total = 0.0
    maximum = getattr(trainer.model.config, "max_position_embeddings", None)
    limit = 512 if not isinstance(maximum, int) or maximum >= 1024 else math.ceil(maximum / 2)
    try:
        for row in dataset:
            inputs = tokenizer(row["prompt"], padding=True, truncation=True, max_length=limit, return_tensors="pt")
            inputs = {key: value.to(trainer.accelerator.device) for key, value in inputs.items()}
            # Base forward first, so disable_adapter never toggles the active
            # adapter while a policy computation graph is waiting for backward.
            with reference_policy(trainer) as ref:
                base = ref(**inputs).logits
            ft = policy(**inputs).logits
            positions = []
            for start in range(0, ft.shape[1], 128):
                p = torch.nn.functional.softmax(base[:, start:start + 128], dim=-1)
                logq = torch.nn.functional.log_softmax(ft[:, start:start + 128], dim=-1)
                positions.append(torch.nn.functional.kl_div(logq, p, reduction="none").sum(dim=-1))
            total += torch.cat(positions, dim=1).mean(dim=-1).sum().item()
        return total / len(dataset)
    finally:
        policy.train(was_training)


class OfficialGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, *args, **kwargs):
        if model.training:
            from training_plan import observe_tokens
            observe_tokens(self, inputs)
        result = super().compute_loss(model, inputs, *args, **kwargs)
        if model.training:
            current_finite = torch.isfinite((result[0] if isinstance(result, tuple) else result).detach())
            previous_finite = getattr(self, "_last_finite_loss", None)
            self._last_finite_loss = current_finite if previous_finite is None else previous_finite & current_finite
        return result

    def __init__(self, *args, **kwargs):
        self.selection_baseline = None
        self.selection_scales = None
        self.selection_seed_seconds = 60.0
        self.selection_kl_seconds = 60.0
        self.selection_seeds = [int(seed.strip()) for seed in os.environ.get("GRPO_SELECTION_SEEDS", "42,314159").split(",")]
        if not self.selection_seeds or len(self.selection_seeds) > 4 or len(set(self.selection_seeds)) != len(self.selection_seeds):
            raise ValueError("Use one to four distinct GRPO_SELECTION_SEEDS")
        if any(seed < 0 or seed >= 2**32 for seed in self.selection_seeds):
            raise ValueError("GRPO selection seeds must be unsigned 32-bit integers")
        config = kwargs.get("args")
        if getattr(config, "deepspeed", None) and not getattr(config, "use_vllm", False):
            # Retain ZeRO's per-layer gather/release hooks during generation.
            # TRL 0.18's temporary global gather removes/reinstalls hooks and
            # can leave empty weights when evaluation follows training/saving.
            config.ds3_gather_for_generation = False
        model = kwargs.get("model")
        context = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        self.selection_prompt_length = min(512, context // 2) if isinstance(context, int) else 512
        self.selection_completion_length = min(256, context - self.selection_prompt_length) if isinstance(context, int) else 256
        original = (config.max_prompt_length, config.max_completion_length)
        if isinstance(context, int):
            if context < 2:
                raise ValueError("GRPO requires at least two context positions")
            completion = min(max(1, original[1]), context - 1)
            original = (min(max(1, original[0] or 512), context - completion), completion)
        # vLLM must reserve a context large enough for both training and the
        # evaluator, even if adaptive training uses shorter prompts.
        config.max_prompt_length = max(config.max_prompt_length or 0, self.selection_prompt_length)
        config.max_completion_length = max(config.max_completion_length, self.selection_completion_length)
        if isinstance(context, int):
            config.max_completion_length = min(config.max_completion_length, context - 1)
            config.max_prompt_length = min(config.max_prompt_length, context - config.max_completion_length)
        try:
            super().__init__(*args, **kwargs)
        finally:
            config.max_prompt_length, config.max_completion_length = original
        self.max_prompt_length, self.max_completion_length = original
        if hasattr(self, "generation_config"):
            self.generation_config.max_new_tokens = original[1]

    def evaluate(self, *args, **kwargs):
        # Check the WHOLE operation before generating, not only the extra KL.
        required = len(self.selection_seeds) * self.selection_seed_seconds + self.selection_kl_seconds
        if not self._selection_time_available(required):
            prefix = kwargs.get("metric_key_prefix", "eval")
            return {f"{prefix}_selection_skipped": 1, f"{prefix}_runtime": 0.}
        original = (self.max_prompt_length, self.max_completion_length)
        generation = getattr(self, "generation_config", None)
        old_generation = generation.max_new_tokens if generation is not None else None
        self.max_prompt_length = self.selection_prompt_length
        self.max_completion_length = self.selection_completion_length
        if generation is not None:
            generation.max_new_tokens = self.selection_completion_length
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self.max_prompt_length, self.max_completion_length = original
            if generation is not None:
                generation.max_new_tokens = old_generation

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        seeds_raw, weights = [], {}
        for index, seed in enumerate(self.selection_seeds):
            for function in self.reward_funcs:
                function.raw_values.clear()
            started = time.monotonic()
            with selection_randomness(self, seed):
                output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
            self.selection_seed_seconds = max(self.selection_seed_seconds, time.monotonic() - started)
            raw = {}
            for function in self.reward_funcs:
                values = function.raw_values
                local = torch.tensor([sum(values), len(values)], dtype=torch.float64, device=self.accelerator.device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(local)
                if local[1] <= 0 or not torch.isfinite(local).all():
                    raise ValueError("Invalid generated reward statistics")
                raw[function.reward_name] = (local[0] / local[1]).item()
                weights[function.reward_name] = function.reward_weight
            seeds_raw.append(raw)
            remaining = (len(self.selection_seeds) - index - 1) * self.selection_seed_seconds + self.selection_kl_seconds
            if not self._selection_time_available(remaining):
                output.metrics[f"{metric_key_prefix}_selection_skipped"] = 1
                return output
        started = time.monotonic()
        dataset = args[0].dataset if args else kwargs["dataloader"].dataset
        kl = prompt_reverse_kl(self, dataset)
        self.selection_kl_seconds = max(self.selection_kl_seconds, time.monotonic() - started)
        if not math.isfinite(kl):
            raise ValueError("Nonfinite official prompt KL")
        raw = {name: sum(values[name] for values in seeds_raw) / len(seeds_raw) for name in weights}
        if self.selection_baseline is None:
            self.selection_baseline = {"raw": raw.copy(), "kl": kl, "step": self.state.global_step}
            self.selection_scales = {name: max(1., abs(raw[name]), float(np.std([row[name] for row in seeds_raw])))
                                     for name in weights}
        scores = normalize_scores({"anchor": self.selection_baseline["raw"], "candidate": raw}, weights,
                                  {"anchor": self.selection_baseline["kl"], "candidate": kl})
        margin = scores["candidate"] - scores["anchor"]
        utilities = [reward_utility(row, weights, self.selection_scales, kl) for row in seeds_raw]
        # Conservative across-seed proxy, explicitly distinct from official
        # opponent-dependent normalization. Fixed scales remove reward plateaus.
        utility = float(np.mean(utilities) - .5 * np.std(utilities))
        output.metrics.update({f"{metric_key_prefix}_ranking_loss": -utility,
                               f"{metric_key_prefix}_prompt_reverse_kl": kl,
                               f"{metric_key_prefix}_reward_utility": utility,
                               f"{metric_key_prefix}_proxy_margin": margin,
                               f"{metric_key_prefix}_proxy_score": scores["candidate"]})
        if self.is_world_process_zero():
            directory = Path(self.args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"grpo_selection-{self.state.global_step}.json").write_text(
                json.dumps({"raw_rewards": raw, "weights": weights, "prompt_reverse_kl": kl,
                            "dataset_fingerprint": dataset_fingerprint(dataset),
                            "generation_protocol": {"seeds": self.selection_seeds,
                                "prompt_length": self.selection_prompt_length,
                                "completion_length": self.selection_completion_length,
                                "num_generations": self.num_generations,
                                "temperature": self.temperature, "top_p": self.top_p,
                                "top_k": self.top_k, "min_p": self.min_p,
                                "repetition_penalty": self.repetition_penalty,
                                "guided_decoding_regex": getattr(self, "guided_decoding_regex", None)},
                            "baseline": self.selection_baseline, "scores": scores,
                            "proxy_margin": margin, "selection_utility": utility,
                            "selection_scales": self.selection_scales, "seed_raw_rewards": seeds_raw,
                            "seeds": self.selection_seeds, "seed_std_penalty": .5,
                            "seed_control": "explicit_colocate_or_torch; vllm_server_not_seeded",
                            "scope": "local monotone proxy, not actual Boss score",
                            "prompts": len(dataset)}), encoding="utf-8")
        return output

    def _selection_time_available(self, seconds):
        available = TrainingBudget().can_start(seconds, reserve=180)
        if torch.distributed.is_initialized():
            value = torch.tensor([int(available)], device=self.accelerator.device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
            available = bool(value.item())
        return available
