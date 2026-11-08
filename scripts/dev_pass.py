"""Optional reclaim candidate with an independent held-out acceptance gate.

Disabled by default, and unsupported for sharded backends. A rejected candidate
restores in-memory weights. Only an accepted candidate may atomically replace
the existing validated submission. Gate data never enter the reclaim loader.
"""
from training_budget import TrainingBudget
from selection_gate import evaluate_gate, accept_gate
from submission import new_staging, commit_staging
import time

import contextlib
import gc
import os
import shutil
from inspect import signature as _signature

import torch
from transformers.trainer_utils import is_main_process


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def run_dev_pass(
    trainer,
    *,
    submission_dir,
    min_lr,
    max_grad_norm,
    train_per_device,
    train_grad_accum,
    local_rank,
    log,
):
    """One low-LR epoch over the trainer's eval (dev) set, then a weights-only
    save into submission_dir. Uses trainer.get_eval_dataloader() so batches are
    built correctly for THIS trainer (plain LM for instruct; tokenized
    chosen/rejected pairs for DPO) and trainer.training_step() so the loss is the
    trainer's own (incl. DPO's reference-model term)."""
    if os.environ.get("ENABLE_DEV_PASS") != "1" or getattr(trainer, "selection_gate", None) is None:
        return {"accepted": False, "reason": "disabled_or_no_independent_gate"}
    if getattr(trainer, "is_deepspeed_enabled", False) or getattr(trainer, "is_fsdp_enabled", False):
        return {"accepted": False, "reason": "sharded_dev_pass_not_supported"}
    budget = TrainingBudget()
    if not budget.can_start(300, reserve=180):
        return {"accepted": False, "reason": "insufficient_budget"}
    before = evaluate_gate(trainer)
    accelerator = trainer.accelerator
    # Reuse the model the trainer already wrapped — do NOT re-wrap (that would
    # register a second set of DDP reducer hooks on the same params).
    ddp_model = getattr(trainer, "model_wrapped", None) or trainer.model
    unwrapped = _unwrap(ddp_model)
    device = next(unwrapped.parameters()).device

    trainable = [p for p in unwrapped.parameters() if p.requires_grad]
    original = {n: p.detach().cpu().clone() for n, p in unwrapped.named_parameters() if p.requires_grad}
    if not trainable:
        log("[sn56][devfit] sem parâmetros treináveis, pulando")
        return

    # Drop the training optimizer + stale grads to free room for a fresh,
    # momentum-free dev optimizer.
    try:
        trainer.optimizer = None
    except Exception:
        pass
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev_opt = torch.optim.AdamW(trainable, lr=min_lr, weight_decay=0.0)

    # The trainer's own eval dataloader: correctly processed + sharded for THIS
    # trainer. It uses the eval batch size, so match training's effective batch by
    # accumulating proportionally more micro-steps.
    loader = trainer.get_eval_dataloader()
    per_device_eval = max(1, int(getattr(trainer.args, "per_device_eval_batch_size", 1) or 1))
    accum = max(1, round(train_per_device * train_grad_accum / per_device_eval))

    steps_in_epoch = len(loader)
    # Defensive lockstep: don't trust even_batches — force every rank to run the
    # SAME number of micro-batches (the global min). Identical step counts ->
    # identical boundary/all-reduce pattern -> DDP can't deadlock.
    if torch.distributed.is_initialized():
        _t = torch.tensor([steps_in_epoch], device=device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.MIN)
        steps_in_epoch = int(_t.item())
    if steps_in_epoch == 0:
        log("[sn56][devfit] dev loader vazio, pulando")
        return

    # Use compute_loss explicitly: Transformers 4.51 normalizes training_step
    # by args.gradient_accumulation_steps, not the dev loop's accumulation.
    # Accelerator.backward also has its own divisor. Account for it exactly.
    ddp_model.train()
    n_opt = 0
    completed = True
    for step, inputs in enumerate(loader):
        remaining = budget.remaining()
        if torch.distributed.is_initialized():
            remaining_tensor = torch.tensor([remaining], device=device)
            torch.distributed.all_reduce(remaining_tensor, op=torch.distributed.ReduceOp.MIN)
            remaining = remaining_tensor.item()
        if remaining < 180:
            completed = False
            break
        if step >= steps_in_epoch:   # honor the synced step count on every rank
            break
        do_sync = ((step + 1) % accum == 0) or ((step + 1) == steps_in_epoch)
        # We bypass the Trainer's prefetch-aware loop, so set the flag ourselves.
        accelerator.gradient_state._set_sync_gradients(do_sync)
        sync_ctx = (contextlib.nullcontext() if do_sync
                    else accelerator.no_sync(model=ddp_model))
        with sync_ctx:
            # training_step computes the RIGHT loss for this trainer and handles
            # autocast + _prepare_inputs + accelerator.backward.
            inputs = trainer._prepare_inputs(inputs)
            with trainer.compute_loss_context_manager():
                loss = trainer.compute_loss(ddp_model, inputs, num_items_in_batch=None)
            window = min(accum, steps_in_epoch - (step // accum) * accum)
            accelerator.backward(loss * accelerator.gradient_accumulation_steps / window)
        if do_sync:
            if max_grad_norm and max_grad_norm > 0:
                accelerator.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
            dev_opt.step()
            ddp_model.zero_grad(set_to_none=True)
            n_opt += 1

    eff = per_device_eval * accum * max(1, getattr(accelerator, "num_processes", 1))
    log(f"[sn56][devfit] {steps_in_epoch} micro / {n_opt} opt-steps sobre dev "
        f"@ lr={min_lr:.2e} (lote efetivo ~{eff}, igual ao treino)")

    # Free the dev optimizer before the save / any NCCL.
    del dev_opt
    unwrapped.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accepted = False
    may_evaluate = completed and budget.can_start(60, reserve=120)
    if torch.distributed.is_initialized():
        flag = torch.tensor([int(may_evaluate)], device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        may_evaluate = bool(flag.item())
    if may_evaluate:
        after = evaluate_gate(trainer)
        accepted = accept_gate(before, after)
    # All ranks have the same gathered gate metrics. Keep policy parameters in
    # sync on rejection and never change the existing committed model on error.
    if not accepted:
        with torch.no_grad():
            for name, parameter in unwrapped.named_parameters():
                if name in original:
                    parameter.copy_(original[name].to(parameter.device))
    if accepted and is_main_process(local_rank):
        try:
            _save_weights_only(unwrapped, submission_dir, log)
        except Exception as exc:
            log(f"Gate passed but publication failed; preserving prior submission: {exc}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return {"accepted": accepted}


def _is_weight_file(name):
    return (name.endswith(".safetensors")
            or name == "model.safetensors.index.json"
            or name.endswith(".bin"))


def _save_weights_only(unwrapped, submission_dir, log):
    """Stage weights and metadata, then atomically exchange real directories."""
    if not submission_dir or not os.path.isdir(submission_dir):
        raise ValueError("Cannot replace an absent validated submission")
    staging = new_staging(submission_dir)
    try:
        unwrapped.save_pretrained(staging, safe_serialization=True)
        for name in os.listdir(submission_dir):
            src = os.path.join(submission_dir, name)
            if os.path.isfile(src) and not _is_weight_file(name):
                shutil.copy2(src, staging / name)
        previous_manifest = os.path.join(submission_dir, "submission_manifest.json")
        metadata = {}
        if os.path.isfile(previous_manifest):
            import json
            metadata = json.loads(open(previous_manifest, encoding="utf-8").read()).get("selection", {})
        commit_staging(staging, submission_dir, metadata={**metadata, "postprocess": "validated_weights_candidate"})
        log("Validated candidate committed")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
