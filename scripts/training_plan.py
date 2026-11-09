"""Resource resolution with explicit block/token batch accounting."""
import json
import hashlib
import math
import os
from pathlib import Path


def observe_tokens(trainer, inputs):
    """Accumulate detached device counters without synchronizing each microbatch."""
    import torch
    completion = inputs.get("completion_mask")
    prompt = inputs.get("prompt_mask", inputs.get("prompt_attention_mask"))
    attention = inputs.get("attention_mask")
    if completion is not None:
        supervised = (completion != 0).sum()
        total = supervised + ((prompt != 0).sum() if prompt is not None else 0)
        if attention is not None:
            total = (attention != 0).sum()
    elif "labels" in inputs:
        supervised = (inputs["labels"][:, 1:] != -100).sum()
        total = (attention != 0).sum() if attention is not None else torch.tensor(inputs["input_ids"].numel(), device=supervised.device)
    elif "chosen_attention_mask" in inputs and "rejected_attention_mask" in inputs:
        supervised = inputs["chosen_attention_mask"].sum() + inputs["rejected_attention_mask"].sum()
        total = supervised + (2 * prompt.sum() if prompt is not None else 0)
    else:
        return
    values = torch.stack((total, supervised)).detach().to(torch.int64)
    if getattr(trainer, "_token_counts", None) is None:
        trainer._token_counts = values
    else:
        trainer._token_counts += values


def search_plan_key(args, request):
    """Cache identity includes the task, local artifacts and loss/resource plan."""
    model = Path(request["model_path"])
    artifacts = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(model.glob("*")) if p.is_file()]
    dataset = Path(request["dataset"])
    digest = hashlib.sha256()
    if dataset.is_file():
        with dataset.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    identity = {"task": request["task_id"], "model": str(model), "artifacts": artifacts,
                "dataset_sha256": digest.hexdigest(), "length": request.get("max_length"),
                "microbatch": args.per_device_train_batch_size, "world": args.world_size,
                "accum": args.gradient_accumulation_steps, "optimizer": str(args.optim),
                "use_kl": os.environ.get("USE_KL"), "kl_coef": os.environ.get("KL_COEF"),
                "use_lora": getattr(args, "use_lora", getattr(args, "use_peft", False)),
                "training_controls": request.get("runtime_config", {}),
                "baseline_stats": request.get("baseline_stats"),
                "objective": {name: getattr(args, name, None) for name in
                              ("label_smoothing_factor", "beta", "loss_type", "rpo_alpha", "ld_alpha", "use_weighting")}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def resolve_batch(config, target=64, max_accum=None, generations=None):
    world = max(1, int(config["gpu_nums"]))
    micro = max(1, int(config["batch_size"]))
    if generations:
        # TRL samples groups inside each global generation batch, not across
        # gradient accumulation. Keep global micro/eval batch divisible.
        divisor = int(generations) // math.gcd(world, int(generations))
        micro = max(divisor, micro // divisor * divisor)
    accum = max(1, math.ceil(target / (micro * world)))
    if max_accum:
        accum = min(accum, max_accum)
    if "requested_gradient_accumulation_steps" in config:
        accum = int(config["requested_gradient_accumulation_steps"])
    config.update(batch_size=micro, gradient_accumulation_steps=accum)
    return micro * accum * world


def record_plan(args, request):
    import torch
    actual = int(args.world_size)
    if actual < 1 or not len(request.get("task_id", "")):
        raise ValueError("Invalid world size/task id")
    plan = {"world_size": actual, "microbatch": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "effective_blocks_per_update": actual * args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "max_length": request.get("max_length"), "learning_rate": args.learning_rate,
            "optimizer": str(args.optim), "deadline": request["end_time"],
            "use_kl": os.environ.get("USE_KL") == "1", "step_seconds": None}
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        path = Path(request["output_dir"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "resolved_training_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
