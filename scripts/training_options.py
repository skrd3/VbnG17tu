"""Validate training boundaries before loading weights or running LR probes."""
from transformers import PretrainedConfig


def apply_request_limits(args, request):
    if getattr(args, "save_total_limit", None) is None:
        # The selected submission is committed separately from rolling saves.
        args.save_total_limit = 1
    limits = [int(value) for value in (args.max_steps, request.get("max_steps", -1))
              if value is not None and int(value) > 0]
    if limits:
        args.max_steps = min(limits)


def native_training_kwargs(model_path, quantization_config=None):
    """Decode native MXFP4 weights through the library's BF16 loader.

    Packed blocks/scales must be decoded, never relabelled as dense or NF4.
    BF16 experts work on A100 and are differentiable, unlike native MXFP4.
    """
    config, _ = PretrainedConfig.get_config_dict(model_path)
    quantization = config.get("quantization_config", {}) or {}
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if quantization.get("quant_method") == "mxfp4":
        if quantization_config is not None:
            raise ValueError("Native MXFP4 cannot be replaced by NF4; use its BF16 dequantization path")
        from transformers import Mxfp4Config
        return {"quantization_config": Mxfp4Config.from_dict({**quantization, "dequantize": True}),
                "attn_implementation": "eager"}
    return {}


def validate_lora_base(model_path, q_lora=False):
    options = native_training_kwargs(model_path)
    if options and q_lora:
        raise ValueError("Native MXFP4 cannot be replaced by NF4; use BF16 LoRA instead of q_lora")
    return options


def grpo_attention_implementation(model_path, disable_fa=False):
    if disable_fa:
        return "eager"
    config, _ = PretrainedConfig.get_config_dict(model_path)
    # GRPO right-pads completions after EOS. Qwen's FA2 path rejects those
    # batches even though the tokenizer correctly left-pads the prompts.
    if config.get("model_type") in {"qwen2", "qwen3", "qwen2_moe", "qwen3_moe"}:
        return "sdpa"
    return "flash_attention_2"


def cap_unfused_logits_batch(config, model_path, max_length):
    """Bound dense LM-head buffers as well as parameter memory."""
    if str(config.get("use_liger", False)).lower() == "true":
        return
    import torch
    model, _ = PretrainedConfig.get_config_dict(model_path)
    vocab = int(model.get("vocab_size", 0))
    if vocab <= 0:
        return
    budget = 8 * 1024**3
    if torch.cuda.is_available():
        budget = min(budget, int(min(torch.cuda.get_device_properties(i).total_memory
                                     for i in range(torch.cuda.device_count())) * .10))
    # BF16 logits, FP32 conversion/CE and backward buffers. OOM retries remain
    # a fallback for activation/optimizer memory, not the first sizing method.
    maximum = max(1, budget // (12 * vocab * max(1, max_length)))
    config["batch_size"] = min(config["batch_size"], maximum)
