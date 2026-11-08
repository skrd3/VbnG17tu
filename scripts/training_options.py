"""Validate training boundaries before loading weights or running LR probes."""
from transformers import PretrainedConfig


def apply_request_limits(args, request):
    limits = [int(value) for value in (args.max_steps, request.get("max_steps", -1))
              if value is not None and int(value) > 0]
    if limits:
        args.max_steps = min(limits)


def validate_lora_base(model_path):
    config, _ = PretrainedConfig.get_config_dict(model_path)
    quantization = config.get("quantization_config", {}) or {}
    if hasattr(quantization, "to_dict"):
        quantization = quantization.to_dict()
    if quantization.get("quant_method") == "mxfp4":
        raise ValueError(
            "LoRA training of pretrained MXFP4 checkpoints is not supported by this trainer; "
            "use a BF16 base for LoRA or the configured gpt-oss full-parameter path. "
            "MXFP4 quantization must not be replaced with None or NF4."
        )
