"""Optional, validated training controls shared by the public container entrypoint.

The validator's standard CLI needs no overrides. These controls select real
training backends/hyperparameters without changing cache, output or deadlines.
"""
import json
from pathlib import Path

from process_runner import replace_argument

BOOLS = {"use_lora", "use_liger", "packing", "disable_fa", "q_lora", "gradient_checkpointing",
         "use_vllm", "explicit_reference", "enable_dev_pass", "grpo_eval_initial"}
INTS = {"micro_batch_size", "gradient_accumulation_steps", "max_length", "max_prompt_length",
        "max_completion_length", "num_generations", "lora_r", "lora_alpha", "periodic_save_steps",
        "grpo_selection_prompts", "vllm_server_port", "min_steps"}
FLOATS = {"learning_rate", "label_smoothing_factor", "label_smoothing", "rpo_alpha",
          "lr_search_fraction", "vllm_gpu_memory_utilization", "save_before_remaining_minutes"}
ENUMS = {"backend": {"ddp", "ds", "fsdp"}, "launcher": {"auto", "torchrun", "deepspeed"},
         "packing_mode": {"fa", "naive"}, "vllm_mode": {"colocate", "server"}}
STRINGS = {"vllm_server_host", "fsdp_layer", "grpo_selection_seeds"}


def load_config(value):
    if not value:
        return {}
    config = json.loads(Path(value).read_text()) if not value.lstrip().startswith("{") else json.loads(value)
    if not isinstance(config, dict):
        raise ValueError("training-config must be a JSON object")
    unknown = set(config) - BOOLS - INTS - FLOATS - set(ENUMS) - STRINGS
    if unknown:
        raise ValueError(f"Unknown training controls: {sorted(unknown)}")
    import math
    for key, item in config.items():
        if key in BOOLS and type(item) is not bool:
            raise ValueError(f"{key} must be boolean")
        if key in INTS and (type(item) is not int or item < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key in FLOATS and (type(item) not in (int, float) or not math.isfinite(item) or item < 0):
            raise ValueError(f"{key} must be a finite nonnegative number")
        if key in ENUMS and item not in ENUMS[key]:
            raise ValueError(f"Unsupported {key}: {item}")
        if key in STRINGS and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("label_smoothing", "label_smoothing_factor", "lr_search_fraction"):
        if config.get(key, 0) > 1:
            raise ValueError(f"{key} must not exceed 1")
    if not 0 < config.get("vllm_gpu_memory_utilization", .4) < 1:
        raise ValueError("vllm_gpu_memory_utilization must be between 0 and 1")
    if config.get("vllm_server_port", 8000) > 65535:
        raise ValueError("Invalid vLLM port")
    return config


def configure_environment(config):
    import os
    names = {"lr_search_fraction": "LR_SEARCH_FRACTION", "enable_dev_pass": "ENABLE_DEV_PASS",
             "grpo_eval_initial": "GRPO_EVAL_INITIAL", "grpo_selection_prompts": "GRPO_SELECTION_PROMPTS",
             "grpo_selection_seeds": "GRPO_SELECTION_SEEDS"}
    for key, variable in names.items():
        if key in config:
            value = config[key]
            os.environ[variable] = str(int(value)) if isinstance(value, bool) else str(value)


def apply_plan(info, plan, max_length=None):
    options = info.get("runtime_config", {})
    aliases = {"backend": "distributed", "micro_batch_size": "batch_size",
               "gradient_accumulation_steps": "requested_gradient_accumulation_steps"}
    for key in ("backend", "launcher", "micro_batch_size", "gradient_accumulation_steps", "learning_rate",
                "use_lora", "use_liger", "packing", "disable_fa", "gradient_checkpointing", "packing_mode",
                "use_vllm", "vllm_gpu_memory_utilization", "num_generations"):
        if key in options:
            value = options[key]
            # Existing config templates use the strings True/False.
            plan[aliases.get(key, key)] = str(value) if key in {"use_liger", "packing", "disable_fa", "gradient_checkpointing"} else value
    if max_length is not None and "max_length" in options:
        from transformers import PretrainedConfig
        model, _ = PretrainedConfig.get_config_dict(info["model_path"])
        max_length = min(options["max_length"], model.get("max_position_embeddings", options["max_length"]))
    if plan.get("packing_mode") == "fa" and str(plan.get("disable_fa", False)).lower() == "true":
        plan["packing_mode"] = "naive"
    return max_length


def finish(info, plan, request, command, task):
    options = info.get("runtime_config", {})
    if not options:
        return {"train_request": request, "run_cmd": command}
    for key in ("periodic_save_steps", "min_steps", "packing_mode"):
        if key in options:
            request[key] = options[key]
    if "save_before_remaining_minutes" in options:
        request["save_before_remaining_time"] = options["save_before_remaining_minutes"]
    if options.get("explicit_reference"):
        request["ref_model"] = info["model_path"]
    if task == "dpo" and "max_length" in options:
        request["max_length"] = options["max_length"]
    for key in ("max_prompt_length",):
        if key in options:
            request[key] = options[key]
    if task == "instruct":
        args = {"q_lora", "lora_r", "lora_alpha", "label_smoothing_factor"}
    elif task == "dpo":
        args = {"lora_r", "lora_alpha", "label_smoothing", "rpo_alpha", "max_length", "max_prompt_length"}
    else:
        args = {"lora_r", "lora_alpha", "max_prompt_length", "max_completion_length", "vllm_mode",
                "vllm_server_host", "vllm_server_port"}
    for key in args & options.keys():
        command = replace_argument(command, key, options[key])
    if plan["distributed"] == "fsdp":
        from transformers import PretrainedConfig
        config, _ = PretrainedConfig.get_config_dict(info["model_path"])
        layers = {"qwen2": "Qwen2DecoderLayer", "qwen3": "Qwen3DecoderLayer", "llama": "LlamaDecoderLayer", "gpt_oss": "GptOssDecoderLayer"}
        layer = options.get("fsdp_layer", layers.get(config.get("model_type")))
        if layer is None:
            raise ValueError("Specify fsdp_layer for this architecture")
        command = replace_argument(command, "fsdp", "full_shard auto_wrap")
        command = replace_argument(command, "fsdp_transformer_layer_cls_to_wrap", layer)
        command = replace_argument(command, "fsdp_config", json.dumps({"use_orig_params": True, "sync_module_states": True, "state_dict_type": "FULL_STATE_DICT"}))
    return {"train_request": request, "run_cmd": command}
