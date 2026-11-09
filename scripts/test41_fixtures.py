"""Generate public-data-free architecture fixtures in the external test cache."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GptOssConfig, GptOssForCausalLM, Qwen2Config, Qwen2ForCausalLM


def prepare(cache):
    tokenizer = AutoTokenizer.from_pretrained(cache / "models/unsloth--Qwen2.5-0.5B", local_files_only=True)
    for kind in ("oss-bf16-tiny", "oss-mxfp4-tiny", "qwen-context-tiny"):
        target = cache / "models" / ("test41--" + kind)
        if (target / "fixture.json").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        options = dict(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       max_position_embeddings=128 if kind == "qwen-context-tiny" else 2048,
                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if kind == "qwen-context-tiny":
            model = Qwen2ForCausalLM(Qwen2Config(**options))
        else:
            model = GptOssForCausalLM(GptOssConfig(**options, head_dim=16, num_local_experts=4,
                                                experts_per_token=2, sliding_window=128,
                                                layer_types=["sliding_attention", "full_attention"]))
        model = model.to(torch.bfloat16)
        model.save_pretrained(target)
        if kind == "oss-mxfp4-tiny":
            # MXFP4 E2M1 nibbles with E8M0 per-32-element scales. The loader
            # must decode these real tensors; changing config alone cannot work.
            state = dict(model.state_dict())
            lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
            for key in list(state):
                if key.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                    dense = state.pop(key).transpose(1, 2).contiguous().float()
                    indices = (dense[..., None] * 64 - lut).abs().argmin(-1).to(torch.uint8)
                    packed = indices[..., 0::2] | (indices[..., 1::2] << 4)
                    blocks = packed.reshape(*packed.shape[:-1], dense.shape[-1] // 32, 16)
                    state[key + "_blocks"] = blocks.contiguous()
                    state[key + "_scales"] = torch.full(blocks.shape[:-1], 121, dtype=torch.uint8)
            model.config.quantization_config = {"quant_method": "mxfp4", "dequantize": False}
            model.config.save_pretrained(target)
            save_file({key: value.contiguous() for key, value in state.items()}, str(target / "model.safetensors"), metadata={"format": "pt"})
        tokenizer.save_pretrained(target)
        (target / "fixture.json").write_text(json.dumps({"kind": kind, "seed": 42, "synthetic": True}))
        del model


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
