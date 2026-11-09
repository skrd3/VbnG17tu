"""Read-only artifact preflight; --reload performs a real local forward."""
import argparse
import json
from pathlib import Path

from submission import validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", required=True, help="Local task base cache directory")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    result = validate_submission(args.output, args.base, check_finite=True)
    if args.reload:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"local_files_only": True, "trust_remote_code": False,
                  "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        if (Path(args.output) / "adapter_config.json").is_file():
            from peft import PeftModel
            from training_options import native_training_kwargs
            base = AutoModelForCausalLM.from_pretrained(args.base, **{**kwargs, **native_training_kwargs(args.base)})
            model = PeftModel.from_pretrained(base, args.output, local_files_only=True)
        else:
            model, loading = AutoModelForCausalLM.from_pretrained(args.output, output_loading_info=True, **kwargs)
            if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
                raise RuntimeError(f"Submission state_dict mismatch: {loading}")
        tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
        inputs = tokenizer("Submission validation example.", return_tensors="pt")
        inputs = {key: value.to(model.get_input_embeddings().weight.device) for key, value in inputs.items()
                  if key in ("input_ids", "attention_mask")}
        model.eval()
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Reloaded submission produced nonfinite causal loss")
        result["reload_loss"] = float(loss)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
