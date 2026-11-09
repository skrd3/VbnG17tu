"""Regressions for validator entrypoint failures; run in the built oss overlay."""
import datetime
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import TrainerControl

from grpo_selection import selection_randomness
from process_runner import argv, training_launcher
from training_control import TrainingControlCallback
from training_options import cap_unfused_logits_batch, grpo_attention_implementation, validate_lora_base
from runtime_config import load_config, apply_plan


class RuntimeReliabilityTests(unittest.TestCase):
    def config_dir(self, config):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Path(temp.name, "config.json").write_text(json.dumps(config))
        return temp.name

    def callback(self, finite):
        trainer = types.SimpleNamespace(_token_counts=torch.tensor([10, 3]), _last_finite_loss=finite)
        end = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        return TrainingControlCallback(trainer, {"end_time": end})

    def test_finite_replica_vector_is_reduced(self):
        for finite in [torch.tensor(True), torch.tensor([True, True])]:
            callback = self.callback(finite)
            callback.on_step_end(types.SimpleNamespace(device="cpu", world_size=1),
                                 types.SimpleNamespace(global_step=1, max_steps=3), TrainerControl())
            self.assertEqual(callback.token_count, 10)

    def test_any_nonfinite_replica_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            self.callback(torch.tensor([True, False])).on_step_end(
                types.SimpleNamespace(device="cpu", world_size=1),
                types.SimpleNamespace(global_step=1, max_steps=3), TrainerControl())

    def test_implicit_cuda_index_and_rng_restore(self):
        trainer = types.SimpleNamespace(accelerator=types.SimpleNamespace(device=torch.device("cuda")))
        from contextlib import nullcontext
        with patch("torch.cuda.current_device", return_value=1), patch("torch.random.fork_rng", return_value=nullcontext()) as fork:
            with selection_randomness(trainer, 42):
                pass
            fork.assert_called_once_with(devices=[1])

    def test_single_gpu_launcher_has_rank_and_loopback(self):
        command = argv(training_launcher("ddp", 1))
        self.assertIn("--nproc_per_node=1", command)
        self.assertIn("--master_addr=127.0.0.1", command)
        self.assertNotIn("--standalone", command)

    def test_qwen_grpo_uses_padding_safe_sdpa(self):
        model = self.config_dir({"model_type": "qwen2"})
        self.assertEqual(grpo_attention_implementation(model), "sdpa")
        self.assertEqual(grpo_attention_implementation(model, True), "eager")

    def test_other_architectures_keep_flash(self):
        model = self.config_dir({"model_type": "llama"})
        self.assertEqual(grpo_attention_implementation(model), "flash_attention_2")

    def test_mxfp4_is_decoded_not_relabelled(self):
        model = self.config_dir({"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}})
        options = validate_lora_base(model)
        self.assertTrue(options["quantization_config"].dequantize)
        self.assertEqual(options["quantization_config"].quant_method, "mxfp4")
        with self.assertRaisesRegex(ValueError, "NF4"):
            validate_lora_base(model, q_lora=True)

    def test_large_vocab_logits_are_budgeted(self):
        model = self.config_dir({"vocab_size": 200000})
        config = {"batch_size": 140, "use_liger": "False"}
        with patch("torch.cuda.is_available", return_value=False):
            cap_unfused_logits_batch(config, model, 2048)
        self.assertEqual(config["batch_size"], 1)

    def test_training_controls_are_validated_and_applied(self):
        controls = load_config('{"backend":"fsdp","micro_batch_size":2,"use_lora":true}')
        plan = {}
        apply_plan({"runtime_config": controls}, plan)
        self.assertEqual(plan["distributed"], "fsdp")
        self.assertEqual(plan["batch_size"], 2)
        self.assertTrue(plan["use_lora"])

    def test_controls_cannot_change_output_or_deadline(self):
        for text in ['{"output_dir":"/cache"}', '{"hours_to_complete":99}', '{"micro_batch_size":0}']:
            with self.assertRaises(ValueError):
                load_config(text)


if __name__ == "__main__":
    unittest.main()
