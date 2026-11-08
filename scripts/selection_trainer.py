"""Unpacked per-example CE plus globally token-weighted instruct KL."""
import json
import hashlib
import math
from pathlib import Path

import torch

from kl_trainer import KLRegularizedTrainer


class OfficialInstructTrainer(KLRegularizedTrainer):
    def _selection_fingerprint(self, dataset):
        cache = getattr(self, "_selection_fingerprints", {})
        key = (id(dataset), len(dataset))
        if key not in cache:
            digest = hashlib.sha256()
            for index in range(len(dataset)):
                row = dataset[index]
                normalized = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in row.items()}
                digest.update(json.dumps(normalized, sort_keys=True).encode())
                digest.update(b"\n")
            cache[key] = digest.hexdigest()
            self._selection_fingerprints = cache
        return cache[key]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        stats = getattr(self, "_last_eval_statistics", None)
        if stats is not None:
            # gather_for_metrics drops duplicated distributed sampler tail rows.
            stats = self.accelerator.gather_for_metrics(stats)
            self._official_rows.extend(stats.detach().cpu().tolist())
        return result

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        self._official_rows = []
        dataloader = args[0] if args else kwargs["dataloader"]
        fingerprint = self._selection_fingerprint(dataloader.dataset)
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        rows = self._official_rows
        valid = [r for r in rows if math.isfinite(r[0]) and r[2] > 0]
        if not valid:
            raise ValueError("No valid supervised examples in selection set")
        raw = sum(r[0] for r in valid) / len(valid)
        tokens = sum(r[2] for r in valid)
        kl = sum(r[1] for r in valid) / tokens
        ranking = raw + self.kl_coef * kl
        output.metrics.update({f"{metric_key_prefix}_loss": ranking,
                               f"{metric_key_prefix}_ranking_loss": ranking,
                               f"{metric_key_prefix}_raw_ce": raw,
                               f"{metric_key_prefix}_completion_forward_kl": kl,
                               f"{metric_key_prefix}_invalid_samples": len(rows) - len(valid)})
        if self.is_world_process_zero():
            path = Path(self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{metric_key_prefix}_samples-{self.state.global_step}.json").write_text(
                json.dumps({"raw_per_sample_loss": [r[0] for r in rows],
                            "sample_ids": list(range(len(rows))), "dataset_fingerprint": fingerprint,
                            "kl_sums": [r[1] for r in rows], "completion_tokens": [r[2] for r in rows],
                            "ranking_scalar": ranking}), encoding="utf-8")
        return output
