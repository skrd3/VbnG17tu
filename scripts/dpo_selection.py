"""Export real TRL per-pair losses at eval batch size one."""
import json
import math
from pathlib import Path

import torch
from trl import DPOTrainer
from trl.trainer.dpo_config import FDivergenceType
from paired_selection import accept_dev_candidate, dataset_fingerprint


class OfficialDPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        if train_eval != "eval":
            return super().get_batch_loss_metrics(model, batch, train_eval)
        # Official selection always uses beta=.1 sigmoid DPO, independent of
        # experimental training smoothing, weighting, RPO and MoE aux losses.
        names = ("beta", "loss_type", "label_smoothing", "reference_free", "use_weighting", "aux_loss_enabled",
                 "f_divergence_type")
        previous = {name: getattr(self, name) for name in names}
        rpo, ld = self.args.rpo_alpha, self.args.ld_alpha
        try:
            self.beta, self.loss_type, self.label_smoothing = .1, "sigmoid", 0.
            self.reference_free, self.use_weighting, self.aux_loss_enabled = False, False, False
            self.f_divergence_type = FDivergenceType.REVERSE_KL.value
            self.args.rpo_alpha, self.args.ld_alpha = None, None
            return super().get_batch_loss_metrics(model, batch, train_eval)
        finally:
            for name, value in previous.items():
                setattr(self, name, value)
            self.args.rpo_alpha, self.args.ld_alpha = rpo, ld

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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if self.args.per_device_eval_batch_size != 1:
            raise ValueError("Per-pair DPO comparison requires eval batch size 1")
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        loss = result[0]
        if loss is not None:
            values = self.accelerator.gather_for_metrics(loss.detach().reshape(1))
            self._pair_losses.extend(values.cpu().tolist())
        return result

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        self._pair_losses = []
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        output.metrics[f"{metric_key_prefix}_ranking_loss"] = output.metrics[f"{metric_key_prefix}_loss"]
        dataset = args[0].dataset if args else kwargs.get("dataloader").dataset
        identity = dataset_fingerprint(dataset)
        losses = self._pair_losses
        if len(losses) != len(dataset) or not all(math.isfinite(v) for v in losses):
            raise ValueError("DPO selection vector is incomplete or nonfinite")
        incumbent = getattr(self, "_selection_incumbent", None)
        diagnostics = {}
        accepted = True
        if metric_key_prefix == "eval" and not getattr(self, "_selection_diagnostics_only", False):
            if incumbent is not None and incumbent["fingerprint"] == identity:
                accepted, diagnostics = accept_dev_candidate(incumbent["losses"], losses)
            if accepted and not getattr(self, "_selection_diagnostics_only", False):
                self._selection_incumbent = {"fingerprint": identity, "losses": list(losses),
                                              "step": self.state.global_step}
            output.metrics[f"{metric_key_prefix}_selection_accepted"] = int(accepted)
        if diagnostics:
            output.metrics.update({f"{metric_key_prefix}_paired_gap_lb90": diagnostics["mean_gap_lower_bound"],
                                   f"{metric_key_prefix}_worst_5pct_gap": diagnostics["worst_5pct_gap"]})
        if self.is_world_process_zero():
            path = Path(self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / f"{metric_key_prefix}_pairs-{self.state.global_step}.json").write_text(
                json.dumps({"raw_per_pair_losses": self._pair_losses,
                            "sample_ids": list(range(len(self._pair_losses))),
                            "dataset_fingerprint": identity, "selection_accepted": accepted,
                            "paired_diagnostics_vs_incumbent": diagnostics,
                            "beta": .1, "training_beta": self.beta,
                            "selection_loss_type": "sigmoid", "max_length": self.args.max_length,
                            "scope": "local dev at training truncation; not hidden-test Boss verdict"}), encoding="utf-8")
        return output
