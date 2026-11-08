"""Held-out acceptance gate; opt-in reclaim pass never trains gate examples."""
from contextlib import contextmanager
import json
import math
import os


def configure_gate(trainer):
    trainer.selection_gate = None
    if trainer.eval_dataset is None:
        return
    if os.environ.get("ENABLE_DEV_PASS") != "1" or len(trainer.eval_dataset) < 40:
        return
    from val_split import stratified_split
    dataset = trainer.eval_dataset
    # Group serialized tokenized records before splitting; a large inseparable
    # group leaves dev-pass disabled instead of contaminating its gate.
    records = [{"row": i, "sample": dataset[i]} for i in range(len(dataset))]
    def text(row):
        sample = row["sample"]
        return json.dumps({k: v.tolist() if hasattr(v, "tolist") else v for k, v in sample.items()}, sort_keys=True)
    gate, select = stratified_split(records, max(8, len(records) // 5), text_fn=text)
    if len(gate) < 8 or len(select) < 16:
        return
    def take(rows):
        indices = [row["row"] for row in rows]
        if hasattr(dataset, "select"):
            return dataset.select(indices)
        from torch.utils.data import Subset
        return Subset(dataset, indices)
    trainer.eval_dataset = take(select)
    trainer.selection_gate = take(gate)


@contextmanager
def no_selection_callbacks(trainer):
    callbacks = trainer.callback_handler.callbacks
    trainer.callback_handler.callbacks = []
    try:
        yield
    finally:
        trainer.callback_handler.callbacks = callbacks


@contextmanager
def isolated_candidate_evaluation(trainer):
    previous = getattr(trainer, "_selection_diagnostics_only", False)
    trainer._selection_diagnostics_only = True
    try:
        with no_selection_callbacks(trainer):
            yield
    finally:
        trainer._selection_diagnostics_only = previous


def evaluate_gate(trainer):
    with no_selection_callbacks(trainer):
        metrics = trainer.evaluate(eval_dataset=trainer.selection_gate, metric_key_prefix="gate")
    loss = float(metrics["gate_loss"])
    raw = float(metrics.get("gate_raw_ce", loss))
    if not math.isfinite(loss) or not math.isfinite(raw):
        raise ValueError("Nonfinite acceptance-gate loss")
    return loss, raw


def accept_gate(before, after):
    return after[0] < before[0] - 1e-6 and after[1] <= before[1] + 1e-6
