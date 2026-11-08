"""Deterministic paired diagnostics; training-dev gates are not Boss verdicts."""
import hashlib
import json
import math

import numpy as np


def dataset_fingerprint(dataset):
    if getattr(dataset, "_fingerprint", None):
        return dataset._fingerprint
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        row = dataset[index]
        row = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in row.items()}
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def paired_diagnostics(reference, candidate, *, confidence=.99, resamples=10000, seed=20260808):
    """Same dead zone and full-row bootstrap as the official paired gate."""
    ref, cand = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 1 or ref.shape != cand.shape or ref.size == 0:
        raise ValueError("Paired vectors must be nonempty and have identical shape")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("Invalid bootstrap configuration")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("Selection vectors must contain only finite values")
    gaps = ref - cand
    wins, decided = gaps > .01, np.abs(gaps) > .01
    rng = np.random.default_rng(seed)
    means, rates = np.empty(resamples), np.empty(resamples)
    chunk = max(1, min(resamples, 2_000_000 // len(gaps)))
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        indices = rng.integers(0, len(gaps), size=(size, len(gaps)))
        counts = decided[indices].sum(axis=1)
        rates[start:start + size] = np.divide(wins[indices].sum(axis=1), counts,
                                             out=np.zeros(size), where=counts > 0)
        means[start:start + size] = gaps[indices].mean(axis=1)
    percentile = 100 * (1 - confidence)
    gap_bound, win_bound = float(np.percentile(means, percentile)), float(np.percentile(rates, percentile))
    threshold = max(.01, .01 * abs(float(ref.mean())))
    return {"n_examples": len(gaps), "n_decided": int(decided.sum()),
            "win_rate": float(wins.sum() / decided.sum()) if decided.any() else 0.,
            "win_rate_lower_bound": win_bound, "mean_gap_nats": float(gaps.mean()),
            "mean_gap_lower_bound": gap_bound, "required_mean_gap_nats": threshold,
            "worst_5pct_gap": float(np.sort(gaps)[:max(1, math.ceil(.05 * len(gaps)))].mean()),
            "max_regression": float(max(0., -gaps.min())),
            "paired_gate_passed": bool(decided.any() and win_bound >= .55 and gap_bound >= threshold)}


def accept_dev_candidate(reference, candidate):
    """Reject apparent mean wins supported only by unstable large outliers.

    No 1% margin or 55% win-rate demand on every training checkpoint. Small
    dev sets keep mean selection, explicitly flagged as statistically limited.
    """
    stats = paired_diagnostics(reference, candidate, confidence=.90, resamples=512)
    sufficient = stats["n_examples"] >= 32
    accept = stats["mean_gap_nats"] > 1e-8 and (not sufficient or stats["mean_gap_lower_bound"] >= 0.)
    return bool(accept), {**stats, "dev_confidence": .90, "dev_resamples": 512,
                          "small_dev_mean_fallback": not sufficient}
