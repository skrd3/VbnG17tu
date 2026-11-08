"""Offline, read-only comparison of legal same-split experiment exports.

No model downloads or hidden evaluation access. This reports a matchup on the
provided exports, never a claim that the tournament has been won.
"""
import argparse
import json
import math
from pathlib import Path

from paired_selection import paired_diagnostics
from reward_scoring import normalize_scores


def compare(reference, candidate, kind="paired"):
    identity = reference.get("dataset_fingerprint")
    if not identity or identity != candidate.get("dataset_fingerprint"):
        raise ValueError("Exports must identify the same ordered dataset")
    if kind == "grpo":
        if not reference.get("generation_protocol") or reference["generation_protocol"] != candidate.get("generation_protocol"):
            raise ValueError("GRPO exports must use the same generation protocol")
        if reference["weights"] != candidate["weights"]:
            raise ValueError("GRPO reward weights differ")
        scores = normalize_scores({"boss": reference["raw_rewards"], "candidate": candidate["raw_rewards"]},
                                  reference["weights"], {"boss": reference["prompt_reverse_kl"],
                                                         "candidate": candidate["prompt_reverse_kl"]})
        threshold = scores["boss"] + .01 * abs(scores["boss"])
        return {"scores": scores, "required_candidate_score": threshold,
                "candidate_won_on_provided_exports": scores["candidate"] >= threshold}
    def vector(payload):
        for key in ("raw_per_pair_losses", "raw_per_sample_loss", "per_example_losses"):
            if key in payload:
                return payload[key]
        raise ValueError("No per-example loss vector")
    for key in ("beta", "max_length", "selection_loss_type"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Selection protocols differ: {key}")
    if reference.get("sample_ids") != candidate.get("sample_ids"):
        raise ValueError("Sample orders differ")
    ref, cand = vector(reference), vector(candidate)
    if "raw_per_pair_losses" in reference and reference.get("beta") != .1:
        raise ValueError("Official DPO comparison requires beta=0.1")
    stats = paired_diagnostics(ref, cand)
    def scalar(payload, losses):
        value = float(payload.get("ranking_scalar", payload.get("eval_loss", sum(losses) / len(losses))))
        if not math.isfinite(value):
            raise ValueError("Nonfinite ranking scalar")
        return value
    boss_scalar, candidate_scalar = scalar(reference, ref), scalar(candidate, cand)
    return {**stats, "boss_ranking_scalar": boss_scalar, "candidate_ranking_scalar": candidate_scalar,
            "candidate_won_on_provided_exports": stats["paired_gate_passed"] and candidate_scalar <= boss_scalar}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--kind", choices=("paired", "grpo"), default="paired")
    parser.add_argument("--output")
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    cand = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = {"scope": "provided legal experiment split; not an official hidden-test verdict",
              **compare(ref, cand, args.kind)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
