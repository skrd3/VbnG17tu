"""Official cross-model reward normalization (raw components, not sums)."""
import math


def reward_utility(raw, weights, scales, prompt_kl):
    """Strictly monotone local proxy; deliberately not an official Boss score.

    Fixed scales prevent zero-baseline normalization plateaus. asinh handles
    signed/unbounded rewards without a changing per-checkpoint denominator.
    """
    if set(raw) != set(weights) or set(raw) != set(scales):
        raise ValueError("Reward utility components differ")
    if not math.isfinite(prompt_kl):
        raise ValueError("Invalid prompt KL")
    score = -.5 * prompt_kl
    for name, weight in weights.items():
        value, scale = float(raw[name]), float(scales[name])
        if not all(math.isfinite(v) for v in (value, scale, float(weight))) or scale <= 0:
            raise ValueError("Nonfinite rewards or invalid fixed scale")
        score += weight * math.asinh(value / scale)
    return score


def normalize_scores(rewards, weights, prompt_kl):
    names = list(rewards)
    if len(names) < 2:
        raise ValueError("Cross-model normalization needs at least two models")
    keys = set(weights)
    for name in names:
        if set(rewards[name]) != keys or not math.isfinite(prompt_kl[name]):
            raise ValueError("Missing reward components or invalid prompt KL")
        if not all(math.isfinite(float(v)) for v in rewards[name].values()):
            raise ValueError("Nonfinite reward")
    scores = {name: -0.5 * prompt_kl[name] for name in names}
    for key, weight in weights.items():
        if not math.isfinite(float(weight)):
            raise ValueError("Nonfinite reward weight")
        values = [(name, float(rewards[name][key])) for name in names]
        minimum = min(v for _, v in values)
        shifted = [(n, v - minimum) for n, v in values] if minimum < 0 else values
        maximum = max(v for _, v in shifted)
        if len(names) == 2 and minimum < 0:
            ordered = sorted(shifted, key=lambda item: item[1])
            normalized = {ordered[0][0]: .25, ordered[1][0]: .75}
        elif maximum > 0:
            normalized = {n: v / maximum for n, v in shifted}
        else:
            normalized = {n: 1.0 for n in names}
        for name in names:
            scores[name] += float(weight) * normalized[name]
    return scores
