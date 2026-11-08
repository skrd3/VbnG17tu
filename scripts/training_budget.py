"""One deadline for preprocessing, attempts, training and optional final work."""
import datetime as dt
import math
import os
import time


def establish_deadline(hours):
    hours = float(hours)
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("hours-to-complete must be finite and positive")
    started = float(os.environ.get("TRAINING_STARTED_AT", time.time()))
    expires = started + hours * 3600
    if "TRAINING_DEADLINE" in os.environ:
        expires = min(expires, float(os.environ["TRAINING_DEADLINE"]))
    os.environ["TRAINING_DEADLINE"] = str(expires)
    return dt.datetime.fromtimestamp(expires, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class TrainingBudget:
    def __init__(self, end_time=None, *, clock=time.monotonic):
        candidates = []
        if os.environ.get("TRAINING_DEADLINE"):
            candidates.append(float(os.environ["TRAINING_DEADLINE"]))
        if end_time:
            candidates.append(dt.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp())
        self.clock = clock
        self.expires = clock() + (min(candidates) - time.time()) if candidates else float("inf")

    def remaining(self):
        return max(0.0, self.expires - self.clock())

    def can_start(self, seconds, reserve=60):
        return self.remaining() > max(0, seconds) + max(0, reserve)


def remaining_hours(end_time=None, reserve=180):
    return max(0.0, TrainingBudget(end_time).remaining() - reserve) / 3600
