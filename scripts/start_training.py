"""Stdlib-only launcher: starts the budget before importing ML libraries."""
import os
from pathlib import Path
import sys
import time


def main():
    from process_runner import extract_argument
    from training_budget import establish_deadline
    os.environ.setdefault("TRAINING_STARTED_AT", str(time.time()))
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        os.execv(sys.executable, [sys.executable, "-m", "text_trainer", "--help"])
    hours = extract_argument(sys.argv[1:], "hours-to-complete")
    if hours is None:
        raise SystemExit("Missing --hours-to-complete")
    establish_deadline(hours)
    model = extract_argument(sys.argv[1:], "model") or ""
    interpreter = sys.executable
    if "gpt-oss" in model.lower():
        interpreter = "/workspace/oss_py/bin/python"
        if not Path(interpreter).is_file():
            raise SystemExit("gpt-oss build environment missing; rebuild the provided Dockerfile")
        os.environ["PATH"] = str(Path(interpreter).parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ["VIRTUAL_ENV"] = "/workspace/oss_py"
    os.execv(interpreter, [interpreter, "-m", "text_trainer", *sys.argv[1:]])


if __name__ == "__main__":
    main()
