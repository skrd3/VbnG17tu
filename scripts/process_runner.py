"""Argument-safe subprocess execution with a process-group deadline."""
import math
import os
import shlex
import signal
import subprocess
import sys
import threading

from training_budget import TrainingBudget


def argv(command):
    return shlex.split(command) if isinstance(command, str) else list(command)


def replace_argument(command, name, value):
    parts = argv(command)
    flag = "--" + name
    try:
        index = parts.index(flag)
    except ValueError:
        parts.extend([flag, str(value)])
    else:
        if index + 1 >= len(parts):
            raise ValueError(f"Missing value for {flag}")
        parts[index + 1] = str(value)
    return shlex.join(parts)


def extract_argument(command, name):
    parts = argv(command)
    try:
        return parts[parts.index("--" + name) + 1]
    except (ValueError, IndexError):
        return None


def run_command(command, log_path, env_vars=None):
    env = os.environ.copy()
    env.update(env_vars or {})
    process = subprocess.Popen(argv(command), shell=False, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               env=env, start_new_session=os.name != "nt")

    def kill_group():
        if process.poll() is None:
            try:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    seconds = TrainingBudget().remaining()
    timer = threading.Timer(max(0.01, seconds - 30), kill_group) if math.isfinite(seconds) else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            code = process.wait()
            log.write(f"\nProcess completed with return code: {code}\n")
            return code
    finally:
        if timer is not None:
            timer.cancel()
        process.stdout.close()
        if process.poll() is None:
            kill_group()
            process.wait()
