"""Validate artifacts and atomically exchange real directories on Linux.

No symlinks and no delete-before-copy. If atomic exchange is unsupported, keep
the previous validated submission; a caller must never fall back to two renames.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time


class InvalidSubmission(ValueError):
    pass


def _tensor_header(path):
    if path.is_symlink():
        raise InvalidSubmission("Weight files must not be symlinks")
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise InvalidSubmission(f"Truncated safetensors: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if length > min(size - 8, 100_000_000):
            raise InvalidSubmission(f"Invalid tensor header: {path.name}")
        header = json.loads(stream.read(length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise InvalidSubmission("Empty tensor file")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
              "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
              "F8_E4M3": 1, "F8_E5M2": 1}
    for value in tensors.values():
        start, end = value["data_offsets"]
        if not 0 <= start <= end <= size - 8 - length:
            raise InvalidSubmission(f"Truncated tensor data: {path.name}")
        elements = 1
        for dim in value["shape"]:
            if not isinstance(dim, int) or dim < 0:
                raise InvalidSubmission("Invalid tensor shape")
            elements *= dim
        if value["dtype"] not in widths:
            raise InvalidSubmission(f"Unsupported tensor dtype: {value['dtype']}")
        if end - start != elements * widths[value["dtype"]]:
            raise InvalidSubmission("Tensor shape/size mismatch")
    offsets = sorted(value["data_offsets"] for value in tensors.values())
    cursor = 0
    for start, end in offsets:
        if start != cursor:
            raise InvalidSubmission("Overlapping or incomplete tensor offsets")
        cursor = end
    if cursor != size - 8 - length:
        raise InvalidSubmission("Unindexed tensor bytes")
    return tensors


def _finite_weights(path, header):
    """Bounded-memory, read-only scan; BF16 is checked without converting it."""
    import numpy as np
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        for tensor in header.values():
            dtype = tensor["dtype"]
            if dtype not in {"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"}:
                continue
            start, end = tensor["data_offsets"]
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise InvalidSubmission("Weight file changed during validation")
                if dtype == "BF16":
                    values = np.frombuffer(block, dtype="<u2")
                    valid = not np.any((values & 0x7f80) == 0x7f80)
                elif dtype == "F8_E4M3":
                    values = np.frombuffer(block, dtype="u1")
                    valid = not np.any((values & 0x7f) == 0x7f)
                elif dtype == "F8_E5M2":
                    values = np.frombuffer(block, dtype="u1")
                    valid = not np.any((values & 0x7c) == 0x7c)
                else:
                    values = np.frombuffer(block, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
                    valid = bool(np.isfinite(values).all())
                if not valid:
                    raise InvalidSubmission(f"Nonfinite model weights in {path.name}")
                remaining -= len(block)


def validate_submission(folder, base_model=None, *, check_finite=False):
    root = Path(folder)
    if root.is_symlink() or not root.is_dir():
        raise InvalidSubmission("Submission must be a real directory")
    adapter = (root / "adapter_config.json").is_file()
    config_path = root / ("adapter_config.json" if adapter else "config.json")
    if not config_path.is_file():
        raise InvalidSubmission("Missing model configuration")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stem = "adapter_model" if adapter else "model"
    index = root / f"{stem}.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping:
            raise InvalidSubmission("Empty weight index")
        headers = {}
        for filename in set(mapping.values()):
            if Path(filename).name != filename:
                raise InvalidSubmission("Unsafe shard path")
            headers[filename] = _tensor_header(root / filename)
        if any(key not in headers[filename] for key, filename in mapping.items()):
            raise InvalidSubmission("Index/tensor key mismatch")
    else:
        filename = f"{stem}.safetensors"
        headers = {filename: _tensor_header(root / filename)}
    if check_finite:
        for filename, header in headers.items():
            _finite_weights(root / filename, header)
    if base_model and not adapter:
        base_path = Path(base_model) / "config.json"
        if base_path.is_file():
            base = json.loads(base_path.read_text(encoding="utf-8"))
            for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "model_type"):
                if name in base and base[name] != config.get(name):
                    raise InvalidSubmission(f"Base architecture mismatch: {name}")
            if isinstance(base.get("max_position_embeddings"), int):
                if config.get("max_position_embeddings", 0) < base["max_position_embeddings"]:
                    raise InvalidSubmission("Submitted context window shrank")
    if adapter and not any("lora_" in key or "adapter" in key or "prompt_embeddings" in key
                           for header in headers.values() for key in header):
        # Other PEFT methods can have different parameter names; their config
        # is authoritative, so enforce LoRA names only for LoRA adapters.
        if str(config.get("peft_type", "")).upper() == "LORA":
            raise InvalidSubmission("LoRA adapter contains no LoRA parameters")
    return {"format": "adapter" if adapter else "full", "validated_at": time.time(),
            "finite_weights_checked": check_finite}


def preflight_publication(destination):
    """Check the actual target filesystem without touching any live output."""
    target = Path(destination).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publication-probe-", dir=target.parent) as probe:
        left, right = Path(probe) / "left", Path(probe) / "right"
        left.mkdir()
        right.mkdir()
        (left / "identity").write_text("left", encoding="utf-8")
        (right / "identity").write_text("right", encoding="utf-8")
        _exchange(left, right)
        if (left / "identity").read_text() != "right" or (right / "identity").read_text() != "left":
            raise InvalidSubmission("Atomic publication probe did not exchange directories")
    return {"atomic_exchange": True, "filesystem_path": str(target.parent)}


def _exchange(left, right):
    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "Atomic directory exchange requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "renameat2 unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(left), -100, os.fsencode(right), 2):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def new_staging(destination):
    target = Path(destination).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.candidate-", dir=target.parent))


def commit_staging(staging, destination, base_model=None, *, metadata=None):
    stage, target = Path(staging).absolute(), Path(destination).absolute()
    if stage.parent != target.parent or stage == target or stage.is_symlink():
        raise InvalidSubmission("Staging must be a distinct sibling directory")
    if target.is_symlink():
        raise InvalidSubmission("Submission target cannot be a symlink")
    manifest = validate_submission(stage, base_model, check_finite=True)
    hashes = {}
    for path in sorted(stage.glob("*.safetensors")):
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    manifest.update(task_id=target.parent.name, output_name=target.name, weight_sha256=hashes,
                    selection=metadata or {})
    (stage / "submission_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for path in stage.iterdir():
        if path.is_file():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
    if sys.platform == "linux":
        directory_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if target.exists():
        # Empty destinations are created by older entrypoints. No valid result
        # can be lost by removing an empty directory before the first commit.
        if not any(target.iterdir()):
            target.rmdir()
            os.replace(stage, target)
        else:
            _exchange(stage, target)
    else:
        os.replace(stage, target)
    if sys.platform == "linux":
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if stage.exists():
        shutil.rmtree(stage)
    return manifest


def publish_checkpoint(source, destination, base_model=None, step=None, metric=None):
    stage = new_staging(destination)
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*", "global_step*"))
        if base_model and (Path(base_model) / "config.json").is_file() and (stage / "config.json").is_file():
            base = json.loads((Path(base_model) / "config.json").read_text(encoding="utf-8"))
            config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
            if base.get("architectures"):
                config["architectures"] = base["architectures"]
            (stage / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (stage / "loss.txt").write_text(f"{step},{metric}", encoding="utf-8")
        return commit_staging(stage, destination, base_model,
                              metadata={"step": step, "metric": metric, "source": Path(source).name})
    finally:
        if stage.exists():
            shutil.rmtree(stage)
