"""Self-contained host orchestration. Every training job uses the image ENTRYPOINT."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parent.parent
BASE = "unsloth/Qwen2.5-0.5B"
REVISION = "c96fd10340df5cf3780725aceb6f123e0987eea2"
NAMES = [
    "sft_liger_flash_packing_search_gate_soup", "lora_kl_naive_packing_plw_dedup_search",
    "qlora4bit_kl_adapter_reload", "fullft_kl_flash_position_ids", "dpo_modifiers_search_paired_gate",
    "dpo_lora_reference_adapter", "grpo_torch_multiseed_selection", "grpo_initial_audit_short_context",
    "zero3_fullft_gather_checkpoint", "zero3_dpo_reference_save", "fsdp_fullft_save", "ddp_deadline_early_save",
    "sft_longer_context_1024", "grpo_vllm_colocate_multiseed", "sft_gate_soup_diverse", "dpo_gate_soup_diverse",
    "sft_liger_label_smoothing", "gpt_oss_20b_overlay_lora", "gpt_oss_tiny_fullft_overlay",
    "zero3_dpo_explicit_reference", "grpo_vllm_server_multiseed", "measured_low_coverage_gaussian_repacking",
    "zero3_native_deepspeed_launcher", "fsdp_dpo_reference_selection", "zero3_grpo_generation_selection",
    "fsdp_grpo_generation_selection", "lr_cache_reuse", "container_instruct", "container_chat", "container_dpo",
    "container_grpo", "container_gptoss_fixture", "nccl", "reliability", "cuda_operators", "gpu_state",
    "control_compare", "guards", "dpo_soup_reproduction", "final_save_ddp", "final_save_zero3",
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def execute(command, log, timeout=None):
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only stop the container created by this invocation, preserving all logs.
            if "--name" in command:
                subprocess.run(["docker", "stop", "-t", "10", command[command.index("--name") + 1]], capture_output=True)
            process.terminate()
            process.wait()
            return 124


def fingerprint():
    files = [REPO / "dockerfiles/standalone-text-trainer.dockerfile", *sorted((REPO / "scripts").rglob("*.py")), REPO / "scripts/training_requirements.txt"]
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def build_image(args, digest):
    image = "test41-official:" + digest[:12]
    probe = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if probe.returncode == 0:
        return image
    # Keep build state in a private RAM-backed Docker volume. This avoids
    # simultaneously storing build snapshots and loaded image snapshots on a
    # small validator-development disk. No existing images/cache are removed.
    name = "test41-build-" + digest[:10]
    volume = name + "-state"
    sockdir = args.out / "build-socket"
    sockdir.mkdir(exist_ok=True)
    subprocess.run(["docker", "volume", "create", "--driver", "local", "--opt", "type=tmpfs", "--opt", "device=tmpfs", "--opt", "o=size=80g", volume], check=True, capture_output=True)
    try:
        subprocess.run(["docker", "run", "-d", "--name", name, "--privileged", "-v", f"{volume}:/var/lib/buildkit", "-v", f"{sockdir}:/run/buildkit", "moby/buildkit:v0.26.2", "--addr", "unix:///run/buildkit/buildkitd.sock"], check=True, capture_output=True)
        for _ in range(60):
            if (sockdir / "buildkitd.sock").exists():
                break
            time.sleep(1)
        subprocess.run(["docker", "buildx", "create", "--name", name, "--driver", "remote", "unix://" + str(sockdir / "buildkitd.sock")], check=True, capture_output=True)
        command = ["docker", "buildx", "build", "--builder", name, "--progress", "plain", "--load", "--label", f"test41.source={digest}", "-t", image, "-f", str(REPO / "dockerfiles/standalone-text-trainer.dockerfile"), str(REPO)]
        write(args.out / "build-command.json", command)
        if execute(command, args.out / "build.log"):
            raise RuntimeError("Official Dockerfile build failed; see build.log")
    finally:
        subprocess.run(["docker", "buildx", "rm", name], capture_output=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", volume], capture_output=True)
    return image


def download_model(cache):
    directory = cache / "models/unsloth--Qwen2.5-0.5B"
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"https://huggingface.co/api/models/{BASE}/revision/{REVISION}")
    with urllib.request.urlopen(request, timeout=60) as response:
        api = json.load(response)
    write(cache / "model-api-test41.json", api)
    files = [x["rfilename"] for x in api["siblings"] if x["rfilename"].endswith((".json", ".txt", ".safetensors"))]

    def fetch(name):
        path = directory / name
        if path.is_file() and path.stat().st_size > 0:
            return
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"https://huggingface.co/{BASE}/resolve/{REVISION}/{name}", timeout=120) as source, path.with_suffix(path.suffix + ".part").open("wb") as output:
                    shutil.copyfileobj(source, output)
                path.with_suffix(path.suffix + ".part").replace(path)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))


def case_spec(index, name):
    task, model, gpus, steps = "InstructTextTask", BASE, "all", 2
    config = dict(micro_batch_size=1, gradient_accumulation_steps=1, max_length=128,
                  min_steps=1, periodic_save_steps=2, lr_search_fraction=0, enable_dev_pass=False)
    env = {}
    if "dpo" in name:
        task = "DpoTask"
        config.update(max_prompt_length=64)
    if "grpo" in name:
        task = "GrpoTask"
        config.update(micro_batch_size=2, max_prompt_length=32, max_completion_length=16, num_generations=2,
                      use_vllm=False, grpo_selection_prompts=2, grpo_selection_seeds="42,314159")
    if "kl" in name:
        env.update(USE_KL="1", KL_COEF="0.1")
    if "lora" in name:
        config.update(use_lora=True, lora_r=8, lora_alpha=16)
        gpus = '"device=0"'
    if "qlora" in name:
        config["q_lora"] = True
    if "naive" in name:
        config.update(packing=True, packing_mode="naive", disable_fa=True)
    elif "packing" in name or "position_ids" in name or "repacking" in name:
        config.update(packing=True, packing_mode="fa")
    if "liger" in name:
        config["use_liger"] = True
    if "search" in name or name == "lr_cache_reuse":
        config["lr_search_fraction"] = .08
    if "gate" in name or "soup" in name:
        steps = 8
        config.update(enable_dev_pass="gate" in name)
    if name in {"sft_gate_soup_diverse", "dpo_gate_soup_diverse", "dpo_modifiers_search_paired_gate", "lr_cache_reuse"}:
        config.update(use_lora=True, lora_r=8, lora_alpha=16)
        gpus = '"device=0"'
    if name == "dpo_modifiers_search_paired_gate":
        config.update(label_smoothing=.1, rpo_alpha=.1)
        steps = 4
    if "zero3" in name:
        config.update(backend="ds", launcher="torchrun")
    if "fsdp" in name:
        config["backend"] = "fsdp"
    if name == "zero3_native_deepspeed_launcher":
        config["launcher"] = "deepspeed"
    small = {"fsdp_fullft_save", "zero3_native_deepspeed_launcher", "zero3_grpo_generation_selection",
             "fsdp_grpo_generation_selection", "final_save_ddp", "final_save_zero3", "grpo_initial_audit_short_context"}
    if name in small:
        model = "test41/qwen-context-tiny"
        config.update(disable_fa=True, use_liger=False)
    if name == "grpo_initial_audit_short_context":
        config["grpo_eval_initial"] = True
    if name == "sft_longer_context_1024":
        config["max_length"] = 1024
    if name == "sft_liger_label_smoothing":
        config["label_smoothing_factor"] = .1
    if name == "zero3_dpo_explicit_reference":
        config["explicit_reference"] = True
    if name == "grpo_vllm_colocate_multiseed":
        config.update(use_vllm=True, vllm_gpu_memory_utilization=.15)
        gpus = '"device=0"'
    if name == "grpo_vllm_server_multiseed":
        config.update(use_vllm=True, vllm_mode="server", vllm_server_host="generator", vllm_server_port=8000)
        gpus = '"device=0"'
    if name == "measured_low_coverage_gaussian_repacking":
        config.update(use_lora=True, lora_r=8, lora_alpha=16, lr_search_fraction=.08, max_length=1024)
        gpus = '"device=0"'
        steps = 4
    if name in {"nccl", "reliability", "cuda_operators", "gpu_state", "control_compare", "guards"}:
        config.update(use_lora=True, lora_r=8, lora_alpha=16)
    if name in {"ddp_deadline_early_save", "final_save_ddp", "final_save_zero3"}:
        config["save_before_remaining_minutes"] = 20
    if name == "gpt_oss_20b_overlay_lora":
        model = "test41/oss-mxfp4-tiny"
        config.update(use_lora=True, lora_r=8, lora_alpha=16, disable_fa=True, use_liger=False)
    if name == "gpt_oss_tiny_fullft_overlay":
        model = "test41/oss-bf16-tiny"
    if name in {"lr_cache_reuse", "measured_low_coverage_gaussian_repacking"}:
        env["SN56_VERBOSE"] = "1"
    if name.startswith("container_"):
        task = {"instruct": "InstructTextTask", "chat": "ChatTask", "dpo": "DpoTask", "grpo": "GrpoTask", "gptoss_fixture": "InstructTextTask"}[name[10:]]
        model = "test41/oss-bf16-tiny" if name.endswith("gptoss_fixture") else BASE
        config = {}  # Official production defaults: no test-side training overrides.
    return dict(index=index, name=name, task=task, model=model, gpus=gpus, steps=steps, config=config, env=env)


def data_for(spec):
    size = 16384 if spec["name"] == "measured_low_coverage_gaussian_repacking" else 64
    rows = []
    for i in range(size):
        key = hashlib.sha256(str(i).encode()).hexdigest()
        prompt = f"Context {key}. Exercise {i}: add {i % 17} and {i % 7}."
        if spec["name"] == "sft_longer_context_1024":
            prompt += " Check every elementary arithmetic operation." * 180
        if spec["name"] == "measured_low_coverage_gaussian_repacking":
            prompt += " Check every elementary arithmetic operation." * 140
        answer = f"The result is {i % 17 + i % 7}."
        rows.append(dict(instruction=prompt, output=answer, prompt=prompt, chosen=answer, rejected="An unrelated purple elephant.", messages=[dict(role="user", content=prompt), dict(role="assistant", content=answer)]))
    if "dedup" in spec["name"]:
        rows += rows[:8]
    return rows


def dataset_type(task):
    if task == "ChatTask":
        return dict(chat_template="chatml", chat_column="messages", chat_role_field="role", chat_content_field="content", chat_user_reference="user", chat_assistant_reference="assistant")
    if task == "DpoTask":
        return dict(field_prompt="prompt", field_chosen="chosen", field_rejected="rejected", prompt_format="{prompt}", chosen_format="{chosen}", rejected_format="{rejected}")
    if task == "GrpoTask":
        return dict(field_prompt="prompt", reward_functions=[dict(reward_func="def reward(completions, **kwargs):\n    return [float(len(c) % 13) / 13 for c in completions]\n", reward_weight=1.0)])
    return dict(field_instruction="instruction", field_output="output")


def run_case(args, image, spec):
    folder = args.out / spec["name"]
    folder.mkdir(parents=True, exist_ok=True)
    task_id = f"t41_{spec['index']:02}_{spec['name']}"
    data = args.cache / "datasets" / f"{task_id}_train_data.json"
    write(data, data_for(spec))
    write(folder / "training-config.json", spec["config"])
    write(folder / "spec.json", spec)
    output = folder / "checkpoints"
    output.mkdir(exist_ok=True)
    temporary = folder / "tmp"
    temporary.mkdir(exist_ok=True)
    name = "test41-" + str(spec["index"])
    network, server = None, None
    try:
        if spec["name"] == "grpo_vllm_server_multiseed":
            network, server = name + "-net", name + "-generator"
            subprocess.run(["docker", "network", "create", "--internal", network], check=True, capture_output=True)
            server_out = folder / "server-output"
            server_out.mkdir(exist_ok=True)
            command = ["docker", "run", "-d", "--name", server, "--network", network, "--network-alias", "generator", "--gpus", '"device=1"', "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{args.cache}:/cache:ro", "-v", f"{server_out}:/app/checkpoints", "--entrypoint", "trl", image, "vllm-serve", "--model", "/cache/models/unsloth--Qwen2.5-0.5B", "--gpu_memory_utilization", "0.15", "--max_model_len", "1024"]
            write(folder / "server-command.json", command)
            subprocess.run(command, check=True, capture_output=True)
            for _ in range(60):
                probe = subprocess.run(["docker", "exec", server, "python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health/',timeout=2)"], capture_output=True)
                if probe.returncode == 0:
                    break
                time.sleep(3)
            else:
                raise RuntimeError("vLLM sidecar health timeout")
        command = ["docker", "run", "--rm", "--name", name, "--gpus", spec["gpus"], "--network", network or "none", "--ipc", "host", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "220g", "--cpus", "48", "-v", f"{args.cache}:/cache:ro", "-v", f"{output}:/app/checkpoints", "-v", f"{folder}:/test41:ro", "-v", f"{temporary}:/tmp"]
        for key, value in spec["env"].items():
            command += ["-e", f"{key}={value}"]
        # No --entrypoint, no script mount, no direct train_*.py invocation.
        command += [image, "--task-id", task_id, "--model", spec["model"], "--dataset", f"https://example.invalid/test41/{task_id}.json", "--dataset-type", json.dumps(dataset_type(spec["task"])), "--task-type", spec["task"], "--file-format", "s3", "--hours-to-complete", "0.2", "--expected-repo-name", "model", "--max-steps", str(spec["steps"]), "--min-steps", "1"]
        if spec["config"]:
            command += ["--training-config", "/test41/training-config.json"]
        if "dedup" in spec["name"]:
            command += ["--baseline-stats", json.dumps({"dataset": {"near_duplicate_rate": .4, "prompt_tokens": 1000, "completion_tokens": 20}})]
        write(folder / "command.json", command)
        started = time.time()
        rc = execute(command, folder / "run.log", 900)
        if rc == 0 and spec["name"] == "lr_cache_reuse":
            (folder / "run.log").rename(folder / "first-run.log")
            rc = execute(command, folder / "run.log", 900)
            if rc == 0 and "Usando LR cached" not in (folder / "run.log").read_text(errors="replace"):
                raise RuntimeError("Second official entrypoint did not reuse LR cache")
        result = dict(scenario=spec["name"], task_id=task_id, model=spec["model"], returncode=rc, seconds=time.time() - started, status="失败")
        manifest = output / task_id / "model/submission_manifest.json"
        if rc == 0 and manifest.is_file():
            metadata = json.loads(manifest.read_text())
            result.update(global_step=int(metadata["selection"]["step"]), manifest=metadata)
            interpreter = "/workspace/oss_py/bin/python" if "test41/oss-" in spec["model"] else "python"
            verify = ["docker", "run", "--rm", "--gpus", '"device=0"', "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{args.cache}:/cache:ro", "-v", f"{output}:/app/checkpoints", "--entrypoint", interpreter, image, "-m", "check_submission", "--output", str(Path('/app/checkpoints') / task_id / 'model'), "--base", "/cache/models/" + spec["model"].replace("/", "--"), "--reload"]
            write(folder / "reload-command.json", verify)
            result["reload_returncode"] = execute(verify, folder / "reload.log", 180)
            if result["global_step"] > 0 and result["reload_returncode"] == 0:
                result["status"] = "通过"
        if spec["name"] == "guards" and result["status"] == "通过":
            bad = list(command)
            bad[bad.index("--task-id") + 1] = "../unsafe"
            code = execute(bad, folder / "guard-rejection.log", 60)
            if code == 0 or "Unsafe task id" not in (folder / "guard-rejection.log").read_text(errors="replace"):
                result.update(status="失败", error="Unsafe task identifier was not rejected")
        if spec["name"] == "gpt_oss_20b_overlay_lora" and result["status"] == "通过":
            result["status"] = "微型 MXFP4 通过；真实 20B 未验证"
        return result
    finally:
        if server:
            with (folder / "server.log").open("w") as log:
                subprocess.run(["docker", "logs", server], stdout=log, stderr=subprocess.STDOUT)
            subprocess.run(["docker", "rm", "-f", server], capture_output=True)
        if network:
            subprocess.run(["docker", "network", "rm", network], capture_output=True)


def report(args, results, image, digest):
    from collections import Counter
    counts = dict(Counter(r["status"] for r in results))
    lines = ["# test41_rou1：正式容器入口全量短程复测", "", f"更新：{datetime.datetime.now(datetime.timezone.utc).isoformat()}", "", f"镜像 `{image}`；源码 SHA-256 `{digest}`。", "", f"结果：{counts}；已执行 {len(results)}/41。", "", "每项训练均调用 Dockerfile 默认 ENTRYPOINT，未挂载/替换生产脚本；禁公网、只读 /cache、cap-drop ALL、no-new-privileges。vLLM server 项的辅助服务由脚本自动启动在 internal 网络。准备模型和只读重载校验使用独立工具容器。", "", "41 项由原报告场景名称重建，不能视为原测试脚本和参数的逐字复现。训练目标为正更新步数、有效权重与有限 loss 重载；不验证质量、收敛、官方 uploader、真实 GPT-OSS 20B 或 4/8 GPU。基础项使用真实入口训练验证相应基础能力，未实现中途 kill 等故障注入。", "", "| 场景 | 模型 | 结果 | step | 证据 |", "|---|---|---|---:|---|"]
    indexed = {r["scenario"]: r for r in results}
    for name in NAMES:
        r = indexed.get(name, {})
        link = f"{args.out.name}/{name}"
        lines.append(f"| {name} | {r.get('model','—')} | {r.get('status','待执行')} | {r.get('global_step','—')} | [结果]({link}/result.json) · [日志]({link}/run.log) |")
    lines += ["", "说明：gpt_oss_20b_overlay_lora 使用真实 blocks/scales 格式的微型随机初始化 MXFP4 夹具，验证 BF16 反量化 → LoRA → 保存与重载；没有训练真实 20B 权重。其余微型夹具同样只提供架构/后端功能证据。", "", "所有配置、标准入口命令、日志与 checkpoint 位于本报告同名目录。历史权重可能按 save_total_limit 清理；字节相同的已完成产物可用硬链接共享存储。"]
    args.report.write_text("\n".join(lines) + "\n")
    write(args.out / "report.json", dict(counts=counts, source_sha256=digest, image=image, results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/root/out/test41_rou1"))
    parser.add_argument("--cache", type=Path, default=Path("/root/out/cache"))
    parser.add_argument("--report", type=Path, default=Path("/root/out/test41_rou1.MD"))
    parser.add_argument("--only", help="Optional comma-separated case names; default all 41")
    parser.add_argument("--resume", action="store_true", help="Keep completed case results")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.cache = args.cache.resolve()
    args.report = args.report.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for executable in ["docker", "nvidia-smi"]:
        if not shutil.which(executable):
            raise SystemExit(f"Required tool missing: {executable}")
    digest, hashes = fingerprint()
    write(args.out / "source-sha256.json", hashes)
    results = []
    image = "not-built"
    try:
        image = build_image(args, digest)
        download_model(args.cache)
        prepare = ["docker", "run", "--rm", "--gpus", '"device=0"', "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "-v", f"{args.cache}:/cache", "--entrypoint", "/workspace/oss_py/bin/python", image, "/workspace/scripts/test41_fixtures.py", "/cache"]
        write(args.out / "prepare-command.json", prepare)
        if execute(prepare, args.out / "prepare.log", 300):
            raise RuntimeError("Fixture preparation failed")
        selected = set(args.only.split(",")) if args.only else set(NAMES)
        if selected - set(NAMES):
            raise ValueError("Unknown scenario in --only")
        for index, name in enumerate(NAMES, 1):
            if name not in selected:
                continue
            folder = args.out / name
            if args.resume and (folder / "result.json").exists():
                result = json.loads((folder / "result.json").read_text())
            else:
                if folder.exists():
                    archive = args.out / "attempts" / (name + "_" + str(time.time_ns()))
                    archive.parent.mkdir(exist_ok=True)
                    folder.rename(archive)
                spec = case_spec(index, name)
                print(f"[{index}/41] {name}", flush=True)
                try:
                    result = run_case(args, image, spec)
                except Exception as error:
                    folder.mkdir(exist_ok=True)
                    result = dict(scenario=name, model=spec["model"], status="失败", error=str(error))
                write(folder / "result.json", result)
            results.append(result)
            report(args, results, image, digest)
            print(result["status"], flush=True)
    except Exception as error:
        write(args.out / "fatal.json", dict(error=str(error)))
        report(args, results, image, digest)
        raise
    report(args, results, image, digest)
    return int(any(r["status"] == "失败" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
