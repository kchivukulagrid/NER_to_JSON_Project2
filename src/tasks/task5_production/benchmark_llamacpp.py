"""Benchmark llama.cpp quantized models and compute metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from typing import Any
from urllib.request import Request, urlopen

from datasets import load_dataset

from src.core.metrics import compute_metrics
from src.core.parsing import extract_json
from src.core.schema import empty_output


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
LLAMA_BIN = "./llama.cpp/build/bin/llama-completion"
SERVER_BIN = "./llama.cpp/build/bin/llama-server"
MODEL_PATH = "models/gguf/model.Q4_K_M.gguf"
INPUT_FILE = "data/processed/task1_test.jsonl"
OUTPUT_DIR = "experiments/task5_production"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp quantized model.")
    parser.add_argument("--llama_bin", default=LLAMA_BIN)
    parser.add_argument("--server_bin", default=SERVER_BIN)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--sample_count", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="none", choices=["none", "metal"])
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--mode", default="spawn", choices=["spawn", "server"])
    parser.add_argument("--server_url", default="http://127.0.0.1:8080")
    parser.add_argument("--start_server", action="store_true")
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8080)
    parser.add_argument("--server_parallel", type=int, default=16)
    parser.add_argument("--server_http_threads", type=int, default=8)
    parser.add_argument("--server_start_timeout", type=int, default=120)
    parser.add_argument("--request_timeout", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--label", default="Q4_K_M")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _run_llama(
    llama_bin: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    threads: int,
    device: str,
    n_gpu_layers: int,
) -> str:
    cmd = [
        llama_bin,
        "-m",
        model_path,
        "-n",
        str(max_tokens),
        "-t",
        str(threads),
        "-ngl",
        str(n_gpu_layers),
        "--device",
        device,
        "--no-conversation",
        "--single-turn",
        "--no-display-prompt",
        "--simple-io",
        "--temp",
        "0",
        "--top-p",
        "1.0",
        "--seed",
        "42",
        "--prompt",
        prompt,
    ]
    env = os.environ.copy()
    if device == "none":
        env["GGML_METAL"] = "0"
        env["LLAMA_METAL"] = "0"
        env["LLAMA_NO_METAL"] = "1"
    else:
        env.pop("GGML_METAL", None)
        env.pop("LLAMA_METAL", None)
        env.pop("LLAMA_NO_METAL", None)
    proc = subprocess.run(cmd, capture_output=True, check=False, env=env)
    stdout = proc.stdout.decode("utf-8", errors="ignore")
    return stdout


def _http_post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def _server_healthcheck(server_url: str, timeout: int) -> bool:
    try:
        with urlopen(f"{server_url.rstrip('/')}/health", timeout=timeout):
            return True
    except Exception:
        return False


def _spawn_llama_server(args: argparse.Namespace) -> tuple[subprocess.Popen, str]:
    log_path = os.path.join(
        args.output_dir,
        f"llama_server_bench_{int(time.time())}.log",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        args.server_bin,
        "-m",
        args.model_path,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--parallel",
        str(args.server_parallel),
        "--threads-http",
        str(args.server_http_threads),
        "-t",
        str(args.threads),
    ]

    if args.device == "none":
        cmd.extend(["--device", "none", "-ngl", "0"])
    elif args.n_gpu_layers > 0:
        cmd.extend(["-ngl", str(args.n_gpu_layers)])

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

    server_url = f"http://{args.server_host}:{args.server_port}"
    deadline = time.time() + args.server_start_timeout
    while time.time() < deadline:
        if _server_healthcheck(server_url, timeout=2):
            return proc, server_url
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    raise RuntimeError(
        "llama-server failed to start for benchmark. "
        f"Check logs at: {log_path}"
    )


def _run_llama_server(
    server_url: str,
    prompt: str,
    max_tokens: int,
    request_timeout: int,
) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    resp = _http_post_json(
        f"{server_url.rstrip('/')}/completion",
        payload,
        timeout=request_timeout,
    )
    return str(resp.get("content", ""))


def _extract_completion(stdout: str) -> str:
    # llama.cpp prints the prompt + completion; take the tail after prompt marker if present.
    return stdout.strip()


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    server_proc: subprocess.Popen | None = None

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"pred_{args.label}.jsonl")

    if args.mode == "server" and args.start_server:
        server_proc, server_url = _spawn_llama_server(args)
        args.server_url = server_url
    if args.mode == "server" and not _server_healthcheck(args.server_url, timeout=5):
        raise RuntimeError(
            f"Server mode selected but health check failed at {args.server_url}. "
            "Start llama-server manually or use --start_server."
        )

    # Warmup to avoid cold-start bias.
    warmup_count = min(args.warmup, len(dataset))
    for i in range(warmup_count):
        prompt = dataset[i]["prompt"]
        try:
            if args.mode == "server":
                _run_llama_server(args.server_url, prompt, args.max_tokens, args.request_timeout)
            else:
                _run_llama(
                    args.llama_bin,
                    args.model_path,
                    prompt,
                    args.max_tokens,
                    args.threads,
                    args.device,
                    args.n_gpu_layers,
                )
        except Exception:
            pass

    start = time.time()
    try:
        with open(pred_path, "w") as f:
            for row in dataset:
                prompt = row["prompt"]
                if args.mode == "server":
                    completion = _run_llama_server(
                        args.server_url,
                        prompt,
                        args.max_tokens,
                        args.request_timeout,
                    )
                else:
                    stdout = _run_llama(
                        args.llama_bin,
                        args.model_path,
                        prompt,
                        args.max_tokens,
                        args.threads,
                        args.device,
                        args.n_gpu_layers,
                    )
                    completion = _extract_completion(stdout)
                parsed = extract_json(completion)
                if parsed is None:
                    parsed = empty_output()
                prediction = json.dumps(parsed, ensure_ascii=False)
                f.write(
                    json.dumps(
                        {"ground_truth": row["output"], "prediction": prediction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        elapsed = time.time() - start

        metrics = compute_metrics(pred_path)

        bench_path = os.path.join(args.output_dir, "quant_benchmark.csv")
        write_header = not os.path.exists(bench_path)
        with open(bench_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "label",
                        "model_path",
                        "sample_count",
                        "elapsed_sec",
                        "per_sample_ms",
                        "precision",
                        "recall",
                        "f1",
                        "validity",
                    ]
                )
            writer.writerow(
                [
                    args.label,
                    args.model_path,
                    args.sample_count,
                    elapsed,
                    (elapsed / max(args.sample_count, 1)) * 1000,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["validity"],
                ]
            )

        print(f"Wrote predictions to {pred_path}")
        print(f"Appended benchmark to {bench_path}")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
