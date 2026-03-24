"""Benchmark concurrent extraction latency with llama.cpp."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.request import Request, urlopen

from datasets import load_dataset


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
LLAMA_BIN = "./llama.cpp/build/bin/llama-completion"
SERVER_BIN = "./llama.cpp/build/bin/llama-server"
MODEL_PATH = "models/gguf/model.Q4_K_M.gguf"
INPUT_FILE = "data/processed/task1_test.jsonl"
OUTPUT_FILE = "experiments/task5_production/concurrency.csv"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrency benchmark with llama.cpp.")
    parser.add_argument("--llama_bin", default=LLAMA_BIN)
    parser.add_argument("--server_bin", default=SERVER_BIN)
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--input_file", default=INPUT_FILE)
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--sample_count", type=int, default=64)
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
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--label", default="Q4_K_M")
    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
        os.path.dirname(args.output_file),
        f"llama_server_{int(time.time())}.log",
    )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

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
        "llama-server failed to start. "
        f"Check logs at: {log_path}"
    )


def _run_one_spawn(args: argparse.Namespace, prompt: str) -> float:
    cmd = [
        args.llama_bin,
        "-m",
        args.model_path,
        "-n",
        str(args.max_tokens),
        "-t",
        str(args.threads),
        "-ngl",
        str(args.n_gpu_layers),
        "--device",
        args.device,
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
    if args.device == "none":
        env["GGML_METAL"] = "0"
        env["LLAMA_METAL"] = "0"
        env["LLAMA_NO_METAL"] = "1"
    else:
        env.pop("GGML_METAL", None)
        env.pop("LLAMA_METAL", None)
        env.pop("LLAMA_NO_METAL", None)
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, env=env)
    return time.time() - start


def _run_one_server(args: argparse.Namespace, prompt: str) -> float:
    payload = {
        "prompt": prompt,
        "n_predict": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    start = time.time()
    _http_post_json(
        f"{args.server_url.rstrip('/')}/completion",
        payload,
        timeout=args.request_timeout,
    )
    return time.time() - start


def _run_one(args: argparse.Namespace, prompt: str) -> float:
    if args.mode == "server":
        return _run_one_server(args, prompt)
    return _run_one_spawn(args, prompt)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return sorted(values)[k]


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    conc_levels = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    server_proc: subprocess.Popen | None = None

    dataset = load_dataset("json", data_files={"data": args.input_file})["data"]
    dataset = dataset.select(range(min(args.sample_count, len(dataset))))
    prompts = [row["prompt"] for row in dataset]

    if args.mode == "server" and args.start_server:
        server_proc, server_url = _spawn_llama_server(args)
        args.server_url = server_url

    if args.mode == "server":
        if not _server_healthcheck(args.server_url, timeout=5):
            raise RuntimeError(
                f"Server mode selected but health check failed at {args.server_url}. "
                "Start llama-server manually or use --start_server."
            )

    # Warmup to avoid cold-start bias in latency percentiles.
    for i in range(min(args.warmup, len(prompts))):
        try:
            _run_one(args, prompts[i])
        except Exception:
            # Warmup failures should fail fast only when all runs are broken.
            pass

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    write_header = not os.path.exists(args.output_file)

    try:
        with open(args.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "label",
                        "concurrency",
                        "sample_count",
                        "p50_ms",
                        "p95_ms",
                        "p99_ms",
                        "throughput_rps",
                    ]
                )

            for conc in conc_levels:
                start = time.time()
                latencies = []
                with ThreadPoolExecutor(max_workers=conc) as ex:
                    futures = [ex.submit(_run_one, args, p) for p in prompts]
                    for fut in as_completed(futures):
                        latencies.append(fut.result())
                elapsed = time.time() - start
                p50 = _percentile(latencies, 50) * 1000
                p95 = _percentile(latencies, 95) * 1000
                p99 = _percentile(latencies, 99) * 1000
                throughput = len(latencies) / elapsed if elapsed > 0 else 0.0
                writer.writerow([args.label, conc, len(latencies), p50, p95, p99, throughput])

                print(
                    json.dumps(
                        {
                            "concurrency": conc,
                            "p50_ms": p50,
                            "p95_ms": p95,
                            "p99_ms": p99,
                            "throughput_rps": throughput,
                        }
                    )
                )
        print(f"Saved concurrency results to {args.output_file}")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
