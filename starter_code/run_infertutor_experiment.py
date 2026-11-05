"""
One-command experiment runner for InferTutor Arena.

This script:
1. Copies modal_infertutor_app.py.
2. Patches the serving constants based on CLI flags.
3. Deploys the Modal app.
4. Waits for /health and warms up every replica.
5. Runs the load tester.
6. Downloads each replica's server metrics and adds a summary to the results.
7. Stops the Modal app so GPUs stop billing (unless --keep-app).

Students should vary the CLI flags, not rewrite deployment code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel


console = Console()
ROOT = Path(__file__).parent
TEMPLATE = ROOT / "modal_infertutor_app.py"
RESULTS_DIR = ROOT / "results_infertutor"
METRICS_VOLUME = "infertutor-metrics"


def app_name_for(label: str) -> str:
    """Modal app names: lowercase letters, digits and dashes."""

    slug = re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-")
    if not slug:
        raise SystemExit(f"--label {label!r} has no usable characters (use letters, digits, dashes)")
    return f"infertutor-{slug}"[:60]


def image_size() -> tuple[int, int]:
    spec = json.loads((ROOT / "prompts.json").read_text()).get("image_spec", {})
    return (int(spec.get("width", 1024)), int(spec.get("height", 768)))


def patch_modal_app(args, app_name: str, run_id: str) -> Path:
    """Create a per-experiment Modal app file with static config values."""

    source = TEMPLATE.read_text()
    # repr() produces valid Python literals, so odd label/model strings can't break the file.
    replacements = {
        'MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")': f"MODEL_NAME = {args.model!r}",
        'TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))': f"TENSOR_PARALLEL = {args.tp!r}",
        'GPU_TYPE = os.environ.get("GPU_TYPE", "H100")': f"GPU_TYPE = {args.gpu_type!r}",
        'GPU_COUNT = int(os.environ.get("GPU_COUNT", str(TENSOR_PARALLEL)))': f"GPU_COUNT = {gpus_per_replica(args)!r}",
        'DTYPE = os.environ.get("DTYPE", "bfloat16")': f"DTYPE = {args.dtype!r}",
        'ENABLE_PREFIX_CACHING = os.environ.get("ENABLE_PREFIX_CACHING", "true").lower() == "true"': f"ENABLE_PREFIX_CACHING = {args.prefix_cache!r}",
        'ENABLE_CHUNKED_PREFILL = os.environ.get("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"': f"ENABLE_CHUNKED_PREFILL = {args.chunked_prefill!r}",
        'MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))': f"MAX_MODEL_LEN = {args.max_model_len!r}",
        'MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "4096"))': f"MAX_NUM_BATCHED_TOKENS = {args.max_batch_tokens!r}",
        'MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))': f"MAX_NUM_SEQS = {args.max_seqs!r}",
        'CONCURRENT_INPUTS = int(os.environ.get("CONCURRENT_INPUTS", "64"))': f"CONCURRENT_INPUTS = {args.concurrent_inputs!r}",
        'MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "1"))': f"MIN_CONTAINERS = {args.replicas!r}",
        'MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "1"))': f"MAX_CONTAINERS = {args.replicas!r}",
        'FAST_BOOT = os.environ.get("FAST_BOOT", "true").lower() == "true"': f"FAST_BOOT = {args.fast_boot!r}",
        'MM_MAX_PIXELS = int(os.environ.get("MM_MAX_PIXELS", str(512 * 28 * 28)))': f"MM_MAX_PIXELS = {args.mm_max_pixels!r}",
        'RUN_ID = os.environ.get("RUN_ID", "manual")': f"RUN_ID = {run_id!r}",
        'CPU_CORES = float(os.environ.get("CPU_CORES", "4"))': f"CPU_CORES = {float(args.cpu)!r}",
        'MEMORY_MB = int(os.environ.get("MEMORY_MB", "32768"))': f"MEMORY_MB = {args.memory_mb!r}",
        'MM_CACHE_GB = float(os.environ.get("MM_CACHE_GB", "-1"))': f"MM_CACHE_GB = {float(args.mm_cache_gb)!r}",
        'QUANTIZATION = os.environ.get("QUANTIZATION", "")': f"QUANTIZATION = {args.quantization!r}",
        'VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.21.0")': f"VLLM_VERSION = {args.vllm_version!r}",
        'PROBE_IMAGE_SIZE = (1024, 768)': f"PROBE_IMAGE_SIZE = {image_size()!r}",
        'app = modal.App("infertutor-arena")': f"app = modal.App({app_name!r})",
    }
    missing = [old for old in replacements if old not in source]
    if missing:
        # Fail loudly: a silent miss would deploy defaults while results claim otherwise.
        raise SystemExit(
            "modal_infertutor_app.py no longer contains these lines, so the runner can't patch them:\n  "
            + "\n  ".join(missing)
        )
    for old, new in replacements.items():
        source = source.replace(old, new)

    # One generated file per experiment, so parallel runs don't overwrite each other.
    generated = ROOT / f"modal_infertutor_app_generated_{app_name}.py"
    generated.write_text(source)
    return generated


def deploy(app_path: Path) -> str:
    """Deploy to Modal and return the web endpoint."""

    proc = subprocess.run(
        ["modal", "deploy", str(app_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    console.print(proc.stdout)
    if proc.returncode != 0:
        console.print(proc.stderr, style="red")
        raise SystemExit(proc.returncode)

    # Modal sometimes wraps long URLs in terminal output. Remove whitespace before parsing.
    # Endpoint hosts look like <workspace>--<app>-serve[.<region>].modal.run.
    # Only [a-z0-9-] labels are allowed, so other links in the output
    # (e.g. https://modal.com/apps/...) can't be matched by mistake.
    normalized = re.sub(r"\s+", "", proc.stdout)
    match = re.search(r"https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.modal\.run", normalized, re.IGNORECASE)
    if not match:
        raise RuntimeError("Could not find Modal endpoint in deploy output")
    return match.group(0)


def start_reaper(app_name: str, minutes: float) -> None:
    """Stop the app after `minutes` even if this runner is killed.

    The finally block below can't run if the process is killed (closed
    terminal, lost SSH session, SIGKILL), and min_containers would then keep
    the GPUs billing indefinitely. The reaper is a separate process in its own
    session, so it survives the runner. Stopping an already stopped app is a
    no-op, so it is harmless after a normal run.
    """

    modal_bin = shutil.which("modal") or "modal"
    subprocess.Popen(
        ["/bin/sh", "-c", f"sleep {int(minutes * 60)}; {shlex.quote(modal_bin)} app stop --yes {shlex.quote(app_name)}"],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    console.print(f"Safety stop: {app_name} will be stopped after {minutes:.0f} min even if this runner dies.")


def stop_app(app_name: str):
    """Stop the deployed app. min_containers keeps GPUs billing until this runs."""

    console.print(f"[bold]Stopping Modal app[/bold] {app_name} (pass --keep-app to leave it running)")
    try:
        proc = subprocess.run(
            ["modal", "app", "stop", "--yes", app_name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:  # noqa: BLE001 - must never mask the original error
        proc = None
        err = str(exc)
    else:
        err = proc.stderr.strip()
    if proc is None or proc.returncode != 0:
        console.print(
            f"[bold red]Could not stop {app_name}: {err}\n"
            f"Run `modal app stop {app_name}` manually, or GPUs will keep billing.[/bold red]"
        )
    else:
        console.print(f"[green]Stopped {app_name}[/green]")


def fetch_metrics(run_id: str, results_file: Path) -> Path | None:
    """Download every replica's metrics file and attach a summary to the results."""

    import server_metrics

    # Samplers commit to the volume every ~10 s; wait for the last commit.
    time.sleep(12)
    dest = results_file.with_suffix("")  # results_infertutor/<run>/  next to <run>.json
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["modal", "volume", "get", "--force", METRICS_VOLUME, f"/{run_id}", str(dest)],
        cwd=ROOT, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        console.print(f"[yellow]Could not download server metrics: {proc.stderr.strip()}[/yellow]")
        return None
    try:
        agg = server_metrics.attach_to_results(results_file, dest)["aggregate"]
    except Exception as exc:  # noqa: BLE001 - metrics are a bonus, never fail the run
        console.print(f"[yellow]Could not summarize server metrics: {exc}[/yellow]")
        return None
    console.print(Panel(json.dumps(agg, indent=2), title="Server metrics (measured window)"))
    return dest


def wait_for_health(url: str, timeout_s: int = 900):
    """Wait until vLLM reports healthy (on at least one replica)."""

    console.print(f"[bold]Waiting for vLLM health:[/bold] {url}/health")
    deadline = time.time() + timeout_s
    last_error = ""
    with httpx.Client(timeout=20) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{url.rstrip('/')}/health")
                if resp.status_code == 200:
                    console.print("[green]Endpoint is healthy[/green]")
                    return
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(10)
    raise TimeoutError(f"Endpoint did not become healthy: {last_error}")


async def _warmup_round(url: str, model: str, n: int) -> tuple[float, int]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    limits = httpx.Limits(max_connections=n, max_keepalive_connections=n)
    async with httpx.AsyncClient(timeout=900, limits=limits) as client:

        async def one() -> bool:
            try:
                resp = await client.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
                return resp.status_code == 200
            except Exception:
                return False

        start = time.perf_counter()
        ok = await asyncio.gather(*(one() for _ in range(n)))
    return time.perf_counter() - start, sum(ok)


def warmup(url: str, args):
    """
    /health only proves ONE replica is up. Send bursts larger than a single
    container's concurrency limit so Modal has to spread them across every
    replica; this blocks until the remaining replicas finish booting, so cold
    starts don't land inside the measured window.
    """

    n = args.concurrent_inputs * max(args.replicas - 1, 0) + 4 * args.replicas
    for i in range(args.warmup_rounds):
        elapsed, ok = asyncio.run(_warmup_round(url, args.model, n))
        console.print(f"Warmup round {i + 1}/{args.warmup_rounds}: {ok}/{n} ok in {elapsed:.1f}s")


def gpus_per_replica(args) -> int:
    return args.gpu_count or args.tp


def run_load_test(url: str, args, gpu_count_source: str, out_file: Path):
    """Run the fixed prompt workload against the deployed endpoint."""

    cmd = [
        sys.executable,
        str(ROOT / "load_test_infertutor.py"),
        "--url",
        url,
        "--model",
        args.model,
        "--mode",
        args.mode,
        "--users",
        str(args.users),
        "--duration",
        str(args.duration),
        "--ramp-up",
        str(args.ramp_up),
        "--max-tokens",
        str(args.max_tokens),
        "--label",
        args.label,
        "--total-gpus",
        str(gpus_per_replica(args) * args.replicas),
        "--gpu-count-source",
        gpu_count_source,
        "--deployment-json",
        json.dumps(vars(args)),
        "--processes",
        str(args.processes),
        "--slo-ttft-ms",
        str(args.slo_ttft_ms),
        "--slo-tpot-ms",
        str(args.slo_tpot_ms),
        "--out",
        str(out_file),
    ]
    if args.arrival_rate:
        cmd += ["--arrival-rate", str(args.arrival_rate)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Deploy and benchmark InferTutor")
    parser.add_argument("--label", required=True, help="Short name for this experiment")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--gpu-type", default="H100", choices=["H100", "H200", "A100", "L40S"])
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast-boot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--concurrent-inputs", type=int, default=64)
    parser.add_argument("--mm-max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--cpu", type=float, default=4.0,
                        help="CPU cores reserved per container (0 = Modal default of 0.125)")
    parser.add_argument("--memory-mb", type=int, default=32768, help="Memory reserved per container (0 = Modal default)")
    parser.add_argument("--seed", type=int, default=None, help="Seed so repeated runs send identical requests")
    parser.add_argument("--mm-cache-gb", type=float, default=-1,
                        help="vLLM multimodal processor cache size (0 disables it; default: vLLM's 4 GiB)")
    parser.add_argument("--quantization", default="", help='vLLM --quantization, e.g. "fp8" (default: none)')
    parser.add_argument("--vllm-version", default="0.21.0", help="vLLM version installed in the server image")
    parser.add_argument("--sweep-users", default=None,
                        help="Comma-separated user counts, e.g. 20,50,100: one deploy, one load test per count")
    parser.add_argument("--mode", choices=["text", "long", "image", "mixed"], default="mixed")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--ramp-up", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--arrival-rate", type=float, default=None,
                        help="Open-loop traffic: requests/s instead of --users (diagnostic, not scored)")
    parser.add_argument("--processes", type=int, default=1, help="Load-test client processes")
    parser.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    parser.add_argument("--warmup-rounds", type=int, default=2, help="Warmup bursts before measuring (0 to skip)")
    parser.add_argument("--url", default=None, help="Reuse an existing endpoint instead of deploying")
    parser.add_argument("--deploy-only", action="store_true", help="Deploy and leave the app running")
    parser.add_argument("--keep-app", action="store_true", help="Don't stop the Modal app after the run")
    parser.add_argument("--max-app-minutes", type=float, default=None,
                        help="Safety stop: the app is stopped after this many minutes even if the runner is killed "
                             "(default: estimated from the run; 0 disables, as does --keep-app/--deploy-only)")
    args = parser.parse_args()

    total_gpus = gpus_per_replica(args) * args.replicas
    console.print(Panel(json.dumps(vars(args) | {"total_gpus": total_gpus}, indent=2), title="InferTutor Experiment"))

    if total_gpus > 8:
        raise SystemExit("This starter runner caps experiments at 8 GPUs.")
    if args.duration <= args.ramp_up:
        raise SystemExit("--duration must be longer than --ramp-up (metrics are measured after ramp-up).")

    stamp = int(time.time())
    if args.url:
        # We didn't deploy it, so we can't vouch for the GPU count, fetch its
        # metrics (its run id is unknown), or stop it.
        url = args.url
        wait_for_health(url)
        if not args.deploy_only:
            warmup(url, args)
            out_file = RESULTS_DIR / f"{app_name_for(args.label)}-{stamp}_{args.mode}.json"
            run_load_test(url, args, gpu_count_source="user-supplied", out_file=out_file)
        return

    app_name = app_name_for(args.label)
    run_id = f"{app_name}-{stamp}"
    out_file = RESULTS_DIR / f"{run_id}_{args.mode}.json"
    deploy_started = time.time()
    generated = patch_modal_app(args, app_name, run_id)
    keep = args.keep_app or args.deploy_only
    # Everything from deploy onward is inside try/finally: if anything fails
    # (even parsing the deploy output), the app is still stopped.
    if not keep and args.max_app_minutes != 0:
        levels = len(args.sweep_users.split(",")) if args.sweep_users else 1
        estimate = 25 + levels * (args.duration + 60) / 60  # cold start/compile + each load test, with slack
        start_reaper(app_name, args.max_app_minutes or estimate)
    try:
        url = deploy(generated)
        wait_for_health(url)
        args.deploy_to_healthy_s = round(time.time() - deploy_started, 1)  # saved in results["deployment"]
        console.print(f"Deploy to first healthy replica: {args.deploy_to_healthy_s:.0f}s")
        if not args.deploy_only:
            warmup(url, args)
            if args.sweep_users:
                # One deployment, several load levels. Each result keeps its own
                # window, so server metrics are summarized per level.
                for users in [int(u) for u in args.sweep_users.split(",")]:
                    args.users = users
                    level_file = RESULTS_DIR / f"{run_id}-{users}u_{args.mode}.json"
                    run_load_test(url, args, gpu_count_source="runner-deploy", out_file=level_file)
                    fetch_metrics(run_id, level_file)
            else:
                run_load_test(url, args, gpu_count_source="runner-deploy", out_file=out_file)
                fetch_metrics(run_id, out_file)
    finally:
        if keep:
            console.print(
                f"[yellow]App {app_name} is still running and billing GPUs. "
                f"Stop it with: modal app stop {app_name}[/yellow]"
            )
        else:
            stop_app(app_name)


if __name__ == "__main__":
    main()
