"""
InferTutor Arena - Modal + vLLM server.

Students should usually not edit this file first. Start by changing
configuration from run_infertutor_experiment.py CLI flags.

This app runs Qwen/Qwen3-VL-4B-Instruct behind an OpenAI-compatible vLLM
HTTP server on Modal.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import modal


# vLLM 0.21.0 supports Qwen3-VL and the OpenAI-compatible multimodal API.
# The CUDA base image gives vLLM access to the GPU runtime it needs.
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.21.0")
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "qwen-vl-utils==0.0.14")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("infertutor-arena")

# Persistent caches reduce repeated model download and compilation overhead.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
# Per-replica server metrics, written by a sampler thread in each container.
metrics_vol = modal.Volume.from_name("infertutor-metrics", create_if_missing=True)
METRICS_DIR = "/metrics_out"


# These constants are patched by run_infertutor_experiment.py before deploy.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct")
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")
GPU_COUNT = int(os.environ.get("GPU_COUNT", str(TENSOR_PARALLEL)))
DTYPE = os.environ.get("DTYPE", "bfloat16")
ENABLE_PREFIX_CACHING = os.environ.get("ENABLE_PREFIX_CACHING", "true").lower() == "true"
ENABLE_CHUNKED_PREFILL = os.environ.get("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "4096"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
CONCURRENT_INPUTS = int(os.environ.get("CONCURRENT_INPUTS", "64"))
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "1"))
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", "1"))
FAST_BOOT = os.environ.get("FAST_BOOT", "true").lower() == "true"
MM_MAX_PIXELS = int(os.environ.get("MM_MAX_PIXELS", str(512 * 28 * 28)))
MM_CACHE_GB = float(os.environ.get("MM_CACHE_GB", "-1"))  # <0: vLLM default (4 GiB)
QUANTIZATION = os.environ.get("QUANTIZATION", "")  # e.g. "fp8"; empty = full-precision weights
RUN_ID = os.environ.get("RUN_ID", "manual")
# CPU cores and memory reserved per container. vLLM needs at least 2 + N
# physical cores (API server, engine core, one worker per GPU). 0 = Modal's
# default, which reserves only 0.125 cores.
CPU_CORES = float(os.environ.get("CPU_CORES", "4"))
MEMORY_MB = int(os.environ.get("MEMORY_MB", "32768"))
PROBE_IMAGE_SIZE = (1024, 768)  # same size as the load tester's images
METRICS_INTERVAL_S = 2.0
INGRESS_MODULE_DIR = "/root/infertutor_mw"

MINUTES = 60
VLLM_PORT = 8000


# ---------------------------------------------------------------------------
# Server-side metrics sampler
#
# The public Modal URL load-balances across replicas and doesn't say which one
# answered, so an external Prometheus can't scrape replicas individually.
# Instead each container samples its own vLLM /metrics and GPU stats and
# appends them to <METRICS_DIR>/<RUN_ID>/<replica>.jsonl on a Modal Volume.
# The runner downloads the files after the load test.
# ---------------------------------------------------------------------------


IGNORED_LABELS = {"model_name", "engine"}
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse vLLM's Prometheus text.

    Two kinds of keys, so totals and breakdowns are both available:
      name / name|le=X                totals, summed over all other labels
      name|k1=v1,k2=v2[,le=X]         one entry per label set (model name and
                                      engine index dropped), e.g. by source,
                                      finish reason or cache config
    """

    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            series, value = line.rsplit(" ", 1)
            val = float(value)
        except ValueError:
            continue
        name, _, labels = series.partition("{")
        pairs = dict(_LABEL_RE.findall(labels)) if labels else {}
        le = pairs.pop("le", None)
        extra = sorted((k, v) for k, v in pairs.items() if k not in IGNORED_LABELS)
        total_key = f"{name}|le={le}" if le is not None else name
        out[total_key] = out.get(total_key, 0.0) + val
        if extra:
            parts = [f"{k}={v}" for k, v in extra] + ([f"le={le}"] if le is not None else [])
            key = f"{name}|" + ",".join(parts)
            out[key] = out.get(key, 0.0) + val
    return out


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def gpu_stats() -> list[dict]:
    raw = _run([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            idx, util, used, total, power = parts
            rows.append({"gpu": int(idx), "util_pct": float(util), "mem_used_mib": float(used),
                         "mem_total_mib": float(total), "power_w": float(power)})
        except ValueError:
            continue
    return rows


CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def process_cpu_seconds() -> dict[str, float]:
    """CPU seconds used so far by every process in the container, grouped by
    process name (vLLM names its processes, e.g. VLLM::EngineCore). The sampler
    turns deltas of these into cores used per process."""

    out: dict[str, float] = {}
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
        except OSError:
            continue
        # comm is in parentheses and may contain spaces.
        name = stat[stat.index("(") + 1 : stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2 :].split()
        try:
            secs = (int(fields[11]) + int(fields[12])) / CLK_TCK  # utime + stime
        except (IndexError, ValueError):
            continue
        out[name] = out.get(name, 0.0) + secs
    return out


def cpu_info() -> dict:
    info: dict = {"os_cpu_count": os.cpu_count()}
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            info["cgroup_" + Path(path).name] = Path(path).read_text().strip()
        except OSError:
            pass
    info["reserved_cores"] = CPU_CORES or 0.125
    return info


# Facts vLLM only prints at startup. Filled from its log by watch_vllm_log().
STARTUP: dict = {}
STARTUP_PATTERNS = {
    "kv_cache_tokens": (r"GPU KV cache size: ([\d,]+) tokens", lambda m: int(m.group(1).replace(",", ""))),
    "max_concurrency_x": (r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", lambda m: float(m.group(1))),
    "model_load_gib": (r"Model loading took ([\d.]+) GiB", lambda m: float(m.group(1))),
    "model_load_s": (r"Model loading took [\d.]+ GiB (?:memory )?and ([\d.]+) seconds", lambda m: float(m.group(1))),
    "torch_compile_s": (r"torch\.compile takes ([\d.]+) s in total", lambda m: float(m.group(1))),
    "cudagraph_capture_s": (r"Graph capturing finished in ([\d.]+) secs", lambda m: float(m.group(1))),
    "cudagraph_sizes": (r"cudagraph_capture_sizes['\"]?[:=]\s*(\[[\d, ]*\])", lambda m: json.loads(m.group(1))),
    "engine_init_s": (r"init engine .* took ([\d.]+) (?:seconds|s)", lambda m: float(m.group(1))),
}


def watch_vllm_log(stream):
    """Echo vLLM's log (so it still shows in Modal) and pick out startup facts."""

    for line in stream:
        print(line, end="", flush=True)
        for key, (pattern, parse) in STARTUP_PATTERNS.items():
            if key in STARTUP:
                continue
            m = re.search(pattern, line)
            if m:
                try:
                    STARTUP[key] = parse(m)
                except (ValueError, IndexError):
                    pass


def probe_image_tokens() -> dict:
    """Exact image-token count from vLLM's own processor: the prompt-token
    difference between the same request with and without the test image."""

    import base64
    import io

    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", PROBE_IMAGE_SIZE, (245, 247, 250))
        ImageDraw.Draw(img).rectangle([100, 100, 600, 300], fill=(38, 92, 135))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        question = {"type": "text", "text": "What is shown?"}

        def prompt_tokens(content) -> int:
            body = json.dumps({"model": MODEL_NAME, "max_tokens": 1,
                               "messages": [{"role": "user", "content": content}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["usage"]["prompt_tokens"]

        with_image = prompt_tokens([{"type": "image_url", "image_url": {"url": url}}, question])
        without = prompt_tokens([question])
        return {"image_size": list(PROBE_IMAGE_SIZE), "prompt_with_image": with_image,
                "prompt_without_image": without, "image_tokens": with_image - without}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


# Pure-ASGI middleware loaded into vLLM's API server with --middleware. It adds
# two server-clock timestamps to every response's headers: when the request
# entered the container's HTTP server, and when the response started (headers
# sent, before the first token). The client compares them with its own clock
# deltas to split TTFT into network/proxy time and server time.
INGRESS_MIDDLEWARE = '''
import time


class IngressTimer:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        t_in = time.time()

        async def timed_send(message):
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-ingress-ts", repr(t_in).encode()))
                headers.append((b"x-response-start-ts", repr(time.time()).encode()))
                message = dict(message, headers=headers)
            await send(message)

        await self.app(scope, receive, timed_send)
'''


def _get(path: str, timeout: float = 3.0) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}{path}", timeout=timeout) as resp:
            return resp.read().decode() if resp.status == 200 else None
    except Exception:  # noqa: BLE001
        return None


def metrics_sampler(container_start: float, commit=None, stop: threading.Event | None = None):
    """Runs in a daemon thread for the life of the container."""

    commit = commit or metrics_vol.commit
    replica = os.environ.get("MODAL_TASK_ID", f"pid-{os.getpid()}")
    out_dir = Path(METRICS_DIR) / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{replica}.jsonl"

    # Wait until vLLM serves requests: that moment defines the cold start.
    while _get("/health", timeout=2) is None:
        if stop is not None and stop.is_set():
            return
        time.sleep(1)
    ready = time.time()

    version = _get("/version") or ""
    try:
        vllm_version = json.loads(version).get("version", version)
    except ValueError:
        vllm_version = version
    probe = probe_image_tokens()
    meta = {
        "type": "meta",
        "replica": replica,
        "run_id": RUN_ID,
        "container_start": container_start,
        "vllm_ready": ready,
        "cold_start_s": ready - container_start,
        "vllm_version": vllm_version,
        "torch_cuda": _run([sys.executable, "-c", "import torch; print(torch.version.cuda)"]),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "cpu": cpu_info(),
        "startup": dict(STARTUP),
        "image_token_probe": probe,
        "config": {
            "model": MODEL_NAME, "tp": TENSOR_PARALLEL, "gpu_type": GPU_TYPE, "gpu_count": GPU_COUNT,
            "max_num_seqs": MAX_NUM_SEQS, "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_model_len": MAX_MODEL_LEN, "prefix_caching": ENABLE_PREFIX_CACHING,
            "chunked_prefill": ENABLE_CHUNKED_PREFILL, "fast_boot": FAST_BOOT,
            "mm_max_pixels": MM_MAX_PIXELS, "concurrent_inputs": CONCURRENT_INPUTS,
            "cpu_cores": CPU_CORES, "memory_mb": MEMORY_MB, "mm_cache_gb": MM_CACHE_GB,
            "quantization": QUANTIZATION, "vllm_pinned": VLLM_VERSION,
        },
    }
    with path.open("a") as f:
        f.write(json.dumps(meta) + "\n")
        n = 0
        while stop is None or not stop.is_set():
            t = time.time()
            text = _get("/metrics")
            sample = {"type": "sample", "t": t, "vllm": parse_prometheus(text) if text else None,
                      "gpu": gpu_stats(), "cpu_s": process_cpu_seconds()}
            f.write(json.dumps(sample) + "\n")
            f.flush()
            n += 1
            if n % 5 == 0:  # make data visible to the runner about every 10 s
                try:
                    commit()
                except Exception as exc:  # noqa: BLE001
                    print(f"metrics commit failed: {exc}", flush=True)
            time.sleep(max(METRICS_INTERVAL_S - (time.time() - t), 0))


# NOTE: min_containers keeps replicas warm, which means they never scale down
# on their own. run_infertutor_experiment.py stops the app after each run;
# if you deploy by hand, run `modal app stop <app-name>` when finished.
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=10 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=15 * MINUTES,
    cpu=CPU_CORES or None,
    memory=MEMORY_MB or None,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        METRICS_DIR: metrics_vol,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    """Start vLLM inside the Modal container."""

    container_start = time.time()

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL),
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        "0.90",
        "--uvicorn-log-level=warning",
        "--middleware",
        "ingress_timer.IngressTimer",
        "--limit-mm-per-prompt",
        '{"image": 1, "video": 0}',
        "--mm-processor-kwargs",
        f'{{"min_pixels": 784, "max_pixels": {MM_MAX_PIXELS}, "fps": 1}}',
    ]

    # Eager mode starts faster. Compiled mode can improve text-only throughput
    # after a longer warmup.
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]

    # vLLM's V1 engine enables both features by default, so "off" must be
    # passed explicitly. Omitting the flag would silently leave them on.
    cmd += ["--enable-prefix-caching" if ENABLE_PREFIX_CACHING else "--no-enable-prefix-caching"]
    cmd += ["--enable-chunked-prefill" if ENABLE_CHUNKED_PREFILL else "--no-enable-chunked-prefill"]
    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]
    if MM_CACHE_GB >= 0:
        cmd += ["--mm-processor-cache-gb", str(MM_CACHE_GB)]

    # The API server imports the middleware from this directory.
    Path(INGRESS_MODULE_DIR).mkdir(parents=True, exist_ok=True)
    (Path(INGRESS_MODULE_DIR) / "ingress_timer.py").write_text(INGRESS_MIDDLEWARE)
    env = dict(os.environ, PYTHONPATH=INGRESS_MODULE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""))

    print("Starting vLLM:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=watch_vllm_log, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=metrics_sampler, args=(container_start,), daemon=True).start()

