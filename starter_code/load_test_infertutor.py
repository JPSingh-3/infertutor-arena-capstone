"""Fixed workload load tester for InferTutor Arena.

Measurement rules (metrics_version 3):
- Users ramp up during --ramp-up seconds. Metrics only cover requests that
  START after ramp-up ends (the measured window), and throughput is divided
  by the window length only.
- ITL percentiles use every individual inter-chunk gap, so mid-stream stalls
  show up in p95/p99.
- Token counts come from the server (stream_options.include_usage), so
  tokens/s is exact rather than estimated from streamed chunks.
- Every metric is also broken down by request type (text / long / image).

Traffic models:
- Closed loop (default): N users, each waits for its answer, pauses, asks again.
- Open loop (--arrival-rate R): new requests arrive at R per second whether or
  not earlier ones finished. A slow server can't hide by receiving fewer
  requests, so this shows true capacity.

Use --processes N to spread the load over N client processes when one Python
process can't keep up (the results report client event-loop lag and CPU use
so you can tell).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import multiprocessing
import os
import random
import statistics
import struct
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table


METRICS_VERSION = 3
console = Console()
ROOT = Path(__file__).parent
PROMPTS_BYTES = (ROOT / "prompts.json").read_bytes()
PROMPTS = json.loads(PROMPTS_BYTES)
PROMPTS_SHA256 = hashlib.sha256(PROMPTS_BYTES).hexdigest()

# ---------------------------------------------------------------------------
# Image workload
# ---------------------------------------------------------------------------

IMAGE_SPEC = PROMPTS.get("image_spec", {"width": 1024, "height": 768})
PALETTE = [bytes(c) for c in [(245, 247, 250), (38, 92, 135), (228, 111, 71), (81, 168, 129)]]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def make_base_raw(width: int, height: int) -> bytearray:
    """Raw (filtered) RGB scanlines for a diagram-like image: grid plus two bars."""

    rows: dict[tuple, bytes] = {}
    raw = bytearray()
    bar_x0, bar_x1 = width * 3 // 16, width * 13 // 16
    for y in range(height):
        bar = 1 if height * 11 // 30 < y < height * 14 // 30 else 2 if height * 20 // 30 < y < height * 23 // 30 else 0
        key = ((y // 24) % len(PALETTE), bar)
        if key not in rows:
            row = bytearray([0])  # PNG filter type 0
            for x in range(width):
                idx = bar if bar and bar_x0 < x < bar_x1 else ((x // 24) + key[0]) % len(PALETTE)
                row += PALETTE[idx]
            rows[key] = bytes(row)
        raw += rows[key]
    return raw


BASE_RAW = make_base_raw(IMAGE_SPEC["width"], IMAGE_SPEC["height"])


def encode_png(raw: bytes, width: int, height: int) -> str:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, 1))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


STATIC_IMAGE_URL = encode_png(bytes(BASE_RAW), IMAGE_SPEC["width"], IMAGE_SPEC["height"])


def make_unique_image_url(stamp: bytes | None = None) -> str:
    """
    Stamp random pixels into the first scanline so every request has different
    pixel content. vLLM hashes image pixels for its multimodal/prefix caches, so
    a single repeated image would make the vision path look artificially cheap.
    Runs in a worker thread (zlib releases the GIL) to keep the event loop free.
    `stamp` (48 bytes) comes from a seeded RNG so runs can be repeated exactly.
    """

    raw = bytearray(BASE_RAW)
    raw[1:49] = stamp if stamp is not None else os.urandom(48)  # 16 RGB pixels after the filter byte
    return encode_png(bytes(raw), IMAGE_SPEC["width"], IMAGE_SPEC["height"])


# ---------------------------------------------------------------------------
# Long-context workload
# ---------------------------------------------------------------------------

LONG_TRACE_LINES = int(PROMPTS.get("long_trace_lines", 0))
COMPONENTS = ["scheduler", "engine", "api_server", "kv_cache", "worker"]


def make_debug_trace(rng: random.Random, lines: int) -> str:
    """Synthetic vLLM-style server log. Different per request, so prefix caching
    can only reuse the shared system prompt, as with real user-pasted traces."""

    t = rng.uniform(0, 3600)
    step = rng.randint(1000, 90000)
    out = []
    for _ in range(lines):
        t += rng.uniform(0.01, 0.4)
        step += rng.randint(1, 9)
        out.append(
            f"[{t:9.3f}] INFO {rng.choice(COMPONENTS)} step={step} running={rng.randint(1, 64)} "
            f"waiting={rng.randint(0, 200)} kv_usage={rng.uniform(0.2, 0.99):.3f} "
            f"prefill_tok={rng.randint(0, 8192)} decode_tok={rng.randint(0, 64)} "
            f"ttft_ms={rng.uniform(80, 6000):.1f} itl_ms={rng.uniform(6, 90):.1f}"
        )
    return "\n".join(out)


def long_prompt(rng: random.Random) -> str:
    question = rng.choice(PROMPTS["long"])
    if not LONG_TRACE_LINES:
        return question
    trace = make_debug_trace(random.Random(rng.getrandbits(64)), LONG_TRACE_LINES)
    return f"Server log excerpt from the incident:\n```\n{trace}\n```\n\n{question}"


# ---------------------------------------------------------------------------
# Raw measurements (one per client process, merged at the end)
# ---------------------------------------------------------------------------

KINDS = ("text", "long", "image")


@dataclass
class Raw:
    window_start: float  # unix time; shared by all processes
    window_end: float
    requests: list[dict] = field(default_factory=list)  # requests that started in the window
    gaps_ms: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in KINDS})
    window_chunks: int = 0  # chunks received during the window (incl. ramp-phase requests)
    ramp_requests: int = 0
    ramp_errors: int = 0
    loop_lag_ms: list[float] = field(default_factory=list)
    cpu_util: float = 0.0
    active_users: int = 0
    draining: bool = False  # window over: no new requests, in-flight ones finish

    def in_window(self, start_unix: float) -> bool:
        return start_unix >= self.window_start

    def measuring(self, now_unix: float) -> bool:
        return self.window_start <= now_unix <= self.window_end

    def record(self, start_unix: float, rec: dict, gaps: list[float]):
        if not self.in_window(start_unix):
            self.ramp_requests += 1
            self.ramp_errors += 0 if rec["ok"] else 1
            return
        self.requests.append(rec)
        if rec["ok"]:
            self.gaps_ms[rec["kind"]].extend(gaps)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p / 100), len(ordered) - 1)]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def make_rng(args, *parts) -> random.Random:
    """Seeded per user (or per open-loop worker) when --seed is set, so every
    user sends the same sequence of requests in every run, however the
    requests interleave. Unseeded runs stay random."""

    if args.seed is None:
        return random.Random()
    return random.Random("-".join(str(p) for p in (args.seed, *parts)))


def pick_kind(mode: str, rng: random.Random) -> str:
    if mode != "mixed":
        return mode
    # Official product workload: mostly text, with some long and image traffic.
    roll = rng.random()
    return "image" if roll < 0.25 else "long" if roll < 0.45 else "text"


async def build_messages(kind: str, unique_images: bool, rng: random.Random) -> list[dict]:
    system = {"role": "system", "content": PROMPTS["system_prompt"]}
    if kind == "text":
        return [system, {"role": "user", "content": rng.choice(PROMPTS["text"])}]
    if kind == "long":
        return [system, {"role": "user", "content": long_prompt(rng)}]
    if unique_images:
        stamp = bytes(rng.getrandbits(8) for _ in range(48))
        url = await asyncio.to_thread(make_unique_image_url, stamp)
    else:
        url = STATIC_IMAGE_URL
    content = [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": rng.choice(PROMPTS["image"])},
    ]
    return [system, {"role": "user", "content": content}]


def _header_ts(resp, name: str) -> float | None:
    try:
        return float(resp.headers[name])
    except (KeyError, ValueError):
        return None


async def one_request(client: httpx.AsyncClient, args, raw: Raw, rng: random.Random):
    kind = pick_kind(args.mode, rng)
    payload = {
        "model": args.model,
        "messages": await build_messages(kind, args.unique_images, rng),
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},  # exact token counts in the last chunk
    }

    start_unix = time.time()
    start = time.perf_counter()
    first = last = None
    gaps: list[float] = []
    chunks = 0
    usage: dict = {}
    rec = {"kind": kind, "ok": False, "error": None}
    hop: dict = {}

    try:
        async with client.stream("POST", f"{args.url.rstrip('/')}/v1/chat/completions", json=payload) as resp:
            # Response headers arrive once the server has set up the stream,
            # before the first token. The server stamps two times on them
            # (its own clock), so only differences are compared across clocks.
            headers_at = time.perf_counter()
            ingress, resp_start = _header_ts(resp, "x-ingress-ts"), _header_ts(resp, "x-response-start-ts")
            hop["to_headers_ms"] = (headers_at - start) * 1000
            if ingress is not None and resp_start is not None:
                hop["server_pre_ms"] = (resp_start - ingress) * 1000
                hop["outside_server_ms"] = hop["to_headers_ms"] - hop["server_pre_ms"]
            if resp.status_code != 200:
                await resp.aread()
                rec["error"] = f"http_{resp.status_code}"
                raw.record(start_unix, rec, gaps)
                return
            async for line in resp.aiter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                content = choices[0].get("delta", {}).get("content") if choices else None
                if content:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    else:
                        gaps.append((now - last) * 1000)
                    last = now
                    chunks += 1
                    if raw.measuring(time.time()):
                        raw.window_chunks += 1
    except asyncio.CancelledError:
        if raw.draining:
            # Still running after the drain period: a real slow request. Count
            # it instead of silently dropping the slowest requests.
            rec["error"] = "unfinished_after_drain"
            raw.record(start_unix, rec, gaps)
        raise
    except httpx.TimeoutException:
        rec["error"] = "timeout"
    except Exception as exc:  # noqa: BLE001
        rec["error"] = type(exc).__name__
    else:
        if first is None:
            rec["error"] = "empty_stream"
        else:
            latency = (time.perf_counter() - start) * 1000
            ttft = (first - start) * 1000
            if "to_headers_ms" in hop:
                hop["headers_to_first_token_ms"] = ttft - hop["to_headers_ms"]
            completion = usage.get("completion_tokens")
            if completion and completion > 1:
                tpot = (latency - ttft) / (completion - 1)
            else:
                tpot = sum(gaps) / len(gaps) if gaps else 0.0
            rec.update(
                ok=True,
                ttft_ms=ttft,
                latency_ms=latency,
                tpot_ms=tpot,
                chunks=chunks,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=completion,
                **hop,
            )
    raw.record(start_unix, rec, gaps)


# ---------------------------------------------------------------------------
# Traffic drivers
# ---------------------------------------------------------------------------


async def closed_loop(args, raw: Raw, n_users: int, ramp_start: float, stop: asyncio.Event, tasks: list,
                      worker_idx: int = 0):
    async def user(uid: str):
        rng = make_rng(args, "user", uid)
        async with httpx.AsyncClient(timeout=args.request_timeout) as client:
            while not stop.is_set():
                await one_request(client, args, raw, rng)
                # Always pause, including after errors: instant retries would
                # hammer an overloaded server and inflate the error count.
                await asyncio.sleep(rng.uniform(args.min_pause, args.max_pause))

    delay = args.ramp_up / max(n_users, 1) if args.ramp_up else 0
    for i in range(n_users):
        if stop.is_set():
            return
        tasks.append(asyncio.create_task(user(f"{worker_idx}.{i}")))
        raw.active_users = i + 1
        if delay:
            await asyncio.sleep(delay)


async def open_loop(args, raw: Raw, rate: float, ramp_start: float, stop: asyncio.Event, tasks: list,
                    worker_idx: int = 0):
    """Poisson arrivals. The rate ramps linearly from 0 during --ramp-up."""

    arrivals = make_rng(args, "arrivals", worker_idx)
    n_fired = 0
    limits = httpx.Limits(max_connections=args.max_inflight, max_keepalive_connections=args.max_inflight)
    inflight = 0
    async with httpx.AsyncClient(timeout=args.request_timeout, limits=limits) as client:

        async def fire(rng: random.Random):
            nonlocal inflight
            inflight += 1
            raw.active_users = inflight
            try:
                await one_request(client, args, raw, rng)
            finally:
                inflight -= 1
                raw.active_users = inflight

        while not stop.is_set():
            elapsed = time.time() - ramp_start
            current = rate * min(max(elapsed / args.ramp_up, 0.05), 1.0) if args.ramp_up else rate
            await asyncio.sleep(arrivals.expovariate(current))
            if stop.is_set():
                break
            if inflight >= args.max_inflight:
                # The client can't hold more open requests: the server is
                # falling behind the arrival rate. Count it, don't hide it.
                raw.record(time.time(), {"kind": pick_kind(args.mode, arrivals), "ok": False, "error": "client_backlog"}, [])
                continue
            n_fired += 1
            tasks.append(asyncio.create_task(fire(make_rng(args, "request", worker_idx, n_fired))))
        # Keep the shared client open until in-flight requests finish (or the
        # caller cancels them after the drain timeout).
        pending = [t for t in tasks if not t.done()]
        if pending:
            await asyncio.wait(pending)


async def loop_lag_monitor(raw: Raw, stop: asyncio.Event):
    """If the event loop is late waking up, the client itself is overloaded and
    its latency numbers include its own delay."""

    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.1)
        lag = (time.perf_counter() - t - 0.1) * 1000
        if raw.measuring(time.time()):
            raw.loop_lag_ms.append(max(lag, 0.0))


def live_table(raw: Raw, start_unix: float) -> Table:
    # Cheap on purpose: sorting millions of gaps every refresh would block the
    # event loop and distort the latencies being measured.
    now = time.time()
    phase = "ramp-up (not measured)" if now < raw.window_start else "measuring"
    window = max(min(now, raw.window_end) - raw.window_start, 0.0)
    recent = raw.requests[-3000:]
    ok = [r for r in recent if r["ok"]]
    gaps = [g for k in KINDS for g in raw.gaps_ms[k][-7000:]]
    errors = sum(1 for r in raw.requests if not r["ok"])
    table = Table(title=f"InferTutor Load Test - {now - start_unix:.0f}s - {phase}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Active users / in flight", str(raw.active_users))
    table.add_row("Requests (window)", str(len(raw.requests)))
    table.add_row("Errors (window)", str(errors))
    table.add_row("TTFT p95 (recent)", f"{percentile([r['ttft_ms'] for r in ok], 95):.1f} ms")
    table.add_row("ITL p95 (recent)", f"{percentile(gaps, 95):.1f} ms")
    table.add_row("Throughput", f"{raw.window_chunks / window if window else 0:.1f} chunks/s")
    return table


async def run_worker(args, idx: int, n_workers: int, t0: float, live: bool) -> dict:
    """One client process. t0 is a shared unix start time so all processes
    use the same measured window."""

    raw = Raw(window_start=t0 + args.ramp_up, window_end=t0 + args.duration)
    stop = asyncio.Event()
    tasks: list[asyncio.Task] = []
    await asyncio.sleep(max(t0 - time.time(), 0))

    if args.arrival_rate:
        driver = open_loop(args, raw, args.arrival_rate / n_workers, t0, stop, tasks, idx)
    else:
        n_users = args.users // n_workers + (1 if idx < args.users % n_workers else 0)
        driver = closed_loop(args, raw, n_users, t0, stop, tasks, idx)
    driver_task = asyncio.create_task(driver)
    lag_task = asyncio.create_task(loop_lag_monitor(raw, stop))

    cpu0 = None

    async def tick_until_end(on_tick=None):
        nonlocal cpu0
        while time.time() < raw.window_end:
            nxt = time.time() + 2
            if cpu0 is None:
                nxt = min(nxt, raw.window_start)
            await asyncio.sleep(max(min(nxt, raw.window_end) - time.time(), 0))
            if cpu0 is None and time.time() >= raw.window_start:
                cpu0 = time.process_time()
            if on_tick:
                on_tick()

    if live:
        with Live(live_table(raw, t0), refresh_per_second=0.5, console=console) as view:
            await tick_until_end(lambda: view.update(live_table(raw, t0)))
    else:
        await tick_until_end()
    if cpu0 is not None:
        raw.cpu_util = (time.process_time() - cpu0) / max(raw.window_end - raw.window_start, 1e-9)

    # Drain: stop starting requests, let in-flight ones finish so the slowest
    # requests of the window are measured rather than cancelled.
    stop.set()
    raw.draining = True
    pending = [t for t in tasks if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=args.drain_timeout)
    # Cancel requests before the driver: the driver owns the HTTP client(s).
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    driver_task.cancel()
    await asyncio.gather(driver_task, lag_task, return_exceptions=True)
    return asdict(raw)


def _worker_entry(args_dict: dict, idx: int, n_workers: int, t0: float) -> dict:
    return asyncio.run(run_worker(argparse.Namespace(**args_dict), idx, n_workers, t0, live=False))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


HOPS = ("to_headers_ms", "server_pre_ms", "outside_server_ms", "headers_to_first_token_ms")


def ttft_breakdown(ok: list[dict]) -> dict:
    """p50/p95 of each TTFT segment. outside_server = client wait until headers
    minus the server's own ingress->headers time: network, Modal proxy and any
    queueing before the container's HTTP server."""

    out: dict = {"requests_with_server_timestamps": sum(1 for r in ok if "server_pre_ms" in r)}
    for hop in HOPS:
        vals = [r[hop] for r in ok if hop in r]
        if vals:
            out[hop] = {"p50": percentile(vals, 50), "p95": percentile(vals, 95)}
    return out


def latency_block(reqs: list[dict], gaps: list[float]) -> dict:
    ok = [r for r in reqs if r["ok"]]
    prompt = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    return {
        "requests": len(reqs),
        "errors": len(reqs) - len(ok),
        "ttft_p50_ms": percentile([r["ttft_ms"] for r in ok], 50),
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in ok], 95),
        "itl_p95_ms": percentile(gaps, 95),
        "tpot_p95_ms": percentile([r["tpot_ms"] for r in ok], 95),
        "latency_p95_ms": percentile([r["latency_ms"] for r in ok], 95),
        "prompt_tokens_mean": statistics.mean(prompt or [0]),
        "prompt_tokens_min": min(prompt) if prompt else None,
        "prompt_tokens_max": max(prompt) if prompt else None,
        "ttft_breakdown": ttft_breakdown(ok),
        "completion_tokens_mean": statistics.mean([r["completion_tokens"] for r in ok if r.get("completion_tokens")] or [0]),
    }


def compute_results(raws: list[dict], args) -> dict:
    ws, we = raws[0]["window_start"], raws[0]["window_end"]
    window = max(we - ws, 1e-9)
    reqs = [r for raw in raws for r in raw["requests"]]
    gaps = {k: [g for raw in raws for g in raw["gaps_ms"][k]] for k in KINDS}
    all_gaps = [g for k in KINDS for g in gaps[k]]
    window_chunks = sum(raw["window_chunks"] for raw in raws)
    ok = [r for r in reqs if r["ok"]]
    errors = len(reqs) - len(ok)

    error_kinds: dict[str, int] = {}
    for r in reqs:
        if not r["ok"]:
            error_kinds[r["error"]] = error_kinds.get(r["error"], 0) + 1

    with_usage = [r for r in ok if r.get("completion_tokens")]
    tokens_per_chunk = (
        sum(r["completion_tokens"] for r in with_usage) / sum(r["chunks"] for r in with_usage) if with_usage else None
    )

    met = [r for r in ok if r["ttft_ms"] <= args.slo_ttft_ms and r["tpot_ms"] <= args.slo_tpot_ms]
    lag = [x for raw in raws for x in raw["loop_lag_ms"]]
    cpu = max(raw["cpu_util"] for raw in raws)
    client = {
        "processes": len(raws),
        "loop_lag_p99_ms": percentile(lag, 99),
        "loop_lag_max_ms": max(lag) if lag else 0.0,
        "cpu_util_max_process": cpu,
    }
    client["overloaded"] = client["loop_lag_p99_ms"] > 20 or cpu > 0.85

    return {
        "metrics_version": METRICS_VERSION,
        "traffic": "open_loop" if args.arrival_rate else "closed_loop",
        "window_start_unix": ws,
        "window_end_unix": we,
        "measured_window_s": window,
        "total_requests": len(reqs),
        "total_errors": errors,
        "error_rate": errors / max(len(reqs), 1),
        "error_kinds": error_kinds,
        "window_stream_chunks": window_chunks,
        "ttft_p50_ms": percentile([r["ttft_ms"] for r in ok], 50),
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in ok], 95),
        "ttft_p99_ms": percentile([r["ttft_ms"] for r in ok], 99),
        "itl_p50_ms": percentile(all_gaps, 50),
        "itl_p95_ms": percentile(all_gaps, 95),
        "itl_p99_ms": percentile(all_gaps, 99),
        "tpot_p95_ms": percentile([r["tpot_ms"] for r in ok], 95),
        "latency_p50_ms": percentile([r["latency_ms"] for r in ok], 50),
        "latency_p95_ms": percentile([r["latency_ms"] for r in ok], 95),
        "aggregate_stream_chunks_per_s": window_chunks / window,
        "tokens_per_chunk": tokens_per_chunk,
        "output_tokens_per_s": window_chunks * tokens_per_chunk / window if tokens_per_chunk else None,
        "requests_per_s": len(ok) / window,
        "ttft_breakdown": ttft_breakdown(ok),
        "slo": {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms},
        "slo_attainment": len(met) / max(len(reqs), 1),
        "slo_goodput_rps": len(met) / window,
        "slo_goodput_tokens_per_s": sum(r.get("completion_tokens") or 0 for r in met) / window,
        "by_kind": {k: latency_block([r for r in reqs if r["kind"] == k], gaps[k]) for k in KINDS
                    if any(r["kind"] == k for r in reqs)},
        "ramp_phase_requests": sum(raw["ramp_requests"] for raw in raws),
        "ramp_phase_errors": sum(raw["ramp_errors"] for raw in raws),
        "client": client,
    }


def summary_table(r: dict) -> Table:
    table = Table(title="InferTutor results (measured window)")
    for col in ["type", "requests", "errors", "TTFT p95", "ITL p95", "TPOT p95", "prompt tok", "output tok"]:
        table.add_column(col, justify="right")
    rows = [("all", {**r, "requests": r["total_requests"], "errors": r["total_errors"],
                     "prompt_tokens_mean": None, "completion_tokens_mean": None})]
    rows += list(r["by_kind"].items())
    for name, b in rows:
        table.add_row(
            name, str(b["requests"]), str(b["errors"]),
            f'{b["ttft_p95_ms"]:.0f} ms', f'{b["itl_p95_ms"]:.1f} ms', f'{b["tpot_p95_ms"]:.1f} ms',
            "" if b.get("prompt_tokens_mean") is None else f'{b["prompt_tokens_mean"]:.0f}',
            "" if b.get("completion_tokens_mean") is None else f'{b["completion_tokens_mean"]:.0f}',
        )
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--mode", choices=["text", "long", "image", "mixed"], default="mixed")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--arrival-rate", type=float, default=None,
                        help="Open-loop mode: requests per second (Poisson). Ignores --users.")
    parser.add_argument("--max-inflight", type=int, default=2000, help="Open-loop cap on open requests")
    parser.add_argument("--duration", type=int, default=60, help="Total seconds, including ramp-up")
    parser.add_argument("--ramp-up", type=int, default=15, help="Seconds of ramp-up, excluded from metrics")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--min-pause", type=float, default=0.2)
    parser.add_argument("--max-pause", type=float, default=1.2)
    parser.add_argument("--processes", type=int, default=1, help="Client processes to spread the load over")
    parser.add_argument("--drain-timeout", type=float, default=30.0,
                        help="Seconds to let in-flight requests finish after the window ends")
    parser.add_argument("--slo-ttft-ms", type=float, default=1000.0, help="SLO for time to first token")
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0, help="SLO for time per output token")
    parser.add_argument(
        "--unique-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vary image pixels per request so vLLM caches can't skip the vision encoder",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed per-user request sequences so repeated runs send identical requests")
    parser.add_argument("--label", default="manual")
    parser.add_argument("--total-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-count-source",
        choices=["runner-deploy", "user-supplied"],
        default="user-supplied",
        help="Set by run_infertutor_experiment.py when it deployed the endpoint itself",
    )
    parser.add_argument("--deployment-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help="Result file path (default: results_infertutor/<auto>.json)")
    args = parser.parse_args()
    if args.duration <= args.ramp_up:
        parser.error("--duration must be longer than --ramp-up (metrics are measured after ramp-up)")
    if args.processes < 1:
        parser.error("--processes must be at least 1")

    if args.processes == 1:
        t0 = time.time()
        raws = [asyncio.run(run_worker(args, 0, 1, t0, live=True))]
    else:
        # Start a few seconds out so every process is ready before t0.
        t0 = time.time() + 5
        console.print(f"Running {args.processes} client processes for {args.duration}s...")
        with ProcessPoolExecutor(args.processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_worker_entry, vars(args), i, args.processes, t0) for i in range(args.processes)]
            raws = [f.result() for f in futures]

    results = compute_results(raws, args)
    config = vars(args).copy()
    deployment = config.pop("deployment_json", None)
    config.pop("out", None)
    data = {
        "config": config,
        "deployment": json.loads(deployment) if deployment else None,
        "prompts_sha256": PROMPTS_SHA256,
        "results": results,
    }
    if args.out:
        out_file = Path(args.out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ROOT / "results_infertutor"
        out_dir.mkdir(exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in args.label)
        out_file = out_dir / f"{safe_label}_{args.mode}_{args.users}u_{int(time.time())}.json"
    out_file.write_text(json.dumps(data, indent=2))

    console.print(summary_table(results))
    console.print(
        f"Throughput {results['aggregate_stream_chunks_per_s']:.0f} chunks/s"
        + (f", {results['output_tokens_per_s']:.0f} tokens/s" if results["output_tokens_per_s"] else "")
        + f" · SLO met by {100 * results['slo_attainment']:.1f}% of requests"
    )
    if results["client"]["overloaded"]:
        console.print(
            f"[bold yellow]Warning: the load generator itself is overloaded "
            f"(loop lag p99 {results['client']['loop_lag_p99_ms']:.0f} ms, "
            f"CPU {100 * results['client']['cpu_util_max_process']:.0f}%). "
            f"Latencies include client delay; re-run with more --processes.[/bold yellow]"
        )
    console.print(f"[green]Saved {out_file}[/green]")


if __name__ == "__main__":
    main()
