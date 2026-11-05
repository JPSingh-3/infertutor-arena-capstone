"""Summarize per-replica vLLM metrics collected during a load test.

Each replica's sampler (modal_infertutor_app.py) writes JSON lines:
  {"type": "meta", ...versions, cold_start_s...}
  {"type": "sample", "t": <unix time>, "vllm": {metric: value}, "gpu": [...]}

summarize() restricts everything to the load tester's measured window, so
server-side numbers line up with the client-side ones.

Usage (manual):  python server_metrics.py <results.json> <metrics_dir>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# vLLM renames metrics between versions; try each candidate in order.
GAUGES = {
    "requests_running": ["vllm:num_requests_running"],
    "requests_waiting": ["vllm:num_requests_waiting"],
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
}
COUNTERS = {
    "prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "generation_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "prefix_cache_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits_total"],
    "prefix_cache_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries_total"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "requests_finished": ["vllm:request_success_total", "vllm:request_success"],
    "engine_steps": ["vllm:iteration_tokens_total_count"],
    "prefill_computed_tokens": ["vllm:request_prefill_kv_computed_tokens_sum"],
}
HISTOGRAMS = {
    "queue_time_s": ["vllm:request_queue_time_seconds"],
    "prefill_time_s": ["vllm:request_prefill_time_seconds"],
    "decode_time_s": ["vllm:request_decode_time_seconds"],
    "ttft_s": ["vllm:time_to_first_token_seconds"],
    "itl_s": ["vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"],
}


def load_replica_files(metrics_dir: Path) -> dict[str, dict]:
    replicas = {}
    for f in sorted(Path(metrics_dir).rglob("*.jsonl")):
        meta, samples = {}, []
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a partially written last line
            if rec.get("type") == "meta":
                meta = rec
            elif rec.get("type") == "sample" and rec.get("vllm"):
                samples.append(rec)
        replicas[meta.get("replica", f.stem)] = {"meta": meta, "samples": samples}
    return replicas


def _pick(values: dict, candidates: list[str]) -> float | None:
    for name in candidates:
        if name in values:
            return values[name]
    return None


def _hist(values: dict, candidates: list[str]) -> tuple[str, float, float, dict[float, float]] | None:
    for name in candidates:
        if f"{name}_count" in values:
            buckets = {}
            prefix = f"{name}_bucket|le="
            for k, v in values.items():
                if k.startswith(prefix):
                    le = k[len(prefix):]
                    buckets[float("inf") if le in ("+Inf", "inf") else float(le)] = v
            return name, values.get(f"{name}_sum", 0.0), values[f"{name}_count"], buckets
    return None


def _hist_delta(before: dict, after: dict, candidates: list[str]) -> dict | None:
    a, b = _hist(after, candidates), _hist(before, candidates) if before else None
    if a is None:
        return None
    _, sum_a, cnt_a, bk_a = a
    sum_b, cnt_b, bk_b = (b[1], b[2], b[3]) if b else (0.0, 0.0, {})
    count = cnt_a - cnt_b
    if count <= 0:
        return {"count": 0}
    deltas = sorted((le, bk_a[le] - bk_b.get(le, 0.0)) for le in bk_a)
    p50 = p95 = None
    for le, cum in deltas:  # cumulative buckets: first upper bound covering the quantile
        if p50 is None and cum >= 0.50 * count:
            p50 = le
        if p95 is None and cum >= 0.95 * count:
            p95 = le
    return {"count": count, "mean": (sum_a - sum_b) / count, "p50_le": p50, "p95_le": p95}


def summarize_replica(rep: dict, window_start: float, window_end: float) -> dict:
    samples = rep["samples"]
    inside = [s for s in samples if window_start <= s["t"] <= window_end]
    before = [s for s in samples if s["t"] < window_start]
    base = before[-1]["vllm"] if before else (inside[0]["vllm"] if inside else None)
    last = inside[-1] if inside else None
    out: dict = {"meta": rep["meta"], "samples_in_window": len(inside)}
    if not inside or last is None:
        return out
    span = max(last["t"] - (before[-1]["t"] if before else inside[0]["t"]), 1e-9)

    gauges = {}
    for key, names in GAUGES.items():
        vals = [v for v in (_pick(s["vllm"], names) for s in inside) if v is not None]
        if vals:
            gauges[key] = {"mean": statistics.mean(vals), "max": max(vals)}
    out["gauges"] = gauges

    counters = {}
    for key, names in COUNTERS.items():
        a, b = _pick(last["vllm"], names), _pick(base, names) if base else None
        if a is not None:
            counters[key] = a - (b or 0.0)
    out["counters"] = counters
    if "generation_tokens" in counters:
        out["generation_tokens_per_s"] = counters["generation_tokens"] / span
    if counters.get("prefix_cache_queries"):
        out["prefix_cache_hit_rate"] = counters.get("prefix_cache_hits", 0.0) / counters["prefix_cache_queries"]

    out["histograms"] = {k: h for k, names in HISTOGRAMS.items() if (h := _hist_delta(base, last["vllm"], names))}

    if counters.get("engine_steps"):
        out["step_time_ms"] = 1000 * span / counters["engine_steps"]
    if "requests_finished" in counters:
        out["requests_per_s"] = counters["requests_finished"] / span

    # CPU cores used, per process name (API server, VLLM::EngineCore, ...).
    first_cpu = (before[-1] if before else inside[0]).get("cpu_s")
    last_cpu = last.get("cpu_s")
    if first_cpu is not None and last_cpu:
        by_proc = {name: (secs - first_cpu.get(name, 0.0)) / span for name, secs in last_cpu.items()}
        by_proc = {k: v for k, v in by_proc.items() if v > 0.01}
        out["cpu_cores_used"] = sum(by_proc.values())
        out["cpu_cores_by_process"] = dict(sorted(by_proc.items(), key=lambda kv: -kv[1])[:6])

    gpu_rows = [g for s in inside for g in s.get("gpu", [])]
    if gpu_rows:
        out["gpu"] = {
            "util_pct_mean": statistics.mean(g["util_pct"] for g in gpu_rows),
            "mem_used_mib_max": max(g["mem_used_mib"] for g in gpu_rows),
            "power_w_mean": statistics.mean(g["power_w"] for g in gpu_rows),
        }
    return out


def summarize(metrics_dir: Path, window_start: float, window_end: float) -> dict:
    replicas = load_replica_files(metrics_dir)
    per = {name: summarize_replica(r, window_start, window_end) for name, r in replicas.items()}
    live = [p for p in per.values() if p.get("samples_in_window")]
    agg: dict = {"replicas_reporting": len(live), "replicas_total": len(per)}
    if live:
        agg["generation_tokens_per_s"] = sum(p.get("generation_tokens_per_s", 0.0) for p in live)
        hits = sum(p.get("counters", {}).get("prefix_cache_hits", 0.0) for p in live)
        queries = sum(p.get("counters", {}).get("prefix_cache_queries", 0.0) for p in live)
        if queries:
            agg["prefix_cache_hit_rate"] = hits / queries
        agg["preemptions"] = sum(p.get("counters", {}).get("preemptions", 0.0) for p in live)
        for key in GAUGES:
            maxes = [p["gauges"][key]["max"] for p in live if key in p.get("gauges", {})]
            means = [p["gauges"][key]["mean"] for p in live if key in p.get("gauges", {})]
            if maxes:
                agg[f"{key}_max"] = max(maxes)
                agg[f"{key}_mean_per_replica"] = statistics.mean(means)
        for key in HISTOGRAMS:
            hs = [p["histograms"][key] for p in live if p.get("histograms", {}).get(key, {}).get("count")]
            if hs:
                total = sum(h["count"] for h in hs)
                agg[f"{key}_mean"] = sum(h["mean"] * h["count"] for h in hs) / total
                agg[f"{key}_p95_le_worst_replica"] = max(h["p95_le"] for h in hs if h["p95_le"] is not None)
        # Load balance: share of generated tokens per replica (1/N each is perfect).
        gens = [p.get("counters", {}).get("generation_tokens", 0.0) for p in live]
        if sum(gens):
            agg["load_share_per_replica"] = [g / sum(gens) for g in gens]
        # Share of requests separates "fewer requests" from "shorter requests".
        reqs = [p.get("counters", {}).get("requests_finished", 0.0) for p in live]
        if sum(reqs):
            agg["request_share_per_replica"] = [r / sum(reqs) for r in reqs]
        steps = [p["step_time_ms"] for p in live if "step_time_ms" in p]
        if steps:
            agg["step_time_ms_per_replica"] = steps
        cpu = [p["cpu_cores_used"] for p in live if "cpu_cores_used" in p]
        if cpu:
            agg["cpu_cores_used_per_replica"] = cpu
        gpu = [p["gpu"]["util_pct_mean"] for p in live if "gpu" in p]
        if gpu:
            agg["gpu_util_pct_mean"] = statistics.mean(gpu)
    colds = [p["meta"]["cold_start_s"] for p in per.values() if p.get("meta", {}).get("cold_start_s") is not None]
    if colds:
        agg["cold_start_s_max"] = max(colds)
    metas = [p["meta"] for p in per.values() if p.get("meta")]
    if metas:
        agg["versions"] = {k: sorted({str(m.get(k)) for m in metas}) for k in ("vllm_version", "torch_cuda", "gpu")}
        agg["cpu_models"] = sorted({str(m.get("cpu", {}).get("model")) for m in metas if m.get("cpu")})
        startup = [m.get("startup") for m in metas if m.get("startup")]
        if startup:
            agg["startup"] = startup[0]
        probe = [m.get("image_token_probe") for m in metas if m.get("image_token_probe")]
        if probe:
            agg["image_token_probe"] = probe[0]
    return {"aggregate": agg, "per_replica": per}


def attach_to_results(results_path: Path, metrics_dir: Path) -> dict:
    data = json.loads(Path(results_path).read_text())
    r = data["results"]
    summary = summarize(metrics_dir, r["window_start_unix"], r["window_end_unix"])
    data["server_metrics"] = summary
    Path(results_path).write_text(json.dumps(data, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(attach_to_results(Path(sys.argv[1]), Path(sys.argv[2]))["aggregate"], indent=2))
