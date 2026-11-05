# InferTutor Arena: Inference Engineering Capstone

Welcome to **InferTutor Arena**, the final capstone project for the Inference Engineering workshop.

You will deploy a real multimodal LLM serving system on Modal using vLLM, load test it under concurrent traffic, measure production inference metrics, and improve the system through careful engineering. The goal is not to write a toy script. The goal is to think like an inference engineer: read metrics, form hypotheses, change serving parameters, and justify the result.

## What You Will Build

You will launch **InferTutor**, a personalized AI tutor for inference engineering students. Users ask questions about KV cache behavior, batching, vLLM, latency spikes, GPU bottlenecks, and simple uploaded diagrams. Your system must serve those users quickly and reliably.

The assignment uses:

- **Model:** `Qwen/Qwen3-VL-4B-Instruct`
- **Serving engine:** vLLM OpenAI-compatible server
- **Deployment platform:** Modal GPU containers
- **Primary GPU target:** H100
- **Main workload:** fixed text, long-context, image, and mixed tutor prompts
- **Core metrics:** TTFT, ITL, throughput, request rate, error rate, GPU efficiency

## Why This Model

We chose `Qwen/Qwen3-VL-4B-Instruct` because it is:

- Multimodal, so the capstone feels like a real product rather than a synthetic text benchmark.
- Small enough to iterate on quickly with 1-4 H100 GPUs.
- Supported by recent vLLM releases.
- Large enough to expose real serving tradeoffs around batching, prefix caching, chunked prefill, compiled mode, replicas, image preprocessing, and concurrency.

## Repository Contents

| Path | What it is |
|---|---|
| [`InferTutor_Arena_Capstone.pdf`](./InferTutor_Arena_Capstone.pdf) | Main student assignment PDF: goals, rules, tracks, metrics, baselines, required submission |
| [`Modal_vLLM_Runbook.pdf`](./Modal_vLLM_Runbook.pdf) | Setup guide: what Modal is, how vLLM runs, cost estimates, commands, cleanup |
| [`InferTutor_Arena_Capstone.md`](./InferTutor_Arena_Capstone.md) | Editable source for the main assignment PDF |
| [`Modal_vLLM_Runbook.md`](./Modal_vLLM_Runbook.md) | Editable source for the setup/runbook PDF |
| [`render_pdfs.py`](./render_pdfs.py) | Regenerates both PDFs from Markdown |
| [`starter_code/`](./starter_code) | Runnable Modal + vLLM starter implementation |

## Starter Code

The [`starter_code`](./starter_code) folder contains:

| File | Purpose |
|---|---|
| [`modal_infertutor_app.py`](./starter_code/modal_infertutor_app.py) | Modal app that launches vLLM on GPU |
| [`run_infertutor_experiment.py`](./starter_code/run_infertutor_experiment.py) | One-command deploy + benchmark runner |
| [`load_test_infertutor.py`](./starter_code/load_test_infertutor.py) | Async streaming load tester |
| [`score_infertutor.py`](./starter_code/score_infertutor.py) | Local leaderboard scorer |
| [`server_metrics.py`](./starter_code/server_metrics.py) | Summarizes per-replica vLLM and GPU metrics for each run |
| [`prompts.json`](./starter_code/prompts.json) | Fixed official prompt set |
| [`requirements.txt`](./starter_code/requirements.txt) | Local Python dependencies |
| [`README.md`](./starter_code/README.md) | Quick-start commands for the code |

The starter code is intentionally complete. You should not spend your capstone time fighting Modal boilerplate. Your job is to improve the serving configuration.

## Competition Goal

Maximize:

```text
Score =
  goodput_tokens_per_second
  * sustained_users
  * quality_pass_rate
  * (1 - error_rate)
  / (p95_TTFT_seconds * p95_ITL_seconds * total_GPU_count)
```

The starter harness reports streamed content chunks per second. The final evaluator may tokenize outputs to compute exact tokens per second, but the optimization logic is the same.

You are rewarded for:

- Low p95 time to first token.
- Low p95 inter-token latency.
- High throughput.
- High sustained concurrency.
- Low error rate.
- Efficient GPU use.

You are penalized for:

- Dropping requests.
- Using extra GPUs without proportional improvement.
- Improving text speed while breaking multimodal traffic.
- Reporting averages while hiding bad p95 behavior.

## Tracks

**Note on baselines:** the 1-, 2- and 4-GPU numbers below were re-measured on 2026-11-03 with the current harness (H100, vLLM 0.21.0, 0 errors in ~15,600 requests), with the load generator on a laptop over the internet. ITL p95 here is measured per token at the client, so it includes network bunching: vLLM itself spaced tokens about 5 ms apart in compiled mode and 22-26 ms in eager mode. The 8-GPU table is from the earlier dry run with the old harness. Notice that the 4-replica mixed run does worse than the 2-replica one; finding out why is a good first experiment (look at `server_metrics`).

### 1. Multimodal Product Track

This is the main capstone track.

Use:

```bash
--mode mixed
```

Recommended budget:

```text
Up to 4 H100 GPUs
```

Reference baseline (re-measured 2026-11-03):

| Config | Users | GPUs | Errors | TTFT p95 | ITL p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| eager, 4 replicas, seq32/b4096 | 120 | 4 | 0.0% | 2,655 ms | 124.4 ms | 2,618 chunks/s |
| eager, 2 replicas, seq32/b4096 | 100 | 2 | 0.0% | 1,125 ms | 91.4 ms | 2,695 chunks/s |

### 2. Text Speed Track

This is the pure serving optimization track.

Use:

```bash
--mode text
```

Recommended budget:

```text
Up to 4 H100 GPUs
```

Reference baseline (re-measured 2026-11-03):

| Config | Users | GPUs | Errors | TTFT p95 | ITL p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| compiled, 4 replicas, seq32/b4096 | 400 | 4 | 0.0% | 1,167 ms | 30.5 ms | 18,476 chunks/s |
| compiled, 1 replica, seq32/b4096 | 100 | 1 | 0.0% | 1,786 ms | 38.5 ms | 3,349 chunks/s |

### Optional Boss Fight

You may try up to 8 H100 GPUs if quota is available. In our internal dry run, 8 GPUs produced strong throughput but errors appeared at high user counts. This is part of the lesson: GPU compute is not the only bottleneck.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/VizuaraAI/infertutor-arena-capstone.git
cd infertutor-arena-capstone/starter_code
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Modal:

```bash
modal token new
```

Create the Hugging Face secret expected by the Modal app:

```bash
modal secret create huggingface HF_TOKEN=<YOUR_HF_TOKEN>
```

Run a tiny smoke test:

```bash
python run_infertutor_experiment.py \
  --label smoke \
  --gpu-type H100 \
  --replicas 1 \
  --mode text \
  --users 5 \
  --duration 30 \
  --ramp-up 5 \
  --max-tokens 64
```

If the smoke test passes with zero errors, the infrastructure is working.

## Example Runs

### Single-GPU Baseline

```bash
python run_infertutor_experiment.py \
  --label baseline-text \
  --gpu-type H100 \
  --replicas 1 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 50 \
  --duration 60 \
  --ramp-up 15 \
  --max-tokens 96
```

### Main Multimodal Baseline

```bash
python run_infertutor_experiment.py \
  --label mixed-r2 \
  --gpu-type H100 \
  --replicas 2 \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode mixed \
  --users 100 \
  --duration 90 \
  --ramp-up 25 \
  --max-tokens 96
```

### Text Speed Baseline

```bash
python run_infertutor_experiment.py \
  --label compiled-r4 \
  --gpu-type H100 \
  --replicas 4 \
  --no-fast-boot \
  --max-seqs 32 \
  --max-batch-tokens 4096 \
  --mode text \
  --users 400 \
  --duration 90 \
  --ramp-up 40 \
  --max-tokens 96
```

## What You Should Tune

Start with:

- `--replicas`
- `--max-seqs`
- `--max-batch-tokens`
- `--prefix-cache` / `--no-prefix-cache`
- `--chunked-prefill` / `--no-chunked-prefill`
- `--fast-boot` / `--no-fast-boot`
- `--concurrent-inputs`
- `--mm-max-pixels` (the official test image is 1024x768 = 786k pixels, so caps below that resize it)
- `--cpu` / `--memory-mb`: CPU cores and memory reserved per container (defaults 4 and 32768; measured use is 1.5–2.8 cores). vLLM needs at least 2 + GPUs physical cores; `--cpu 0` falls back to Modal's default of 0.125 reserved cores.

Measurement options (they don't change the server):

- `--processes N`: spread the load over N client processes. Use it if a run warns that the load generator is overloaded.
- `--arrival-rate R`: open-loop traffic at R requests/s instead of a fixed user count. It shows true capacity; it's diagnostic and not scored.
- `--slo-ttft-ms` / `--slo-tpot-ms`: latency targets for SLO goodput (defaults 1000 ms / 50 ms).
- `--seed N`: every simulated user sends the same request sequence in every run, so repeated runs are comparable.

Each result also records a TTFT breakdown (`ttft_breakdown`: client wait until response headers, server time from container ingress to headers, the difference as network/proxy time, and headers to first token). The server metrics add CPU cores used per vLLM process, requests and step time per replica, the container's CPU model, vLLM's startup facts (KV cache size, compile and CUDA-graph capture time) and an exact image-token count from vLLM's processor.

Keep notes. A good capstone submission explains not just the final number, but why the final configuration won.

## Scoring Your Runs

Results are saved under:

```text
starter_code/results_infertutor/
```

Score them:

```bash
python score_infertutor.py results_infertutor
```

## Cleanup

Modal charges while GPU containers are active or warming. Each experiment deploys its own app with `min_containers = --replicas`, so its GPUs never scale down on their own. `run_infertutor_experiment.py` stops the app automatically at the end of each run (pass `--keep-app` to keep it). If a run was interrupted, check for leftovers:

```bash
modal app list
modal app stop <APP_ID_OR_NAME>
```

## Regenerating PDFs

From the repository root:

```bash
python render_pdfs.py
```

This regenerates:

- `InferTutor_Arena_Capstone.pdf`
- `Modal_vLLM_Runbook.pdf`

## Submission Checklist

Submit:

- Final benchmark JSON.
- Exact command used for the final run.
- One-page engineering report.
- Table of at least five experiments.
- Explanation of the best configuration.
- One surprising failure or tradeoff.

The point of this capstone is simple: prove that you can operate a real inference system, measure it honestly, and make it better.
