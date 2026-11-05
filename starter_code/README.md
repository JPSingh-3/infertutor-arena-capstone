# InferTutor Arena Starter Code

This folder contains the starter code for the capstone.

The goal is to remove infrastructure friction. You should spend your time on inference-engineering decisions, not on figuring out how to deploy vLLM.

## Files

| File | Purpose |
|---|---|
| `modal_infertutor_app.py` | Modal app that launches vLLM as an OpenAI-compatible server |
| `run_infertutor_experiment.py` | One-command deploy + load-test runner |
| `load_test_infertutor.py` | Async streaming load tester for text, long, image, and mixed workloads |
| `score_infertutor.py` | Summarizes result JSONs and computes a leaderboard score |
| `server_metrics.py` | Summarizes the per-replica vLLM/GPU metrics the runner downloads after each run |
| `prompts.json` | Fixed official prompt set |
| `requirements.txt` | Local Python dependencies |

## Quick Start

```bash
pip install -r requirements.txt
modal token new
modal secret create huggingface HF_TOKEN=<YOUR_HF_TOKEN>
```

Smoke test:

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

Main multimodal baseline:

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

Text speed baseline:

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

Score results:

```bash
python score_infertutor.py results_infertutor
```

The runner stops each experiment's Modal app when the run ends (use `--keep-app` to keep it for reuse with `--url`). If a run was interrupted, stop leftovers manually:

```bash
modal app list
modal app stop <APP_ID_OR_NAME>
```

