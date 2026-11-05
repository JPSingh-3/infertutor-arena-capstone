# InferTutor Arena: benchmark results

Fork of [VizuaraAI/infertutor-arena-capstone](https://github.com/VizuaraAI/infertutor-arena-capstone) with a fixed harness and 44 measured load tests of **Qwen3-VL-4B-Instruct on vLLM 0.21 / Modal H100s** (86,336 requests, 0 errors).

**Full report with charts:** [docs/index.html](docs/index.html) (GitHub Pages: `https://JPSingh-3.github.io/infertutor-arena-capstone/`)

## Key results (single runs unless noted; differences under ~10% may be noise)

| Change | Effect |
|---|---|
| Default execution mode (torch.compile + CUDA graphs) vs `--enforce-eager` | 2× throughput on mixed traffic, 2.8× on text (1 H100) |
| `max-num-seqs` 32 → 128 | +35% throughput (3,709 tok/s), engine ITL 11 → 20 ms |
| Modal `concurrent_inputs` 64 → 32 on 4 replicas | Balanced replicas, TTFT p95 up to 2.9 s → 0.4–0.5 s (2 runs; mechanism is a hypothesis) |
| Modal `concurrent_inputs` 64 → 160, text, 1 GPU | +42% (5,667 tok/s, $0.19 per 1M output tokens) |
| Prefix cache, chunked prefill, pixel cap, image cache, token budget, FP8, vLLM 0.33 | Within about ±13% under 150 closed-loop users |
| CPU reservation 0.125 vs 8 cores | No effect; vLLM used 1.5–2.8 cores |

## What changed in the harness

See commit message on branch `harness-fixes`. The most important fix: several experiment flags were not passed to the generated Modal app, so earlier ablations compared identical server configs.

## Reproduce

```bash
cd starter_code
python run_infertutor_experiment.py --gpu-type H100 --replicas 1 --mode mixed --users 100 \
  --duration 90 --ramp-up 25 --max-tokens 96 --seed 1 --label my-run
```

`results/` holds the summary JSON for every run (client latency plus per-replica vLLM metrics).

## Limitations

Closed-loop load only, mostly single runs, laptop client over the internet, roofline numbers are analytic estimates. See section 9 of the report.
