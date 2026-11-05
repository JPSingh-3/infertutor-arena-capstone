"""Score InferTutor benchmark result JSON files.

This is a LOCAL, unofficial score. It does not compute quality_pass_rate
(the instructor's evaluator does), and it flags runs whose settings differ
from the official benchmark so they can't be compared like-for-like.
"""

import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


console = Console()
ROOT = Path(__file__).parent
CURRENT_PROMPTS_SHA256 = hashlib.sha256((ROOT / "prompts.json").read_bytes()).hexdigest()

# Settings every official run must use. Changing them changes the workload.
OFFICIAL = {
    "max_tokens": 96,
    "min_pause": 0.2,
    "max_pause": 1.2,
    "request_timeout": 180,
    "unique_images": True,
}
TRACK_MODES = {"text", "mixed"}
MIN_WINDOW_S = 30
TRACK_GPU_BUDGET = 4  # both tracks
BOSS_GPU_BUDGET = 8   # optional boss fight


def score(data: dict) -> float:
    config = data["config"]
    result = data["results"]
    users = config.get("users", 1)
    total_gpus = max(config.get("total_gpus", 1), 1)
    error_rate = result.get("error_rate", 1)
    goodput = result.get("aggregate_stream_chunks_per_s", 0) * max(0, 1 - error_rate)
    ttft = max(result.get("ttft_p95_ms", 1) / 1000, 0.001)
    itl = max(result.get("itl_p95_ms", 1) / 1000, 0.001)
    # quality_pass_rate is omitted here; the official evaluator applies it.
    return goodput * users / (ttft * itl * total_gpus)


def problems(data: dict) -> list[str]:
    """Reasons a run is not comparable to official runs. Empty list = OK."""

    config = data.get("config", {})
    result = data.get("results", {})
    issues = []
    if result.get("metrics_version") != 3:
        return ["older harness version; re-run"]
    if data.get("prompts_sha256") != CURRENT_PROMPTS_SHA256:
        issues.append("prompt set differs from prompts.json")
    if config.get("mode") not in TRACK_MODES:
        issues.append(f"mode={config.get('mode')} is diagnostic, not a track")
    for key, expected in OFFICIAL.items():
        if config.get(key) != expected:
            issues.append(f"{key}={config.get(key)} (official {expected})")
    gpus = config.get("total_gpus", 1)
    if gpus > BOSS_GPU_BUDGET:
        issues.append(f"{gpus} GPUs exceeds the {BOSS_GPU_BUDGET}-GPU maximum")
    elif gpus > TRACK_GPU_BUDGET:
        issues.append(f"{gpus} GPUs: boss fight only (track budget is {TRACK_GPU_BUDGET})")
    if config.get("arrival_rate"):
        issues.append("open-loop run (diagnostic; score assumes closed-loop users)")
    if result.get("client", {}).get("overloaded"):
        issues.append("load generator overloaded; re-run with more --processes")
    if config.get("gpu_count_source") != "runner-deploy":
        issues.append("GPU count self-reported (endpoint not deployed by runner)")
    if result.get("measured_window_s", 0) < MIN_WINDOW_S:
        issues.append(f"measured window < {MIN_WINDOW_S}s")
    return issues


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        console.print("Usage: python score_infertutor.py [results_file_or_directory]")
        return

    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results_infertutor")
    if not path.exists():
        console.print(f"[red]No result path found:[/red] {path}")
        return

    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        console.print(f"[red]No JSON results found in:[/red] {path}")
        return

    rows = []
    for file in files:
        try:
            data = json.loads(file.read_text())
            rows.append((score(data), problems(data), file, data))
        except (json.JSONDecodeError, KeyError) as exc:
            console.print(f"[yellow]Skipping {file.name}: {exc}[/yellow]")
    # Comparable runs first, then by score.
    rows.sort(key=lambda r: (len(r[1]) == 0, r[0]), reverse=True)

    table = Table(title="InferTutor Leaderboard (local, excludes quality_pass_rate)")
    for col in ["file", "ok", "mode", "users", "gpus", "err%", "TTFT p95", "ITL p95", "chunks/s", "tok/s", "SLO%", "score"]:
        table.add_column(col, justify="left" if col == "file" else "right")

    for value, issues, file, data in rows:
        c = data["config"]
        r = data["results"]
        table.add_row(
            file.name,
            "[green]yes[/green]" if not issues else "[red]no[/red]",
            c.get("mode", ""),
            str(c.get("users", "")),
            str(c.get("total_gpus", 1)),
            f'{100 * r.get("error_rate", 0):.1f}',
            f'{r.get("ttft_p95_ms", 0):.0f} ms',
            f'{r.get("itl_p95_ms", 0):.1f} ms',
            f'{r.get("aggregate_stream_chunks_per_s", 0):.0f}',
            f'{r["output_tokens_per_s"]:.0f}' if r.get("output_tokens_per_s") else "-",
            f'{100 * r["slo_attainment"]:.0f}' if "slo_attainment" in r else "-",
            f"{value:.0f}",
        )

    console.print(table)
    flagged = [(file, issues) for _, issues, file, _ in rows if issues]
    if flagged:
        console.print("\n[bold]Runs not comparable to official settings:[/bold]")
        for file, issues in flagged:
            console.print(f"  {file.name}: " + "; ".join(issues))


if __name__ == "__main__":
    main()
