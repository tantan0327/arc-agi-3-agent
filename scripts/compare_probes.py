"""Compare throughput probes: turns per game are the reading, levels are noise.

    python scripts/compare_probes.py <dir> [<dir> ...]

Each dir is a `kaggle kernels output` directory of an arc3-probe-* run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def read(d: Path) -> dict:
    out = {"name": d.name.replace("arc3-probe-", "").replace("_out", "")}
    bm = json.loads((d / "benchmark.json").read_text()) if (d / "benchmark.json").exists() else {"game_runs": []}
    runs = bm.get("game_runs", [])
    out["games"] = len(runs)
    out["levels"] = sum(int(r.get("levels_completed") or 0) for r in runs)
    out["actions"] = sum(len(r.get("history") or []) for r in runs)
    out["seconds"] = max([float(r.get("final_wallclock_seconds") or 0) for r in runs] or [0])
    p = d / "vllm-metrics-final.prom"
    m = {}
    if p.exists():
        text = p.read_text()
        for key in ("prompt_tokens_total", "generation_tokens_total", "time_to_first_token_seconds_sum",
                    "time_to_first_token_seconds_count", "request_queue_time_seconds_sum",
                    "request_queue_time_seconds_count", "prefix_cache_hits_total", "prefix_cache_queries_total"):
            hit = re.search(rf"^vllm:{key}\{{[^}}]*\}}\s+([0-9.e+-]+)$", text, re.M)
            if hit:
                m[key] = float(hit.group(1))
    out["calls"] = m.get("time_to_first_token_seconds_count", 0)
    out["prompt_tokens"] = m.get("prompt_tokens_total", 0)
    out["gen_tokens"] = m.get("generation_tokens_total", 0)
    out["prompt_per_call"] = out["prompt_tokens"] / out["calls"] if out["calls"] else 0
    out["ttft"] = m.get("time_to_first_token_seconds_sum", 0) / out["calls"] if out["calls"] else 0
    out["queue"] = (m.get("request_queue_time_seconds_sum", 0) /
                    m.get("request_queue_time_seconds_count", 1)) if m.get("request_queue_time_seconds_count") else 0
    q = m.get("prefix_cache_queries_total", 0)
    out["cache_hit_rate"] = (m.get("prefix_cache_hits_total", 0) / q) if q else 0.0
    return out


def main() -> None:
    rows = [read(Path(d)) for d in sys.argv[1:]]
    if not rows:
        sys.exit("give me probe output directories")
    base = rows[0]
    cols = [("name", "{:<12}"), ("levels", "{:>6.0f}"), ("actions", "{:>8.0f}"), ("calls", "{:>7.0f}"),
            ("prompt_per_call", "{:>9.0f}"), ("gen_tokens", "{:>10.0f}"), ("ttft", "{:>7.1f}"),
            ("queue", "{:>7.1f}"), ("cache_hit_rate", "{:>7.1%}")]
    print("".join(h.rjust(10) if h != "name" else h.ljust(12) for h, _ in cols))
    for r in rows:
        print("".join(f.format(r[k]) for k, f in cols))
    print("\nrelative to the first row (higher is better for calls and actions):")
    for r in rows[1:]:
        for k in ("calls", "actions", "prompt_per_call", "ttft"):
            if base[k]:
                print(f"  {r['name']:<12} {k:<16} x{r[k]/base[k]:.2f}")
        print()


if __name__ == "__main__":
    main()
