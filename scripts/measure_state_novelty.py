"""GP1 v2: is progress visible in state space where behaviour hid it?

The v1 measurement (measure_progress_signal.py) returned an informative null:
humans do not converge behaviourally within a level, so surface statistics
cannot mark progress. This probe drops below behaviour to the boards actions
produce. The quantity is the NOVELTY RATE — the fraction of actions whose
resulting board, chrome-masked, has not been seen before within the level —
using the same isolation-learned chrome mask the determinism work validated
(a cell is chrome once it changes while at most 2 cells change in total),
via the FrontierMap digest machinery.

Hypothesis: humans sustain novelty to the end of a level (every action still
moves the world somewhere new), while the agent's stalled tail shows novelty
collapse (wandering among near-duplicate states without exact cycling, which
we already know is rare at 2.3%). If true, falling masked-state novelty is an
online progress-loss signal, and it is computable by the harness from data it
already holds.

Cohorts: human levels (engine replay, all completed); agent completed levels;
agent stalled tails. Thirds within each segment, plus the terminal 20%%.

    python scripts/measure_state_novelty.py <agent_run_dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAYS = ROOT / "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor/taaf-kaggle-source/src/re-arc-3"))

from duck_delta.frontier_map import FrontierMap  # noqa: E402


def novelty_seq(boards: list) -> list[int]:
    """1 if board i is novel within the segment under the evolving chrome mask."""
    fm = FrontierMap()
    seen: set[str] = set()
    out = []
    prev = None
    for b in boards:
        if prev is not None:
            fm._learn_chrome(prev, b)
        d = fm._digest(b)
        out.append(0 if d in seen else 1)
        seen.add(d)
        prev = b
    return out[1:]  # novelty of the *result* of each action


def thirds_rates(seqs: list[list[int]]):
    rows = []
    for idx in range(4):
        vals = []
        for s in seqs:
            n = len(s)
            if n < 6:
                continue
            if idx < 3:
                a, b = (n * idx) // 3, (n * (idx + 1)) // 3
                part = s[a:b]
            else:
                part = s[-max(1, n // 5):]
            if part:
                vals.append(sum(part) / len(part))
        rows.append((sum(vals) / len(vals) if vals else math.nan, len(vals)))
    return rows


def human_board_segments():
    from re_arc import EnvSampler
    from arcengine import GameAction
    import numpy as np
    by_value = {m.value: m for m in GameAction}
    segs = []
    for p in sorted(REPLAYS.glob("*.json")):
        data = json.loads(p.read_text())
        gid = data["game_id"]
        try:
            env = EnvSampler(include=[gid]).make(gid)
        except Exception:
            continue
        obs = env.reset()
        for actions in data["levels"]:
            boards = [np.array(obs.frame[-1]).tolist()]
            for value, adata in actions:
                ga = by_value.get(int(value))
                if ga is None:
                    break
                obs, _r, _d, _i = env.step(ga, adata or None)
                boards.append(np.array(obs.frame[-1]).tolist())
            segs.append(boards)
    return segs


def agent_board_segments(run_dir: Path):
    from load_recorded_states import load
    import numpy as np
    bm = json.loads((run_dir / "benchmark.json").read_text())
    games_states = load(str(run_dir / "intermediate_states.pkl"))
    by_gid = {}
    for states in games_states:
        gid = getattr(states[0].raw, "game_id", None) if states else None
        if gid:
            by_gid[str(gid)] = states
    completed, stalled = [], []
    for g in bm["game_runs"]:
        states = by_gid.get(str(g.get("game_id")))
        if not states:
            continue
        boards = [np.array(s.raw.frame)[-1].tolist() for s in states]
        apl = g.get("actions_per_level") or []
        done = int(g.get("levels_completed") or 0)
        off = 0
        for k in range(done):
            n = int(apl[k] or 0)
            completed.append(boards[off:off + n + 1])
            off += n
        if off + 1 < len(boards):
            stalled.append(boards[off:])
    return completed, stalled


def show(label, rows):
    print(f"\n{label}")
    names = ("early", "mid", "late", "last20%")
    print(f"  {'phase':<9}{'novelty rate':>13}{'n':>6}")
    for name, (v, n) in zip(names, rows):
        print(f"  {name:<9}{v:>13.1%}{n:>6}")


def main() -> None:
    run_dir = Path(sys.argv[1])
    hs = human_board_segments()
    comp, stall = agent_board_segments(run_dir)
    print(f"human levels: {len(hs)} | agent completed: {len(comp)} | stalled: {len(stall)}")
    show("HUMAN (all levels)", thirds_rates([novelty_seq(b) for b in hs]))
    show("AGENT, completed levels", thirds_rates([novelty_seq(b) for b in comp]))
    show("AGENT, stalled tails", thirds_rates([novelty_seq(b) for b in stall]))


if __name__ == "__main__":
    main()
