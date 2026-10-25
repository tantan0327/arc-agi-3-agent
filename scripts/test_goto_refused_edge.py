"""Unit test for the goto tool's refused-edge learning, on a synthetic engine.

The 9/20 full run (sp80) showed the failure this guards against: the body's UP
was refused by an invisible boundary (floor-coloured cells ahead), the tool
learned nothing, re-planned the same move and retried it three times while the
level timer ran out. A ticking HUD cell hid the refusal from the old
"nothing changed on the board" test as well.

Three checks, no sandbox needed (the tool's helpers read the module globals):
  1. with a gap in the invisible wall, goto reaches the target and never
     repeats an identical (position, keys) attempt;
  2. with no gap, goto stops after two invisible refusals and says so, instead
     of burning _MAX_REPLANS x chunk actions;
  3. _learn derives the refused edge from history alone (so a fresh sandbox
     process, which is what every python-tool call is, inherits it).

    python3 scripts/test_goto_refused_edge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from duck_delta import goto_tool as GT  # noqa: E402

H = W = 24; FLOOR = 0; WALL = 3; BODY = 5; STEP = 2
BAND = (11, 12)           # the invisible wall: rows 11-12 cannot be entered from below except through the gap


class F:
    def __init__(self, grid, level=1): self._grid = grid; self.level = level


class E:
    def __init__(self, action, grid): self.action = action; self.frame = F(grid)


class Engine:
    def __init__(self, top, left, gap_cols):
        self.top, self.left, self.gap = top, left, gap_cols; self.tick = 0; self.hist = []

    def grid(self):
        g = [[FLOOR] * W for _ in range(H)]
        for i in range(H):
            g[i][0] = g[i][W - 1] = WALL
        for j in range(W):
            g[0][j] = g[H - 1][j] = WALL
        g[0][0] = 7 + (self.tick % 2)                       # a HUD cell that ticks every action
        for oy in (0, 1):
            for ox in (0, 1):
                g[self.top + oy][self.left + ox] = BODY
        return g

    def step(self, key):
        dy, dx = GT._DIRS[key]; nt, nl = self.top + dy * STEP, self.left + dx * STEP
        ok = 1 <= nt and nt + 1 <= H - 2 and 1 <= nl and nl + 1 <= W - 2
        crosses = self.top > BAND[1] and nt <= BAND[1] and self.left not in self.gap
        self.tick += 1
        if ok and not crosses:
            self.top, self.left = nt, nl
        self.hist.append((key, self.grid()))
        return ok and not crosses


def install(engine, history_keys):
    """Play `history_keys` on the engine as the recorded history, then wire the tool's globals."""
    for k in history_keys:
        engine.step(k)
    GT.history = [E(k, g) for k, g in engine.hist]
    GT.current_frame = F(engine.grid())
    GT._CACHE["n"] = -1; GT._CACHE["model"] = None

    def action(actions):
        changed = False
        for a in actions:
            changed = engine.step(str(a["action"]).upper()) or changed
        GT.history = [E(k, g) for k, g in engine.hist]
        GT.current_frame = F(engine.grid())
        return {"executed": True, "executed_count": len(actions), "board_changed": changed, "level": 1,
                "level_completed": False, "game_over": False, "run_complete": False}
    GT.action = action


def no_repeats(trace):
    seen = set()
    for t in trace:
        key = (tuple(t["from"]), t["keys"])
        if key in seen:
            return False
        seen.add(key)
    return True


def main():
    warm = ["RIGHT", "RIGHT", "UP", "LEFT", "DOWN", "RIGHT"]        # six moves that change the board

    # 1. a gap exists, far from the target: each call probes at most _MAX_INVISIBLE
    #    edges (one action each), hands the board back, and the next call inherits the
    #    refused edges from history -- so repeated calls converge without ever
    #    retrying a refused move.
    eng = Engine(top=18, left=4, gap_cols={16, 17}); install(eng, warm)
    m = GT._model(); assert m is not None and m["mover"] == BODY, m
    traces = []; total = 0; calls = 0; r = None
    for calls in range(1, 7):
        r = GT.goto(4, 4); traces += r["trace"]; total += r["actions_taken"]
        if r.get("reached"):
            break
        assert "refused" in str(r.get("reason", "")), r
        for t in r["trace"]:                      # a refusal costs exactly one action
            if t.get("refused_at"):
                assert t["executed"] == 1, t
    assert r.get("reached"), r
    assert no_repeats(traces), traces
    assert total <= 40, total
    print(f"1 gap:    reached after {calls} calls / {total} actions; refused edges now {len(GT._model().get('refused', ()))}")

    # 2. no gap: stop early with the reason, not after _MAX_REPLANS retries
    eng = Engine(top=18, left=4, gap_cols=set()); install(eng, warm)
    r = GT.goto(4, 4)
    assert not r.get("reached"), r
    assert "refused" in str(r.get("reason", "")), r
    assert r["actions_taken"] <= 12, r
    print(f"2 no gap: stopped after {r['actions_taken']} actions, reason: {r['reason'][:70]}...")

    # 3. a refused move already in the history becomes a refused edge in a fresh model
    eng = Engine(top=14, left=4, gap_cols=set()); install(eng, warm + ["UP", "UP"])   # from 14: UP -> 12 crosses (refused)
    m = GT._learn(GT._transitions(GT.history))
    assert m is not None and any(k == "UP" for _, _, k in m["refused"]), m and m["refused"]
    print(f"3 history: refused edges from history alone: {sorted(m['refused'])}")
    print("OK")


if __name__ == "__main__":
    main()
