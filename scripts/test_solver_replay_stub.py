"""Stub test of the solver-replay patch's wrapper logic against Franzen's patched tool_agent.

Loads his module (PYTHONPATH must point at the patched bundle src), execs the patch, and
drives the hooks with a fake agent: a fake `_run_python_tool` that calls the budgeted
step callback, a fake `_load_state`, and a fake original `_build_user_prompt`.

    PYTHONPATH=<bundle>/src python3 scripts/test_solver_replay_stub.py
"""
from __future__ import annotations

import os, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import inference.agent.tool_agent as TA  # noqa: E402

ns = {"__name__": "solver_replay_patch"}
exec(compile((ROOT / "scripts/patches/solver_replay.py").read_text(), "solver_replay", "exec"), ns)
assert TA.ToolAgent._build_user_prompt is ns["_build_with_replay"], "prompt hook not installed"
assert TA.ToolAgent.analyze is ns["_analyze_with_replay"], "analyze hook not installed"
sp = TA._build_system_prompt(tool_output_tokens=3072)
assert ns["SYSTEM_LINE"].strip() in sp, "system line missing"
print("1 hooks installed; system prompt carries the solve_level rule")


class Frame:
    def __init__(self, level, step=0): self.level = level; self.step = step


class Fake:
    """Only what the wrapper touches."""
    def __init__(self, level, kept, n_actions_per_call=4, progress=True, clear_at=None):
        self._kept_functions = kept; self._current_valid_actions = ["UP", "DOWN"]; self._last_step_summary = None
        self._replay_state_path = Path("/tmp/game-x_p0.json"); self.level = level; self.calls = []; self.exec_total = 0
        self.n = n_actions_per_call; self.progress = progress; self.clear_at = clear_at

        def cb(step_args):
            acts = list(step_args.get("actions") or []); k = len(acts); self.exec_total += k
            done = self.clear_at is not None and self.exec_total >= self.clear_at
            if done: self.level += 1
            return {"executed": True, "executed_count": k, "executed_actions": acts, "gameplay_changed": self.progress or done,
                    "board_changed": self.progress or done, "level_completed": done, "level": self.level}
        self._step_env_callback = cb

    def _run_python_tool(self, state_path, arguments):
        self.calls.append(arguments["code"])
        # the model's solve_level: 5 calls of n actions, stops when a call is refused
        total = 0; last = None
        for _ in range(5):
            r = self._step_env_callback({"actions": ["UP"] * self.n})
            last = r
            if not r.get("executed"): break
            total += r.get("executed_count", 0)
            if r.get("level_completed"): break
        self._last_step_summary = {"executed_count": total, "executed_actions": ["UP"] * total, "level": self.level,
                                   "level_transition": bool(last and last.get("level_completed"))}
        return TA._ToolDispatchResult(content='{"returncode": 0, "stdout": "solve_level returned None"}' if last and last.get("executed") else '{"error": "replay refused"}')


def fake_build(self, action_num, *a, **k):
    f = k.get("current_frame"); return "opener for level %d. Level completed - the board was replaced" % f.level if k.get("transition") else "opener for level %d" % f.level


ns["_orig_build"] = fake_build
ns["_load_state"] = lambda p: (Frame(STATE["agent"].level), ["a"] * STATE["agent"].exec_total)
STATE = {}

# 2. no solver retained -> no replay, but the transition opener carries the consolidation request
ag = Fake(level=2, kept={}); STATE["agent"] = ag
txt = ns["_build_with_replay"](ag, 10, current_frame=Frame(2), history_entries=["a"] * 20, transition=True)
assert not ag.calls and ns["CONSOLIDATE_LINE"] in txt, txt[:200]
print("2 no solver: no replay, consolidation line present on the transition opener")

# 3. solver retained, budget caps the replay: previous level cost 10 actions -> budget 12; 4 per call -> 12 then refused
ag = Fake(level=2, kept={"solve_level": "def solve_level(): pass"}, n_actions_per_call=4, progress=True); STATE["agent"] = ag
ag._replay_actions_at_level_start = 0
txt = ns["_build_with_replay"](ag, 10, current_frame=Frame(2), history_entries=["a"] * 10, transition=True)
assert len(ag.calls) == 1 and ag.exec_total == 12, (ag.calls, ag.exec_total)
assert "Harness replay" in txt and "budget 12" in txt and "stopped at the action budget" in txt, txt
print("3 budget: replay executed exactly the budget (12) and the opener reports it")

# 4. no-progress rule: actions that change nothing stop the replay after 3
ag = Fake(level=3, kept={"solve_level": "def solve_level(): pass"}, n_actions_per_call=1, progress=False); STATE["agent"] = ag
ag._replay_actions_at_level_start = 0
txt = ns["_build_with_replay"](ag, 10, current_frame=Frame(3), history_entries=["a"] * 30, transition=True)
assert ag.exec_total == 3 and "without a board change" in txt, (ag.exec_total, txt[:300])
print("4 no-progress: stopped after 3 unchanged actions")

# 5. a replay that clears the level chains into the next level (and stops at the chain cap)
ag = Fake(level=2, kept={"solve_level": "def solve_level(): pass"}, n_actions_per_call=5, progress=True, clear_at=5); STATE["agent"] = ag
ag._replay_actions_at_level_start = 0
txt = ns["_build_with_replay"](ag, 10, current_frame=Frame(2), history_entries=["a"] * 10, transition=True)
assert txt.count("Harness replay") == 3 and "CLEARED" in txt, (txt.count("Harness replay"), txt[:400])
assert ag.level == 5, ag.level
print("5 chain: three consecutive levels cleared by replay, chain capped at 3")
print("OK")
