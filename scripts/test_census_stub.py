"""Stub test of the orienting-census patch against Franzen's patched tool_agent.

    PYTHONPATH=<bundle>/src python3 scripts/test_census_stub.py
"""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import inference.agent.tool_agent as TA  # noqa: E402

ns = {"__name__": "census_patch"}
exec(compile((ROOT / "scripts/patches/orienting_census.py").read_text(), "orienting_census", "exec"), ns)
assert TA.ToolAgent._build_user_prompt is ns["_build_with_census"]
assert TA.ToolAgent.analyze is ns["_analyze_with_census"]
assert ns["SYSTEM_LINE"].strip() in TA._build_system_prompt(tool_output_tokens=3072)
print("1 hooks installed; system prompt carries the census rule")


class Frame:
    def __init__(self, level, step=0): self.level = level; self.step = step; self.shape = (64, 64)


class Fake:
    def __init__(self, level, valid, effects, clear_on=None, die_on=None):
        self._current_valid_actions = valid; self._last_step_summary = None; self._step_env_callback = lambda a: {}
        self._census_state_path = Path("/tmp/game-x_p0.json"); self.level = level; self.calls = []
        self.effects = effects; self.clear_on = clear_on; self.die_on = die_on; self.exec_total = 0; self.clear_level = level

    def _run_python_tool(self, state_path, arguments):
        code = arguments["code"]; name = code.split("action(")[1].split(")")[0].strip("'\"")
        self.calls.append(name); self.exec_total += 1
        diff = {"changed_cell_count": 0, "moved": [], "rotated": [], "appeared": [], "disappeared": [], "changed_color": [], "resized": []}
        res = {"executed": True, "executed_actions": [name], "gameplay_changed": False, "board_changed": False, "level": self.level}
        if name == self.clear_on and self.level == self.clear_level:
            res.update(level_completed=True, gameplay_changed=True, board_changed=True); self.level += 1
        elif name == self.die_on:
            res.update(game_over=True, gameplay_changed=True, board_changed=True)
        elif name in self.effects:
            diff = {"changed_cell_count": 4, "moved": [{"color": "G", "pixels": 4, "from": [10, 5], "to": self.effects[name]}],
                    "rotated": [], "appeared": [], "disappeared": [], "changed_color": [], "resized": []}
            res.update(gameplay_changed=True, board_changed=True)
        self._last_step_summary = {"executed_count": 1, "executed_actions": [name], "level": self.level,
                                   "level_transition": bool(res.get("level_completed")), "game_over": bool(res.get("game_over"))}
        return TA._ToolDispatchResult(content=json.dumps({"returncode": 0, "stdout": "CENSUS_JSON " + json.dumps({"diff": diff, "result": res})}))


SEEN = {}
def fake_build(self, action_num, *a, **k):
    SEEN["kwargs"] = dict(k); f = k.get("current_frame"); s = k.get("previous_step_summary") or {}
    return "opener for level %d%s" % (f.level, " [LEVEL START GUIDANCE]" if s.get("level_transition") else "")
ns["_orig_build"] = fake_build
ns["_load_state"] = lambda p: (Frame(STATE["agent"].level), ["h"] * STATE["agent"].exec_total)
STATE = {}

# 2. arrow game at level 1: all five probed, effects tabulated, opener describes the post-census board
ag = Fake(level=1, valid=["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO"], effects={"UP": [9, 5], "RIGHT": [10, 6]}); STATE["agent"] = ag
txt = ns["_build_with_census"](ag, 0, current_frame=Frame(1), history_entries=[], previous_step_summary=None)
assert ag.calls == ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], ag.calls
assert "- UP: 4 cells changed. - moved: color G 4px [10, 5]->[9, 5]" in txt.replace("\n", " ") or "- UP: 4 cells changed." in txt, txt
assert "- DOWN: no change" in txt and "5 action(s) are already spent" in txt, txt
assert len(SEEN["kwargs"]["history_entries"]) == 5, SEEN["kwargs"]["history_entries"]
print("2 arrow game: five probes, effects table, post-census history passed to the opener")

# 3. once per level: the same level again runs nothing
txt2 = ns["_build_with_census"](ag, 5, current_frame=Frame(1), history_entries=["h"] * 5, previous_step_summary=ag._last_step_summary)
assert len(ag.calls) == 5 and "Orienting census" not in txt2, (ag.calls, txt2)
print("3 once per level")

# 4. level transition: the level-start guidance survives the census (merged summary)
ag = Fake(level=2, valid=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], effects={"LEFT": [10, 4]}); STATE["agent"] = ag
txt = ns["_build_with_census"](ag, 30, current_frame=Frame(2), history_entries=["h"] * 30, previous_step_summary={"level": 2, "level_transition": True, "executed_count": 3})
assert "[LEVEL START GUIDANCE]" in txt and "- LEFT: 4 cells changed" in txt, txt
print("4 transition opener keeps the level-start guidance")

# 5. a probe causes game over: census stops there, remaining actions listed as not probed
ag = Fake(level=2, valid=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], effects={}, die_on="DOWN"); STATE["agent"] = ag
txt = ns["_build_with_census"](ag, 30, current_frame=Frame(2), history_entries=["h"] * 30, previous_step_summary={"level": 2, "level_transition": True})
assert ag.calls == ["UP", "DOWN"] and "GAME OVER" in txt and "not probed: LEFT, RIGHT, SPACE" in txt, (ag.calls, txt)
assert "[LEVEL START GUIDANCE]" not in txt, "a game-over summary must not be relabelled as a level start"
print("5 game over stops the census; guidance not forced over a game-over summary")

# 6. a probe clears the level: the next level is censused too, once
ag = Fake(level=3, valid=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], effects={"UP": [9, 5]}, clear_on="RIGHT"); STATE["agent"] = ag
txt = ns["_build_with_census"](ag, 40, current_frame=Frame(3), history_entries=["h"] * 40, previous_step_summary={"level": 3, "level_transition": True})
assert ag.calls == ["UP", "DOWN", "LEFT", "RIGHT", "UP", "DOWN", "LEFT", "RIGHT", "SPACE"], ag.calls
assert "LEVEL CLEARED by this action" in txt and txt.count("Orienting census") == 2 and ag._census_levels_done == {3, 4}, txt
print("6 a clearing probe chains one census into the next level")

# 7. click-only game: nothing executed, the opener says so
ag = Fake(level=1, valid=["MOUSE", "UNDO"], effects={}); STATE["agent"] = ag
txt = ns["_build_with_census"](ag, 0, current_frame=Frame(1), history_entries=[], previous_step_summary=None)
assert ag.calls == [] and "executed nothing" in txt, txt
print("7 click-only game: no probes")
print("OK")
