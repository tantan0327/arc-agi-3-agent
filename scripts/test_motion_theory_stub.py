"""Stub test of the motion_theory patch against Franzen's patched harness (no GPU, no model).

    PYTHONPATH=<bundle>/src/ARC3-Inference:<bundle>/src/tufa-arc-agi-framework/src \
    .venv/bin/python scripts/test_motion_theory_stub.py [game ...]

1. installs the patch on the harness: wrappers in place, system prompt gains the addendum once;
2. runs the prelude through the REAL sandbox (restricted builtins, function retention on) on a
   replayed level, and checks error line numbers are reported in the model's own numbering;
3. replays official human solutions of direction games through the local engine, renders the
   frames with the harness's own ASCII legend, and scores `predict_move` against the avatar's
   actual displacement, `path_to` at level starts (plans executed in the engine), and build time.
"""
from __future__ import annotations
import importlib.util, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts/patches/motion_theory.py"
os.environ.setdefault("ARC3_PERSISTENT_FUNCTIONS", "1")
os.environ.setdefault("ARC3_ACTION_INFO", "1")

import inference.agent.tool_agent as TA  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.utils.grid_utils import format_grid_ascii  # noqa: E402

GAMES = sys.argv[1:] or ["re86", "ar25", "sk48", "tr87", "tu93"]   # direction games (sb26 is click-only)
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
MARK = "Motion theory (three helpers"

# 1. install: wrappers in place, addendum appended exactly once, original prompt preserved verbatim
before = TA._build_system_prompt(tool_output_tokens=3072)
assert MARK not in before
orig_run, orig_system = TA.run_sandboxed_python, TA._build_system_prompt
ns = {"__name__": "motion_theory_patch"}
exec(compile(SRC.read_text(), "motion_theory", "exec"), ns)
assert TA.run_sandboxed_python is ns["_run_with_motion"], "run_sandboxed_python not wrapped"
assert TA._build_system_prompt is ns["_system_with_motion"], "_build_system_prompt not wrapped"
after = TA._build_system_prompt(tool_output_tokens=3072)
assert after.startswith(before.rstrip("\n")), "the original prompt is not preserved verbatim"
assert after.count(MARK) == 1 and TA._build_system_prompt(tool_output_tokens=3072).count(MARK) == 1, "addendum not appended exactly once"
PRELUDE = ns["_PRELUDE"]; n_lines = PRELUDE.count("\n")
assert n_lines < 420, n_lines  # 346 + the 35-line push block
print("1 installed: run_sandboxed_python + _build_system_prompt wrapped; addendum %d chars appended once (idempotent); prelude %d lines"
      % (len(ns["SYSTEM_LINES"]), n_lines))

# --- the local engine and the official human replays (as scripts/test_theory_planner.py does) ---
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
spec = importlib.util.spec_from_file_location("ti", ROOT / "scripts/test_theory_induction.py"); ti = importlib.util.module_from_spec(spec); spec.loader.exec_module(ti)
import logging; logging.disable(logging.CRITICAL)  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine.enums import GameAction  # noqa: E402
REPLAYS = ROOT / ti.REPLAYS
ARCADE = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
LABEL = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE"}


def label_of(a, p):   # the harness's history labels: solver._format_action_display
    return "MOUSE(row=%d, col=%d)" % (p["y"], p["x"]) if a == 6 else LABEL[a]


class F:   # what the sandbox exposes as a frame: .ascii / .step / .level / .shape (+ the grid for the checks here)
    def __init__(self, grid, step, level):
        self.grid = grid; self.np = ti.np.array(grid); self.ascii = format_grid_ascii(grid); self.step = step; self.level = level
        self.shape = (len(grid), len(grid[0]))


class E:   # a history entry: .action / .frame / .result
    def __init__(self, action, frame):
        self.action = action; self.frame = frame; self.result = {}


def engine_action(a, p):
    return GameAction[f"ACTION{a}"], ({"x": p["x"], "y": p["y"]} if a == 6 and p else None)


def replay(game, levels):
    """history entries exactly as the harness builds them: entry 0 is the seeded frame (action ''), entry i the frame after action i."""
    env = ARCADE.make(game); f = env.reset(); n_levels = len(levels)
    mk = lambda f, step: F(ti.grid_of(f).tolist(), step, min(f.levels_completed + 1, n_levels))
    entries = [E("", mk(f, 0))]
    for lvl in levels:
        for a, p in lvl:
            f = env.step(*engine_action(a, p))
            entries.append(E(label_of(a, p), mk(f, len(entries))))
    return entries


def fresh_namespace():
    g = {"__name__": "python_tool_stub", "history": [], "current_frame": None}
    exec(compile(PRELUDE, "<python_tool>", "exec"), g, g)
    return g


# 2. the real sandbox: restricted builtins, persistence on -> the helpers work, nothing is reported to the model
levels0 = [[(a, p) for a, p in lvl] for lvl in json.load(open(REPLAYS / f"{GAMES[0]}.json"))["levels"]]
ent0 = replay(GAMES[0], levels0)
L1 = [i for i in range(1, len(ent0)) if ent0[i - 1].frame.level == 1 and ent0[i].frame.level == 1]
cut = L1[min(5, len(L1) - 1)] + 1
hist = [HistoryEntry(action=e.action, frame=Frame(grid=tuple(tuple(r) for r in e.frame.grid), step=e.frame.step, level=e.frame.level)) for e in ent0[:cut]]
state = {"current_frame": TA._ascii_frame_view_payload(hist[-1].frame), "history": TA._ascii_history_view_payload(hist),
         "valid_actions": list(DIRS), "last_action_call_result": {}, "death_ledger": {"attempts": []}}
code = ('import json\nth = motion_theory()\np = predict_move("UP")\nr = path_to("%s")\n'
        'print(json.dumps({"avatar": th.get("avatar"), "n": {k: v["n"] for k, v in th["keys"].items()}, "pred": p["result"], '
        '"path": r.get("keys"), "conf": th["confidence"], "err": [x.get("error") for x in (th, p, r)]}))\n') % ent0[cut - 1].frame.ascii[0]
def no_actions(*a, **k): raise RuntimeError("the stub never acts")
res = TA.run_sandboxed_python(code=code, timeout_seconds=30, initial_state=state, action_handler=no_actions,
                              kept_functions=[], retain_imports=True)
assert not res.get("error"), res
out = json.loads(res["stdout"].strip().splitlines()[-1])
assert out["err"] == [None, None, None] and out["avatar"] is not None, out
assert res.get("keepable_functions") == [] and res.get("retention_rejected") == [], "the prelude leaked into function retention: %s" % res
res2 = TA.run_sandboxed_python(code="x = 1\ny = undefined_name\n", timeout_seconds=30, initial_state=state, action_handler=no_actions)
assert 'line 2' in res2.get("error", "") and "NameError" in res2["error"], res2
res3 = TA.run_sandboxed_python(code="x = 1\nx = (\n", timeout_seconds=30, initial_state=state, action_handler=no_actions)
assert "SyntaxError" in res3.get("error", "") and "line 2)" in res3["error"], res3
print("2 real sandbox (%s, %d history entries): avatar=%s presses=%s predict_move(UP)=%s path_to=%s; retention untouched; error line numbers are the model's own (NameError at line 2, SyntaxError at line 2)"
      % (GAMES[0], cut, out["avatar"], out["n"], out["pred"], out["path"]))


# 3. accuracy / path_to / timing on real frames
def unit(o):
    return ti.RichTheory.unit(tuple(o))


def evaluate(game, entries):
    g = fresh_namespace(); counted = correct = 0; per_level = {}; first_avatar = None
    presses_l1 = 0
    for i in range(1, len(entries)):
        key = entries[i].action.split("(")[0]
        if key not in DIRS or entries[i].frame.level != entries[i - 1].frame.level:
            continue
        g["history"] = entries[:i]; g["current_frame"] = entries[i - 1].frame
        pred = g["predict_move"](key)
        assert "error" not in pred, pred
        if entries[i - 1].frame.level == 1 and first_avatar is None:
            presses_l1 += 1
            if g["motion_theory"]().get("avatar"):
                first_avatar = presses_l1 - 1   # presses observed before the avatar was known
        actual = None
        for _, off in ti.movers_any(entries[i - 1].frame.np, entries[i].frame.np).items():
            actual = off
        if pred["result"] == "unknown":
            continue
        if pred["result"] == "blocked":
            ok = actual is None
        else:
            po = pred["offset"]
            ok = actual is not None and unit(po) == unit(actual) and abs(max(abs(po[0]), abs(po[1])) - max(abs(actual[0]), abs(actual[1]))) <= 1
        counted += 1; correct += int(ok)
        lv = per_level.setdefault(entries[i].frame.level, [0, 0]); lv[0] += 1; lv[1] += int(ok)
    return counted, correct, first_avatar, per_level, g


def level_segments(entries):
    """{level: [indices i of the actions taken on that level]}"""
    seg = {}
    for i in range(1, len(entries)):
        seg.setdefault(entries[i - 1].frame.level, []).append(i)
    return seg


def goal_colour(g, entries, j):
    """The rarest non-background colour the human's completing press entered (the planner's goal)."""
    key = entries[j].action.split("(")[0]
    if key not in DIRS:
        return None
    g["history"] = entries[:j]; g["current_frame"] = entries[j - 1].frame
    p = g["predict_move"](key)
    if p.get("result") != "move":
        return None
    rows = entries[j - 1].frame.ascii.split("\n"); off = p["offset"]
    cells = {(r, c) for r, c in p["cells_after"]}; was = {(r - off[0], c - off[1]) for r, c in cells}
    entered = [rows[r][c] for r, c in cells - was if 0 <= r < len(rows) and 0 <= c < len(rows[0])]
    s = "".join(rows); bg = max(set(s), key=s.count)
    cands = [c for c in entered if c != bg] or entered
    return min(cands, key=s.count) if cands else None


def run_plan(game, levels, upto, keys):
    """Replay the human's first `upto` actions, then the plan -> (levels completed, every press moved the avatar as predicted)."""
    env = ARCADE.make(game); f = env.reset(); flat = [ap for lvl in levels for ap in lvl]
    for a, p in flat[:upto]:
        f = env.step(*engine_action(a, p))
    done = f.levels_completed; before = ti.grid_of(f); as_predicted = True
    for k in keys:
        f = env.step(GameAction[f"ACTION{DIRS.index(k) + 1}"], None)
        if f is None or f.levels_completed > done:
            break
        if f.levels_completed < done or ti.np.array(f.frame).ndim < 3:   # the press ended the attempt (reset or empty frame)
            as_predicted = False; break
        after = ti.grid_of(f); as_predicted &= bool(ti.movers_any(before, after)); before = after
    return (max(0, f.levels_completed - done) if f is not None else 0), as_predicted


tot_counted = tot_correct = 0; plans = plans_ok = plans_won = plan_levels = 0; max_build = 0.0; max_path = 0.0; biggest = None
print("%-6s %5s %5s | %-9s | %-7s | %s" % ("game", "pred", "ok", "accuracy", "avatar", "per level (counted/ok) ; path_to at level starts"))
for game in GAMES:
    levels = [[(a, p) for a, p in lvl] for lvl in json.load(open(REPLAYS / f"{game}.json"))["levels"]]
    entries = replay(game, levels)
    counted, correct, first_avatar, per_level, g = evaluate(game, entries)
    assert first_avatar is not None and first_avatar <= 3, "%s: avatar not found on level 1 within the first presses (%s)" % (game, first_avatar)
    tot_counted += counted; tot_correct += correct
    # path_to at the start of each level (after the level's 2nd direction press), executed in the engine
    seg = level_segments(entries); notes = []
    for L, idx in sorted(seg.items()):
        dir_idx = [i for i in idx if entries[i].action in DIRS]
        if len(dir_idx) < 3:
            continue
        goal = goal_colour(g, entries, idx[-1])
        if goal is None:
            continue
        s = dir_idx[1]
        g["history"] = entries[:s + 1]; g["current_frame"] = entries[s].frame
        t0 = time.perf_counter(); r = g["path_to"](goal); max_path = max(max_path, time.perf_counter() - t0)
        assert "error" not in r, r
        plan_levels += 1
        if r["keys"]:
            plans += 1; won, moved = run_plan(game, levels, s, r["keys"]); plans_won += int(won > 0); plans_ok += int(moved)
            notes.append("L%d:%s->%d keys%s%s" % (L, goal, len(r["keys"]), "" if moved else " (deviated)", " WIN" if won else ""))
        else:
            notes.append("L%d:%s->none" % (L, goal))
        if biggest is None or len(idx) > biggest[2]:
            biggest = (game, entries, len(idx), idx)
    pl = " ".join("L%d:%d/%d" % (L, v[1], v[0]) for L, v in sorted(per_level.items()))
    print("%-6s %5d %5d | %7.0f%%  | %-7s | %s ; %s" % (game, counted, correct, 100.0 * correct / max(1, counted), "press %d" % first_avatar, pl, " ".join(notes)))

# timing: a full rebuild from 150 transitions of one level (cache cleared), then the cached path
game, entries, _, idx = biggest
window = entries[max(1, idx[-1] - 150) - 1: idx[-1] + 1]
g = fresh_namespace(); g["history"] = window; g["current_frame"] = window[-1].frame
for _ in range(3):
    g["_MT_CACHE"].clear(); t0 = time.perf_counter(); th = g["motion_theory"](150); max_build = max(max_build, time.perf_counter() - t0)
assert "error" not in th and th["transitions"] <= 150, th
t0 = time.perf_counter(); g["predict_move"]("UP"); t_cached = time.perf_counter() - t0
t0 = time.perf_counter(); g["_MT_CACHE"].clear(); exec(compile(PRELUDE, "<python_tool>", "exec"), g, g); t_def = time.perf_counter() - t0
acc = 100.0 * tot_correct / max(1, tot_counted)
print("\nALL: predict_move agreed with the actual avatar displacement on %d/%d direction presses = %.0f%% (numpy RichTheory ~68%%; gate 55%%)" % (tot_correct, tot_counted, acc))
print("path_to: %d/%d level starts got a key list; executed in the engine, %d plans moved the avatar on every press as predicted and %d completed the level (touching the goal colour is rarely the whole objective)" % (plans, plan_levels, plans_ok, plans_won))
print("timing: full build from %d transitions of %s = %.3f s (worst of 3; gate 2 s); cached predict_move %.4f s; path_to worst %.3f s; prelude definition %.4f s"
      % (th["transitions"], game, max_build, t_cached, max_path, t_def))
assert acc >= 55.0, acc
assert plans >= 1, "path_to never returned a key list"
assert max_build < 2.0, max_build

# 4. disabled by flag -> the harness is untouched
os.environ["ARC3_MOTION_THEORY"] = "0"
TA.run_sandboxed_python, TA._build_system_prompt = orig_run, orig_system
ns2 = {"__name__": "motion_theory_patch"}
exec(compile(SRC.read_text(), "motion_theory", "exec"), ns2)
assert TA.run_sandboxed_python is orig_run and TA._build_system_prompt is orig_system, "flag did not disable the patch"
print("4 ARC3_MOTION_THEORY=0 leaves the harness untouched")
print("ALL OK")
