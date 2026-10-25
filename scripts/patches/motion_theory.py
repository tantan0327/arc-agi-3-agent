# PATCH: motion theory (CC0, ours) on Franzen's M2 harness.
#
# What it adds: three cheap helpers inside every sandboxed `python` call - a code-owned theory of
# how the avatar moves, fitted by the harness from the recorded frames of the CURRENT level, so the
# model stops re-deriving its own mechanics (measured: a counting theory predicts ~70% of human
# direction moves; the model spends 6-10 turns per level re-establishing them).
#   motion_theory()      -> avatar (the object that moved after a direction key), per-key direction /
#                           step / blocking colours, and how often the theory's own predictions held
#   predict_move(key)    -> move | blocked | unknown, with the avatar's cells after the press
#   path_to(target)      -> BFS over avatar placements under the theory -> key list for action([...])
# The theory is the pure-Python port of RichTheory (scripts/test_theory_induction.py) and plan()
# (scripts/test_theory_planner.py): avatar = the thing that moved last; per-key direction and step
# counters; blocked / passed colour counters; border mask (the step bar lives at the edge); relocate.
# The prelude is wrapped in `if True:` so the sandbox's function-retention scan (which only reads
# top-level defs) never reports our helpers to the model. Nothing else changes.
# Flags: ARC3_MOTION_THEORY=0 disables; ARC3_MOTION_BORDER=4 (edge cells ignored when detecting motion).
import os as _os
import re as _re

from inference.agent import tool_agent as _TA

_ENABLED = _os.environ.get("ARC3_MOTION_THEORY", "1").strip().lower() not in ("0", "false", "off", "")
_BORDER = int(_os.environ.get("ARC3_MOTION_BORDER", "4") or 4)
_PUSH = _os.environ.get("ARC3_MOTION_PUSH", "0").strip().lower() not in ("0", "false", "off", "")

_PRELUDE = r'''
if True:  # motion theory prelude (harness-owned; hidden from function retention)
    _MT_KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
    _MT_PRIOR = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
    _MT_BORDER = %BORDER%
    _MT_CACHE = {}

    def _mt_key(a):
        return str(a or "").split("(")[0].strip().upper()

    def _mt_rows(frame):
        try:
            rows = str(frame.ascii).split("\n")
            return rows if rows[0] and all(len(r) == len(rows[0]) for r in rows) else None
        except Exception:
            return None

    def _mt_unit(dr, dc):
        if dr == 0 and dc == 0:
            return None
        return (1 if dr > 0 else -1, 0) if abs(dr) >= abs(dc) else (0, 1 if dc > 0 else -1)

    def _mt_top(d):
        return max(d, key=d.get) if d else None

    def _mt_inc(d, key, item):
        sub = d.setdefault(key, {})
        sub[item] = sub.get(item, 0) + 1

    def _mt_changed(b, a, H, W, border, cap):
        out = []
        for r in range(border, H - border):
            br = b[r]; ar = a[r]
            if br != ar:
                for c in range(border, W - border):
                    if br[c] != ar[c]:
                        out.append((r, c))
                if len(out) > cap:
                    return None
        return out

    def _mt_fill(rows, r0, c0, col, lo, hi_r, hi_c, cap, seen):
        comp = {(r0, c0)}; seen.add((r0, c0)); stack = [(r0, c0)]
        while stack:
            r, c = stack.pop()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if lo <= rr < hi_r and lo <= cc < hi_c and (rr, cc) not in seen and rows[rr][cc] == col:
                    seen.add((rr, cc)); comp.add((rr, cc)); stack.append((rr, cc))
                    if len(comp) > cap:
                        return None
        return comp

    def _mt_movers(b, a, changed, H, W, border):
        """{colour: (components in `b`, offset)} for colours whose vacated cells map onto arrived cells by one offset."""
        out = {}
        if not changed:
            return out
        for col in sorted({b[r][c] for r, c in changed} | {a[r][c] for r, c in changed}):   # sorted: ties must not depend on set order
            vac = [(r, c) for r, c in changed if b[r][c] == col and a[r][c] != col]
            arr = [(r, c) for r, c in changed if a[r][c] == col and b[r][c] != col]
            if not vac or not arr:
                continue
            off = (min(r for r, _ in arr) - min(r for r, _ in vac), min(c for _, c in arr) - min(c for _, c in vac))
            if off == (0, 0):
                continue
            comps = []; seen = set(); shifted = set(); ok = True
            for r, c in vac:
                if (r, c) in seen:
                    continue
                comp = _mt_fill(b, r, c, col, border, H - border, W - border, 500, seen)
                if comp is None:
                    ok = False; break
                comps.append(comp); shifted.update((rr + off[0], cc + off[1]) for rr, cc in comp)
            if not ok:
                continue
            seen = set(); after_cells = set()
            for r, c in arr:
                if (r, c) in seen:
                    continue
                comp = _mt_fill(a, r, c, col, border, H - border, W - border, 500, seen)
                if comp is None:
                    ok = False; break
                after_cells |= comp
            if ok and shifted == after_cells:
                out[col] = (comps, off)
        return out

    def _mt_dest(rows, cells, off, H, W):
        out = set()
        for r, c in cells:
            rr, cc = r + off[0], c + off[1]
            if (rr, cc) not in cells:
                out.add(rows[rr][cc] if 0 <= rr < H and 0 <= cc < W else "edge")
        return out

    def _mt_dir(th, key):
        d = th["dir"].get(key)
        return _mt_top(d) if d else (_MT_PRIOR.get(key) if any(th["dir"].values()) else None)

    def _mt_step(th, key):
        s = th["step"].get(key); pooled = {}
        for sk in ([s] if s else th["step"].values()):   # this key's steps, else every key's pooled
            for k, n in sk.items():
                pooled[k] = pooled.get(k, 0) + n
        return _mt_top(pooled) if pooled else 1

    def _mt_blocked(th, key, cols):
        bl = th["block"].get(key) or {}; ps = th["passed"].get(key) or {}
        return any(bl.get(d, 0) > ps.get(d, 0) for d in cols)

    def _mt_slide(th, rows, cells, u, key, H, W):
        k = 0; cur = set(cells)
        while k < 64:
            nxt = {(r + u[0], c + u[1]) for r, c in cur}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in nxt):
                break
            if _mt_blocked(th, key, {rows[r][c] for r, c in nxt - cur}):
                break
            k += 1; cur = nxt
        return k

    def _mt_relocate(th, rows, H, W):
        av = th["avatar"]; col = th["colour"]
        if av is None or col is None:
            return
        if all(0 <= r < H and 0 <= c < W and rows[r][c] == col for r, c in av):
            return
        best = None; seen = set(); n = len(av)
        for r in range(H):
            row = rows[r]; c = row.find(col)
            while c != -1:
                if (r, c) not in seen:
                    comp = _mt_fill(rows, r, c, col, 0, H, W, 2000, seen)
                    if comp is not None and (best is None or abs(len(comp) - n) < abs(len(best) - n)):
                        best = comp
                c = row.find(col, c + 1)
        th["avatar"] = set(best) if best else None

    def _mt_new(level):
        return {"level": level, "avatar": None, "colour": None, "dir": {}, "step": {}, "passed": {}, "block": {},
                "score_step": 0, "score_slide": 0, "presses": 0, "moves": 0, "ok": 0, "bad": 0, "n": 0}

    def _mt_predict(th, rows, key, H, W):
        unknown = {"result": "unknown", "cells_after": None, "offset": None}
        if key not in _MT_KEYS:
            return dict(unknown, reason="%s is not a direction key" % key)
        if th["avatar"] is None:
            return dict(unknown, reason="no avatar yet on this level: press one direction key first")
        _mt_relocate(th, rows, H, W)
        if th["avatar"] is None:
            return dict(unknown, reason="the avatar's colour %r is no longer on the board" % th["colour"])
        u = _mt_dir(th, key)
        if u is None:
            return dict(unknown, reason="no direction key observed yet on this level")
        n = sum((th["dir"].get(key) or {}).values())
        how = "seen %dx" % n if n else "assumed from the key name, never pressed here"
        ks = _mt_slide(th, rows, th["avatar"], u, key, H, W)
        if ks == 0:
            front = sorted(_mt_dest(rows, th["avatar"], u, H, W))
            return {"result": "blocked", "cells_after": None, "offset": [0, 0],
                    "reason": "%s (%s) moves (%d,%d) but the cells in front hold %s, which stopped it before" % (key, how, u[0], u[1], front)}
        kf = _mt_step(th, key)
        k = ks if th["score_slide"] > th["score_step"] else min(kf, ks)
        off = (u[0] * k, u[1] * k)
        cells = sorted((r + off[0], c + off[1]) for r, c in th["avatar"])
        front = sorted(_mt_dest(rows, th["avatar"], off, H, W))
        return {"result": "move", "cells_after": [[r, c] for r, c in cells[:20]], "offset": [off[0], off[1]], "unit": u, "k": k,
                "reason": "%s (%s): direction (%d,%d), step %d%s; cells entered hold %s, not known to block" % (
                    key, how, u[0], u[1], k, " (slides until blocked)" if k != kf else "", front)}

    def _mt_observe(th, b, a, key, H, W):
        border = _MT_BORDER if min(H, W) > 2 * _MT_BORDER + 2 else 0
        ch = _mt_changed(b, a, H, W, border, 400)
        mv = _mt_movers(b, a, ch, H, W, border) if ch else {}
        if not mv:
            chf = _mt_changed(b, a, H, W, 0, 400)
            mv = _mt_movers(b, a, chf, H, W, 0) if chf else {}
        actual = comp_moved = col_moved = None
        if mv:
            s = "".join(b); bg = max(sorted(set(s)), key=s.count); expect = _mt_top(th["dir"].get(key) or {}); best = None
            for col, (comps, off) in mv.items():
                u = _mt_unit(off[0], off[1])
                for comp in comps:  # a plausible body first, then continuity (Spelke), contingency, size
                    score = (int(len(comp) >= 2 and col != bg), int(bool(th["avatar"]) and bool(comp & th["avatar"])),
                             int(expect is not None and u == expect), len(comp))
                    if best is None or score > best[0]:
                        best = (score, comp, off, col)
            _, comp_moved, actual, col_moved = best
        th["presses"] += 1
        pred = _mt_predict(th, b, key, H, W)   # scored before the update: the theory's own track record
        if pred["result"] != "unknown":
            k_act = 0 if actual is None else max(abs(actual[0]), abs(actual[1]))
            if pred["result"] == "blocked":
                ok = actual is None
            else:
                ok = actual is not None and _mt_unit(actual[0], actual[1]) == pred["unit"] and abs(pred["k"] - k_act) <= 1
            th["ok" if ok else "bad"] += 1
        if th["avatar"] is not None:
            u_pred = _mt_dir(th, key)
            if u_pred is not None:
                ks = _mt_slide(th, b, th["avatar"], u_pred, key, H, W)
                kf = _mt_top(th["step"].get(key) or {}) or 1
                k_act = 0 if actual is None else max(abs(actual[0]), abs(actual[1]))
                th["score_step"] += int(k_act == min(kf, ks)); th["score_slide"] += int(k_act == ks)
        if actual is not None:
            u_act = _mt_unit(actual[0], actual[1])
            _mt_inc(th["dir"], key, u_act); _mt_inc(th["step"], key, max(abs(actual[0]), abs(actual[1])))
            for dc in _mt_dest(b, comp_moved, u_act, H, W):
                _mt_inc(th["passed"], key, dc)
            th["avatar"] = {(r + actual[0], c + actual[1]) for r, c in comp_moved}; th["colour"] = col_moved
            th["moves"] += 1
        elif th["avatar"] is not None and ch == []:   # nothing changed inside the border: what was in front blocks
            u = _mt_dir(th, key)
            if u is not None:
                kf = _mt_top(th["step"].get(key) or {}) or 1
                for k in range(1, kf + 1):
                    for dc in _mt_dest(b, th["avatar"], (u[0] * k, u[1] * k), H, W):
                        _mt_inc(th["block"], key, dc)

    def _mt_level(frame):
        return int(getattr(frame, "level", 0) or 0)

    def _mt_theory(max_transitions):
        hist = list(history) if history else []
        if current_frame is None:
            raise ValueError("no current frame")
        level = _mt_level(current_frame); n = len(hist); C = _MT_CACHE
        last_step = int(getattr(hist[-1].frame, "step", -1)) if n else -1
        th = C.get("th")
        if (th is not None and C.get("level") == level and C.get("max") == max_transitions and 0 < C["n"] <= n
                and int(getattr(hist[C["n"] - 1].frame, "step", -2)) == C["last_step"]):
            todo = range(C["n"], n)   # history only grew: observe the new transitions
        else:
            th = _mt_new(level)
            todo = [i for i in range(1, n) if _mt_level(hist[i].frame) == level and _mt_level(hist[i - 1].frame) == level][-max_transitions:]
        for i in todo:
            e = hist[i]; p = hist[i - 1]
            if _mt_level(e.frame) != level or _mt_level(p.frame) != level:
                continue
            b = _mt_rows(p.frame); a = _mt_rows(e.frame)
            if b is None or a is None or len(b) != len(a) or len(b[0]) != len(a[0]):
                continue
            th["n"] += 1
            key = _mt_key(e.action)
            if key in _MT_KEYS:
                _mt_observe(th, b, a, key, len(b), len(b[0]))
        C.update(th=th, level=level, max=max_transitions, n=n, last_step=last_step)
        return th

    def _mt_board():
        rows = _mt_rows(current_frame)
        if rows is None:
            raise ValueError("current_frame has no readable ascii board")
        return rows, len(rows), len(rows[0])

    def motion_theory(max_transitions=150):
        """What the harness has learned about the avatar on THIS level from the recorded frames."""
        try:
            th = _mt_theory(int(max_transitions)); rows, H, W = _mt_board()
            _mt_relocate(th, rows, H, W); av = None
            if th["avatar"]:
                rs = [r for r, _ in th["avatar"]]; cs = [c for _, c in th["avatar"]]
                av = {"cells": len(th["avatar"]), "bbox": [min(rs), min(cs), max(rs), max(cs)], "colors": [th["colour"]]}
            keys = {}
            for k in _MT_KEYS:
                u = _mt_dir(th, k); d = th["dir"].get(k) or {}; bl = th["block"].get(k) or {}; ps = th["passed"].get(k) or {}
                keys[k] = {"dir": [u[0], u[1]] if u else None, "step": _mt_step(th, k), "n": sum(d.values()),
                           "blocked_by": sorted(c for c in bl if bl[c] > ps.get(c, 0)),
                           "passes": sorted(c for c in ps if ps[c] >= bl.get(c, 0))}
                if u and not d:
                    keys[k]["assumed"] = True
            return {"level": th["level"], "transitions": th["n"], "avatar": av, "keys": keys,
                    "slides": th["score_slide"] > th["score_step"],
                    "confidence": {"moves_seen": th["presses"], "predicted_ok": th["ok"], "predicted_bad": th["bad"]}}
        except Exception as exc:
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    def predict_move(key):
        """What pressing `key` now would do to the avatar, under the learned theory."""
        try:
            th = _mt_theory(150); rows, H, W = _mt_board()
            p = _mt_predict(th, rows, _mt_key(key), H, W)
            return {"result": p["result"], "cells_after": p["cells_after"], "offset": p["offset"], "reason": p["reason"]}
        except Exception as exc:
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    def path_to(target, max_depth=80):
        """Shortest key sequence (BFS under the theory) whose last press touches `target`: a colour letter or [[row, col], ...]."""
        try:
            th = _mt_theory(150); rows, H, W = _mt_board(); goal_col = goal_cells = None
            if isinstance(target, str) and len(target.strip()) == 1:
                goal_col = target.strip()
                if not any(goal_col in row for row in rows):
                    return {"keys": None, "length": 0, "reason": "colour %r is not on the board" % goal_col}
            else:
                cells = list(target) if isinstance(target, (list, tuple, set)) else []
                try:
                    goal_cells = {(int(r), int(c)) for r, c in ([cells] if len(cells) == 2 and all(isinstance(x, int) for x in cells) else cells)}
                except Exception:
                    goal_cells = set()
                if not goal_cells:
                    return {"keys": None, "length": 0, "reason": "target must be a colour letter or a list of [row, col] cells"}
            if th["avatar"] is not None:
                _mt_relocate(th, rows, H, W)
            if th["avatar"] is None:
                return {"keys": None, "length": 0, "reason": "no avatar yet on this level: press one direction key first"}
            moves = []
            start = th["avatar"]; rs = [r for r, _ in start]; cs = [c for _, c in start]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            for k in _MT_KEYS:
                u = _mt_dir(th, k)
                if u is None:
                    continue
                st = _mt_step(th, k); sweep = set()
                for i in range(1, st + 1):
                    sweep.update((r + u[0] * i, c + u[1] * i) for r, c in start)
                moves.append((k, (u[0] * st, u[1] * st), sorted(sweep - start)))
            if not moves:
                return {"keys": None, "length": 0, "reason": "no direction key observed yet on this level"}
            if goal_cells and goal_cells & start:
                return {"keys": [], "length": 0, "reason": "the avatar is already on the target"}
            seen = {(0, 0)}; queue = [((0, 0), [])]; qi = 0
            while qi < len(queue):
                (dr, dc), path = queue[qi]; qi += 1
                if len(path) >= max_depth:
                    continue
                for k, off, sweep in moves:
                    nd = (dr + off[0], dc + off[1])
                    if r0 + nd[0] < 0 or r1 + nd[0] >= H or c0 + nd[1] < 0 or c1 + nd[1] >= W:
                        continue
                    cols = set(); hit = False
                    for r, c in sweep:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            cols.add(rows[rr][cc])
                            if goal_cells is not None and (rr, cc) in goal_cells:
                                hit = True
                    if hit or (goal_col is not None and goal_col in cols):
                        return {"keys": path + [k], "length": len(path) + 1,
                                "reason": "%d presses; the last one touches %s" % (len(path) + 1, goal_col if goal_col else "the target cell")}
                    if _mt_blocked(th, k, cols) or nd in seen:
                        continue
                    seen.add(nd); queue.append((nd, path + [k]))
            return {"keys": None, "length": 0, "reason": "no route within %d presses under the current theory (an unseen colour may block, or the target is not reachable by walking)" % max_depth}
        except Exception as exc:
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    if %PUSH%:
        def _mt_push_line():
            try:
                th = _mt_theory(150); rows, H, W = _mt_board()
                if th["avatar"] is None:
                    return "MOTION: no avatar seen yet on this level (press one direction key; the theory learns from it)"
                _mt_relocate(th, rows, H, W)
                if th["avatar"] is None:
                    return "MOTION: avatar lost (its cells changed colour); press one direction key to re-find it"
                rs = [r for r, _ in th["avatar"]]; cs = [c for _, c in th["avatar"]]
                parts = ["avatar %s rows %d-%d cols %d-%d" % (th.get("colour"), min(rs), max(rs), min(cs), max(cs))]
                for k in _MT_KEYS:
                    pr = _mt_predict(th, rows, k, H, W)
                    u = _mt_dir(th, k); st = _mt_step(th, k)
                    bl = th["block"].get(k) or {}; ps = th["passed"].get(k) or {}
                    blockers = "".join(sorted(c for c in bl if bl[c] > ps.get(c, 0)))[:6]
                    if pr["result"] == "move" and u is not None:
                        parts.append("%s:(%+d,%+d)%s" % (k, u[0] * st, u[1] * st, (" blocked-by[" + blockers + "]") if blockers else ""))
                    elif pr["result"] == "blocked":
                        parts.append("%s:blocked%s" % (k, ("[" + blockers + "]") if blockers else ""))
                    else:
                        parts.append("%s:?" % k)
                parts.append("theory %d/%d" % (th["ok"], th["ok"] + th["bad"]))
                return "MOTION: " + " | ".join(parts)
            except Exception as exc:
                return "MOTION: unavailable (%s)" % type(exc).__name__
        _mt_orig_action = action
        def action(actions):
            _r = _mt_orig_action(actions)
            try:
                print(_mt_push_line())
            except Exception:
                pass
            return _r
'''.replace("%BORDER%", str(_BORDER)).replace("%PUSH%", "True" if _PUSH else "False")

_SEPARATOR = "\n# ---- the model's code ----\n"
_OFFSET = (_PRELUDE + _SEPARATOR).count("\n")   # the model's line L is line L + _OFFSET of the combined code
_LINE_RE = _re.compile(r'(File "<python_tool>", line |\(<python_tool>, line )(\d+)')

SYSTEM_LINES = (
    "\nMotion theory (three helpers the harness computes from the recorded frames of the CURRENT level - "
    "measurements, not guesses - available in every python call, reading them costs no in-game action):\n"
    "- `motion_theory()` -> {'avatar': {'cells', 'bbox', 'colors'} or None, 'keys': {'UP'|'DOWN'|'LEFT'|'RIGHT': {'dir': [drow, dcol], "
    "'step': cells per press, 'n': presses seen, 'blocked_by': [colour letters that stopped it], 'passes': [colour letters it walked over]}}, "
    "'confidence': {'moves_seen', 'predicted_ok', 'predicted_bad'}}. The avatar is the object that moved after a direction key; "
    "it is None until one direction key has moved something on this level.\n"
    "- `predict_move(key)` -> {'result': 'move'|'blocked'|'unknown', 'cells_after': [[row, col], ...], 'offset': [drow, dcol], 'reason'} for pressing `key` now.\n"
    "- `path_to(target, max_depth=80)` -> {'keys': ['UP', 'UP', 'LEFT', ...] or None, 'length', 'reason'}: breadth-first search over avatar "
    "positions under the learned theory; `target` is a colour letter (the plan ends with the press whose front touches that colour) or a "
    "list of [row, col] cells. Pass the keys straight to `action(keys)`.\n"
    "- When to use them: on the first turn of every level press ONE direction key, then call `motion_theory()` instead of re-deriving how "
    "the avatar moves; call `path_to(...)` before planning any walk; call `predict_move(key)` before a press you are unsure about.\n"
    "- Limits: the theory knows only what the frames showed - a colour never walked into is assumed passable, and a press that changed "
    "nothing is how it learns blockers. `confidence` is its own track record on this level; if `predicted_bad` grows, the controls are "
    "not a fixed step per key (turning, momentum, pushing, teleports) - model that yourself. The helpers say nothing about the objective.\n"
    "- They never raise: a failure returns {'error': ...}.\n"
)
if _PUSH:
    SYSTEM_LINES = SYSTEM_LINES.rstrip("\n") + (
        "\n- After every `action(...)` call the tool output ends with a `MOTION:` line: the avatar's colour and box, what each "
        "direction key does now ((drow,dcol) per press, blocked[colours], or ? when unseen) and the theory's track record on this "
        "level. Read it instead of re-deriving movement; call `path_to(...)` for routes.\n"
    )


_orig_run = _TA.run_sandboxed_python


def _fix_line_numbers(text):
    """Error line numbers refer to the combined code; report them in the model's own numbering."""
    def _sub(m):
        n = int(m.group(2))
        return m.group(1) + str(n - _OFFSET if n > _OFFSET else n)
    return _LINE_RE.sub(_sub, text)


def _run_with_motion(*, code, **kwargs):
    try:
        code = _PRELUDE + _SEPARATOR + str(code)
    except Exception:
        pass
    result = _orig_run(code=code, **kwargs)
    try:
        if isinstance(result, dict) and isinstance(result.get("error"), str) and "<python_tool>" in result["error"]:
            result["error"] = _fix_line_numbers(result["error"])
    except Exception:
        pass
    return result


_orig_system = _TA._build_system_prompt


def _system_with_motion(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and "Motion theory (three helpers" not in prompt:
        prompt = prompt.rstrip("\n") + "\n" + SYSTEM_LINES
    return prompt


if _ENABLED:
    _TA.run_sandboxed_python = _run_with_motion
    _TA._build_system_prompt = _system_with_motion
print("MOTION_THEORY_PATCH installed: enabled=%s push=%s border=%d prelude=%d lines/%d chars addendum=%d chars" % (
    _ENABLED, _PUSH, _BORDER, _PRELUDE.count("\n"), len(_PRELUDE), len(SYSTEM_LINES)), flush=True)
