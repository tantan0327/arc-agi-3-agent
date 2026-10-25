# goto_tool: the physics planner as TOOLS the model calls from its python tool.
#
# Why this shape. Across eleven full runs the agent that reached level 2 often
# had the mechanic right and lost the level to execution: it mistook the goal
# bracket for its own body (g50t), estimated the cell pitch as 6.125 (g50t),
# walked 13 moves into walls and died to the step budget (ls20), and found the
# correct route one turn before the clock (m0r0). Every one of those is a
# planning-and-execution error over rules the agent had already stated. The
# harness can learn those rules from the agent's own moves and execute a stated
# plan in one turn. Handing the facts to the model in the prompt was measured
# null (9/19); handing it a *lever* is the untested configuration.
#
# Sandbox constraints: `python -I -S`, stdlib only (collections, math, ...),
# a restricted builtins set, no numpy. Frames expose `._grid` (list of lists of
# ints 0-15); `action(actions)` executes real actions and refreshes the globals
# `current_frame` and `history`. This file is prepended to the model's code on
# every python-tool call, so it must be self-contained and fast (~1 s to learn
# from 60 transitions).
#
# Public helpers (documented to the model in the user prompt):
#   body()                 -> facts about the body the arrow keys move
#   goto(row, col)         -> plan + execute moving the body's top-left to (row, col)
#   goto_colour(colour)    -> plan + execute until the body touches a cell of `colour`
from collections import Counter, defaultdict, deque

_DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
_KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
_MAX_STEP = 16
_LEARN_LAST = 80          # transitions used for learning
_MAX_NODES = 40000
_CHUNK = 8                # actions per action() call inside goto
_MAX_REPLANS = 12
_MAX_INVISIBLE = 4        # refusals with nothing visible in the way before goto gives the board back
_CACHE = {"n": -1, "model": None}


def _gl():
    """The sandbox runtime globals (`history`, `current_frame`, `action`): the
    builtins whitelist has no globals(), but a function's __globals__ is that dict."""
    return _gl.__globals__


def _grid_of(frame):
    g = getattr(frame, "_grid", None)
    return g if g is not None else frame


def _action_name(text):
    t = str(text or "").strip()
    return t.split("(")[0].strip().upper() if t else ""


def _comps(grid, colour, cap=12):
    """4-connected components of `colour`: [(top, left, offs, cells)], largest first."""
    H = len(grid); W = len(grid[0]) if H else 0
    seen = set(); out = []
    for r in range(H):
        row = grid[r]
        for c in range(W):
            if row[c] == colour and (r, c) not in seen:
                stack = [(r, c)]; seen.add((r, c)); cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen and grid[ny][nx] == colour:
                            seen.add((ny, nx)); stack.append((ny, nx))
                top = min(y for y, _ in cells); left = min(x for _, x in cells)
                offs = tuple(sorted((y - top, x - left) for y, x in cells))
                out.append((top, left, offs, cells))
    out.sort(key=lambda t: (-len(t[3]), t[0], t[1]))
    return out[:cap]


def _unit(dy, dx):
    if abs(dy) > abs(dx):
        return (1 if dy > 0 else -1, 0)
    if abs(dx) > abs(dy):
        return (0, 1 if dx > 0 else -1)
    return None


def _transitions(history):
    """[(grid_before, key, grid_after)] for consecutive same-level entries."""
    out = []
    prev = None
    for e in history or []:
        f = getattr(e, "frame", None)
        if f is None:
            prev = None; continue
        g = _grid_of(f); lv = getattr(f, "level", 0)
        if prev is not None and prev[1] == lv:
            out.append((prev[0], _action_name(getattr(e, "action", "")), g))
        prev = (g, lv)
    return out


def _learn(transitions):
    """Which colour the keys move, how each key moves it, what blocks it."""
    trs = [t for t in transitions if t[1] in _DIRS][-_LEARN_LAST:]
    if len(trs) < 4:
        return None
    H = len(trs[0][0]); W = len(trs[0][0][0])
    votes = Counter(); nblobs = {}; disp = defaultdict(lambda: defaultdict(Counter))
    passable = Counter(); floor = Counter(); shapes = defaultdict(set)
    refused = []                     # (grid, key) with no change at all
    for g0, key, g1 in trs:
        diff = [(r, c) for r in range(H) for c in range(W) if g0[r][c] != g1[r][c]]
        if not diff:
            refused.append((g0, key)); continue
        if len(diff) > 0.5 * H * W:
            continue
        cols = set(g0[r][c] for r, c in diff) | set(g1[r][c] for r, c in diff)
        for col in cols:
            b0 = _comps(g0, col); b1 = _comps(g1, col)
            if not b0 or len(b0) != len(b1) or len(b0) > 6:
                continue
            if sorted(x[2] for x in b0) != sorted(x[2] for x in b1):
                continue
            if sorted((x[0], x[1]) for x in b0) == sorted((x[0], x[1]) for x in b1):
                continue
            votes[col] += 1; nblobs.setdefault(col, len(b0))
            # body 0 = the component with the smallest (left, top); match by shape + nearest
            b0s = sorted(b0, key=lambda t: (t[1], t[0])); t0, l0, offs0, cells0 = b0s[0]
            best = None
            for t1, l1, offs1, _ in b1:
                if offs1 != offs0:
                    continue
                d = abs(t1 - t0) + abs(l1 - l0)
                if best is None or d < best[0]:
                    best = (d, t1 - t0, l1 - l0)
            if best is None or best[0] == 0:
                continue
            u = _unit(best[1], best[2]); mag = max(abs(best[1]), abs(best[2]))
            if u is None or mag > _MAX_STEP:
                continue
            disp[col][key][(u, mag)] += 1
            shapes[col].add(offs0)
            for oy, ox in offs0:                       # destination cells before the move
                y, x = t0 + best[1] + oy, l0 + best[2] + ox
                if 0 <= y < H and 0 <= x < W and g0[y][x] != col:
                    passable[(col, g0[y][x])] += 1
            for y, x in cells0:                        # vacated cells after the move
                if g1[y][x] != col:
                    floor[(col, g1[y][x])] += 1
    if not votes:
        return None
    mover = max(votes, key=lambda c: (votes[c], -nblobs.get(c, 99)))
    rules = {}
    for key, cnt in disp[mover].items():
        (u, mag), _ = cnt.most_common(1)[0]
        rules[key] = (u, mag)
    if not rules:
        return None
    for a, opp in (("UP", "DOWN"), ("DOWN", "UP"), ("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
        if a not in rules and opp in rules:
            (u, mag) = rules[opp]; rules[a] = ((-u[0], -u[1]), mag)
    fl = Counter({c: n for (m, c), n in floor.items() if m == mover})
    floor_c = fl.most_common(1)[0][0] if fl else None
    pas = {c for (m, c) in passable if m == mover}
    if floor_c is not None:
        pas.add(floor_c)
    blocking = Counter(); refused_edges = set()
    shape_set = shapes[mover]
    # A refused move is one where the BODY stayed put -- judged on the mover's
    # cells, not on the whole board, because a timer or HUD that ticks every
    # action would otherwise hide the refusal (sp80, 9/20 run). What stopped it
    # is a blocking colour when one sits in the way; when nothing visible does,
    # the (position, key) pair itself is remembered as a refused edge.
    for g0, key, g1 in trs:
        if key not in rules:
            continue
        b0 = [c for c in _comps(g0, mover, cap=40) if c[2] in shape_set]
        if not b0:
            continue
        b1 = [c for c in _comps(g1, mover, cap=40) if c[2] in shape_set]
        if sorted((t, l) for t, l, _, _ in b0) != sorted((t, l) for t, l, _, _ in b1):
            continue                                   # something moved: not a refusal
        (u, mag) = rules[key]
        for t0, l0, offs0, _ in b0:
            ahead = []
            for oy, ox in offs0:
                y, x = t0 + u[0] * mag + oy, l0 + u[1] * mag + ox
                if 0 <= y < H and 0 <= x < W:
                    if g0[y][x] != mover and g0[y][x] not in pas:
                        blocking[g0[y][x]] += 1
                    ahead.append(g0[y][x])
                else:
                    ahead.append(None)                 # off the board
            if ahead and all(v is not None and (v == mover or v in pas) for v in ahead):
                refused_edges.add((t0, l0, key))
    visited = set()
    for g0, key, g1 in trs:
        for g in (g0, g1):
            for t, l, offs, _ in _comps(g, mover, cap=40):
                if offs in shape_set:
                    visited.add((t, l))
    return {"mover": mover, "rules": rules, "floor": floor_c, "passable": pas, "visited": visited,
            "blocking": set(blocking), "refused": refused_edges, "shapes": shape_set,
            "n_bodies": nblobs.get(mover, 1), "moves_seen": sum(votes.values()), "H": H, "W": W}


def _model():
    hist = _gl().get("history") or []
    n = len(hist)
    if _CACHE["n"] == n and _CACHE["model"] is not None:
        return _CACHE["model"]
    m = _learn(_transitions(hist))
    _CACHE["n"] = n; _CACHE["model"] = m
    return m


def _locate(grid, m, near=None):
    """The body: a component of the mover colour with a learned shape, nearest to `near`."""
    cands = [c for c in _comps(grid, m["mover"], cap=40) if c[2] in m["shapes"]] or _comps(grid, m["mover"], cap=4)
    if not cands:
        return None
    if near is None:
        return sorted(cands, key=lambda t: (t[1], t[0]))[0]
    return min(cands, key=lambda t: abs(t[0] - near[0]) + abs(t[1] - near[1]))


def _static(grid, m):
    """The board with the body colour erased to floor (so the body never blocks itself)."""
    fl = m["floor"] if m["floor"] is not None else 0
    mv = m["mover"]
    return [[fl if v == mv else v for v in row] for row in grid]


def _plan(static, start, offs, m, goal_fn, optimistic=False):
    H = m["H"]; W = m["W"]; pas = m["passable"]; blk = m["blocking"]
    hmax = max(o[0] for o in offs); wmax = max(o[1] for o in offs)

    def can(t, l):
        if t < 0 or l < 0 or t + hmax >= H or l + wmax >= W:
            return False
        for oy, ox in offs:
            v = static[t + oy][l + ox]
            if optimistic:
                if v in blk:
                    return False
            elif v not in pas:
                return False
        return True

    if goal_fn(start, None):
        return []
    seen = {start}; q = deque([(start, [])]); nodes = 0
    while q:
        s, path = q.popleft()
        for key in _KEYS:
            if key not in m["rules"] or (s[0], s[1], key) in m.get("refused", ()):
                continue
            (u, mag) = m["rules"][key]
            t, l = s[0] + u[0] * mag, s[1] + u[1] * mag
            if goal_fn((t, l), key):
                return path + [key]
            if not can(t, l):
                continue
            if (t, l) in seen:
                continue
            seen.add((t, l)); q.append(((t, l), path + [key])); nodes += 1
            if nodes > _MAX_NODES:
                return None
    return None


def body():
    """Facts about the body the arrow keys move, measured from your own moves."""
    m = _model()
    if m is None:
        return {"error": "not enough arrow-key moves recorded yet (need a few that changed the board)"}
    cf = _gl().get("current_frame")
    b = _locate(_grid_of(cf), m) if cf is not None else None
    out = {"colour": m["mover"], "n_bodies": m["n_bodies"], "learned_from_moves": m["moves_seen"],
           "step": {k: v[1] for k, v in m["rules"].items()},
           "direction": {k: [v[0][0], v[0][1]] for k, v in m["rules"].items()},
           "floor": m["floor"], "passable_colours": sorted(m["passable"]), "blocking_colours": sorted(m["blocking"])}
    if b is not None:
        t, l, offs, cells = b
        out.update({"top": t, "left": l, "height": max(o[0] for o in offs) + 1, "width": max(o[1] for o in offs) + 1, "cells": len(cells)})
    return out


def _execute(m, goal_fn, target_desc):
    act = _gl().get("action")
    cf = _gl().get("current_frame")
    if cf is None or act is None:
        return {"reached": False, "reason": "no frame or action() available"}
    grid = _grid_of(cf); level0 = getattr(cf, "level", None)
    b = _locate(grid, m)
    if b is None:
        return {"reached": False, "reason": "cannot find the body on the board"}
    pos = (b[0], b[1]); offs = b[2]
    taken = 0; replans = 0; stopped = None; last = None; trace = []
    while True:
        static = _static(grid, m)
        plan = _plan(static, pos, offs, m, goal_fn)
        mode = "strict"
        if plan is None:
            plan = _plan(static, pos, offs, m, goal_fn, optimistic=True); mode = "optimistic"
        if plan is None:
            return {"reached": False, "reason": "no path found with the measured rules", "body_at": [pos[0], pos[1]],
                    "target": target_desc, "actions_taken": taken, "replans": replans, "trace": trace}
        if not plan:
            return {"reached": True, "body_at": [pos[0], pos[1]], "target": target_desc, "actions_taken": taken, "replans": replans, "trace": trace}
        # Execute in one batch only the prefix whose every destination the body has
        # stood on before; the first step onto unproven ground goes alone, so an
        # invisible refusal costs one action, not a batch (sp80 lost 5 of 8 that way).
        visited = m.setdefault("visited", set()); proven = 0; ct, cl = pos
        for key in plan[:_CHUNK]:
            (u, mag) = m["rules"][key]; ct, cl = ct + u[0] * mag, cl + u[1] * mag
            if (ct, cl) in visited:
                proven += 1
            else:
                break
        chunk = plan[:max(1, proven)]
        # predicted position after the chunk
        pt, pl = pos
        for key in chunk:
            (u, mag) = m["rules"][key]; pt, pl = pt + u[0] * mag, pl + u[1] * mag
        res = act([{"action": k} for k in chunk]) or {}
        last = res
        taken += int(res.get("executed_count") or len(chunk))
        cf = _gl().get("current_frame"); grid = _grid_of(cf)
        _b = _locate(grid, m, near=(pt, pl))
        trace.append({"keys": "".join(k[0] for k in chunk), "from": [pos[0], pos[1]], "predicted": [pt, pl],
                      "actual": [_b[0], _b[1]] if _b else None, "executed": int(res.get("executed_count") or 0)})
        if res.get("game_over"):
            return {"reached": False, "reason": "game over during the plan (a hazard?)", "actions_taken": taken, "replans": replans, "target": target_desc, "trace": trace}
        if res.get("level_completed") or res.get("run_complete") or getattr(cf, "level", level0) != level0:
            return {"reached": True, "level_completed": True, "actions_taken": taken, "replans": replans, "target": target_desc, "trace": trace}
        b = _locate(grid, m, near=(pt, pl))
        if b is None:
            return {"reached": False, "reason": "lost the body after moving", "actions_taken": taken, "replans": replans, "target": target_desc, "trace": trace}
        pos = (b[0], b[1]); offs = b[2]
        m.setdefault("visited", set()).add(pos)
        if len(chunk) == len(plan) and goal_fn(pos, None):
            return {"reached": True, "body_at": [pos[0], pos[1]], "target": target_desc, "actions_taken": taken, "replans": replans, "mode": mode, "trace": trace}
        if pos != (pt, pl):
            replans += 1
            if replans > _MAX_REPLANS:
                return {"reached": False, "reason": "the board kept disagreeing with the measured rules", "body_at": [pos[0], pos[1]],
                        "target": target_desc, "actions_taken": taken, "replans": replans, "trace": trace}
            # which keys took effect: the subset of the chunk's displacements that sums
            # to what was observed (prefer the most keys executed, refusals earliest)
            start = trace[-1]["from"]; need = (pos[0] - start[0], pos[1] - start[1])
            vecs = [(m["rules"][k][0][0] * m["rules"][k][1], m["rules"][k][0][1] * m["rules"][k][1]) for k in chunk]
            best = None
            for mask in range(1 << len(chunk)):
                dy = sum(v[0] for i, v in enumerate(vecs) if mask >> i & 1)
                dx = sum(v[1] for i, v in enumerate(vecs) if mask >> i & 1)
                if (dy, dx) == need:
                    score = (bin(mask).count("1"), -mask)
                    if best is None or score > best[0]:
                        best = (score, mask)
            invisible = 0
            if best is not None:
                mask = best[1]; cur = (start[0], start[1]); g_before = trace[-1].get("_grid")
                for i, k in enumerate(chunk):
                    if mask >> i & 1:
                        cur = (cur[0] + vecs[i][0], cur[1] + vecs[i][1]); continue
                    # key k was refused at `cur`: a visible blocker, or a refused edge
                    (u, mag) = m["rules"][k]; seen_block = False
                    for oy, ox in offs:
                        y, x = cur[0] + u[0] * mag + oy, cur[1] + u[1] * mag + ox
                        if 0 <= y < m["H"] and 0 <= x < m["W"]:
                            v = grid[y][x]
                            if v != m["mover"] and v not in m["passable"]:
                                m["blocking"].add(v); seen_block = True
                    if not seen_block:
                        m.setdefault("refused", set()).add((cur[0], cur[1], k)); invisible += 1
                        trace[-1].setdefault("refused_at", []).append([cur[0], cur[1], k])
            else:
                # moved somewhere the rules do not explain: fall back to colours ahead of the last key
                key = chunk[-1]; (u, mag) = m["rules"][key]
                for oy, ox in offs:
                    y, x = pos[0] + u[0] * mag + oy, pos[1] + u[1] * mag + ox
                    if 0 <= y < m["H"] and 0 <= x < m["W"]:
                        v = grid[y][x]
                        if v != m["mover"] and v != m["floor"]:
                            m["blocking"].add(v); m["passable"].discard(v)
            stopped = (stopped or 0) + (1 if invisible else 0)
            if stopped >= _MAX_INVISIBLE:
                return {"reached": False, "reason": "%d moves refused with nothing visible in the way (a door, a lock, or an edge of the level?) at %s"
                        % (stopped, sorted(m.get("refused", ()))), "body_at": [pos[0], pos[1]], "target": target_desc,
                        "actions_taken": taken, "replans": replans, "trace": trace}


def goto(row, col):
    """Move the body so its top-left cell is at (row, col). Plans with the measured
    rules, executes, re-plans if the board disagrees. Returns a dict."""
    m = _model()
    if m is None:
        return {"reached": False, "reason": "not enough arrow-key moves recorded yet to learn the rules"}
    target = (int(row), int(col))
    return _execute(m, lambda p, key: p == target, [target[0], target[1]])


def goto_colour(colour):
    """Move until the body touches a cell of `colour` (the last step may push into it)."""
    m = _model()
    if m is None:
        return {"reached": False, "reason": "not enough arrow-key moves recorded yet to learn the rules"}
    colour = int(colour)
    cf = _gl().get("current_frame"); grid = _grid_of(cf)
    b = _locate(grid, m)
    if b is None:
        return {"reached": False, "reason": "cannot find the body"}
    offs = b[2]; H = m["H"]; W = m["W"]

    def touches(p, key):
        for oy, ox in offs:
            y, x = p[0] + oy, p[1] + ox
            if 0 <= y < H and 0 <= x < W and grid[y][x] == colour:
                return True
        return False

    return _execute(m, touches, ["colour", colour])
