"""GP-G v4: a learned rigid-body physics model + in-head planning.

The v10 explorer pays one real action per edge of the abstract graph. On
two-body puzzles (m0r0: a key-following twin and a horizontally mirrored
twin, blocked independently by walls, level complete when they collide) the
joint state space is ~10^4 states, so model-free exploration cannot clear a
level in a 2,000-action budget while a human needs 29 moves. The human does
not explore: after a few moves they know the RULES and plan in their head.

This module does the same mechanically for the class of games whose movers
are rigid bodies with a constant step per key press:

  learn  - per entity ROLE: the unit direction each key moves it (a mirrored
           twin has LEFT -> +x); the STEP magnitude per level; which colours
           are PASSABLE (seen under a successful move's destination) and
           which BLOCK (seen under a refused move's destination); the FLOOR
           (what a vacated cell becomes). All fitted from observed
           (board, action, next_board) transitions - the human's level-1
           replay, then the agent's own moves.
  plan   - BFS over joint entity positions in a simulator built from those
           rules and the CURRENT level's static board, to a goal event
           inferred from level 1's completion (collide / reach colour).
  verify - after every real action, the observed positions are compared to
           the prediction; a mismatch updates passability or roles and the
           planner replans. A wrong model costs one action, not a burst.

No model in the loop; pure Python + numpy; CC0/MIT-0.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
KEYS = ("UP", "DOWN", "LEFT", "RIGHT")
MAX_STEP = 16          # one key press never moves a body further than this


def _unit(dy, dx):
    """Axis-snapped unit direction, or None for an ambiguous diagonal (a
    scrolling world drifts both axes; the dominant one is the real direction,
    a tie is not evidence)."""
    if abs(dy) > abs(dx):
        return (int(np.sign(dy)), 0)
    if abs(dx) > abs(dy):
        return (0, int(np.sign(dx)))
    return None


# ------------------------------------------------------------- components
def components(mask):
    """4-connected components of a boolean mask -> list of (top, left, offsets, cells)."""
    m = np.asarray(mask, dtype=bool)
    rows, cols = m.shape
    seen = np.zeros_like(m)
    out = []
    for r in range(rows):
        for c in range(cols):
            if m[r, c] and not seen[r, c]:
                stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and m[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
                top = min(y for y, _ in cells); left = min(x for _, x in cells)
                offs = tuple(sorted((y - top, x - left) for y, x in cells))
                out.append((top, left, offs, cells))
    return out


def decompose(comp, shapes):
    """Split a component into copies of known rigid shapes (touching twins
    form one component). Returns [(top, left, offs, cells)] or None."""
    top, left, offs, cells = comp
    remaining = set(cells)
    out = []
    while remaining:
        y0, x0 = min(remaining)                       # row-major first cell
        placed = False
        for sh in shapes:
            fy, fx = sh[0]                            # the shape's row-major first cell
            t, l = y0 - fy, x0 - fx
            body = [(t + oy, l + ox) for oy, ox in sh]
            if all(c in remaining for c in body):
                remaining.difference_update(body)
                out.append((t, l, sh, body)); placed = True
                break
        if not placed:
            return None
    return out


def anchor_bodies(trans, colour, chrome=None, shapes=None, cap=40):
    """Find the first transition in which blobs of `colour` moved, and return
    (index, [(top, left, offs)]) for those blobs BEFORE the move. On a board
    with dozens of same-coloured specks (a one-cell avatar among one-cell
    decorations) this is the only reliable way to say which one is the body."""
    for i, (g, a, ng) in enumerate(trans):
        if a not in DIRS:
            continue
        b1 = entities_of(g, colour, chrome, cap=cap, shapes=shapes)
        b2 = entities_of(ng, colour, chrome, cap=cap, shapes=shapes)
        pos2 = {(t, l) for t, l, _, _ in b2}
        moved = [(t, l, offs) for t, l, offs, _ in b1 if (t, l) not in pos2]
        if moved and len(moved) <= 4:
            return i, moved
    return None, []


def moving_shapes(trans, colour, chrome=None, shapes=None):
    """Shapes of the blobs that actually moved during these transitions.
    Static decorations in the avatar's colour drop out by construction."""
    out = set()
    for g, a, ng in trans:
        if a not in DIRS:
            continue
        b1 = entities_of(g, colour, chrome, cap=12, shapes=shapes)
        b2 = entities_of(ng, colour, chrome, cap=12, shapes=shapes)
        if not b1 or len(b1) != len(b2):
            continue
        pos2 = {(t, l) for t, l, _, _ in b2}
        for t, l, offs, _ in b1:
            if (t, l) not in pos2:
                out.add(offs)
    return out


def entities_of(board, colour, chrome=None, cap=4, shapes=None, keep=None):
    g = np.asarray(board)
    m = g == colour
    if chrome is not None:
        m &= ~chrome
    comps = components(m)
    if shapes:
        split = []
        for comp in comps:
            if comp[2] in shapes:
                split.append(comp); continue
            parts = decompose(comp, shapes)
            split.extend(parts if parts else [comp])
        comps = split
    if keep:
        kept = [c for c in comps if c[2] in keep]
        if kept:
            comps = kept
    comps.sort(key=lambda t: (-len(t[3]), t[0], t[1]))
    return comps[:cap]


def match(prev, cur):
    """Greedy nearest matching of entity lists -> list of (i_prev, j_cur, dy, dx)."""
    pairs = []
    used = set()
    for i, (t, l, offs, _) in enumerate(prev):
        best = None
        for j, (t2, l2, offs2, _) in enumerate(cur):
            if j in used:
                continue
            d = abs(t - t2) + abs(l - l2) + (0 if offs == offs2 else 50)
            if best is None or d < best[0]:
                best = (d, j, t2 - t, l2 - l)
        if best is not None:
            used.add(best[1]); pairs.append((i, best[1], best[2], best[3]))
    return pairs


# ------------------------------------------------------------------ rules
class Physics:
    """Rules learned from transitions; per-level state lives in `LevelModel`."""

    def __init__(self, mover, chrome=None):
        self.mover = int(mover)
        self.chrome = None if chrome is None else np.asarray(chrome, dtype=bool)
        # role tables learned at level 1, ordered by the entities' start x
        self.role_tables: list[dict] = []
        self.role_step = None          # step magnitude at level 1
        self.role_steps = {}           # per-action magnitude at level 1
        self.role_size = None          # entity size at level 1 (for the step prior)
        self.shapes = ()               # rigid footprints seen at level 1
        self.floor = None
        self.passable = set()
        self.blocking = set()
        self.sliding = False           # partial moves observed (bp35-style)
        self.deadly = set()            # colours whose entry resets the level
        self.goal_kind = "frontier"
        self.goal_colours = set()
        self.goal_area = None          # how rare the goal colour was where it was learned

    # ---- fitting from a whole level's transitions --------------------
    def fit_level1(self, trans):
        """trans: [(board, action_name, next_board)] of one level (the human's)."""
        first = np.asarray(trans[0][0], dtype=np.int16)
        ents = entities_of(first, self.mover, self.chrome)
        self.shapes = tuple(sorted({e[2] for e in ents}, key=len, reverse=True))
        order = sorted(range(len(ents)), key=lambda i: (ents[i][1], ents[i][0]))
        ids = {i: k for k, i in enumerate(order)}          # entity index -> role id
        tables = [defaultdict(Counter) for _ in ents]
        steps = Counter(); disp_all = []; step_by_action = defaultdict(Counter)
        floor = Counter(); passable = Counter(); blocking = Counter()
        cur = ents
        for b, a, nb in trans:
            if a not in DIRS:
                cur = entities_of(np.asarray(nb, dtype=np.int16), self.mover, self.chrome, shapes=self.shapes)
                continue
            g = np.asarray(b, dtype=np.int16); ng = np.asarray(nb, dtype=np.int16)
            prev = cur; cur = entities_of(ng, self.mover, self.chrome, shapes=self.shapes)
            if len(prev) != len(ents) or len(cur) != len(ents):
                # a merge/vanish: keep going with what we can track
                if len(cur) == len(ents):
                    prev = cur
                continue
            for i, j, dy, dx in match(prev, cur):
                rid = ids.get(i)
                if rid is None or prev[i][2] != cur[j][2]:
                    continue                                  # shape changed: a level change, not a move
                if dy or dx:
                    u = _unit(dy, dx)
                    if u is None:
                        continue                              # ambiguous diagonal: not evidence
                    tables[rid][a][u] += 1
                    mag = max(abs(dy), abs(dx))
                    if mag > MAX_STEP:
                        continue                              # a level change or a teleport, not a step
                    steps[mag] += 1; disp_all.append(mag); step_by_action[a][mag] += 1
                    # vacated cells -> floor; destination cells -> passable
                    t, l, offs, cells = prev[i]
                    for y, x in cells:
                        if ng[y, x] != self.mover:
                            floor[int(ng[y, x])] += 1
                    for oy, ox in offs:
                        y, x = t + dy + oy, l + dx + ox
                        if g[y, x] != self.mover:
                            passable[int(g[y, x])] += 1
                else:
                    tables[rid][a][(0, 0)] += 1
            # keep identity: re-order cur to prev's matching so ids persist
            m = match(prev, cur)
            cur = [cur[j] for _, j, _, _ in sorted(m)] if len(m) == len(cur) else cur
        self.role_tables = []
        for rid in range(len(ents)):
            tab = {}
            for a, cnt in tables[rid].items():
                moved = {u: n for u, n in cnt.items() if u != (0, 0)}
                if moved:
                    tab[a] = max(moved, key=moved.get)
            for a, opp in (("UP", "DOWN"), ("DOWN", "UP"), ("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
                if a not in tab and opp in tab:
                    tab[a] = (-tab[opp][0], -tab[opp][1])     # symmetry prior, verified online
            self.role_tables.append(tab)
        if steps:
            self.role_step = max(steps, key=steps.get)
            mags = sorted(set(disp_all))
            self.sliding = len(mags) > 1
        self.role_steps = {a: max(c, key=c.get) for a, c in step_by_action.items() if c}
        self.role_size = len(ents[0][3]) if ents else None
        self.floor = max(floor, key=floor.get) if floor else None
        self.passable = set(passable) | ({self.floor} if self.floor is not None else set())
        self.blocking = set()
        # blocked moves: colours under the refused destination
        cur = ents
        for b, a, nb in trans:
            if a not in DIRS:
                cur = entities_of(np.asarray(nb, dtype=np.int16), self.mover, self.chrome, shapes=self.shapes); continue
            g = np.asarray(b, dtype=np.int16); ng = np.asarray(nb, dtype=np.int16)
            prev = cur; cur = entities_of(ng, self.mover, self.chrome, shapes=self.shapes)
            if len(prev) != len(cur):
                continue
            m = match(prev, cur)
            for i, j, dy, dx in m:
                if dy == 0 and dx == 0 and i < len(self.role_tables):
                    u = self.role_tables[i].get(a)
                    st = self.role_steps.get(a, self.role_step)
                    if u is None or st is None:
                        continue
                    t, l, offs, _ = prev[i]
                    for oy, ox in offs:
                        y, x = t + u[0] * st + oy, l + u[1] * st + ox
                        if 0 <= y < 64 and 0 <= x < 64 and g[y, x] != self.mover:
                            blocking[int(g[y, x])] += 1
            cur = [cur[j] for _, j, _, _ in sorted(m)] if len(m) == len(cur) else cur
        self.blocking = {c for c in blocking if c not in self.passable}

    def infer_goal(self, trans):
        """Goal event from the completing transition of level 1."""
        b, a, nb = trans[-1]
        g = np.asarray(b, dtype=np.int16)
        ents = entities_of(g, self.mover, self.chrome, shapes=self.shapes)
        if a not in DIRS or not ents:
            self.goal_kind = "frontier"; return
        # collide: two entities whose tentative destinations intersect or cross
        if len(ents) >= 2 and self.role_tables and self.role_step:
            lm = LevelModel(self, g)
            if lm.ok and lm.collides(lm.state, a):
                self.goal_kind = "collide"; return
        # reach: colours in the tentative destination footprint that are not passable
        if self.role_tables and self.role_step:
            ahead = Counter()
            st = self.role_steps.get(a, self.role_step)
            for k, (t, l, offs, _) in enumerate(ents):
                u = self.role_tables[min(k, len(self.role_tables) - 1)].get(a)
                if u is None:
                    continue
                for oy, ox in offs:
                    y, x = t + u[0] * st + oy, l + u[1] * st + ox
                    if 0 <= y < 64 and 0 <= x < 64:
                        c = int(g[y, x])
                        if c != self.mover and c != self.floor:
                            ahead[c] += 1
            if ahead:
                # a door is RARE; a corridor colour under the avatar's last step
                # is not the goal, it is the floor of the last room
                area = {c: int((g == c).sum()) for c in ahead}
                c = min(area, key=area.get)
                if area[c] <= 0.05 * g.size:
                    self.goal_kind = "reach"; self.goal_colours = {c}
                    self.goal_area = area[c]
                    return
        self.goal_kind = "frontier"


class LevelModel:
    """The simulator for one level: static board + entity footprints + rules."""

    def __init__(self, phys: Physics, board, roles=None, step=None, keep=None, bodies=None):
        self.P = phys
        self.keep = keep
        g = np.asarray(board, dtype=np.int16)
        if bodies:
            self.ents = [(t, l, offs, [(t + oy, l + ox) for oy, ox in offs]) for t, l, offs in bodies]
        else:
            self.ents = entities_of(g, phys.mover, phys.chrome, shapes=phys.shapes, keep=keep)
        self.ok = bool(self.ents) and bool(phys.role_tables)
        self.state = (); self.offs = []; self.visited = set(); self.observed = {}; self.tried = defaultdict(set)
        self.shapes = phys.shapes; self.steps = {a: (phys.role_step or 1) for a in KEYS}
        self.confirmed = set(); self.obs_dirs = []; self.roles = []
        self.passable = set(phys.passable); self.blocking = set(phys.blocking); self.deadly = set(phys.deadly)
        self.start_state = (); self.optimistic = False
        self.hits = 0; self.misses = 0        # predictions checked on this level
        if not self.ok:
            return
        self.shapes = tuple(sorted({e[2] for e in self.ents}, key=len, reverse=True))
        order = sorted(range(len(self.ents)), key=lambda i: (self.ents[i][1], self.ents[i][0]))
        self.ents = [self.ents[i] for i in order]
        n = len(self.ents)
        self.roles = list(roles) if roles is not None else [min(k, len(phys.role_tables) - 1) for k in range(n)]
        # step prior: level 1's step (the footprint's side if the size changed);
        # the first successful move on this level CONFIRMS the real step
        self.confirmed = set()         # actions whose step was observed on this level
        same_size = phys.role_size == len(self.ents[0][3])
        if step is not None:
            prior = step; self.confirmed = set(KEYS)
        elif same_size or phys.role_step is None:
            prior = phys.role_step or 1
        else:
            offs = self.ents[0][2]
            h = max(o[0] for o in offs) + 1; w = max(o[1] for o in offs) + 1
            prior = max(1, min(h, w))
        self.steps = {a: (phys.role_steps.get(a, prior) if same_size else prior) for a in KEYS}
        self.observed = {}             # (state, action) -> next state or None (death): ground truth
        self.visited = set()
        self.tried = defaultdict(set)  # state -> extra (unmodelled) actions already tried there
        self.start_board = g.copy()
        self.obs_dirs = [dict() for _ in self.ents]   # entity -> {action: unit dir} seen on this level
        # colour roles swap between levels (a wall colour here is a path colour
        # there), so passability is LOCAL, seeded from the global priors
        self.passable = set(phys.passable); self.blocking = set(phys.blocking); self.deadly = set(phys.deadly)
        self.offs = [offs for _, _, offs, _ in self.ents]
        self.state = tuple((t, l) for t, l, _, _ in self.ents)
        self.start_state = self.state
        self.visited.add(self.state)
        self._set_terrain(g)
        self.rebuild(optimistic=False)

    def _set_terrain(self, g):
        """The obstacle map is the CURRENT frame with the bodies erased: a
        world that scrolls under the avatar, a door that opens, a block that
        was pushed -- all of it is re-read rather than remembered."""
        static = np.asarray(g, dtype=np.int16).copy()
        m = static == self.P.mover
        if self.P.chrome is not None:
            m &= ~self.P.chrome
        static[m] = self.P.floor if self.P.floor is not None else 0
        self.static = static

    def locate(self, board, reach=None):
        """Where are the bodies now? Each is matched to the nearest blob of its
        own shape within a plausible distance -- identity by continuity, not by
        a fresh scan (a scan cannot tell the avatar from an identical speck)."""
        g = np.asarray(board, dtype=np.int16)
        cands = entities_of(g, self.P.mover, self.P.chrome, cap=60, shapes=self.shapes)
        if reach is None:
            reach = 3 * max(self.steps.values()) + 4
        for limit in (reach, 1 << 20):     # nearby first; then anywhere (a level
            out = []; taken = set()        # reset teleports the bodies home)
            for k, (t, l) in enumerate(self.state):
                offs = self.offs[k]; best = None
                for j, (t2, l2, o2, _) in enumerate(cands):
                    if j in taken or o2 != offs:
                        continue
                    d = abs(t2 - t) + abs(l2 - l)
                    if d <= limit and (best is None or d < best[0]):
                        best = (d, j, t2, l2)
                if best is None:
                    break
                taken.add(best[1]); out.append((best[2], best[3]))
            if len(out) == len(self.state):
                return tuple(out)
        return None

    def refresh(self, board):
        """Re-read the terrain and the bodies' positions from a live frame."""
        g = np.asarray(board, dtype=np.int16)
        st = self.locate(g)
        moved = False
        if st is not None:
            moved = st != self.state
            self.state = st; self.visited.add(st)
        self._set_terrain(g)
        self.rebuild(optimistic=self.optimistic)
        return moved

    def rebuild(self, optimistic=False):
        P = self.P
        pas = np.isin(self.static, sorted(self.passable))
        if optimistic:
            pas |= ~np.isin(self.static, sorted(self.blocking | self.deadly))
        if P.chrome is not None:
            pas &= ~P.chrome
        self.optimistic = optimistic
        # can-occupy maps per entity: ok[t, l] iff every footprint cell passable;
        # dead[t, l] iff any footprint cell is a known hazard
        self.occ = []
        self.goal_hit = []
        self.dead_hit = []
        goalmask = np.isin(self.static, sorted(P.goal_colours)) if P.goal_colours else None
        deadmask = np.isin(self.static, sorted(self.deadly)) if self.deadly else None
        for offs in self.offs:
            h = max(o[0] for o in offs) + 1; w = max(o[1] for o in offs) + 1
            ok = np.ones((64 - h + 1, 64 - w + 1), dtype=bool)
            gh = np.zeros_like(ok); dh = np.zeros_like(ok)
            for oy, ox in offs:
                ok &= pas[oy:oy + ok.shape[0], ox:ox + ok.shape[1]]
                if goalmask is not None:
                    gh |= goalmask[oy:oy + ok.shape[0], ox:ox + ok.shape[1]]
                if deadmask is not None:
                    dh |= deadmask[oy:oy + ok.shape[0], ox:ox + ok.shape[1]]
            self.occ.append(ok); self.goal_hit.append(gh); self.dead_hit.append(dh)

    def dir_of(self, k, action):
        """Observed direction on this level beats the level-1 role prior."""
        u = self.obs_dirs[k].get(action)
        return u if u is not None else self.P.role_tables[self.roles[k]].get(action)

    def tentative(self, state, action):
        out = []
        for k, (t, l) in enumerate(state):
            u = self.dir_of(k, action)
            if u is None:
                out.append((t, l, False)); continue
            st = self.steps[action]
            out.append((t + u[0] * st, l + u[1] * st, True))
        return out

    def _can(self, k, t, l):
        ok = self.occ[k]
        return 0 <= t < ok.shape[0] and 0 <= l < ok.shape[1] and bool(ok[t, l])

    def _dead(self, k, t, l):
        dh = self.dead_hit[k]
        return 0 <= t < dh.shape[0] and 0 <= l < dh.shape[1] and bool(dh[t, l])

    def step_state(self, state, action, ignore_dead=False):
        """Joint next state, or None if any entity would enter a hazard.
        An observed transition overrides the model (ignore_dead asks for the
        would-be position even where a hazard is believed)."""
        if (state, action) in self.observed:
            rec = self.observed[(state, action)]
            if rec is not None or not ignore_dead:
                return rec
        nxt = []
        for k, (t, l) in enumerate(state):
            u = self.dir_of(k, action)
            if u is None:
                nxt.append((t, l)); continue
            st = self.steps[action]
            if not ignore_dead and self._dead(k, t + u[0] * st, l + u[1] * st):
                return None
            if self.P.sliding:
                best = (t, l)
                for m in range(1, st + 1):
                    tt, ll = t + u[0] * m, l + u[1] * m
                    if self._can(k, tt, ll):
                        best = (tt, ll)
                    else:
                        break
                nxt.append(best)
            else:
                tt, ll = t + u[0] * st, l + u[1] * st
                nxt.append((tt, ll) if self._can(k, tt, ll) else (t, l))
        return tuple(nxt)

    def collides(self, state, action):
        """Two entities' ACTUAL destinations (after blocking) intersect, or
        their horizontal order flips (they pass through each other)."""
        nxt = self.step_state(state, action)
        if nxt is None:
            return False
        tent = [(t, l, (t, l) != state[k]) for k, (t, l) in enumerate(nxt)]
        n = len(state)
        for a in range(n):
            for b in range(a + 1, n):
                (t1, l1, m1), (t2, l2, m2) = tent[a], tent[b]
                if not (m1 or m2):
                    continue
                c1 = {(t1 + oy, l1 + ox) for oy, ox in self.offs[a]}
                c2 = {(t2 + oy, l2 + ox) for oy, ox in self.offs[b]}
                if c1 & c2:
                    return True
                # crossing: same row band before and after, horizontal order flips
                (pt1, pl1), (pt2, pl2) = state[a], state[b]
                h1 = max(o[0] for o in self.offs[a]) + 1; h2 = max(o[0] for o in self.offs[b]) + 1
                rows_overlap = pt1 < pt2 + h2 and pt2 < pt1 + h1 and t1 < t2 + h2 and t2 < t1 + h1
                if rows_overlap and (pl1 - pl2) * (l1 - l2) < 0:
                    return True
        return False

    def reaches(self, state, action):
        if self.step_state(state, action) is None:
            return False
        for k, (t, l) in enumerate(state):
            u = self.dir_of(k, action)
            if u is None:
                continue
            st = self.steps[action]
            tt, ll = t + u[0] * st, l + u[1] * st
            gh = self.goal_hit[k]
            if 0 <= tt < gh.shape[0] and 0 <= ll < gh.shape[1] and gh[tt, ll]:
                return True
        return False

    def is_goal(self, state, action):
        if self.P.goal_kind == "collide":
            return self.collides(state, action)
        if self.P.goal_kind == "reach":
            return self.reaches(state, action)
        return False

    def reachable_count(self, cap=4000):
        seen = {self.state}; q = deque([self.state])
        while q and len(seen) < cap:
            st = q.popleft()
            for a in KEYS:
                s2 = self.step_state(st, a)
                if s2 is not None and s2 != st and s2 not in seen:
                    seen.add(s2); q.append(s2)
        return len(seen)

    def plan(self, max_states=60000, frontier=False, extra=()):
        """BFS over joint positions to the first goal event (or, in frontier
        mode, to the nearest joint state never visited on this level).
        Returns an action list or None."""
        kind = self.P.goal_kind
        if kind == "reach":
            n = int(np.isin(self.static, sorted(self.P.goal_colours)).sum())
            if n == 0 or n > 0.05 * self.static.size:
                frontier = True          # the colour is absent here, or so common it cannot be a door
        if kind not in ("collide", "reach"):
            frontier = True
        start = self.state
        seen = {start}; q = deque([(start, [])])
        while q:
            s, path = q.popleft()
            if frontier:
                for xa in extra:
                    if xa not in self.tried[s]:
                        return path + [xa]        # an untried lever here: pull it
            for a in KEYS:
                s2 = self.step_state(s, a)
                if s2 is None:
                    continue                                  # a hazard: never plan through it
                if not frontier and self.is_goal(s, a):
                    return path + [a]
                if s2 != s and s2 not in seen:
                    if frontier and s2 not in self.visited:
                        return path + [a]
                    seen.add(s2); q.append((s2, path + [a]))
                    if len(seen) > max_states:
                        return None
        return None


class PhysicsPlanner:
    """Online controller: plan, act one step, verify, repair, replan."""

    def __init__(self, phys: Physics, log=None):
        self.P = phys; self.log = log or (lambda s: None)
        self.level = None; self.lm = None; self.plan = []; self.pending = None
        self.roles = None; self.stats = Counter(); self.frontier_fallback = True
        self.extra_actions = ()        # actions with no dynamics model (SPACE): tried once per state
        self.failed_plans = 0

    def _new_level(self, board, level, keep=None, bodies=None):
        self.level = level; self.roles = None
        self.lm = LevelModel(self.P, board, keep=keep, bodies=bodies)
        self.plan = []; self.pending = None; self.failed_plans = 0

    def observe(self, board_after):
        """Call after the pending action executed; returns True if prediction held."""
        if self.pending is None or self.lm is None or not self.lm.ok or not self.lm.state:
            self.pending = None
            return True
        action, prev_state = self.pending; self.pending = None
        if action not in DIRS:
            lm = self.lm
            lm.tried[prev_state].add(action)
            st = lm.locate(board_after)
            if st is not None:
                lm.state = st; lm.visited.add(st)
            self.stats["extra"] += 1
            return True
        expected_death = self.lm.step_state(prev_state, action) is None
        predicted = self.lm.step_state(prev_state, action, ignore_dead=True)
        g = np.asarray(board_after, dtype=np.int16)
        observed = self.lm.locate(g)
        if observed is None:
            self.stats["lost_bodies"] += 1; self.plan = []
            return False
        prev_state = self.lm.state
        self.lm.visited.add(observed)
        if observed == predicted:
            self.lm.observed[(prev_state, action)] = observed
            if observed != prev_state:
                self.lm.confirmed.add(action)
            self.lm.state = observed; self.lm.hits += 1; self.stats["verified"] += 1; return True
        self.lm.misses += 1; self.stats["mismatch"] += 1
        if expected_death and observed != self.lm.start_state:
            self._unlearn_hazard(action, prev_state)
        jump = any(max(abs(ot - t), abs(ol - l)) > self.lm.steps[action] for (t, l), (ot, ol) in zip(self.lm.state, observed))
        if jump and observed == self.lm.start_state and predicted != self.lm.start_state and self.lm.state != self.lm.start_state:
            self.lm.observed[(prev_state, action)] = None
            self._learn_death(action)
            self.lm.state = observed; self.plan = []
            return False
        self.lm.observed[(prev_state, action)] = observed
        self._repair(action, predicted, observed, g)
        self.lm.state = observed; self.plan = []
        return False

    def _unlearn_hazard(self, action, prev_state):
        """A move we believed fatal was not: the colours under that footprint
        are not hazards on this level."""
        lm = self.lm; freed = set()
        for k, (t, l) in enumerate(prev_state):
            u = lm.dir_of(k, action)
            if u is None:
                continue
            st = lm.steps[action]
            for oy, ox in lm.offs[k]:
                y, x = t + u[0] * st + oy, l + u[1] * st + ox
                if 0 <= y < 64 and 0 <= x < 64:
                    c = int(lm.static[y, x])
                    if c in lm.deadly:
                        freed.add(c)
        if freed:
            lm.deadly -= freed; lm.rebuild(optimistic=lm.optimistic)
            self.log(f"physics: {sorted(freed)} is not fatal after all; deadly now {sorted(lm.deadly)}")

    def _learn_death(self, action):
        """The level reset after this action: every non-floor colour under an
        entity's intended footprint is a hazard (one death per colour)."""
        P = self.P; lm = self.lm; found = set()
        for k, (t, l) in enumerate(lm.state):
            u = lm.dir_of(k, action)
            if u is None:
                continue
            st = lm.steps[action]
            for oy, ox in lm.offs[k]:
                y, x = t + u[0] * st + oy, l + u[1] * st + ox
                if 0 <= y < 64 and 0 <= x < 64:
                    c = int(lm.static[y, x])
                    if c != P.floor and c not in lm.passable and c not in lm.blocking:
                        found.add(c)
        if not found:
            # every colour under the footprints was thought passable: the least trusted one
            for k, (t, l) in enumerate(lm.state):
                u = lm.dir_of(k, action)
                if u is None:
                    continue
                st = lm.steps[action]
                for oy, ox in lm.offs[k]:
                    y, x = t + u[0] * st + oy, l + u[1] * st + ox
                    if 0 <= y < 64 and 0 <= x < 64 and int(lm.static[y, x]) != P.floor:
                        found.add(int(lm.static[y, x]))
        lm.deadly |= found; lm.passable -= found; P.deadly |= found; P.passable -= found
        self.stats["deaths"] += 1
        self.log(f"physics: level reset after {action}; deadly colours now {sorted(lm.deadly)}")
        lm.rebuild(optimistic=lm.optimistic)

    def _repair(self, action, predicted, observed, g):
        P = self.P; lm = self.lm
        if action not in lm.confirmed:
            mags = [m for m in (max(abs(ot - t), abs(ol - l)) for (t, l), (ot, ol) in zip(lm.state, observed))
                    if 0 < m <= MAX_STEP]
            if mags:
                lm.steps[action] = max(set(mags), key=mags.count); lm.confirmed.add(action)
                self.log(f"physics: step of {action} confirmed {lm.steps[action]}")
                # roles may still be wrong; fall through only for direction checks
                for k, (t, l) in enumerate(lm.state):
                    ot, ol = observed[k]
                    if (ot, ol) != (t, l):
                        u = lm.dir_of(k, action); uo = _unit(ot - t, ol - l)
                        if uo is not None and u != uo:
                            self._fix_role(k, action, uo)
                lm.rebuild(optimistic=lm.optimistic)
                return
        for k, (t, l) in enumerate(lm.state):
            u = lm.dir_of(k, action)
            pt, pl = predicted[k]; ot, ol = observed[k]
            if (ot, ol) == (t, l) and (pt, pl) != (t, l):
                # refused: colours under the intended footprint block
                for oy, ox in lm.offs[k]:
                    y, x = pt + oy, pl + ox
                    if 0 <= y < 64 and 0 <= x < 64:
                        c = int(lm.static[y, x])
                        if c != P.floor:
                            lm.passable.discard(c); lm.blocking.add(c); P.blocking.add(c)
                self.log(f"physics: {action} refused for entity {k}; blocking now {sorted(lm.blocking)}")
            elif (ot, ol) != (t, l) and (pt, pl) == (t, l):
                # moved where we predicted a block: colours under it are passable
                for oy, ox in lm.offs[k]:
                    y, x = ot + oy, ol + ox
                    if 0 <= y < 64 and 0 <= x < 64:
                        c = int(lm.static[y, x]); lm.passable.add(c); lm.blocking.discard(c); P.passable.add(c)
                dy, dx = ot - t, ol - l
                uo = _unit(dy, dx)
                if uo is not None and u != uo:
                    self._fix_role(k, action, uo)
                mag = max(abs(dy), abs(dx))
                if mag != lm.steps[action] and 0 < mag <= MAX_STEP:
                    lm.steps[action] = mag; self.log(f"physics: step of {action} -> {mag}")
                self.log(f"physics: entity {k} moved where a block was predicted; passable now {sorted(lm.passable)}")
            elif (ot, ol) != (pt, pl):
                dy, dx = ot - t, ol - l
                uo = _unit(dy, dx)
                if uo is not None and u != uo:
                    self._fix_role(k, action, uo)
                mag = max(abs(dy), abs(dx))
                if mag != lm.steps[action] and 0 < mag <= MAX_STEP:
                    old_st = lm.steps[action]
                    if mag < old_st:
                        P.sliding = True
                    else:
                        lm.steps[action] = mag
                    self.log(f"physics: {action} moved {mag} where {old_st} was expected (sliding={P.sliding})")
        lm.rebuild(optimistic=lm.optimistic)

    def _fix_role(self, k, action, u):
        """Record the direction this entity actually moves under `action` on
        this level (the level-1 role is only a prior)."""
        self.lm.obs_dirs[k][action] = u
        self.log(f"physics: entity {k} moves {u} on {action} (observed)")

    def propose(self, board, level):
        """Next action name for this level, or None when physics cannot plan."""
        if self.level != level or self.lm is None:
            self._new_level(board, level)
        lm = self.lm
        if not lm.ok:
            return None
        if not self.plan:
            plan = lm.plan()
            if plan is None and not lm.optimistic:
                lm.rebuild(optimistic=True); plan = lm.plan()
                if plan is not None:
                    self.stats["optimistic_plans"] += 1
            if plan is None and self.frontier_fallback:
                plan = lm.plan(frontier=True, extra=self.extra_actions)   # explore what the model can reach
                if plan is not None:
                    self.stats["frontier_plans"] += 1
            if plan is None:
                self.stats["no_plan"] += 1
                if self.stats["no_plan"] <= 3 or self.stats["no_plan"] % 50 == 0:
                    n_goal = int(np.isin(lm.static, sorted(self.P.goal_colours)).sum()) if self.P.goal_colours else 0
                    reach = lm.reachable_count()
                    self.log(f"physics L{level}: NO PLAN (goal {self.P.goal_kind} cells {n_goal}, "
                             f"reachable {reach}, visited {len(lm.visited)}, optimistic {lm.optimistic}, "
                             f"state {lm.state}, passable {sorted(lm.passable)}, blocking {sorted(lm.blocking)})")
                return None
            self.plan = plan; self.stats["plans"] += 1; self.stats["plan_len"] += len(plan)
            self.log(f"physics L{level}: plan of {len(plan)} ({'optimistic' if lm.optimistic else 'strict'}), goal {self.P.goal_kind}; state {lm.state} roles {lm.roles} steps {lm.steps}: {' '.join(plan[:12])}")
        a = self.plan.pop(0)
        self.pending = (a, lm.state)
        return a


# ================================================================ harness stage
def chrome_mask(trans):
    """Cells that change in isolation (<=2 cells) or on >30% of within-level
    moves are instrumentation (counters, bars): the determinism-ladder rule."""
    if not trans:
        return None
    shape = np.asarray(trans[0][0]).shape
    chrome = np.zeros(shape, dtype=bool); count = np.zeros(shape); within = 0
    for g, _, ng in trans:
        d = np.asarray(g) != np.asarray(ng); k = int(d.sum())
        if k == 0:
            continue
        if k <= 2:
            chrome |= d
        if k <= 0.3 * d.size:
            within += 1; count += d
    if within:
        chrome |= (count / within) > 0.3
    return chrome


def detect_mover(trans, chrome, cap_cells=900):
    """The avatar's colour: among the colours that change under direction keys,
    the one with the FEWEST blobs (the family-library rule, proven on the 22
    public games), preferring colours whose blobs are seen to shift rigidly."""
    moves = Counter(); rigid = Counter(); boards = []
    for g, a, ng in trans:
        g = np.asarray(g, dtype=np.int16); ng = np.asarray(ng, dtype=np.int16)
        boards.append(g)
        if a not in DIRS:
            continue
        d = g != ng
        if chrome is not None:
            d &= ~chrome
        if not d.any() or d.sum() > 0.5 * d.size:
            continue
        for c in np.unique(np.concatenate([g[d], ng[d]])).tolist():
            moves[int(c)] += 1
        for c in np.unique(np.concatenate([g[d], ng[d]])).tolist():
            c = int(c); n1 = int((g == c).sum())
            if n1 == 0 or n1 > cap_cells or n1 != int((ng == c).sum()):
                continue
            b1 = components((g == c) & ~chrome if chrome is not None else (g == c))
            b2 = components((ng == c) & ~chrome if chrome is not None else (ng == c))
            if len(b1) != len(b2) or len(b1) > 6:
                continue
            if sorted(x[2] for x in b1) != sorted(x[2] for x in b2):
                continue
            if sorted((x[0], x[1]) for x in b1) != sorted((x[0], x[1]) for x in b2):
                rigid[c] += 1
    if not moves:
        return None, 0
    allb = np.concatenate([b.ravel() for b in boards[::max(1, len(boards) // 20)]])
    vals, cnts = np.unique(allb, return_counts=True); bg = int(vals[cnts.argmax()])
    cands = [c for c, _ in moves.most_common(6) if c != bg]
    if not cands:
        return None, 0

    def median_blobs(c):
        counts = []
        for b in boards[::max(1, len(boards) // 8)]:
            m = (b == c)
            if chrome is not None:
                m = m & ~chrome
            counts.append(len(components(m)) or 99)
        counts.sort(); return counts[len(counts) // 2] if counts else 99

    scored = sorted(cands, key=lambda c: (0 if rigid[c] >= 2 else 1, median_blobs(c), -rigid[c]))
    c = scored[0]
    return c, max(rigid[c], 1 if moves[c] >= 3 else 0)


def _parse(text):
    """'UP' -> 'UP'; 'MOUSE(row=3, col=17)' -> 'MOUSE'."""
    t = str(text or "").strip()
    return t.split("(")[0].strip().upper() if t else ""


class PhysicsStage:
    """The harness-facing controller: ingest the agent's history, learn the
    physics from its own moves, and take over a level with in-head plans.

    propose(grid, level, history, valid) -> [action spec] or None
      history: [(action_text, grid, level)] post-action frames (v10's format)

    Activation: after `stall_after` actions on a level without completion
    (the model plays first and its moves are the training data), at most
    `burst` actions per takeover, then `exhaust_yield` turns back to the
    model. Goal: inferred from the first observed completion; before any,
    hypotheses in order (collide if two entities, reach each rare colour,
    frontier), each tried once per level.
    """

    def __init__(self, log=None, *, stall_after=25, burst=300, exhaust_yield=8,
                 min_moves=6, max_bursts=2, max_reach_hyp=4, hyp_budget=60,
                 min_verified=20, max_miss_rate=0.1):
        self.log = log or (lambda s: None)
        self.stall_after = stall_after; self.burst = burst; self.exhaust_yield = exhaust_yield
        self.min_moves = min_moves; self.max_bursts = max_bursts; self.max_reach_hyp = max_reach_hyp
        self.hyp_budget = hyp_budget
        self.min_verified = min_verified; self.max_miss_rate = max_miss_rate
        self.trans = defaultdict(list); self._ingested = 0; self._last = None
        self.completed = {}            # level -> transitions of that level incl. the completing one
        self.phys = None; self.planner = None; self.goal_known = False
        self.level_start_n = {}; self.burst_used = defaultdict(int); self.bursts_done = defaultdict(int)
        self.yield_until = defaultdict(int)
        self.hyps = {}                 # level -> remaining goal hypotheses
        self.hyp = {}                  # level -> current hypothesis
        self.pending = None            # (n_history_at_proposal, action)
        self._hyp_used = {}            # level -> actions spent on the current hypothesis
        self._replayed = {}            # level -> transitions already fed to the model
        self.force_mover = None        # set by set_mover(): skip the guess
        self.stats = Counter()

    # ---- ingestion --------------------------------------------------------
    def seed_initial(self, grid, level):
        if self._last is None:
            self._last = (np.asarray(grid, dtype=np.int16), level)

    def _ingest(self, history):
        for i in range(self._ingested, len(history)):
            a, g, lv = history[i]
            g = np.asarray(g, dtype=np.int16); name = _parse(a)
            if self._last is not None:
                pg, plv = self._last
                if lv == plv:
                    self.trans[lv].append((pg, name, g))
                elif lv > plv and name != "RESET":
                    # the completing transition belongs to the finished level
                    self.trans[plv].append((pg, name, g))
                    self.completed[plv] = list(self.trans[plv])
                    if not self.goal_known and self.hyp.get(plv) is not None and self._hyp_used.get(plv):
                        kind, cols = self.hyp[plv]
                        if kind != "frontier":
                            self.phys.goal_kind, self.phys.goal_colours = kind, set(cols)
                            self.goal_known = True
                            self.log(f"stage: L{plv} completed under hypothesis {kind} {sorted(cols)} -- adopting it as the goal")
            self._last = (g, lv)
        self._ingested = len(history)

    # ---- learning ----------------------------------------------------------
    def set_mover(self, colour):
        """Force the avatar colour (an oracle, or the model's answer)."""
        self.force_mover = int(colour)

    def _fit(self, level):
        """Fit the physics from the level with the most direction moves."""
        best = None
        for lv, tr in self.trans.items():
            n = sum(1 for _, a, _ in tr if a in DIRS)
            if n < self.min_moves:
                continue
            rank = (lv in self.completed, n)
            if best is None or rank > best[0]:
                best = (rank, lv, n)
        if best is None:
            return False
        tr = self.trans[best[1]]
        best = (best[2], best[1])
        chrome = chrome_mask(tr)
        if self.force_mover is not None:
            mover, votes = self.force_mover, 99
        else:
            mover, votes = detect_mover(tr, chrome)
        if mover is None or votes < 3:
            self.stats["no_mover"] += 1
            return False
        phys = Physics(mover, chrome)
        try:
            phys.fit_level1(tr)
        except Exception as e:  # noqa: BLE001
            self.log(f"stage: fit failed {type(e).__name__}: {str(e)[:80]}"); return False
        if not phys.role_tables or phys.role_step is None or phys.floor is None:
            return False
        if not any(phys.role_tables):
            return False
        self.phys = phys
        self.log(f"stage: physics from L{best[1]} ({best[0]} moves): mover {mover} roles {phys.role_tables} "
                 f"steps {phys.role_steps} floor {phys.floor} passable {sorted(phys.passable)} blocking {sorted(phys.blocking)}")
        self._refresh_goal()
        return True

    def _refresh_goal(self):
        if self.phys is None or self.goal_known or not self.completed:
            return
        lv = min(self.completed)
        try:
            self.phys.infer_goal(self.completed[lv])
        except Exception as e:  # noqa: BLE001
            self.log(f"stage: infer_goal failed {type(e).__name__}"); return
        if self.phys.goal_kind in ("collide", "reach"):
            self.goal_known = True
            self.log(f"stage: goal from L{lv} completion: {self.phys.goal_kind} {sorted(self.phys.goal_colours)}")

    def _hypotheses(self, grid, level):
        P = self.phys
        g = np.asarray(grid, dtype=np.int16)
        ents = entities_of(g, P.mover, P.chrome, shapes=P.shapes)
        hyps = []
        if len(ents) >= 2:
            hyps.append(("collide", set()))
        vals, cnts = np.unique(g[~P.chrome] if P.chrome is not None else g, return_counts=True)
        cand = [(int(n), int(c)) for c, n in zip(vals, cnts)
                if c != P.mover and c != P.floor and c not in P.deadly and n <= 200]
        for _n, c in sorted(cand)[:self.max_reach_hyp]:
            hyps.append(("reach", {c}))
        hyps.append(("frontier", set()))
        return hyps

    def _build_level(self, grid, level):
        """Build the level's simulator from the level's FIRST frame and replay
        every recorded transition through it. Two things fall out: the true
        start position (so a level reset is recognisable as a death, not a
        move), and the passability / step facts the model's own play already
        demonstrated."""
        pl = self.planner; tr = self.trans.get(level) or []
        i, bodies = anchor_bodies(tr, self.phys.mover, self.phys.chrome, shapes=self.phys.shapes)
        if bodies:
            pl._new_level(tr[i][0], level, bodies=bodies)
            self._replayed[level] = i
        else:
            pl._new_level(tr[0][0] if tr else grid, level)
            self._replayed[level] = 0
        lm = pl.lm
        if not lm.ok:
            return
        self._absorb(level, grid)
        lm = pl.lm
        self.log(f"stage L{level}: model built from {len(tr)} recorded transitions; start {lm.start_state} now {lm.state} "
                 f"passable {sorted(lm.passable)} blocking {sorted(lm.blocking)} deadly {sorted(lm.deadly)} steps {lm.steps}")

    def _absorb(self, level, grid):
        """Feed the level's not-yet-seen transitions through the model. Every
        move the fallback policy (or the agent's model) makes is evidence:
        passability, step sizes, and which levers were already pulled where."""
        pl = self.planner; lm = pl.lm
        if lm is None or not lm.ok:
            return
        tr = self.trans.get(level) or []
        k = self._replayed.get(level, 0)
        for g, a, ng in tr[k:]:
            pl.pending = (a, lm.state)
            pl.observe(ng)
            lm = pl.lm
        self._replayed[level] = len(tr)
        if lm.refresh(grid):
            pl.plan = []
        pl.pending = None

    # ---- the per-turn call --------------------------------------------------
    def propose(self, grid, level, history, valid):
        self._ingest(history)
        n = len(history)
        if level not in self.level_start_n:
            self.level_start_n[level] = n
        # Every action played since the last turn -- ours, the model's, anyone's
        # -- is verified and learned from in _absorb(); nothing is verified twice.
        self.pending = None
        if n < self.yield_until[level]:
            self.stats["skip_yield"] += 1; return None
        if self.stall_after and n - self.level_start_n[level] < self.stall_after:
            self.stats["skip_stall"] += 1; return None
        if self.max_bursts and self.bursts_done[level] >= self.max_bursts:
            return None
        if self.burst and self.burst_used[level] >= self.burst:
            self.burst_used[level] = 0; self.bursts_done[level] += 1
            self.yield_until[level] = n + self.exhaust_yield; self.stats["burst_yield"] += 1
            return None
        try:
            if self.phys is None and not self._fit(level):
                self.stats["skip_nofit"] += 1; return None
            self._refresh_goal()
            if self.planner is None:
                self.planner = PhysicsPlanner(self.phys, log=self.log)
            pl = self.planner
            pl.extra_actions = tuple(a for a in ("SPACE",) if a in (valid or ()))
            if pl.level != level or pl.lm is None:
                self._build_level(grid, level)
                self.hyps.pop(level, None); self.hyp.pop(level, None); self._hyp_used.pop(level, None)
            if pl.lm is None or not pl.lm.ok or not pl.lm.state:
                self.yield_until[level] = n + self.exhaust_yield
                self.stats["skip_nomodel"] += 1; return None
            self._absorb(level, grid)                    # everything played since the last turn
            lm = pl.lm
            n_pred = lm.hits + lm.misses
            if lm.hits < self.min_verified or lm.misses > self.max_miss_rate * n_pred:
                # The physics has not earned the budget on this level. Every
                # action the mechanism spends is one the model does not get, so
                # it may only plan once its predictions have been right.
                self.stats["untrusted"] += 1
                return None
            if not self.goal_known:
                if level not in self.hyps:
                    self.hyps[level] = self._hypotheses(grid, level)
            if self.goal_known:
                pl.frontier_fallback = True
                a = pl.propose(grid, level)
            else:
                # No completion has been seen yet, so the goal is a hypothesis:
                # collide (two bodies), reach each rare colour, then frontier.
                # Each gets `hyp_budget` actions; they cycle until one completes
                # the level (which then becomes the known goal).
                if level not in self.hyps:
                    self.hyps[level] = self._hypotheses(grid, level)
                    self.hyp[level] = None
                a = None
                for _try in range(2):
                    cur = self.hyp.get(level)
                    if cur is None or self._hyp_used.get(level, 0) >= self.hyp_budget:
                        self.hyps[level].append(self.hyps[level].pop(0)) if cur is not None else None
                        cur = self.hyps[level][0]; self.hyp[level] = cur
                        self._hyp_used[level] = 0
                        self.phys.goal_kind, self.phys.goal_colours = cur[0], set(cur[1])
                        pl.lm.rebuild(optimistic=pl.lm.optimistic); pl.plan = []
                        pl.frontier_fallback = cur[0] == "frontier"
                        self.log(f"stage L{level}: goal hypothesis {cur[0]} {sorted(cur[1])}")
                    a = pl.propose(grid, level)
                    if a is not None:
                        self._hyp_used[level] = self._hyp_used.get(level, 0) + 1
                        break
                    self._hyp_used[level] = self.hyp_budget      # no plan: rotate
            if a is None:
                self.yield_until[level] = n + self.exhaust_yield; self.stats["exhausted"] += 1
                return None
        except Exception as e:  # noqa: BLE001
            self.log(f"stage: error {type(e).__name__}: {str(e)[:120]}")
            self.planner = None; self.stats["errors"] += 1
            return None
        self.burst_used[level] += 1; self.stats["actions"] += 1
        self.pending = (n, a)
        return [{"action": a}]
