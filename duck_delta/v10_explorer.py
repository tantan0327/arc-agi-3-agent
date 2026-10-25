"""v10: harness-forced exploration over an authored abstraction.

The agent's reactive play wanders; on games where an abstraction of the
board exists (a level identity plus the few things that move), systematic
exploration of the ABSTRACT state space -- try every untried action from
every reachable abstract state, Go-Explore style -- covers a level with far
fewer actions than pixel-level wandering, and reaches levels the reactive
loop stalls on. That is the reach mechanism the leaderboard formula pays for
(per-level score is capped at 1.15, so only reaching levels counts).

Where it sits. ToolAgent.analyze() asks this explorer first; if it returns a
plan, the agent issues `action([...])` through its own python tool -- the
same path the model's actions take (history, no-op guard, animation, state
persistence all reused) -- and the model is not called that turn. If it
returns None, the normal model turn runs. The model is still the authoring
engine: after `warmup` transitions on a level the explorer asks it (via
`chat`) for `extract_state` candidates, scores them on the agent's OWN
recorded transitions (next-state determinism, revisit coverage,
responsiveness -- GP-A's dynamics view, no human data needed online), and
keeps the best. Failing that, it stays out of the way.

Pure Python + numpy, no harness imports, so it is testable offline against
the real engine (scripts/test_v10_explorer.py).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict, deque

import numpy as np

# ----------------------------------------------------------------- actions
_ACTION_RE = re.compile(r"^\s*([A-Z0-9_]+)")
_ROWCOL_RE = re.compile(r"row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)")
NON_PLAN = {"RESET"}


def parse_action(text):
    """'UP' -> ('UP', None, None); 'MOUSE(row=3, col=17)' -> ('MOUSE', 3, 17)."""
    m = _ACTION_RE.match(str(text or ""))
    name = m.group(1) if m else ""
    rc = _ROWCOL_RE.search(str(text or ""))
    if rc:
        return name, int(rc.group(1)), int(rc.group(2))
    return name, None, None


def spec_of(name, row=None, col=None):
    if row is not None and col is not None:
        return {"action": name, "row": int(row), "col": int(col)}
    return {"action": name}


def akey_of(name, row=None, col=None):
    """Graph edge label: coarse click cell so nearby clicks share an edge."""
    if row is not None and col is not None:
        return (name, int(row) // 4, int(col) // 4)
    return name


# ------------------------------------------------------------------ boards
def blobs(board):
    """4-connected components per colour: {colour: [(size, top, left, cy, cx)]}."""
    g = np.asarray(board)
    out = {}
    seen = np.zeros(g.shape, dtype=bool)
    rows, cols = g.shape
    for r in range(rows):
        for c in range(cols):
            if not seen[r, c]:
                colour = int(g[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and not seen[ny, nx] and int(g[ny, nx]) == colour:
                            seen[ny, nx] = True; stack.append((ny, nx))
                out.setdefault(colour, []).append(
                    (len(cells), min(y for y, _ in cells), min(x for _, x in cells),
                     sum(y for y, _ in cells) // len(cells), sum(x for _, x in cells) // len(cells)))
    return out


def blobs_for_authoring(board):
    """The helper handed to authored code: {colour: [(size, top, left), ...]}."""
    return {c: [(s, t, l) for s, t, l, _, _ in v] for c, v in blobs(board).items()}


def click_candidates(board, cap=16):
    """Where a human would click: small, rare blobs first (never the background)."""
    g = np.asarray(board)
    vals, cnts = np.unique(g, return_counts=True)
    bg = int(vals[cnts.argmax()])
    cands = []
    for colour, v in blobs(board).items():
        if colour == bg:
            continue
        for size, _t, _l, cy, cx in v:
            if size <= 200:
                cands.append((size, len(v), cy, cx))
    cands.sort()
    out, seen = [], set()
    def take(seq, limit):
        for size, _n, cy, cx in seq:
            cell = (cy // 4, cx // 4)
            if cell in seen:
                continue
            seen.add(cell); out.append((cy, cx))
            if len(out) >= limit:
                break
    take(cands, cap // 2)                      # the smallest, rarest objects first
    rest = cands[len(out):]
    if rest:
        step = max(1, len(rest) // max(1, cap - len(out)))
        take(rest[::step], cap)                # then a spread across the larger ones
    return out


def render(board):
    chars = "0123456789ABCDEF"
    return "\n".join("".join(chars[int(v) % 16] for v in row) for row in np.asarray(board))


def describe_diff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = np.argwhere(a != b)
    if len(d) == 0:
        return "no change"
    before = Counter(int(a[y, x]) for y, x in d).most_common(3)
    after = Counter(int(b[y, x]) for y, x in d).most_common(3)
    return (f"{len(d)} cells changed in rows {d[:,0].min()}-{d[:,0].max()} cols {d[:,1].min()}-{d[:,1].max()}; "
            f"colours before {before} -> after {after}")


def _freeze(k):
    if isinstance(k, (list, tuple)):
        return tuple(_freeze(x) for x in k)
    if isinstance(k, dict):
        return tuple(sorted((str(a), _freeze(b)) for a, b in k.items()))
    if isinstance(k, np.ndarray):
        return tuple(_freeze(x) for x in k.tolist())
    if isinstance(k, (np.integer,)):
        return int(k)
    return k


# --------------------------------------------------------------- authoring
CONTRACT = """You are writing PERCEPTION code for an unknown 64x64 grid game (cells are integers 0-15 = colours).
Write one Python function:

    def extract_state(board) -> tuple

`board` is a 64x64 list of lists of ints. Return a hashable tuple that captures the game's STATE VARIABLES the way a careful player would note them: which level/layout this is (e.g. a hash of the static walls), and the position / size / count of everything that moves or changes in response to actions (an avatar, a cursor, bars that extend, tiles that flip, items collected). Use plain ints and nested tuples.

Rules that the scorer will enforce:
- DETERMINISTIC: the same board must always give the same tuple (no randomness, no global state).
- RESPONSIVE: when an action visibly changes the board, the tuple should change too -- do not throw away the moving parts.
- CLEAN: exclude decorations and step/energy counters that tick on every action regardless of what happened (they usually live on the border or in a corner); including them makes every state unique and useless.
- NUMERIC where things extend or move: lengths and positions as ints, so the same action produces the same delta.
- Available in scope: `np` (numpy) and `blobs(board) -> {colour: [(size, top, left), ...]}` (4-connected components per colour). No imports needed; do not read files or use randomness.

Keep it SHORT: under 80 lines, no docstrings or commentary inside the code, no example usage. Do not reassign the name `blobs`.

Return ONLY the Python code for extract_state (and any small helpers), inside one ```python block."""


def mechanical_hints(trans):
    boards = [np.asarray(t[0]) for t in trans] + [np.asarray(trans[-1][2])]
    n = len(boards) - 1
    if n < 5:
        return ""
    chrome = np.zeros(boards[0].shape, dtype=bool); count = np.zeros(boards[0].shape)
    within = 0; moved = set(); present = set()
    for i in range(n):
        d = boards[i] != boards[i + 1]; k = int(d.sum())
        if k == 0:
            continue
        if k <= 2:
            chrome |= d
        if k > 0.3 * d.size:
            continue
        within += 1; count += d
        present.update(np.unique(boards[i]).tolist())
        moved.update(np.unique(boards[i][d]).tolist()); moved.update(np.unique(boards[i + 1][d]).tolist())
    if within:
        chrome |= (count / within) > 0.3
    lines = []
    cells = np.argwhere(chrome)
    if len(cells):
        rows = sorted(set(cells[:, 0].tolist())); cols = sorted(set(cells[:, 1].tolist()))
        lines.append(f"- INSTRUMENTATION: {len(cells)} cells in rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]} change on their own (a step counter / energy bar / indicator). They are NOT game state -- exclude them completely.")
    static = sorted(int(c) for c in present if c not in moved); mv = sorted(int(c) for c in moved)
    lines.append(f"- Within a level, colours whose cells changed: {mv}. Colours that never changed within a level (static layout -- use only as a level identity): {static}.")
    return "\nMECHANICAL FACTS measured from the recorded play (trust these):\n" + "\n".join(lines)


def build_prompt(trans, feedback=None, n_full=3, n_diffs=8):
    parts = [CONTRACT, mechanical_hints(trans), f"\nThe board at the start of the recorded play:\n```\n{render(trans[0][0])}\n```"]
    span = min(len(trans), 40)
    idx_full = sorted(set([0, span // 3, (2 * span) // 3][:n_full]))
    for i in idx_full:
        b, a, nb = trans[i]
        parts.append(f"\nAfter action {a} at step {i}: {describe_diff(b, nb)}. Board after:\n```\n{render(nb)}\n```")
    diffs = []
    step = max(1, span // n_diffs)
    for i in range(0, span, step):
        if i in idx_full or len(diffs) >= n_diffs:
            continue
        b, a, nb = trans[i]
        diffs.append(f"- step {i}: {a} -> {describe_diff(b, nb)}")
    if diffs:
        parts.append("\nMore transitions (diff summaries only):\n" + "\n".join(diffs))
    if feedback:
        parts.append("\nYOUR PREVIOUS extract_state WAS SCORED AGAINST THE RECORDED PLAY. Fix the problems below and return the full revised function.\n" + feedback)
    return "\n".join(parts)


def compile_fn(code):
    start = code.find("```python")
    if start >= 0:
        body = code[start + len("```python"):]
        end = body.rfind("```")
        src = body[:end] if end >= 0 else body
    else:
        src = code
    ns = {"np": np, "blobs": blobs_for_authoring}
    exec(compile(src, "<authored>", "exec"), ns)
    fn = ns.get("extract_state")
    if fn is None:
        raise ValueError("no extract_state defined")
    return fn, src


def chrome_mask(trans):
    """Cells that change in isolation (<=2 cells in a transition) are
    instrumentation -- the determinism ladder's rule. Also cells flipping on
    >30% of within-level moves."""
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


def score_abstraction(fn, trans, budget_s=20.0):
    """GP-A dynamics view on a list of (grid, action_text, next_grid)."""
    t0 = time.time()
    chrome = chrome_mask(trans)
    keys = []
    for g, _, _ in trans:
        keys.append(_freeze(fn(np.asarray(g).tolist())))
        if time.time() - t0 > budget_s:
            raise TimeoutError("scoring budget exceeded")
    keys.append(_freeze(fn(np.asarray(trans[-1][2]).tolist())))
    for k in keys:
        hash(k)
    tc = defaultdict(Counter)
    sa = det = resp = changed = 0
    n = len(trans)
    for i, (g, a, ng) in enumerate(trans):
        key, nxt = keys[i], keys[i + 1]
        name, r, c = parse_action(a)
        ak = akey_of(name, r, c)
        # responsiveness is judged only where the BOARD changed: a random
        # policy's misses say nothing about the abstraction (BLIND = the board
        # moved and the state did not)
        d = np.asarray(g) != np.asarray(ng)
        if chrome is not None:
            d = d & ~chrome
        if d.any():                      # a real move, not a counter tick
            changed += 1
            if nxt != key:
                resp += 1
        cnt = tc[(key, ak)]
        if cnt:
            sa += 1
            if cnt.most_common(1)[0][0] == nxt:
                det += 1
        cnt[nxt] += 1
    return {"sa_cov": sa / n, "determ": det / sa if sa else 0.0,
            "respons": resp / changed if changed else 1.0, "changed": changed,
            "states": len(set(keys)), "n": n}


# ---------------------------------------------------------------- explorer
class V10Explorer:
    def __init__(self, chat=None, log=None, *, warmup=30, n_author=4, author_temp=0.8,
                 max_plan=10, click_cap=16, min_determ=0.8, min_sa=0.05, min_resp=0.5,
                 exhaust_yield=8, max_author_rounds=3, per_call_budget_s=2.0,
                 stall_after=25, burst=200, max_bursts=0, min_states=4):
        self.chat = chat; self.log = log or (lambda s: None)
        self.warmup = warmup; self.n_author = n_author; self.author_temp = author_temp
        self.max_plan = max_plan; self.click_cap = click_cap
        self.min_determ = min_determ; self.min_sa = min_sa; self.min_resp = min_resp
        self.exhaust_yield = exhaust_yield; self.max_author_rounds = max_author_rounds
        self.per_call_budget_s = per_call_budget_s
        # Stall-breaker: the model's goal-directed play runs first on every
        # level; the explorer takes over only after `stall_after` actions on a
        # level without completing it, explores for at most `burst` actions,
        # then hands back for `exhaust_yield` turns, and so on.
        self.stall_after = stall_after; self.burst = burst
        self.max_bursts = max_bursts          # 0 = unlimited; else bursts per level
        self.min_states = min_states          # reject degenerate abstractions
        self.level_start_n = {}; self.burst_used = defaultdict(int)
        self.bursts_done = defaultdict(int)
        self.fn = None; self.fn_src = ""; self.fn_score = None
        self.level_ok = {}              # level -> True/False (validated on that level)
        self.author_rounds = defaultdict(int)
        self.next_author_at = defaultdict(int)
        self.trans = defaultdict(list)  # level -> [(grid, action_text, next_grid)]
        self._ingested = 0; self._last_grid = None; self._last_level = None
        self.graph = {}                 # level -> {"n": int, "edges": {(s,ak): (s2, spec)}, "tried": {s: set}, "rep": {s: grid}, "keys": list}
        self.yield_until = defaultdict(int)
        self.recheck_at = defaultdict(int)
        self.stats = Counter()

    # ---- external: an abstraction can be supplied (tests) --------------
    def set_abstraction(self, fn, src="<given>"):
        self.fn = fn; self.fn_src = src; self.fn_score = None; self.level_ok = {}; self.graph = {}

    # ---- ingestion ------------------------------------------------------
    def _ingest(self, history):
        """history: list of (action_text, grid, level), post-action frames."""
        for i in range(self._ingested, len(history)):
            a, g, lv = history[i]
            g = np.asarray(g, dtype=np.int16)
            if self._last_grid is not None and self._last_level == lv:
                self.trans[lv].append((self._last_grid, a, g))
            self._last_grid = g; self._last_level = lv
        self._ingested = len(history)

    def seed_initial(self, grid, level):
        """Tell the explorer the pre-action frame (the harness's first history
        entry has no predecessor otherwise)."""
        if self._last_grid is None:
            self._last_grid = np.asarray(grid, dtype=np.int16); self._last_level = level

    # ---- abstraction lifecycle -----------------------------------------
    def _key(self, grid):
        t0 = time.time()
        k = _freeze(self.fn(np.asarray(grid).tolist()))
        if time.time() - t0 > self.per_call_budget_s:
            raise TimeoutError("extract_state too slow")
        hash(k)
        return k

    def _passes(self, s):
        return (s["respons"] >= self.min_resp and s["determ"] >= self.min_determ
                and s["sa_cov"] >= self.min_sa and s["states"] >= self.min_states)

    def _validate(self, level):
        trans = self.trans[level]
        try:
            s = score_abstraction(self.fn, trans)
        except Exception as e:  # noqa: BLE001
            self.log(f"v10 L{level}: abstraction failed to score ({type(e).__name__}); dropping it")
            self.fn = None; self.level_ok = {}; return False
        # Evidence rules: responsiveness needs no revisits to judge; determinism
        # does. With no repeated (state, action) yet the verdict is "undetermined"
        # -- keep using the abstraction optimistically and re-check later.
        if s["respons"] < self.min_resp:
            verdict = "reject"
        elif s["sa_cov"] == 0.0:
            verdict = "undetermined"
        elif s["determ"] >= self.min_determ:
            verdict = "OK"
        else:
            verdict = "reject"
        self.level_ok[level] = verdict != "reject"
        self.recheck_at[level] = s["n"] + (15 if verdict == "undetermined" else 40)
        self.log(f"v10 L{level}: validate n={s['n']} determ={s['determ']:.0%} sa_cov={s['sa_cov']:.0%} respons={s['respons']:.0%} states={s['states']} -> {verdict}")
        return verdict != "reject"

    def _author(self, level):
        trans = self.trans[level]
        if self.chat is None or self.author_rounds[level] >= self.max_author_rounds:
            return False
        self.author_rounds[level] += 1
        self.next_author_at[level] = len(trans) + self.warmup
        prompt = build_prompt(trans)
        best = None
        for i in range(self.n_author):
            try:
                reply = self.chat(prompt, self.author_temp)
                fn, src = compile_fn(reply)
                s = score_abstraction(fn, trans)
            except Exception as e:  # noqa: BLE001
                self.log(f"v10 L{level}: author candidate {i} failed: {type(e).__name__}: {str(e)[:80]}")
                continue
            rank = (self._passes(s), s["determ"] * min(1.0, s["sa_cov"] / 0.1), s["respons"])
            self.log(f"v10 L{level}: author candidate {i}: determ={s['determ']:.0%} sa_cov={s['sa_cov']:.0%} respons={s['respons']:.0%} states={s['states']}")
            if best is None or rank > best[0]:
                best = (rank, fn, src, s)
        if best is None or not best[0][0]:
            self.stats["author_fail"] += 1
            self.log(f"v10 L{level}: authoring round {self.author_rounds[level]} produced no usable abstraction")
            return False
        self.fn, self.fn_src, self.fn_score = best[1], best[2], best[3]
        self.level_ok = {level: True}; self.graph.pop(level, None)
        self.stats["author_ok"] += 1
        self.log(f"v10 L{level}: ADOPTED abstraction determ={best[3]['determ']:.0%} sa_cov={best[3]['sa_cov']:.0%} respons={best[3]['respons']:.0%}")
        return True

    def _ensure_abstraction(self, level):
        trans = self.trans[level]
        if self.fn is not None:
            if level in self.level_ok:
                if self.level_ok[level]:
                    if len(trans) >= self.recheck_at[level]:
                        return self._validate(level)     # periodic re-check as evidence grows
                    return True
                if len(trans) >= self.recheck_at[level] and self.chat is None:
                    return self._validate(level)         # nothing to re-author with: re-try later
            elif len(trans) >= 15:
                if self._validate(level):
                    return True
            elif len(trans) >= 10:
                return True      # inherited, used optimistically until validation
            else:
                return False
        if len(trans) < self.warmup or len(trans) < self.next_author_at[level]:
            return False
        return self._author(level)

    # ---- graph ----------------------------------------------------------
    def _graph(self, level):
        trans = self.trans[level]
        G = self.graph.get(level)
        if G is None:
            G = {"n": 0, "edges": {}, "tried": defaultdict(set), "rep": {}, "cands": {}}
            self.graph[level] = G
        eff = G.setdefault("eff", {})
        while G["n"] < len(trans):
            g, a, ng = trans[G["n"]]
            s, s2 = self._key(g), self._key(ng)
            name, r, c = parse_action(a)
            ak = akey_of(name, r, c)
            G["edges"][(s, ak)] = (s2, spec_of(name, r, c))
            G["tried"][s].add(ak)
            G["rep"].setdefault(s, g); G["rep"].setdefault(s2, ng)
            tries, changes, states = eff.get(name, (0, 0, set()))
            states = set(states); states.add(s)
            eff[name] = (tries + 1, changes + (0 if np.array_equal(g, ng) else 1), states)
            G["n"] += 1
        return G

    def _dead_classes(self, level):
        """Action classes (UP/DOWN/.../MOUSE) that never changed the board after
        enough tries at several states: not worth exploring on this level."""
        eff = self.graph[level].setdefault("eff", {})
        # Only click/interact classes can be declared dead: a direction key
        # that bumps walls at three states is still the way out of the maze.
        return {cls for cls, (tries, changes, states) in eff.items()
                if cls not in ("UP", "DOWN", "LEFT", "RIGHT") and tries >= 8 and changes == 0 and len(states) >= 3}

    def _candidates(self, G, s, valid, dead=frozenset()):
        cached = G["cands"].get(s)
        if cached is None:
            cached = [(name, spec_of(name)) for name in valid if name not in NON_PLAN and name != "MOUSE"]
            if "MOUSE" in valid and s in G["rep"]:
                for cy, cx in click_candidates(G["rep"][s], self.click_cap):
                    cached.append((akey_of("MOUSE", cy, cx), spec_of("MOUSE", cy, cx)))
            G["cands"][s] = cached
        if not dead:
            return cached
        return [(ak, spec) for ak, spec in cached if (ak if isinstance(ak, str) else ak[0]) not in dead]

    def _level_of(self, G):
        for lv, g in self.graph.items():
            if g is G:
                return lv
        return None

    def _plan(self, G, s0, valid):
        """BFS to the nearest abstract state with an untried candidate action."""
        seen = {s0}; q = deque([(s0, [])])
        dead = self._dead_classes(self._level_of(G))
        while q:
            s, path = q.popleft()
            if len(path) >= self.max_plan:
                continue
            tried = G["tried"].get(s, set())
            for ak, spec in self._candidates(G, s, valid, dead):
                if ak not in tried:
                    return path + [spec], len(path)
            for (es, ak), (s2, spec) in G["edges"].items():
                if es == s and s2 not in seen:
                    seen.add(s2); q.append((s2, path + [spec]))
        return None, None

    # ---- the per-turn call ---------------------------------------------
    def propose(self, current_grid, level, history, valid_actions):
        self._ingest(history)
        n = len(history)
        if level not in self.level_start_n:
            self.level_start_n[level] = n
        if n < self.yield_until[level]:
            return None
        if self.stall_after and n - self.level_start_n[level] < self.stall_after:
            return None                      # let the model play the level first
        if self.max_bursts and self.bursts_done[level] >= self.max_bursts:
            return None                      # this level has had its share of exploration
        if self.burst and self.burst_used[level] >= self.burst:
            self.burst_used[level] = 0
            self.bursts_done[level] += 1
            self.yield_until[level] = n + self.exhaust_yield
            self.stats["burst_yield"] += 1
            return None
        try:
            if not self._ensure_abstraction(level):
                return None
            G = self._graph(level)
            s0 = self._key(current_grid)
            G["rep"].setdefault(s0, np.asarray(current_grid, dtype=np.int16))
            plan, depth = self._plan(G, s0, list(valid_actions or []))
        except Exception as e:  # noqa: BLE001
            self.log(f"v10 L{level}: explorer error {type(e).__name__}: {str(e)[:100]}; dropping abstraction")
            self.fn = None; self.level_ok = {}; self.graph.pop(level, None)
            return None
        if plan is None:
            self.yield_until[level] = n + self.exhaust_yield
            self.stats["exhausted"] += 1
            self.log(f"v10 L{level}: frontier exhausted at {len(G['rep'])} abstract states; yielding {self.exhaust_yield} turns")
            return None
        self.stats["plans"] += 1; self.stats["plan_actions"] += len(plan)
        self.burst_used[level] += len(plan)
        self.log(f"v10 L{level}: plan len={len(plan)} (path {depth}) states={len(G['rep'])} edges={len(G['edges'])}")
        return plan
