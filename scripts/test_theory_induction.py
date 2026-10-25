"""Can a code-fitted mechanics theory predict ARC-AGI-3 transitions? (theory-based RL test)

Human players (100%) learn a causal model of a game in a few interactions and plan in it
(Tsividis et al. 2021, EMPA). The LLM agent re-establishes mechanics at every level with
6-10 turns of tokens. If a cheap, code-fitted object-level theory predicts later
transitions well, planning can run in the theory for free and the model's tokens go to
goals. This script measures that premise on the official human replays (22 games, 168
level completions), replayed on the local engine.

Theory space (deliberately the simplest core-knowledge one):
  * movers: for direction actions, a colour's connected components translate by a fixed
    per-action offset unless the destination holds a BLOCKING colour (learned);
  * clicks: the clicked component changes (something in it recolours) or not, per colour.
Fitting regimes: (A) fit on level 1 only, test on levels >= 2; (B) online: fit on every
transition before the current one, test on the current.

    .venv/bin/python scripts/test_theory_induction.py [game ...]
"""
from __future__ import annotations
import glob, json, os, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arcengine.enums import GameAction

REPLAYS = os.environ.get("HUMAN_REPLAYS", "vendor/taaf-kaggle-source/src/re-arc-3/re_arc/dsl/official_human_replays")
DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}   # UP, DOWN, LEFT, RIGHT in (row, col)


def grid_of(frame):
    arr = np.array(frame.frame)
    while arr.ndim > 2:
        arr = arr[-1]
    return arr.astype(int)


def components(grid, colour):
    """4-connected components of one colour -> list of frozenset of (r,c)."""
    H, W = grid.shape; seen = np.zeros_like(grid, dtype=bool); out = []
    rs, cs = np.where(grid == colour)
    for r0, c0 in zip(rs, cs):
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]; seen[r0, c0] = True; comp = []
        while stack:
            r, c = stack.pop(); comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and not seen[rr, cc] and grid[rr, cc] == colour:
                    seen[rr, cc] = True; stack.append((rr, cc))
        out.append(frozenset(comp))
    return out


def movers(before, after):
    """Colours whose vacated cells map one-to-one onto arrived cells under a single offset.
    Returns {colour: (dr, dc)}; colours with no such offset are not movers."""
    changed = np.argwhere(before != after)
    if len(changed) == 0 or len(changed) > 400:
        return {}
    out = {}
    for colour in set(before[tuple(changed.T)]) | set(after[tuple(changed.T)]):
        vac = {(r, c) for r, c in changed if before[r, c] == colour and after[r, c] != colour}
        arr = {(r, c) for r, c in changed if after[r, c] == colour and before[r, c] != colour}
        if not vac or not arr:
            continue
        # candidate offsets from the bounding boxes
        vr = min(r for r, _ in vac); vc = min(c for _, c in vac); ar = min(r for r, _ in arr); ac = min(c for _, c in arr)
        off = (ar - vr, ac - vc)
        # the full component(s) of this colour before, shifted, must equal the component(s) after over the changed area
        comps_before = [c for c in components(before, colour) if c & vac]
        shifted = set()
        for comp in comps_before:
            shifted |= {(r + off[0], c + off[1]) for r, c in comp}
        comps_after = set()
        for comp in components(after, colour):
            if comp & arr:
                comps_after |= comp
        if shifted and shifted == comps_after and off != (0, 0):
            out[colour] = off
    return out


class AvatarTheory:
    """Regime C: track the avatar as an object (the component that moved last), not a colour."""

    def __init__(self):
        self.avatar = None                      # frozenset of cells (current position)
        self.off = collections.defaultdict(collections.Counter)    # action -> Counter(offset)
        self.block = collections.defaultdict(collections.Counter)  # action -> Counter(dest colour) when blocked
        self.passed = collections.defaultdict(collections.Counter)

    def _dest(self, grid, cells, off):
        H, W = grid.shape
        d = {(r + off[0], c + off[1]) for r, c in cells} - cells
        return {int(grid[r, c]) if 0 <= r < H and 0 <= c < W else -1 for r, c in d}

    def observe(self, before, after, action, param):
        if action not in DIRS:
            return
        mv = movers(before, after)
        if mv:
            # the moved object becomes the avatar (largest moved component)
            best = None
            for colour, off in mv.items():
                for comp in components(before, colour):
                    if {(r + off[0], c + off[1]) for r, c in comp} <= set(zip(*np.where(after == colour))):
                        if best is None or len(comp) > len(best[0]):
                            best = (comp, off, colour)
            if best:
                comp, off, colour = best
                self.off[action][off] += 1
                for dc in self._dest(before, comp, off):
                    self.passed[action][dc] += 1
                self.avatar = frozenset((r + off[0], c + off[1]) for r, c in comp)
        elif self.avatar is not None and self.off[action]:
            off = self.off[action].most_common(1)[0][0]
            for dc in self._dest(before, self.avatar, off):
                self.block[action][dc] += 1

    def predict(self, before, action, param):
        if action not in DIRS or self.avatar is None or not self.off[action]:
            return None
        off = self.off[action].most_common(1)[0][0]
        # the avatar must still be where we think it is (same colour footprint)
        cells = list(self.avatar)
        H, W = before.shape
        if not all(0 <= r < H and 0 <= c < W for r, c in cells):
            return None
        dests = self._dest(before, self.avatar, off)
        blocked = any(self.block[action][d] > self.passed[action][d] for d in dests)
        return ("avatar", None if blocked else off)


def evaluate_avatar(before, after, action, pred):
    if pred is None or action not in DIRS:
        return 0, 0
    mv = movers(before, after)
    actual = None
    for colour, off in mv.items():
        actual = off
    return 1, int(actual == pred[1])


BORDER = int(os.environ.get('TI_BORDER', '1'))   # mask of the outer rows/cols where the timer/steps bar lives
REL = os.environ.get('TI_REL', '1') == '1'           # allow the relative-control model


def interior(grid):
    if BORDER <= 0:
        return grid
    g = grid.copy()
    g[:BORDER, :] = -9; g[-BORDER:, :] = -9; g[:, :BORDER] = -9; g[:, -BORDER:] = -9
    return g


def movers_any(before, after):
    """Like movers() (any offset, so slides count). Interior first (so the timer bar at the
    border is never mistaken for a mover); if nothing translates there, the full grid (an
    avatar at the very edge)."""
    mv = movers(interior(before), interior(after))
    return mv if mv else movers(before, after)


class RichTheory(AvatarTheory):
    """Regime D: avatar as object + border mask + slide (variable step along the key's axis)
    + relative controls (key means turn/forward relative to the last heading) + blocking
    remembered per cell as well as per colour. Model selection is online: whichever of
    {absolute, relative} control model predicted better so far is used."""

    def __init__(self):
        super().__init__()
        self.abs_dir = collections.defaultdict(collections.Counter)   # key -> Counter(unit direction)
        self.rel_turn = collections.defaultdict(collections.Counter)  # key -> Counter(turn index 0..3 relative to heading)
        self.heading = None
        self.score_abs = 0; self.score_rel = 0
        self.step = collections.defaultdict(collections.Counter)      # key -> Counter(step length)
        self.block_cells = set(); self.pass_cells = set()
        self.score_step = 0; self.score_slide = 0

    @staticmethod
    def unit(off):
        dr, dc = off
        if dr == 0 and dc == 0:
            return None
        if abs(dr) >= abs(dc):
            return (1 if dr > 0 else -1, 0)
        return (0, 1 if dc > 0 else -1)

    @staticmethod
    def turn_index(prev, cur):
        order = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        return (order.index(cur) - order.index(prev)) % 4

    def _slide_dest(self, grid, cells, unit, k):
        off = (unit[0] * k, unit[1] * k)
        return self._dest(grid, cells, off), off

    def predicted_unit(self, action):
        a = self.abs_dir[action].most_common(1)[0][0] if self.abs_dir[action] else (DIRS.get(action) if any(self.abs_dir.values()) else None)
        r = None
        if self.heading is not None and self.rel_turn[action]:
            order = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            t = self.rel_turn[action].most_common(1)[0][0]
            r = order[(order.index(self.heading) + t) % 4]
        if a is None and r is None:
            return None
        if (not REL) or r is None or self.score_abs >= self.score_rel:
            return a if a is not None else r
        return r

    def slide_len(self, grid, cells, u, action, cap=64):
        """How many unit steps the avatar can take before a known blocking colour or the edge."""
        H, W = grid.shape; k = 0; cur = set(cells)
        while k < cap:
            nxt = {(r + u[0], c + u[1]) for r, c in cur}
            if any(not (0 <= r < H and 0 <= c < W) for r, c in nxt):
                break
            front = nxt - cur
            cols = {int(grid[r, c]) for r, c in front}
            if any(self.block[action][d] > self.passed[action][d] for d in cols):
                break
            k += 1; cur = nxt
        return k

    def relocate(self, grid):
        """If the tracked avatar is no longer where we think (new level, redraw), find the
        component of its colour whose size is closest to the remembered one."""
        if self.avatar is None or getattr(self, "avatar_colour", None) is None:
            return
        H, W = grid.shape
        ok = all(0 <= r < H and 0 <= c < W and int(grid[r, c]) == self.avatar_colour for r, c in self.avatar)
        if ok:
            return
        comps = components(grid, self.avatar_colour)
        if not comps:
            self.avatar = None; return
        n = len(self.avatar)
        self.avatar = frozenset(min(comps, key=lambda c: abs(len(c) - n)))

    def predict(self, before, action, param):
        if action not in DIRS or self.avatar is None:
            return None
        self.relocate(before)
        if self.avatar is None:
            return None
        u = self.predicted_unit(action)
        if u is None:
            return None
        k_slide = self.slide_len(before, self.avatar, u, action)
        if k_slide == 0:
            return ("avatar", None)
        k_fixed = self.step[action].most_common(1)[0][0] if self.step[action] else 1
        k = k_slide if self.score_slide > self.score_step else min(k_fixed, k_slide)
        return ("avatar", (u[0] * k, u[1] * k))

    def observe(self, before, after, action, param):
        if action not in DIRS:
            return
        mv = movers_any(before, after)
        # bookkeeping for the control models: what did the avatar actually do?
        actual = None; comp_moved = None
        if mv:
            cands = []
            ia, ib = before, after
            for colour, off in mv.items():
                for comp in components(ia, colour):
                    if {(r + off[0], c + off[1]) for r, c in comp} <= set(zip(*np.where(ib == colour))):
                        u = self.unit(off)
                        expect = self.abs_dir[action].most_common(1)[0][0] if self.abs_dir[action] else None
                        near = 1 if (self.avatar is not None and (comp & self.avatar)) else 0
                        agree = 1 if (expect is not None and u == expect) else 0
                        bg = collections.Counter(before.flatten().tolist()).most_common(1)[0][0]
                        plausible = 1 if (len(comp) >= 2 and colour != bg) else 0
                        cands.append(((plausible, near, agree, len(comp)), comp, off, colour))   # a plausible body first, then continuity (Spelke), contingency, size
            if cands:
                cands.sort(key=lambda t: t[0], reverse=True)
                _, comp, off, colour = cands[0]; actual = off; comp_moved = comp
        # score the step vs slide models (before updating)
        if self.avatar is not None:
            u_pred = self.predicted_unit(action)
            if u_pred is not None:
                ks = self.slide_len(before, self.avatar, u_pred, action)
                kf = self.step[action].most_common(1)[0][0] if self.step[action] else 1
                k_act = 0 if actual is None else max(abs(actual[0]), abs(actual[1]))
                self.score_step += int(k_act == min(kf, ks))
                self.score_slide += int(k_act == ks)
        # score the two control models on this transition (before updating them)
        if actual is not None:
            u_act = self.unit(actual)
            a = self.abs_dir[action].most_common(1)[0][0] if self.abs_dir[action] else None
            if a is not None:
                self.score_abs += int(a == u_act)
            if self.heading is not None and self.rel_turn[action]:
                order = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                t = self.rel_turn[action].most_common(1)[0][0]
                self.score_rel += int(order[(order.index(self.heading) + t) % 4] == u_act)
            # update models
            self.abs_dir[action][u_act] += 1
            if self.heading is not None:
                self.rel_turn[action][self.turn_index(self.heading, u_act)] += 1
            self.step[action][max(abs(actual[0]), abs(actual[1]))] += 1
            for dc in self._dest(before, comp_moved, (u_act[0], u_act[1])):
                self.passed[action][dc] += 1
            self.heading = u_act
            self.avatar = frozenset((r + actual[0], c + actual[1]) for r, c in comp_moved)
            self.avatar_colour = int(colour)
            self.off[action][actual] += 1
        elif self.avatar is not None:
            u = self.predicted_unit(action)
            if u is not None and not (interior(before) != interior(after)).any():
                kf = self.step[action].most_common(1)[0][0] if self.step[action] else 1
                sweep = set()
                for k in range(1, int(kf) + 1):
                    sweep |= self._dest(before, self.avatar, (u[0] * k, u[1] * k))
                for dc in sweep:
                    self.block[action][dc] += 1


def evaluate_rich(before, after, action, pred):
    if pred is None or action not in DIRS:
        return 0, 0
    mv = movers_any(before, after)
    actual = None
    for colour, off in mv.items():
        actual = off
    if pred[1] is None:
        return 1, int(actual is None)
    if actual is None:
        return 1, 0
    # a slide prediction counts if the direction matches and the step is within 1
    pu, au = RichTheory.unit(pred[1]), RichTheory.unit(actual)
    return 1, int(pu == au and abs(max(abs(pred[1][0]), abs(pred[1][1])) - max(abs(actual[0]), abs(actual[1]))) <= 1)


class Theory:
    """Per-game object-level theory, fitted online."""

    def __init__(self):
        self.move = collections.defaultdict(collections.Counter)    # (action, colour) -> Counter(offset)
        self.block = collections.defaultdict(collections.Counter)   # (action, colour) -> Counter(dest colour) when blocked
        self.passed = collections.defaultdict(collections.Counter)  # (action, colour) -> Counter(dest colour) when moved
        self.click_changes = collections.defaultdict(lambda: [0, 0])  # clicked colour -> [changed, total]
        self.mover_colours = collections.Counter()

    def dest_colours(self, grid, colour, off):
        cells = set()
        for comp in components(grid, colour):
            cells |= comp
        dest = {(r + off[0], c + off[1]) for r, c in cells} - cells
        H, W = grid.shape
        return {int(grid[r, c]) if 0 <= r < H and 0 <= c < W else -1 for r, c in dest}

    def observe(self, before, after, action, param):
        if action in DIRS:
            mv = movers(before, after)
            for colour, off in mv.items():
                self.move[(action, colour)][off] += 1; self.mover_colours[colour] += 1
                for dc in self.dest_colours(before, colour, off):
                    self.passed[(action, colour)][dc] += 1
            if not mv:
                # nothing translated: record what sat in front of every known mover colour
                for colour in list(self.mover_colours):
                    offs = self.move[(action, colour)]
                    if not offs:
                        continue
                    off = offs.most_common(1)[0][0]
                    for dc in self.dest_colours(before, colour, off):
                        self.block[(action, colour)][dc] += 1
        elif action == 6 and param:
            r, c = int(param.get("y", 0)), int(param.get("x", 0))
            if 0 <= r < before.shape[0] and 0 <= c < before.shape[1]:
                colour = int(before[r, c]); changed = bool((before != after).any())
                self.click_changes[colour][0] += int(changed); self.click_changes[colour][1] += 1

    def predict(self, before, action, param):
        """-> None (no opinion) or a prediction: for directions {colour: offset-or-None}; for clicks True/False."""
        if action in DIRS:
            pred = {}
            for colour in self.mover_colours:
                offs = self.move[(action, colour)]
                if not offs:
                    continue
                off = offs.most_common(1)[0][0]
                dests = self.dest_colours(before, colour, off)
                blk = self.block[(action, colour)]; psd = self.passed[(action, colour)]
                blocked = any(blk[d] > psd[d] for d in dests)
                pred[colour] = None if blocked else off
            return pred or None
        if action == 6 and param:
            r, c = int(param.get("y", 0)), int(param.get("x", 0))
            colour = int(before[r, c]) if 0 <= r < before.shape[0] and 0 <= c < before.shape[1] else None
            ch, tot = self.click_changes.get(colour, [0, 0])
            return (ch * 2 > tot) if tot else None
        return None


def evaluate(before, after, action, param, pred):
    """-> (counted, correct) for this transition."""
    if pred is None:
        return 0, 0
    if action in DIRS:
        actual = movers(before, after)
        ok = True
        for colour, off in pred.items():
            if actual.get(colour) != off:
                ok = False
        return 1, int(ok)
    if action == 6:
        return 1, int(pred == bool((before != after).any()))
    return 0, 0


def run_game(arcade, gid, levels):
    env = arcade.make(gid)
    if env is None:
        return None
    f = env.reset(); seen_levels = f.levels_completed
    before = grid_of(f)
    thA, thB, thC, thD = Theory(), Theory(), AvatarTheory(), RichTheory()
    stats = {"transitions": 0, "dir": 0, "click": 0, "A_counted": 0, "A_correct": 0, "B_counted": 0, "B_correct": 0,
             "B1_counted": 0, "B1_correct": 0, "C_counted": 0, "C_correct": 0, "D_counted": 0, "D_correct": 0,
             "levels": len(levels), "goal_cells": []}
    level_idx = 0
    for li, lvl in enumerate(levels):
        for ai, (a, p) in enumerate(lvl):
            data = {"x": p["x"], "y": p["y"]} if a == 6 and p else None
            f = env.step(GameAction[f"ACTION{a}"], data)
            if f is None:
                return stats
            after = grid_of(f)
            stats["transitions"] += 1; stats["dir" if a in DIRS else "click" if a == 6 else "dir"] += (a in DIRS or a == 6)
            # regime B: online prediction before observing
            pb = thB.predict(before, a, p); cb, ok = evaluate(before, after, a, p, pb)
            stats["B_counted"] += cb; stats["B_correct"] += ok
            if li >= 1:
                stats["B1_counted"] += cb; stats["B1_correct"] += ok
                pa = thA.predict(before, a, p); ca, oka = evaluate(before, after, a, p, pa)
                stats["A_counted"] += ca; stats["A_correct"] += oka
            pc = thC.predict(before, a, p); cc, okc = evaluate_avatar(before, after, a, pc)
            stats["C_counted"] += cc; stats["C_correct"] += okc
            pd_ = thD.predict(before, a, p); cd, okd = evaluate_rich(before, after, a, pd_)
            stats["D_counted"] += cd; stats["D_correct"] += okd
            # goal census: at the completing action, what did the avatar enter / what was clicked?
            is_last = (ai == len(lvl) - 1)
            if is_last:
                if a in DIRS and thD.avatar is not None and thD.predicted_unit(a) is not None:
                    u = thD.predicted_unit(a); H, W = before.shape
                    front = {(r + u[0], c + u[1]) for r, c in thD.avatar} - set(thD.avatar)
                    cols = sorted({int(before[r, c]) for r, c in front if 0 <= r < H and 0 <= c < W})
                    stats["goal_cells"].append(("dir", cols))
                elif a == 6 and p:
                    r, c = int(p.get("y", 0)), int(p.get("x", 0))
                    stats["goal_cells"].append(("click", [int(before[r, c])] if 0 <= r < before.shape[0] and 0 <= c < before.shape[1] else []))
            thC.observe(before, after, a, p)
            thD.observe(before, after, a, p)
            thB.observe(before, after, a, p)
            if li == 0:
                thA.observe(before, after, a, p)
            before = after
            if f.levels_completed > seen_levels:
                seen_levels = f.levels_completed
    stats["mover_colours"] = len(thB.mover_colours)
    return stats


def main():
    import logging; logging.disable(logging.CRITICAL)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    files = sorted(glob.glob(f"{REPLAYS}/*.json"))
    want = set(sys.argv[1:])
    print(f"{'game':<6} {'trans':>5} {'dir':>4} {'clk':>4} | {'C: avatar':>14} | {'D: rich':>14} | goal cells entered/clicked at completion")
    tot = collections.Counter()
    for fpath in files:
        g = os.path.basename(fpath)[:-5]
        if want and g not in want:
            continue
        levels = [[(a, p) for a, p in lvl] for lvl in json.load(open(fpath))["levels"]]
        try:
            s = run_game(arcade, g, levels)
        except Exception as exc:
            print(f"{g:<6} EXC {type(exc).__name__}: {str(exc)[:70]}"); continue
        if not s:
            print(f"{g:<6} no env"); continue
        def pct(c, n): return f"{100 * c / n:5.0f}% ({c}/{n})" if n else "   n/a"
        goals = collections.Counter(tuple(c) for kind, c in s["goal_cells"] for c in [c])
        gtxt = " ".join(f"{list(k)}x{v}" for k, v in goals.most_common(4))
        print(f"{g:<6} {s['transitions']:>5} {s['dir']:>4} {s['click']:>4} | {pct(s['C_correct'], s['C_counted']):>14} | {pct(s['D_correct'], s['D_counted']):>14} | {gtxt}")
        for k in ("A_counted", "A_correct", "B1_counted", "B1_correct", "B_counted", "B_correct", "C_counted", "C_correct", "D_counted", "D_correct", "transitions"):
            tot[k] += s[k]
    print(f"\nALL    {tot['transitions']:>5} transitions | A {100*tot['A_correct']/max(1,tot['A_counted']):.0f}% of {tot['A_counted']} | "
          f"B(L2+) {100*tot['B1_correct']/max(1,tot['B1_counted']):.0f}% of {tot['B1_counted']} | B(all) {100*tot['B_correct']/max(1,tot['B_counted']):.0f}% of {tot['B_counted']} | C(avatar) {100*tot['C_correct']/max(1,tot['C_counted']):.0f}% of {tot['C_counted']} | D(rich) {100*tot['D_correct']/max(1,tot['D_counted']):.0f}% of {tot['D_counted']}")
    print("counted = transitions where the theory had an opinion (a known mover colour for the action, or a clicked colour seen before)")


if __name__ == "__main__":
    main()
