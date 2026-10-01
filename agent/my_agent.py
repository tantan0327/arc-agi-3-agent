"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

Only this file's text gets copied into the Kaggle run (see
`scripts/build_notebook.py`), so everything the agent needs — perception,
world model, policy — lives here as a single self-contained module rather
than a package of sibling imports, which would not survive the splice.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.

Strategy
--------
Instead of picking actions uniformly at random, this agent plays the way a
person encountering an unfamiliar game does: look at the screen, notice the
distinct shapes on it, try each control/shape once, remember what moved or
scored, and stop wasting turns repeating whatever did nothing.

  1. Perception  — flood-fill the 64x64 colour grid into connected-component
     "objects" (everything that isn't the background colour), so ACTION6
     clicks target object centroids/corners instead of uniformly random
     pixels out of 4096.
  2. World model — remember, per exact board state, which actions have been
     tried and whether they changed the frame or increased the level count.
     This memory is created once per game and survives RESETs between
     levels of the *same* game, so what's learned (e.g. "ACTION2 moves the
     player up") carries forward as later levels add new mechanics.
  3. Object tracking — link objects across consecutive frames by a
     persistent id (matched by exact position, then shape+color, then cell
     overlap), so "this blob moved" / "that blob got recolored" / "a new
     one appeared" are known facts rather than being re-derived from
     scratch every frame. A summary of these events rides along in each
     action's `reasoning`, for observability.
  4. Goal inference — correlate object events and pairwise touching
     relations against score/level increases, per *color* (not per object
     id, since ids aren't comparable across levels but a color's role
     usually is — e.g. "the collectible is always color 6"). This does two
     things: (a) a soft exploration bias, where untried ACTION6 clicks on
     colors that historically coincided with scoring get tried sooner on
     average without ever fully excluding the rest; and (b) once a
     specific color pair has been directly observed touching at the
     moment score increased, that relation becomes a *confirmed
     predicate* — from then on, any board where those two colors are
     touching is flagged as a milestone for the planner below, even if
     it's a state that's never itself produced a score. That's what turns
     "I got lucky once at this exact pixel layout" into "I know what this
     game wants," generalized across states.
  5. Effect prior — which kinds of action do anything at all, pooled
     across every state rather than per state. A board with 60+ objects
     offers 100+ candidate clicks, so a single state is never exhausted
     and per-state bookkeeping alone would re-test known-inert clicks
     forever. Pooling "does clicking this color ever change the screen"
     by color makes that knowledge reusable the moment the board shifts.
  6. Motion model — what a directional action *means*: "ACTION1 moves this
     object by (-5, 0)". Everything above only asks whether an action
     changed something, which on a movement game is true of every action
     and so says nothing. Learned per tracked object id, this identifies
     the avatar (the thing whose motion depends on which key was pressed)
     and separately the noise (counters and timers, which move the same
     way regardless — those get dropped from the state key in (2)).
  7. Contact model — which colours to chase and which to keep away from,
     learned from what was beside the avatar when the run ended or the
     level did. Scored by *lift* against the board-wide rate, because the
     background colour is beside the avatar for every death there has ever
     been and by raw rate looks like the deadliest thing on screen.
  8. Navigation — with the controls known, search collapses from the whole
     board to the avatar's own position, a few hundred cells instead of a
     state per pixel arrangement. BFS there toward cells beside a goal
     colour, treating cells beside a hazard colour as walls, and falling
     back to the nearest unvisited cell. Walls are learned from moves that
     didn't land where the offset predicted. A random walk in a maze keeps
     recrossing its own path.
  9. Policy — navigate when the controls are understood; otherwise try
     untried actions first (simple actions before coordinate clicks, since
     they're cheaper to probe; clicks weighted by (5) × (4a)). Once a
     state has been fully probed, replay whichever action previously
     changed the frame or scored at that exact state; failing that, BFS
     over the state-transition graph recorded so far — including the
     predicate-generalized milestones from (4b) — to find a path to a
     scoring-relevant state, and take the first step of that path
     (re-planned from scratch every call, so it self-corrects if reality
     ever diverges from the recorded graph).

Known gaps. Goal learning cannot bootstrap: (7) learns a destination from
what was beside the avatar when a level completed, so on a game that has
never once been completed there is nothing to learn from, and five of the
six move-only games are still at zero. Hazard learning does fire — wa30
identifies its killer colour — but avoiding death is not the same as making
progress. Goal inference (4) is limited to pairwise colour-touching: no
containment, counting or ordering, and it waits to notice a predicate rather
than steering to make one true. The simulator (see `_Simulator`) is off:
twice measured, twice a net loss, because a ~12 ms deepcopy has to buy
something that happens more often than once in 400 steps.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import importlib.util
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from arcengine import FrameData, GameAction, GameState
from arcengine.enums import ActionInput

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

Grid = list[list[int]]

# ─────────────────────────────────────────────────────────────────────────
# Perception: raw grid → objects → ACTION6 click candidates
# ─────────────────────────────────────────────────────────────────────────


Cells = frozenset[tuple[int, int]]


def _bbox(cells: Cells) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (min(rows), min(cols), max(rows), max(cols))


def _centroid(cells: Cells) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (sum(rows) // len(rows), sum(cols) // len(cols))


def _shape_signature(cells: Cells) -> tuple[tuple[int, int], ...]:
    """Cell offsets from the bbox top-left, so a translated copy of the same
    blob has the same signature — used by the tracker to recognize "moved"."""
    top, left, _, _ = _bbox(cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


@dataclass(frozen=True)
class _Object:
    color: int
    cells: Cells

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> tuple[int, int]:
        return _centroid(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return _bbox(self.cells)

    def shape_signature(self) -> tuple[tuple[int, int], ...]:
        return _shape_signature(self.cells)


def _background_color(grid: Grid) -> int:
    counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _extract_objects(grid: Grid) -> list[_Object]:
    """4-connected flood fill over non-background cells, grouped by color."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    bg = _background_color(grid)
    seen = [[False] * w for _ in range(h)]
    objects: list[_Object] = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == bg:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr][nc]
                        and grid[nr][nc] == color
                    ):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            objects.append(_Object(color=color, cells=frozenset(cells)))
    return objects


MIN_BAND_PURITY = 0.80


def _ui_bands(grid: Grid, max_share: float = 0.25) -> tuple[int, int, int, int]:
    """How far in from each edge the screen is readout rather than playfield,
    as (top, bottom, left, right) counts of rows/columns.

    Every board looked at keeps its readouts in a strip along an edge: a
    progress bar across the top of cn04 and su15, a one-row status line on
    wa30 and ka59, and on sk48 a legend that spells out the order the targets
    must be collected in. The agent aims clicks at all of it — measured, 17%
    of every click lands there, and 22% on r11l, which is 61% of the score —
    and folds each readout change into the state key, which helps keep 71-86%
    of states unique on a movement game.

    A band is a run of lines from an edge whose dominant colour is not the
    surround's. Two rules were tried and discarded first, both by looking at
    what they produced rather than by reasoning:

      * "a uniform line is a separator" — useless, since 48 of wa30's 64 rows
        are a single colour for the simple reason that most rows are empty.
      * taking the surround from the middle of the board — worse, because on
        sk48 the middle is the yellow playfield, so the grey surround becomes
        the anomaly: it darkened the margins and missed the legend entirely.

    Taking the surround from the whole board gets both right. Where a bar
    covers only part of the width (cn04's) this finds nothing, which is the
    safe way to be wrong.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h < 8 or w < 8:
        return (0, 0, 0, 0)
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    surround = max(counts.items(), key=lambda kv: kv[1])[0]

    def dominant_is_surround(values) -> bool:
        seen: dict[int, int] = {}
        for v in values:
            seen[v] = seen.get(v, 0) + 1
        return max(seen.items(), key=lambda kv: kv[1])[0] == surround

    def dominant(values) -> int:
        seen: dict[int, int] = {}
        for v in values:
            seen[v] = seen.get(v, 0) + 1
        return max(seen.items(), key=lambda kv: kv[1])[0]

    def run(lines, limit: int) -> int:
        """Lines in from an edge that are off-surround *and all alike*.

        The second half is what stops this eating the board. A readout strip
        is the same stuff all the way along — a bar, a status line, a legend
        laid on one background. Content is not: vc33's right quarter is
        off-surround for sixteen columns and was swallowed whole, taking the
        grey bars, the green column and the yellow marker with it, which cost
        that game every level it had been completing. Requiring one dominant
        colour throughout rejects it and keeps the genuine strips, which run
        one to eleven lines.
        """
        n = 0
        first: Optional[int] = None
        kept: list[int] = []
        for values in lines:
            if n >= limit:
                break
            top = dominant(values)
            if top == surround or (first is not None and top != first):
                break
            first = top
            kept.extend(values)
            n += 1
        # A readout strip is nearly all one colour with small markings on it;
        # content is not. Measured against bands verified by eye, the two
        # separate cleanly: kept su15 0.99, wa30 1.00, ka59 1.00, sk48 0.85
        # against rejected vc33 0.77 and dc22 0.60. Without this, vc33's right
        # quarter passes the all-alike test — every one of those columns is
        # black-dominant — and the game loses every level it was completing.
        if kept and kept.count(first) / len(kept) < MIN_BAND_PURITY:
            return 0
        return n

    lim_r, lim_c = max(int(h * max_share), 1), max(int(w * max_share), 1)
    top = run((grid[r] for r in range(h)), lim_r)
    bottom = run((grid[r] for r in range(h - 1, -1, -1)), lim_r)
    left = run(([grid[r][c] for r in range(h)] for c in range(w)), lim_c)
    right = run(([grid[r][c] for r in range(h)] for c in range(w - 1, -1, -1)), lim_c)
    return (top, bottom, left, right)


def _in_bands(row: int, col: int, bands: tuple[int, int, int, int],
              h: int, w: int) -> bool:
    top, bottom, left, right = bands
    return (row < top or row >= h - bottom or col < left or col >= w - right)


def _click_candidates(
    grid: Grid, objects: list[_Object], max_candidates: int = 128,
    bands: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> list[tuple[int, int, int]]:
    """(x, y, color) points worth ACTION6-clicking, tagged with the color of
    the object that produced them (the color feeds goal inference below).
    Falls back to a coarse sample grid tagged with the background color when
    the board has too few distinct objects (e.g. an almost-blank screen).

    Coverage before depth: *every* object contributes its centroid before any
    object contributes a second point. An earlier version spent five points
    (centroid + four bbox corners) per object, largest first, capped at 24 —
    which on a board of 64 objects only ever clicked the biggest five. On
    ft09 that was the difference between zero working actions and eight: the
    only clicks that do anything there are on 36-cell tiles that never made
    the cut.
    """
    ordered = sorted(objects, key=lambda o: o.size, reverse=True)

    points: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    h, w = len(grid), len(grid[0]) if grid else 0

    def add(row: int, col: int, color: int) -> None:
        # ARC-AGI-3's ComplexAction takes (x, y) = (col, row).
        pt = (col, row)
        if pt in seen or len(points) >= max_candidates:
            return
        if _in_bands(row, col, bands, h, w):
            return
        seen.add(pt)
        points.append((col, row, color))

    for obj in ordered:
        add(*obj.centroid, obj.color)

    # Corners are a second tier: worth having for large objects whose centroid
    # can land on a hole or an inert middle, but never at the cost of leaving
    # another object untouched entirely.
    for obj in ordered:
        if len(points) >= max_candidates:
            break
        top, left, bottom, right = obj.bbox
        for r, c in ((top, left), (top, right), (bottom, left), (bottom, right)):
            add(r, c, obj.color)

    if len(points) < 4:
        bg = _background_color(grid)
        h, w = len(grid), len(grid[0]) if grid else 0
        step_r, step_c = max(1, h // 8), max(1, w // 8)
        for r in range(step_r // 2, h, step_r):
            for c in range(step_c // 2, w, step_c):
                add(r, c, bg)

    return points


# ─────────────────────────────────────────────────────────────────────────
# Object tracking: link objects across consecutive frames by persistent id
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _TrackedObject:
    obj_id: int
    color: int
    cells: frozenset[tuple[int, int]]


@dataclass
class _ObjectEvent:
    kind: str  # "moved" | "recolored" | "appeared" | "disappeared"
    obj_id: int
    color: int
    detail: object = None  # (dr, dc) for moved; previous color for recolored


class _ObjectTracker:
    """Assigns a persistent id to each on-screen object and reports what
    happened to it frame over frame (moved / recolored / appeared /
    disappeared), by shape+color+position matching. Raw pixel diffs only
    say "something changed"; this turns that into object-level events, so
    `_GoalModel` below can ask "did the object that moved also cause the
    score to go up?" instead.

    Alive for the whole `Agent.main()` loop, so it survives RESETs between
    levels of one game — a level transition just shows up as a wave of
    "disappeared" + "appeared" events, which is accurate, not a bug.

    Known limitation: an object that changes shape *and* moves in the same
    frame reads as disappeared+appeared (new identity) rather than moved.
    Acceptable for v1; revisit only if it turns out to matter empirically.
    """

    def __init__(self) -> None:
        self._live: dict[int, _TrackedObject] = {}
        self._next_id = 0

    def position_of(self, obj_id: int) -> Optional[tuple[int, int]]:
        tracked = self._live.get(obj_id)
        return None if tracked is None else _centroid(tracked.cells)

    def color_of(self, obj_id: int) -> Optional[int]:
        tracked = self._live.get(obj_id)
        return None if tracked is None else tracked.color

    def update(self, objects: list[_Object]) -> list[_ObjectEvent]:
        old = self._live
        unmatched_old = set(old)
        unmatched_new = list(range(len(objects)))
        matches: dict[int, int] = {}  # new-object index -> obj_id

        # Pass 1: exact match (unchanged).
        for ni in list(unmatched_new):
            for oid in unmatched_old:
                if old[oid].color == objects[ni].color and old[oid].cells == objects[ni].cells:
                    matches[ni] = oid
                    unmatched_old.discard(oid)
                    unmatched_new.remove(ni)
                    break

        # Pass 2: same color + shape, different position -> moved.
        moved_candidates = []
        for ni in unmatched_new:
            sig = objects[ni].shape_signature()
            n_centroid = objects[ni].centroid
            for oid in unmatched_old:
                o = old[oid]
                if o.color == objects[ni].color and _shape_signature(o.cells) == sig:
                    o_centroid = _centroid(o.cells)
                    dist = abs(n_centroid[0] - o_centroid[0]) + abs(n_centroid[1] - o_centroid[1])
                    moved_candidates.append((dist, ni, oid))
        for _, ni, oid in sorted(moved_candidates, key=lambda t: t[0]):
            if ni in matches or oid not in unmatched_old or ni not in unmatched_new:
                continue
            matches[ni] = oid
            unmatched_old.discard(oid)
            unmatched_new.remove(ni)

        # Pass 3: same position (high cell overlap), different color -> recolored.
        recolor_candidates = []
        for ni in unmatched_new:
            for oid in unmatched_old:
                o = old[oid]
                overlap = len(o.cells & objects[ni].cells)
                union = len(o.cells | objects[ni].cells)
                if union and overlap / union >= 0.5:
                    recolor_candidates.append((overlap / union, ni, oid))
        for _, ni, oid in sorted(recolor_candidates, key=lambda t: -t[0]):
            if ni in matches or oid not in unmatched_old or ni not in unmatched_new:
                continue
            matches[ni] = oid
            unmatched_old.discard(oid)
            unmatched_new.remove(ni)

        events: list[_ObjectEvent] = []
        new_live: dict[int, _TrackedObject] = {}
        for ni, obj in enumerate(objects):
            if ni in matches:
                oid = matches[ni]
                prev = old[oid]
                if prev.cells != obj.cells and prev.color == obj.color:
                    pr, pc, _, _ = _bbox(prev.cells)
                    nr, nc, _, _ = obj.bbox
                    events.append(_ObjectEvent("moved", oid, obj.color, (nr - pr, nc - pc)))
                elif prev.color != obj.color:
                    events.append(_ObjectEvent("recolored", oid, obj.color, prev.color))
                new_live[oid] = _TrackedObject(oid, obj.color, obj.cells)
            else:
                oid = self._next_id
                self._next_id += 1
                events.append(_ObjectEvent("appeared", oid, obj.color))
                new_live[oid] = _TrackedObject(oid, obj.color, obj.cells)

        for oid in unmatched_old:
            events.append(_ObjectEvent("disappeared", oid, old[oid].color))

        self._live = new_live
        return events


def _summarize_events(events: list[_ObjectEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────
# Goal inference: correlate object events/relations with score increases
# ─────────────────────────────────────────────────────────────────────────


def _cells_touch(cells_a: Cells, cells_b: Cells) -> bool:
    for r, c in cells_a:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cells_b:
                return True
    return False


def _touching_pairs(objects: list[_Object]) -> set[tuple[int, int]]:
    """Unordered (colorA, colorB) pairs of differently-colored objects with
    at least one adjacent cell between them right now — a candidate for
    "these two are meant to meet"."""
    pairs = set()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a.color != b.color and _cells_touch(a.cells, b.cells):
                pairs.add((min(a.color, b.color), max(a.color, b.color)))
    return pairs


@dataclass
class _GoalModel:
    """Correlates object-level events and relations (from `_ObjectTracker`
    and `_touching_pairs`) with score/level increases, to build a simple,
    generalizable hypothesis of what matters. The correlating keys are
    *colors*, not object ids, since a tracked object's id isn't comparable
    across levels/resets but its color role usually is (e.g. "the
    collectible is always color 6", "color 6 touching color 2 scores")."""

    # color -> [total events of this color, how many coincided with scoring]
    color_totals: dict[int, list[int]] = field(default_factory=dict)
    # (min_color, max_color) -> [total times seen touching, how many
    # coincided with scoring]
    pair_totals: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def observe(
        self,
        events: list[_ObjectEvent],
        touching: set[tuple[int, int]],
        score_increased: bool,
    ) -> None:
        for e in events:
            counts = self.color_totals.setdefault(e.color, [0, 0])
            counts[0] += 1
            if score_increased:
                counts[1] += 1
        for pair in touching:
            counts = self.pair_totals.setdefault(pair, [0, 0])
            counts[0] += 1
            if score_increased:
                counts[1] += 1

    def color_score(self, color: int) -> float:
        """Exploration weight for an object of this color: > 1 if its past
        events correlated with scoring, 1.0 (neutral) if unknown or if
        every past event was a miss. One-sided shrinkage — dividing by
        `total + 2` instead of `total` — keeps a single lucky
        co-occurrence from being overtrusted, while never scoring a color
        *below* neutral just because it has more (all-negative) data than
        an untested one."""
        total, score_linked = self.color_totals.get(color, (0, 0))
        if total == 0:
            return 1.0
        rate = score_linked / (total + 2.0)
        return 1.0 + 4.0 * rate

    def predicate_score(self, pair: tuple[int, int]) -> float:
        """Same one-sided shrinkage as `color_score`, but for "these two
        colors touching" as a hypothesized goal condition. 0.0 (not 1.0)
        when unseen — unlike exploration weighting, an unconfirmed pair
        should never bias planning; only directly-observed correlation
        should."""
        total, hits = self.pair_totals.get(pair, (0, 0))
        if total == 0:
            return 0.0
        return 4.0 * hits / (total + 2.0)

    def confirmed_predicates(self, threshold: float = 0.5) -> set[tuple[int, int]]:
        """Colour pairs directly observed touching at the moment score
        increased, often enough relative to how often they merely touch
        without scoring, to trust as a reusable goal condition — one that
        generalizes beyond the exact board state it was first seen in."""
        return {
            pair
            for pair in self.pair_totals
            if self.predicate_score(pair) >= threshold
        }


@dataclass
class _EffectModel:
    """Which *kinds* of action do anything at all in this game, pooled across
    every state seen so far.

    `_WorldModel` can only answer "have I tried this action at this exact
    state?", and with 60+ objects on screen a state offers 100+ candidate
    clicks — so a state is never exhausted, and the agent re-tests clicks it
    has already learned are inert every time the board shifts. Measured on
    ft09: of 64 objects, only the eight 36-cell color-9 tiles respond to a
    click at all; everything else is scenery. That fact is a property of the
    *game*, not of one board layout, so pooling it by color is what makes it
    reusable.

    Used to weight the choice among untried candidates, never to forbid one:
    an unseen color starts at 0.5 (optimistic enough to get tried), a color
    that has done nothing in 20 attempts falls to ~0.05, and a reliable one
    approaches 1.0.
    """

    simple: dict[int, list[int]] = field(default_factory=dict)
    clicks: dict[int, list[int]] = field(default_factory=dict)

    def observe(
        self, action_id: int, color: Optional[int], frame_changed: bool
    ) -> None:
        table = self.clicks if color is not None else self.simple
        counts = table.setdefault(color if color is not None else action_id, [0, 0])
        counts[0] += 1
        if frame_changed:
            counts[1] += 1

    @staticmethod
    def _weight(counts: Optional[list[int]]) -> float:
        if not counts:
            return 0.5
        tried, changed = counts
        return (changed + 1.0) / (tried + 2.0)

    def simple_weight(self, action_id: int) -> float:
        return self._weight(self.simple.get(action_id))

    def click_weight(self, color: int) -> float:
        return self._weight(self.clicks.get(color))


@dataclass
class _MotionModel:
    """Learns what a directional action *means*: "ACTION1 moves colour 9 by
    (-5, 0)".

    Everything above only ever asks whether an action changed something. On a
    movement game every action always changes something, so all of it is
    uninformative — measured across all 25 games, the six move-only games
    (g50t, ls20, re86, tr87, tu93, wa30) score zero at every budget from 80 up
    to 10,000 actions, while click-capable games keep climbing. Exploration
    volume cannot substitute for knowing what the controls do.

    Two things fall out of the same table of observations:

    * **The avatar** is whatever moves *differently depending on which action
      was pressed*. On ls20 that is colours 9 and 12, which shift by (-5,0),
      (0,-5) or (0,+5) according to the key.
    * **Noise** is whatever moves the same way no matter which action was
      pressed — ls20's colour 11 advances by (0,1) on literally every action
      because it is the step counter. It carries no decision-relevant
      information yet changes the board every frame, which is a large part of
      why 88% of ls20 states were unique.
    """

    # (action_id, obj_id) -> {(dr, dc): count} — per tracked object, because
    # colour alone is not the avatar. On ls20 six separate objects share the
    # avatar's colours; picking "the first object of that colour" made the
    # reported position jump between them every frame, which in turn made the
    # wall detector below mark almost every move as blocked.
    by_object: dict[tuple[int, int], dict[tuple[int, int], int]] = field(
        default_factory=dict
    )
    # (action_id, color) -> {(dr, dc): count} — colour is the right grain for
    # spotting counters, which are usually the only thing of their colour.
    by_color: dict[tuple[int, int], dict[tuple[int, int], int]] = field(
        default_factory=dict
    )

    # An action must have been seen this many times before its motion is
    # trusted; below that a single coincidence could name an avatar.
    MIN_OBSERVATIONS = 3

    def observe(self, action_id: int, events: list[_ObjectEvent]) -> None:
        for e in events:
            if e.kind != "moved" or not isinstance(e.detail, tuple):
                continue
            for table, key in (
                (self.by_object, (action_id, e.obj_id)),
                (self.by_color, (action_id, e.color)),
            ):
                counts = table.setdefault(key, {})
                counts[e.detail] = counts.get(e.detail, 0) + 1

    @classmethod
    def _dominant(
        cls, counts: Optional[dict[tuple[int, int], int]]
    ) -> Optional[tuple[int, int]]:
        if not counts:
            return None
        total = sum(counts.values())
        if total < cls.MIN_OBSERVATIONS:
            return None
        delta, count = max(counts.items(), key=lambda kv: kv[1])
        # A move blocked by a wall shows up as the same action sometimes
        # producing no motion, so this needs a majority, not unanimity. A
        # zero offset is never a useful control — it would make the navigator
        # plan a step that goes nowhere.
        if count * 2 <= total or delta == (0, 0):
            return None
        return delta

    def scheme_for_object(self, obj_id: int) -> dict[int, tuple[int, int]]:
        """action_id -> offset for one tracked object: its control scheme."""
        out: dict[int, tuple[int, int]] = {}
        for (action_id, oid), counts in self.by_object.items():
            if oid != obj_id:
                continue
            delta = self._dominant(counts)
            if delta is not None:
                out[action_id] = delta
        return out

    def avatar_id(self) -> Optional[int]:
        """The tracked object the player is steering: the one whose motion
        depends on *which* action was pressed. Ties break toward the object
        with the most distinct known offsets — the best-understood control —
        and then by id, so the answer is stable frame to frame."""
        best: Optional[tuple[int, int]] = None
        for obj_id in {oid for _, oid in self.by_object}:
            scheme = self.scheme_for_object(obj_id)
            if len(scheme) >= 2 and len(set(scheme.values())) >= 2:
                rank = (len(set(scheme.values())), -obj_id)
                if best is None or rank > best:
                    best = rank
                    best_id = obj_id
        return None if best is None else best_id

    def noise_colors(self) -> set[int]:
        """Colours that move the same way under every action seen — counters,
        timers, animations. ls20's colour 11 advances by (0,1) on literally
        every action because it is the step counter. Worth excluding from the
        state key: it changes the board every frame and means nothing."""
        out = set()
        for color in {c for _, c in self.by_color}:
            deltas = {
                self._dominant(counts)
                for (action_id, c), counts in self.by_color.items()
                if c == color
            }
            deltas.discard(None)
            if len(deltas) == 1:
                seen_actions = {a for a, c in self.by_color if c == color}
                if len(seen_actions) >= 2:
                    out.add(color)
        return out


def _colors_near(
    position: tuple[int, int], objects: list[_Object], radius: int
) -> set[int]:
    """Colours of objects the avatar is close enough to be interacting with.
    Radius is the avatar's own step size, so "near" means "reachable in about
    one move" rather than an arbitrary distance."""
    row, col = position
    near: set[int] = set()
    for obj in objects:
        if obj.color in near:
            continue
        # Reject on the bounding box before touching cells; a large object
        # can hold thousands and this runs every step.
        top, left, bottom, right = obj.bbox
        if row < top - radius or row > bottom + radius:
            continue
        if col < left - radius or col > right + radius:
            continue
        for r, c in obj.cells:
            if abs(r - row) <= radius and abs(c - col) <= radius:
                near.add(obj.color)
                break
    return near


@dataclass
class _ContactModel:
    """Which colours the avatar should chase and which it should keep away
    from, learned from what was beside it when the run ended or the level did.

    `_Navigator` can steer, but "somewhere I haven't been" is not the
    objective — on a game with hazards it is actively dangerous, and ls20's
    hidden life counter falls 3 -> 1 within 120 random actions. Nothing so far
    tells the agent *what* on the board it is supposed to reach or avoid;
    `_GoalModel` only relates colours that touch each other, which never fires
    on a movement game because the avatar is the only thing doing the
    touching.

    Both signals come from the same observation: note what is beside the
    avatar each step, then when the run ends or the level completes, credit
    or blame those colours. Scored with the same one-sided shrinkage used
    elsewhere — an unseen colour is neutral, and a colour only becomes
    dangerous on repeated evidence, never on one unlucky step.
    """

    danger: dict[int, list[int]] = field(default_factory=dict)
    goal: dict[int, list[int]] = field(default_factory=dict)
    observations: int = 0
    deaths: int = 0
    successes: int = 0

    # A colour must be seen this often before it can be judged, and its rate
    # must beat the board-wide rate by this much. The ratio matters more than
    # the rate: the floor colour is beside the avatar for every death there
    # has ever been, so by absolute rate it looks like the deadliest thing on
    # screen. What marks a real hazard is dying near it *more often than you
    # die in general*.
    MIN_SIGHTINGS = 4
    MIN_LIFT = 2.0

    def observe(self, colors: set[int], died: bool, scored: bool) -> None:
        self.observations += 1
        self.deaths += int(died)
        self.successes += int(scored)
        for color in colors:
            for table, hit in ((self.danger, died), (self.goal, scored)):
                counts = table.setdefault(color, [0, 0])
                counts[0] += 1
                if hit:
                    counts[1] += 1

    def _lift(self, counts: Optional[list[int]], total_hits: int) -> float:
        if not counts or not self.observations or not total_hits:
            return 0.0
        seen, hits = counts
        if seen < self.MIN_SIGHTINGS or not hits:
            return 0.0
        base = total_hits / self.observations
        return (hits / seen) / base if base else 0.0

    def danger_colors(self) -> set[int]:
        return {
            c
            for c, counts in self.danger.items()
            if self._lift(counts, self.deaths) >= self.MIN_LIFT
        }

    def goal_colors(self) -> set[int]:
        # A colour that kills is not a destination, however often the level
        # happened to end near it.
        dangerous = self.danger_colors()
        return {
            c
            for c, counts in self.goal.items()
            if self._lift(counts, self.successes) >= self.MIN_LIFT
            and c not in dangerous
        }


@dataclass
class _Navigator:
    """Steers the avatar toward somewhere it has not been, using the control
    scheme `_MotionModel` worked out.

    Once the avatar and its offsets are known, the search space collapses from
    the whole board to the avatar's own position — a few hundred cells instead
    of a state per pixel arrangement. A random walk in a maze keeps crossing
    its own path; BFS to the nearest unvisited cell covers ground in roughly
    the number of moves it takes to walk there.

    Walls are learned the same way everything else here is: press a direction
    whose offset is known, and if the avatar does not arrive, record that edge
    as blocked.
    """

    visited: set[tuple[int, int]] = field(default_factory=set)
    blocked: set[tuple[tuple[int, int], int]] = field(default_factory=set)

    def note_position(self, position: tuple[int, int]) -> None:
        self.visited.add(position)

    def note_blocked(self, position: tuple[int, int], action_id: int) -> None:
        self.blocked.add((position, action_id))

    def next_action(
        self,
        position: tuple[int, int],
        scheme: dict[int, tuple[int, int]],
        targets: Optional[set[tuple[int, int]]] = None,
        avoid: Optional[set[tuple[int, int]]] = None,
        max_nodes: int = 4000,
    ) -> Optional[int]:
        """First action of the shortest known path to somewhere worth going.

        With `targets` — cells beside something `_ContactModel` associates
        with completing a level — head for the nearest of those; that is the
        difference between wandering and pursuing. Without them, fall back to
        the nearest unvisited cell, which is still far better than a random
        walk. `avoid` cells are treated as walls, so a known hazard is routed
        around rather than blundered into.

        Returns None when there is nothing left to head for, in which case the
        caller should fall back to ordinary exploration.
        """
        if not scheme:
            return None
        avoid = avoid or set()

        queue: list[tuple[tuple[int, int], Optional[int]]] = [(position, None)]
        seen = {position}
        head = 0
        while head < len(queue) and head < max_nodes:
            current, first = queue[head]
            head += 1
            for action_id, (dr, dc) in scheme.items():
                if (current, action_id) in self.blocked:
                    continue
                nxt = (current[0] + dr, current[1] + dc)
                if not (0 <= nxt[0] < 64 and 0 <= nxt[1] < 64) or nxt in seen:
                    continue
                if nxt in avoid:
                    continue
                step = first if first is not None else action_id
                wanted = nxt in targets if targets else nxt not in self.visited
                if wanted:
                    return step
                seen.add(nxt)
                queue.append((nxt, step))
        return None


# ─────────────────────────────────────────────────────────────────────────
# Simulator: a private copy of the game, for looking before leaping
# ─────────────────────────────────────────────────────────────────────────

_GAME_SOURCE_GLOBS = (
    "environment_files/{gid}/*/{gid}.py",
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files/{gid}/*/{gid}.py",
    "/kaggle/**/environment_files/{gid}/*/{gid}.py",
    "/tmp/**/environment_files/{gid}/*/{gid}.py",
)


def _find_game_source(game_id: str) -> Optional[str]:
    """Locate a game's own Python source. Confirmed present on Kaggle at
    /kaggle/input/competitions/.../environment_files/{gid}/{hash}/{gid}.py —
    the competition ships every game's implementation alongside the wheels."""
    gid = game_id.split("-")[0]
    for pattern in _GAME_SOURCE_GLOBS:
        hits = glob.glob(pattern.format(gid=gid), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


@dataclass
class _Outcome:
    """What one candidate action would do, according to the simulation."""

    levels_completed: int
    state: str
    frame_changed: bool

    @property
    def scored(self) -> bool:
        return self.levels_completed > 0

    @property
    def fatal(self) -> bool:
        return self.state == "GAME_OVER"


class _Simulator:
    """A private instance of the game, kept in step with the real one, so
    candidate moves can be tried in imagination before being spent.

    Everything else in this agent has to learn by consequence: take the
    action, see what happened, remember. That costs a real action per fact
    and, on a game that can kill you, costs a life to learn that something
    is lethal. With the game's own code in hand the agent can instead look
    one move ahead and simply *see* which candidates score and which end the
    run — the difference between feeling for the edge of a cliff and being
    handed a map.

    Cost measured on ls20: `deepcopy` 11.7 ms, a simulated action 0.6 ms —
    so the copy dominates, which is why lookahead is spent on a bounded
    shortlist rather than all 100+ click candidates.

    The mirror is only trusted while it demonstrably matches reality: every
    step compares its frame against the real one and disables simulation on
    any divergence. A confidently wrong model is worse than none, and the
    rest of the agent already works without this.
    """

    def __init__(self, game_id: str) -> None:
        self.game = None
        self.enabled = False
        self._load(game_id)

    @classmethod
    def disabled(cls) -> "_Simulator":
        """An inert instance, for when lookahead is switched off — avoids
        paying the source load and keeps every call site unconditional."""
        sim = cls.__new__(cls)
        sim.game = None
        sim.enabled = False
        return sim

    def _load(self, game_id: str) -> None:
        source = _find_game_source(game_id)
        if source is None:
            return
        try:
            text = Path(source).read_text(encoding="utf-8")
            spec = importlib.util.spec_from_loader(
                f"_sim_{game_id.replace('-', '_')}", loader=None
            )
            if spec is None:
                return
            module = importlib.util.module_from_spec(spec)
            exec(text, module.__dict__)
            # The class name isn't given to us; take the ARCBaseGame subclass
            # the module defines. Matching on the base by name avoids needing
            # arcengine's class object here.
            cls = None
            for value in module.__dict__.values():
                if not isinstance(value, type):
                    continue
                bases = {b.__name__ for b in getattr(value, "__mro__", ())}
                if "ARCBaseGame" in bases and value.__name__ != "ARCBaseGame":
                    cls = value
                    break
            if cls is None:
                return
            self.game = cls()
            self.enabled = True
        except Exception:
            # Any failure here just means the agent runs as it did before.
            self.game = None
            self.enabled = False

    def _step(self, game, key: ActionKey):
        action = _key_to_action(key)
        payload = ActionInput(id=action, data=action.action_data.model_dump())
        return game.perform_action(payload, raw=True)

    @staticmethod
    def _frame_of(raw) -> Optional[Grid]:
        frames = getattr(raw, "frame", None)
        if not frames:
            return None
        first = frames[0]
        return first.tolist() if hasattr(first, "tolist") else first

    def resync(self, key: ActionKey, real_grid: Grid) -> None:
        """Apply the action the agent actually took, then check the mirror
        still agrees with reality. Disable on divergence."""
        if not self.enabled or self.game is None:
            return
        try:
            raw = self._step(self.game, key)
            mirrored = self._frame_of(raw)
            if mirrored is not None and mirrored != real_grid:
                self.enabled = False
        except Exception:
            self.enabled = False

    def scalars(self) -> dict[str, float]:
        """The mirror's own numeric fields. Free — the mirror is already
        being stepped, so reading it costs nothing, and it exposes state the
        frame doesn't show. On ls20 one of these is the life counter
        (`aqygnziho`), which drops 3 -> 1 within 120 random actions: the
        agent is dying constantly and had no way to know."""
        if not self.enabled or self.game is None:
            return {}
        return {
            k: float(v)
            for k, v in self.game.__dict__.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k != "_action_count"
        }

    def vet(self, key: ActionKey) -> Optional[tuple[_Outcome, dict[str, float]]]:
        """Simulate one candidate — the move about to be played — and report
        what it would cost. One clone, versus the twelve per step that
        made the first design a net loss."""
        if not self.enabled or self.game is None:
            return None
        try:
            clone = self._clone()
            raw = self._step(clone, key)
            after = {
                k: float(v)
                for k, v in clone.__dict__.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and k != "_action_count"
            }
            return (
                _Outcome(
                    levels_completed=getattr(raw, "levels_completed", 0) or 0,
                    state=str(getattr(raw, "state", "")).split(".")[-1],
                    frame_changed=True,
                ),
                after,
            )
        except Exception:
            return None

    # Attributes that hold the level data. Measured with `copy.deepcopy`,
    # these are 92-98% of the cost of cloning a game: 1.03 + 0.95 of vc33's
    # 2.0 ms, and the same shape on every game tried. Everything else — the
    # camera, the action, the sprites, the bookkeeping scalars — rounds to
    # zero.
    _CLEAN_LEVELS = "_clean_levels"
    _LEVELS = "_levels"

    def _clone(self):
        """A copy that shares what a simulated step never writes to.

        `deepcopy` duplicates every level of the game plus a pristine copy of
        every level, on every candidate, when a step can only touch the level
        being played. Checked rather than assumed, across vc33, sp80 and ls20,
        42 stepped clones each: **zero** mutated `_clean_levels`, and **zero**
        mutated any level other than the current one.

        So `_clean_levels` is shared outright and every level but the one in
        play is shared too. If that assumption is ever wrong the mirror
        diverges from the real board and `resync` disables simulation, which
        is the same safety net that already covers every other way this can
        be wrong.

        Done by pre-seeding `deepcopy`'s memo so the shared parts map to
        themselves. Copying attribute by attribute instead does not work, and
        the equivalence test catches it: `deepcopy` keeps objects that appear
        twice in the graph aliased in the copy, and copying each attribute
        separately splits them apart. That produced 440 differing frames in
        1,080 on vc33 and made ls20 *slower*, by duplicating shared
        substructures once per attribute that referenced them.
        """
        game = self.game
        levels = getattr(game, self._LEVELS, None)
        if not isinstance(levels, list):
            return copy.deepcopy(game)
        index = getattr(game, "_current_level_index", 0)
        memo: dict[int, object] = {}
        clean = getattr(game, self._CLEAN_LEVELS, None)
        if clean is not None:
            memo[id(clean)] = clean
        for i, level in enumerate(levels):
            if i != index:
                memo[id(level)] = level
        return copy.deepcopy(game, memo)

    def evaluate(self, keys: list[ActionKey]) -> dict[ActionKey, _Outcome]:
        """Play each candidate on its own throwaway copy and report what
        happens. One clone per candidate, so callers must shortlist."""
        if not self.enabled or self.game is None:
            return {}
        out: dict[ActionKey, _Outcome] = {}
        for key in keys:
            try:
                clone = self._clone()
                raw = self._step(clone, key)
                grid = self._frame_of(raw)
                out[key] = _Outcome(
                    levels_completed=getattr(raw, "levels_completed", 0) or 0,
                    state=str(getattr(raw, "state", "")).split(".")[-1],
                    frame_changed=grid is not None,
                )
            except Exception:
                continue
        return out


# ─────────────────────────────────────────────────────────────────────────
# World model: per-state memory of what's been tried and what it did
# ─────────────────────────────────────────────────────────────────────────

ActionKey = str  # "3" for a simple action, "6:12,34" for a click at (12, 34)


def _state_key(objects: list[_Object], ignore_colors: frozenset[int] = frozenset(),
               bands: tuple[int, int, int, int] = (0, 0, 0, 0),
               shape: tuple[int, int] = (64, 64)) -> str:
    """The state the world model keys on: what objects are on screen and
    where, ignoring their exact pixel shape.

    Hashing raw pixels made states almost never repeat — 88% of 800 ls20
    states and 62% of vc33 states were unique — so `tries()` saw a fresh
    state nearly every step, every action looked untried, and the exploit
    path (`best_known`, `plan_to_milestone`) fired on 11 of 800 ls20 steps
    and 0 of 800 on vc33. The whole world model was inert. Keying on
    (color, centroid) per object drops per-frame pixel churn that carries no
    decision-relevant information while keeping exact positions, which a
    movement game does depend on.

    Readout strips are excluded for the same reason and a sharper one: a
    progress bar or a step counter changes every single action while saying
    nothing about the situation the agent is in, so including it guarantees a
    fresh state every step. See `_ui_bands`.
    """
    h, w = shape
    parts = sorted(
        (o.color, o.centroid)
        for o in objects
        if o.color not in ignore_colors
        and not _in_bands(o.centroid[0], o.centroid[1], bands, h, w)
    )
    return hashlib.sha1(str(parts).encode()).hexdigest()


def _action_key(action_id: int, xy: Optional[tuple[int, int]] = None) -> ActionKey:
    if xy is None:
        return str(action_id)
    x, y = xy
    return f"{action_id}:{x},{y}"


@dataclass
class _Transition:
    tries: int = 0
    frame_changed_ever: bool = False
    score_gains: int = 0


@dataclass
class _WorldModel:
    """Alive for the whole `Agent.main()` loop, so it survives RESETs
    between levels of one game — knowledge carries forward within a game."""

    transitions: dict[tuple[str, ActionKey], _Transition] = field(default_factory=dict)
    # state -> {action_key: next_state}, i.e. the edges discovered so far.
    # Kept separate from `transitions` so BFS can enumerate a state's
    # out-edges directly instead of scanning every recorded transition.
    adjacency: dict[str, dict[ActionKey, str]] = field(default_factory=dict)
    # Every state a score/level-increasing transition ever landed on.
    milestone_states: set[str] = field(default_factory=set)

    def tries(self, state: str, key: ActionKey) -> int:
        t = self.transitions.get((state, key))
        return t.tries if t else 0

    def record(
        self,
        state: str,
        key: ActionKey,
        next_state: str,
        frame_changed: bool,
        score_increased: bool,
    ) -> None:
        t = self.transitions.setdefault((state, key), _Transition())
        t.tries += 1
        t.frame_changed_ever = t.frame_changed_ever or frame_changed
        if score_increased:
            t.score_gains += 1
        self.adjacency.setdefault(state, {})[key] = next_state
        if score_increased:
            self.milestone_states.add(next_state)

    def mark_milestone(self, state: str) -> None:
        """Flag a state as worth planning toward even though no action
        recorded here has actually scored from it yet — used when goal
        inference recognizes a confirmed scoring predicate (e.g. two
        colors touching) independent of the exact pixels around it."""
        self.milestone_states.add(state)

    def best_known(self, state: str, keys: list[ActionKey]) -> Optional[ActionKey]:
        """Among actions already tried at this exact state, prefer whichever
        previously scored or at least changed something on screen."""
        ranked = []
        for key in keys:
            t = self.transitions.get((state, key))
            if t is not None:
                ranked.append(((t.score_gains, int(t.frame_changed_ever), -t.tries), key))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        return ranked[0][1]

    def plan_to_milestone(
        self, start: str, max_nodes: int = 2000
    ) -> Optional[list[ActionKey]]:
        """BFS over the discovered state graph for the shortest known path
        from `start` to any state a score/level increase once landed on."""
        if start in self.milestone_states:
            return []
        visited = {start}
        queue: list[tuple[str, list[ActionKey]]] = [(start, [])]
        head = 0
        expanded = 0
        while head < len(queue) and expanded < max_nodes:
            state, path = queue[head]
            head += 1
            expanded += 1
            for key, next_state in self.adjacency.get(state, {}).items():
                if next_state in visited:
                    continue
                new_path = path + [key]
                if next_state in self.milestone_states:
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
        return None


# ─────────────────────────────────────────────────────────────────────────
# Policy: combine perception + world model into one action choice
# ─────────────────────────────────────────────────────────────────────────

_SIMPLE_ACTION_IDS = (1, 2, 3, 4, 5, 7)  # ACTION6 (click) handled separately


def _candidate_keys(
    grid: Grid, objects: list[_Object], available: set[int],
    bands: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[list[ActionKey], dict[ActionKey, int]]:
    """Returns the candidate action keys plus, for click keys, the color of
    the object each one targets (goal inference uses this to weight which
    untried click is worth trying next)."""
    keys = [_action_key(aid) for aid in _SIMPLE_ACTION_IDS if aid in available]
    color_by_key: dict[ActionKey, int] = {}
    if 6 in available:
        for x, y, color in _click_candidates(grid, objects, bands=bands):
            key = _action_key(6, (x, y))
            keys.append(key)
            color_by_key[key] = color
    return keys, color_by_key


def _key_to_action(key: ActionKey) -> GameAction:
    if ":" in key:
        aid_str, xy_str = key.split(":", 1)
        x_str, y_str = xy_str.split(",")
        action = GameAction.from_id(int(aid_str))
        action.set_data({"x": int(x_str), "y": int(y_str)})
        action.reasoning = {"why": "click candidate", "x": int(x_str), "y": int(y_str)}
        return action
    action = GameAction.from_id(int(key))
    action.reasoning = {"why": "novel or best-known simple action"}
    return action


class MyAgent(Agent):
    """Explores untried (state, action) pairs first, preferring cheap simple
    actions and object-targeted clicks over blind coordinates. Once a state
    is fully probed, replays whatever previously changed the frame or
    scored there, or BFS-plans a path back to a known scoring state,
    instead of re-testing known no-ops."""

    # No per-game action cap — the budget is wall-clock, as in the official
    # sample. At the old cap of 80 the agent quit after about a second: mean
    # levels over 3 seeds went 0.67 → 1.33 → 2.00 on vc33 and 0.00 → 0.33 →
    # 0.67 on ft09 as the budget went 80 → 800 → 4000, so 80 was throwing most
    # of the score away. `scripts/play_local.py` still lowers this for local
    # runs, which is what keeps `make play-local` and preflight quick.
    MAX_ACTIONS = float("inf")

    # Eight hours less a five-minute margin, matching the official sample, so
    # the notebook finishes inside Kaggle's limit. `Swarm` gives each game its
    # own thread and each thread its own agent, so this budget covers every
    # game concurrently rather than being divided between them.
    TIME_BUDGET_SECONDS = 8 * 3600 - 5 * 60

    # `Agent.append_frame` keeps every frame forever, and a 64x64 frame costs
    # ~40 KB (measured: 39.5 KB per action). Unbounded, 100k actions across 25
    # concurrent games would want ~100 GB against Kaggle's ~30 GB — the run
    # would OOM and score nothing, so lifting the action cap without this
    # would be strictly worse than leaving it at 80. Nothing here reads frame
    # history: `is_done` and `choose_action` use only `latest_frame`, and the
    # base class only ever touches `frames[-1]`.
    RETAINED_FRAMES = 4

    def __init__(self, *args, **kwargs) -> None:
        self._dropped_frames = 0
        self._policy_errors = 0
        self._consecutive_failures = 0
        self._forced_resets = 0
        self._session_reopens = 0
        self._last_reopen = 0.0
        self._reached_server = True
        super().__init__(*args, **kwargs)
        self._deadline = time.monotonic() + self.TIME_BUDGET_SECONDS
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        self._rng = random.Random(seed)
        self._model = _WorldModel()
        self._tracker = _ObjectTracker()
        self._goal = _GoalModel()
        self._effect = _EffectModel()
        self._motion = _MotionModel()
        self._nav = _Navigator()
        self._contact = _ContactModel()
        self._recent_contacts: list[set[int]] = []
        self._sim = _Simulator(self.game_id) if self.USE_SIMULATOR else _Simulator.disabled()
        self._pending: Optional[tuple[str, ActionKey, Optional[int]]] = None
        self._pending_avatar: Optional[tuple[tuple[int, int], ActionKey]] = None
        self._avatar_color: Optional[int] = None
        self._last_avatar: Optional[tuple[int, int]] = None
        self._fatal_keys: set[ActionKey] = set()
        self._last_sim_key: Optional[ActionKey] = None
        self._scalar_moves: dict[str, tuple[int, int]] = {}
        self._last_scalars: dict[str, float] = {}
        self._danger_seen = False
        self._bands: Optional[tuple[int, int, int, int]] = None
        self._bands_at = -1
        self._score_before_pending = 0

    @property
    def name(self) -> str:
        return f"{super().name}.symbolic"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        if latest_frame.state is GameState.WIN:
            return True
        # With MAX_ACTIONS unbounded this is what ends the run. `monotonic` so
        # a clock adjustment mid-run can't cut the budget short or extend it.
        return time.monotonic() >= self._deadline

    # A dropped frame must never end the game.
    #
    # `Agent.main` reads `arc_env.observation_space` and calls `take_action`
    # every iteration, and both raise if the environment hands back None —
    # `_convert_raw_frame_data` turns it into a ValueError. `Swarm` runs each
    # game in a daemon thread, so that exception silently ends that game and
    # `join()` returns as if all was well; the game simply stops playing and
    # keeps whatever score it had.
    #
    # Running the real orchestration locally (25 games, threads, HTTP
    # gateway) killed 16 of 25 threads this way. The survivors took 445-709
    # actions in 90 seconds; the casualties managed 1-21 before dying. That
    # shape matches the leaderboard: v5's local score tripled while the
    # leaderboard moved 53%, which is what you would expect if most games
    # were dying in their first seconds rather than playing for hours.
    STEP_RETRIES = 4
    STEP_RETRY_DELAY = 0.25

    # Retrying only rescues a transient fault. Some failures are permanent
    # until the game is reset — the environment answers "Cannot step: game not
    # reset" and will go on answering it forever — and against those, carrying
    # the last frame forward converts a crash into a livelock: the agent keeps
    # choosing ordinary actions, every one fails, and the game runs for hours
    # without advancing a step. Seen in a 45-minute soak of the real
    # orchestration, and again in a 25-game run where a single game accounted
    # for 2,039 of 3,562 dropped frames and every one of the 666 resets while
    # the other 24 sat at 47-78 drops and none. After this many consecutive
    # failures, reset instead of asking again — and see `take_action` for why
    # "consecutive failure" has to be counted there and nowhere else.
    FAILURES_BEFORE_RESET = 3

    # `env.reset()` is a blocking POST with a 10-second timeout, and the
    # failure loop comes back around every other action, so an unreachable
    # gateway would otherwise be asked once every few seconds forever. Wait
    # this long between attempts.
    REOPEN_COOLDOWN_SECONDS = 30.0

    def _reopen_session(self) -> bool:
        """Re-open the game through the wrapper instead of through `step`.

        `GameAction.RESET` is delivered by `take_action` -> `do_action_request`
        -> `arc_env.step(...)`, so it is only as reachable as any other action.
        When the session itself is what broke, the escape hatch runs through
        the very door that is jammed:

            if self._guid is None: ... return None      # step() never sends

        `RemoteEnvironmentWrapper.__init__` calls `reset()`, and `reset()`
        returns None on any request failure, leaving `_guid` as None — so a
        game that loses its startup reset can never be recovered by RESET.
        Measured on a game forced into that state: 60 iterations produced 120
        dropped frames, 30 forced RESETs, and **zero** actions reaching the
        server. Calling `reset()` on the wrapper restored it outright — guid
        reassigned, state NOT_FINISHED, actions available, and the next 60
        iterations all reached the server.

        This is only reached from the failure escape, i.e. after several
        consecutive steps got nothing back, and the caller sends RESET
        immediately afterwards regardless — so there is no progress here left
        to lose that RESET would not have discarded anyway.
        """
        env = getattr(self, "arc_env", None)
        if env is None or not hasattr(env, "reset"):
            return False
        now = time.monotonic()
        if now - self._last_reopen < self.REOPEN_COOLDOWN_SECONDS:
            return False
        self._last_reopen = now
        try:
            if env.reset() is not None:
                self._session_reopens += 1
                return True
        except Exception:
            pass
        return False

    def _convert_raw_frame_data(self, raw):  # type: ignore[no-untyped-def]
        if raw is None:
            # Carry the last known frame forward instead of raising. Acting on
            # a stale board for one step is survivable; losing the game is not.
            self._dropped_frames += 1
            self._reached_server = False
            return self.frames[-1]
        self._reached_server = True
        return super()._convert_raw_frame_data(raw)

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        """Send one action, and judge whether it actually landed.

        `super().take_action` cannot answer that on its own. It routes through
        `_convert_raw_frame_data`, which above turns "no response" into the
        previous frame — so a step that never reached the server comes back
        looking exactly like a step that did. That is deliberate for keeping
        the game alive, and it is why the failure counter is kept here, on the
        outcome of the request, rather than on the shape of the return value.

        This distinction is the whole ballgame. An earlier version cleared the
        counter inside `_convert_raw_frame_data` on any non-None frame, and
        `Agent.main` reads `arc_env.observation_space` at the top of every
        iteration — a *stale* read, still non-None. So production cleared the
        counter once per turn no matter what, and the RESET escape below could
        never fire. Measured against a server refusing every step, same agent,
        same jam, 30 iterations: the local harness saw 9 forced resets while
        `Agent.main`'s loop saw **0**. The escape was dead code in the only
        place it mattered.
        """
        for attempt in range(self.STEP_RETRIES):
            self._reached_server = True
            try:
                frame = super().take_action(action)
            except Exception:
                self._dropped_frames += 1
                frame = None
            if frame is not None and self._reached_server:
                self._consecutive_failures = 0
                return frame
            if attempt + 1 < self.STEP_RETRIES:
                time.sleep(self.STEP_RETRY_DELAY * (attempt + 1))
        self._consecutive_failures += 1
        # `Agent.main` guards this with a walrus, so None just skips a turn.
        return None

    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        # Keep the tail only; see RETAINED_FRAMES for why this is required
        # rather than merely tidy.
        if len(self.frames) > self.RETAINED_FRAMES:
            del self.frames[: -self.RETAINED_FRAMES]

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Guard the policy so a bug cannot forfeit a game.

        Same trap as a dropped frame: `Swarm` runs each game in a daemon
        thread, so an exception raised anywhere below ends that game silently
        and `join()` returns as if it had finished. Twenty-five games, hours
        of play, and boards nothing here has ever been tested against is a lot
        of surface for one unhandled edge case, and the cost of finding it in
        a rerun is the whole game's score.

        Failures are counted rather than swallowed outright — `_policy_errors`
        rides along in the next action's `reasoning`, so a run that limped
        looks different from one that went well.
        """
        try:
            return self._choose_action(frames, latest_frame)
        except Exception:
            self._policy_errors += 1
            available = list(latest_frame.available_actions) or [1, 2, 3, 4, 5]
            action = GameAction.from_id(self._rng.choice(available))
            if action.action_data.model_dump().get("x", None) is not None:
                action.set_data({"x": self._rng.randint(0, 63),
                                 "y": self._rng.randint(0, 63)})
            action.reasoning = {"why": "policy error", "errors": self._policy_errors}
            return action

    def _choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if self._consecutive_failures >= self.FAILURES_BEFORE_RESET:
            # Nothing is getting through. Ordinary actions cannot recover a
            # game the environment considers un-started, so stop asking.
            self._consecutive_failures = 0
            self._forced_resets += 1
            self._pending = None
            # If there is no session at all, RESET cannot reach the server
            # either; re-open first. See `_reopen_session`.
            self._reopen_session()
            self._last_sim_key = _action_key(GameAction.RESET.value)
            return GameAction.RESET

        if latest_frame.state is GameState.GAME_OVER:
            # Proof this game can end the run, which is what turns vetting on.
            self._danger_seen = True
            # Blame whatever was beside the avatar on the way in. This is the
            # only moment the game says "that was fatal", and by the next
            # frame the board has been reset and the evidence is gone.
            self._credit_contact(died=True, scored=False)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._pending = None
            # The mirror has to see resets too. Games are deterministic, so a
            # mirror fed exactly the same action sequence stays identical —
            # but only if *every* action reaches it, and this branch used to
            # be the one that didn't, which desynced it on the very first
            # move and disabled simulation for the whole run.
            self._last_sim_key = _action_key(GameAction.RESET.value)
            return GameAction.RESET

        grid: Grid = latest_frame.frame[0] if latest_frame.frame else [[0]]
        # Replay what we actually did onto the mirror before consulting it,
        # and let it disable itself if it has drifted from reality.
        if self._last_sim_key is not None:
            self._sim.resync(self._last_sim_key, grid)
            self._last_sim_key = None
        self._observe_scalars()
        objects = _extract_objects(grid)
        # Where the screen stops being playfield. Cached rather than computed
        # per frame: it is a full pass over the board, and with the score
        # moving in step with actions per second — v6 1.00x/0.25, v8
        # 1.59x/0.11, v7 3.40x/0.07 — a per-step cost of a few percent would
        # eat the 17% of clicks this saves. Layout only changes with the
        # level, so recompute when that does.
        if self._bands is None or latest_frame.levels_completed != self._bands_at:
            self._bands = _ui_bands(grid)
            self._bands_at = latest_frame.levels_completed
        bands = self._bands
        shape = (len(grid), len(grid[0]) if grid else 0)
        # Counters and timers change every frame while meaning nothing; once
        # they've been identified, keeping them out of the key stops them
        # inflating the state space.
        state = _state_key(objects, frozenset(self._motion.noise_colors()),
                           bands, shape)
        events = self._tracker.update(objects)
        touching = _touching_pairs(objects)
        self._record_pending_outcome(state, latest_frame.levels_completed, events, touching)

        if touching & self._goal.confirmed_predicates():
            # This board doesn't have to be one we've scored from before —
            # a confirmed color-pair predicate generalizes the state graph's
            # exact-pixel milestones to "any state where this relation holds".
            self._model.mark_milestone(state)

        available = set(latest_frame.available_actions) or set(range(1, 8))
        candidates, color_by_key = _candidate_keys(grid, objects, available, bands)
        if not candidates:
            return GameAction.RESET

        # Look one move ahead where a mirror of the game is available: a
        # candidate that scores is worth taking immediately, and one that
        # ends the run is worth never taking. Both facts otherwise cost a
        # real action — or a life — to learn.
        looked = self._lookahead(candidates, color_by_key, latest_frame.levels_completed)
        if looked is not None:
            return self._commit(looked, state, color_by_key, events,
                                latest_frame.levels_completed, "lookahead",
                                self._avatar_position(objects))

        # Once the controls are understood, steer deliberately instead of
        # pressing keys at random: head for ground the avatar hasn't covered.
        avatar = self._avatar_position(objects)
        self._learn_walls(avatar)
        if avatar is not None:
            scheme = self._motion.scheme_for_object(self._motion.avatar_id() or -1)
            self._note_contact(
                _colors_near(avatar, objects, self._step_radius(scheme))
            )
        nav_key = self._navigation_key(avatar, available, objects)
        if nav_key is not None and nav_key not in self._fatal_keys:
            nav_key = self._safe_choice(nav_key, candidates)
            return self._commit(nav_key, state, color_by_key, events,
                                latest_frame.levels_completed, "navigate", avatar)

        # Simulation already showed these end the run; drop them unless that
        # leaves nothing to do.
        survivable = [k for k in candidates if k not in self._fatal_keys]
        if survivable:
            candidates = survivable

        # The label each branch reports. Four very different things used to
        # share the name "explore", which hid the breakdown that matters now
        # that the score charges quadratically for actions: how much of the
        # spend is genuine novelty-seeking, how much is replaying something
        # already known to work, how much is planned, and how much is a coin
        # flip with no information behind it. Measured across 25 games at
        # 1,000 actions: 85% untried, 13% navigate, **1% known, 0% plan** —
        # the world model has never once paid for itself.
        why = "untried"
        untried = [k for k in candidates if self._model.tries(state, k) == 0]
        if untried:
            untried_simple = [k for k in untried if ":" not in k]
            if untried_simple:
                weights = [self._effect.simple_weight(int(k)) for k in untried_simple]
                key = self._rng.choices(untried_simple, weights=weights, k=1)[0]
            else:
                # Two independent signals, multiplied: does clicking this
                # color do *anything* (learned across every state so far),
                # and has it ever coincided with scoring. Both start neutral,
                # so with no data this is still a uniform random pick.
                weights = [
                    self._effect.click_weight(color_by_key[k])
                    * self._goal.color_score(color_by_key[k])
                    for k in untried
                ]
                key = self._rng.choices(untried, weights=weights, k=1)[0]
        else:
            # `plan_to_milestone` is a BFS over the whole discovered graph, so
            # it stays behind the cheap answer rather than beside it. Naming
            # the branches cost that short-circuit once already: an `or` chain
            # was rewritten into an if/elif that computed both, which picks
            # the same action and pays for a search it then discards.
            known = self._model.best_known(state, candidates)
            if known is not None:
                key, why = known, "known"
            else:
                plan = self._model.plan_to_milestone(state)
                if plan:
                    key, why = plan[0], "plan"
                else:
                    key, why = self._rng.choice(candidates), "random"

        key = self._safe_choice(key, candidates)
        return self._commit(key, state, color_by_key, events,
                            latest_frame.levels_completed, why, avatar)

    # Vetting is gated on having seen this game kill us at least once, so
    # click games — which cannot — never pay for it, and how many
    # alternatives to look at before giving up and playing the first choice.
    VET_ALTERNATIVES = 3

    def _resources(self) -> set[str]:
        """Mirror fields that have only ever gone down: lives, timers, fuel.
        Learned rather than named, since the games' identifiers are
        deliberately obfuscated — on ls20 the life counter is `aqygnziho`."""
        return {
            name
            for name, (down, up) in self._scalar_moves.items()
            if down > 0 and up == 0
        }

    def _observe_scalars(self) -> None:
        current = self._sim.scalars()
        for name, value in current.items():
            previous = self._last_scalars.get(name)
            if previous is not None and value != previous:
                down, up = self._scalar_moves.get(name, (0, 0))
                if value < previous:
                    down += 1
                else:
                    up += 1
                self._scalar_moves[name] = (down, up)
        self._last_scalars = current

    def _vet(self, key: ActionKey) -> bool:
        """True if this move looks safe to play. Costs one deepcopy, and is
        only reached on a game already known to be able to end the run."""
        result = self._sim.vet(key)
        if result is None:
            return True
        outcome, after = result
        if outcome.fatal:
            return False
        resources = self._resources()
        return not any(
            after.get(name, self._last_scalars.get(name, 0.0))
            < self._last_scalars.get(name, 0.0)
            for name in resources
        )

    def _safe_choice(self, key: ActionKey, candidates: list[ActionKey]) -> ActionKey:
        """Play `key` unless simulation says it kills us or spends a life, in
        which case try a few alternatives. Falls back to the original rather
        than freezing — being stuck is worse than being hurt."""
        if not self._sim.enabled or not self._danger_seen:
            return key
        if self._vet(key):
            return key
        others = [k for k in candidates if k != key]
        self._rng.shuffle(others)
        for alternative in others[: self.VET_ALTERNATIVES]:
            if self._vet(alternative):
                return alternative
        return key

    # Per-step lookahead is OFF, rejected three times, and the third rejection
    # is the only one whose reasoning still stands. Read it before switching
    # this on a fourth time.
    #
    # The machinery works. The games are deterministic, so a private copy fed
    # the same actions stays frame-identical, and it correctly reports which
    # candidates score and which end the run. It costs a ~12 ms deepcopy per
    # candidate, which is several times fewer actions at equal wall-clock.
    #
    # Rejections one and two scored levels completed, so fewer actions read
    # as fewer levels and the verdict was "loses" (lp85, 45 s per game:
    # 5521 actions / 3 levels off against 374 / 1 on). That reasoning was
    # wrong — the competition scores min((baseline/actions)**2 * 100, 115) per
    # completed level, so three slow levels really are worth less than one
    # fast one. Re-measured against the real formula it looked like a large
    # win: arc_score 0.213 -> 0.619 at 300 s, 208 actions per level -> 40.
    #
    # It was switched on, submitted, and the leaderboard went 0.25 -> 0.07.
    #
    # The formula was right and the inference from it was not. arc_score is a
    # **mean over games**, and a game that never finishes a level contributes
    # zero however efficient it was. Eighteen of twenty-five games are at
    # zero, so the marginal action is not buying efficiency on a game already
    # solved — it is buying the chance to crack a game at all, and cutting
    # throughput 3.4x spends exactly that. The sign was already visible at
    # 300 s and was read past: off reached 15 levels across 7 games, on
    # reached 11 across 6. Off was winning on breadth while losing on the
    # headline number.
    #
    # ON AGAIN, on the fourth attempt, with the two things that were wrong
    # about the third one addressed rather than argued away.
    #
    # **The cost.** `_clone` shares the level data a simulated step never
    # writes to, which is 92-98% of a copy, verified frame-identical against
    # `deepcopy` over 1,080 stepped comparisons on each of five games. And the
    # shortlist is 6 rather than 12, which measurement says is free: quality
    # is identical either way. Together the slowdown goes 3.4x -> 1.59x.
    #
    # **The measurement.** Judged on `--games scoring`, the seven games that
    # have ever finished a level, because a mean over 25 dilutes any change
    # three or four times inside eighteen immovable zeros. At equal actions,
    # 2 seeds x 3,000:
    #
    #     width    arc_score           throughput   slowdown
    #     off      0.784               72.8/s       —
    #     3        1.050  (+34%)       55.6/s       1.31x
    #     6        2.000  (+155%)      45.8/s       1.59x
    #     12       1.996  (+155%)      38.4/s       1.90x
    #
    # Width 3 collapses sp80 and ar25 back to baseline; 6 and 12 are the same
    # answer, so 6 it is. Per game at width 6: r11l 3.358 -> 3.412 (the game
    # that is 61% of the score, untouched), vc33 0.765 -> 5.082, sp80 0.027 ->
    # 3.006. vc33's fourth level goes 2,599 actions -> 450.
    #
    # So this ships v7's exact quality gain at 2.1x less of the cost that
    # sank it. Whether 1.59x is under the line is the one thing no local run
    # can answer — v7 established only that 3.4x is over it. If this loses
    # too, the line is below 1.59x and the next move is gating simulation to
    # the situations where it pays rather than trimming the shortlist further.
    #
    # OFF AGAIN. It lost: v8 scored 0.11 against v6's 0.25. Three points now,
    # and they lie on a line:
    #
    #     version                slowdown     LB    LB x slowdown
    #     v6  no simulation          1.00   0.25             0.25
    #     v8  width 6, fast clone    1.59   0.11             0.17
    #     v7  width 12, deepcopy     3.40   0.07             0.24
    #
    # The leaderboard tracks actions available, not decision quality. v8 made
    # measurably better decisions than v6 — 2.5x better at equal actions on the
    # games that carry the score — and lost more than half the score anyway.
    # So the breakeven is below 1.59x, which is below the cost of any useful
    # shortlist, and gating simulation more narrowly is not the next move: the
    # line is closed.
    #
    # The code stays rather than being deleted. Four attempts is enough to stop
    # trying, not enough to be sure it can never pay — and the metric is
    # efficiency-squared, so an agent that one day finishes levels in tens of
    # actions rather than thousands would be in a different regime entirely,
    # where a slower, better-aimed action is worth what it costs. Flipping
    # these two flags is the whole switch.
    #
    # The safety nets are why leaving it in costs nothing: `_clone` falls back
    # to a plain `deepcopy` if the level attributes are not what it expects,
    # and `resync` disables simulation outright the moment the mirror disagrees
    # with the real board.
    #
    # `_resources()` is still empty on every game: ls20's life counter drops
    # 3 -> 1 under random play but refills on a level restart, so "only ever
    # decreases" never holds. Vetting earns its keep on fatal moves alone.
    USE_SIMULATOR = False
    USE_LOOKAHEAD = False
    LOOKAHEAD_WIDTH = 6

    def _lookahead(
        self,
        candidates: list[ActionKey],
        color_by_key: dict[ActionKey, int],
        levels_now: int,
    ) -> Optional[ActionKey]:
        """Simulate a shortlist and return an action known to score, or None.
        Also records which candidates would end the run so the other branches
        can steer clear of them."""
        self._fatal_keys = set()
        if not self._sim.enabled or not self.USE_LOOKAHEAD:
            return None

        simple = [k for k in candidates if ":" not in k]
        clicks = [k for k in candidates if ":" in k]
        # Prefer clicks on colours already associated with doing something;
        # the shortlist is small, so spending it well matters.
        clicks.sort(
            key=lambda k: self._effect.click_weight(color_by_key.get(k, -1)),
            reverse=True,
        )
        shortlist = (simple + clicks)[: self.LOOKAHEAD_WIDTH]

        outcomes = self._sim.evaluate(shortlist)
        if not outcomes:
            return None

        self._fatal_keys = {k for k, o in outcomes.items() if o.fatal}
        scoring = [
            k for k, o in outcomes.items()
            if o.levels_completed > levels_now and not o.fatal
        ]
        return min(scoring) if scoring else None

    def _learn_walls(self, avatar_now: Optional[tuple[int, int]]) -> None:
        """If the last action was a move with a known offset and the avatar
        did not end up where that offset predicted, something is in the way.
        This is the only source of wall knowledge — the grid alone doesn't say
        which colours are solid."""
        previous = self._pending_avatar
        self._pending_avatar = None
        if previous is None or avatar_now is None:
            return
        before, key = previous
        if ":" in key:
            return
        avatar_id = self._motion.avatar_id()
        if avatar_id is None:
            return
        delta = self._motion.scheme_for_object(avatar_id).get(int(key))
        if delta is None:
            return
        action_id = int(key)
        if avatar_now != (before[0] + delta[0], before[1] + delta[1]):
            self._nav.note_blocked(before, action_id)

    def _avatar_position(self, objects: list[_Object]) -> Optional[tuple[int, int]]:
        """Where the steered object is now, or None while the avatar is still
        unidentified.

        Read by persistent tracker id so it means the same object each frame.
        The tracker does lose an id when the avatar changes shape and moves at
        once (it reads as disappeared + appeared), so when that happens fall
        back to the nearest object of the avatar's colour — the avatar cannot
        have travelled far in one action, and losing the thread entirely costs
        the whole navigation path."""
        avatar_id = self._motion.avatar_id()
        if avatar_id is None:
            return None
        position = self._tracker.position_of(avatar_id)
        if position is not None:
            self._avatar_color = self._tracker.color_of(avatar_id)
            self._last_avatar = position
            return position

        if self._avatar_color is None or self._last_avatar is None:
            return None
        same_color = [o for o in objects if o.color == self._avatar_color]
        if not same_color:
            return None
        nearest = min(
            same_color,
            key=lambda o: abs(o.centroid[0] - self._last_avatar[0])
            + abs(o.centroid[1] - self._last_avatar[1]),
        )
        self._last_avatar = nearest.centroid
        return nearest.centroid

    # How many recent steps' worth of surroundings to credit or blame. A
    # hazard is usually beside the avatar on the step it kills, but a level
    # completion can follow a move or two after touching the thing that
    # caused it, so a short window beats crediting only the last frame.
    CONTACT_WINDOW = 3

    def _note_contact(self, colors: set[int]) -> None:
        """Record this step's surroundings, and retire the step that just
        fell out of the window as an ordinary, uneventful one.

        Every step has to be counted, not only the fatal ones. Crediting
        solely on death made every observation a death, so the board-wide
        rate was 100% and no colour could ever stand out against it — the
        lift test silently answered 1.0 for everything."""
        self._recent_contacts.append(colors)
        while len(self._recent_contacts) > self.CONTACT_WINDOW:
            self._contact.observe(
                self._recent_contacts.pop(0), died=False, scored=False
            )

    def _credit_contact(self, died: bool, scored: bool) -> None:
        """Blame or credit the steps leading up to an outcome, then start a
        fresh window — the board is about to change out from under us."""
        if not (died or scored):
            return
        for colors in self._recent_contacts:
            self._contact.observe(colors, died=died, scored=scored)
        self._recent_contacts.clear()

    def _step_radius(self, scheme: dict[int, tuple[int, int]]) -> int:
        """How far one move carries the avatar — the natural scale for "near"."""
        if not scheme:
            return 4
        return max(max(abs(dr), abs(dc)) for dr, dc in scheme.values()) or 4

    def _cells_beside(
        self, objects: list[_Object], colors: set[int], radius: int
    ) -> set[tuple[int, int]]:
        """Board cells within one move of an object of these colours."""
        out: set[tuple[int, int]] = set()
        for obj in objects:
            if obj.color not in colors:
                continue
            for r, c in obj.cells:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 64 and 0 <= nc < 64:
                            out.add((nr, nc))
        return out

    def _navigation_key(
        self,
        position: Optional[tuple[int, int]],
        available: set[int],
        objects: list[_Object],
    ) -> Optional[ActionKey]:
        if position is None:
            return None
        avatar_id = self._motion.avatar_id()
        if avatar_id is None:
            return None
        self._nav.note_position(position)
        scheme = {
            action_id: delta
            for action_id, delta in self._motion.scheme_for_object(avatar_id).items()
            if action_id in available
        }
        radius = self._step_radius(scheme)
        danger = self._contact.danger_colors()
        goals = self._contact.goal_colors()
        avoid = self._cells_beside(objects, danger, radius) if danger else set()
        # Never rule out the square we're standing on — being beside a hazard
        # already leaves nowhere to plan from.
        avoid.discard(position)
        targets = self._cells_beside(objects, goals, radius) if goals else None

        action_id = self._nav.next_action(position, scheme, targets, avoid)
        if action_id is None and targets:
            # Goal unreachable (or already reached); go back to covering ground.
            action_id = self._nav.next_action(position, scheme, None, avoid)
        return None if action_id is None else _action_key(action_id)

    def _commit(
        self,
        key: ActionKey,
        state: str,
        color_by_key: dict[ActionKey, int],
        events: list[_ObjectEvent],
        levels_completed: int,
        why: str,
        avatar: Optional[tuple[int, int]],
    ) -> GameAction:
        self._pending = (state, key, color_by_key.get(key))
        self._pending_avatar = None if avatar is None else (avatar, key)
        self._last_sim_key = key
        self._score_before_pending = levels_completed
        action = _key_to_action(key)
        if isinstance(action.reasoning, dict):
            action.reasoning["objects"] = _summarize_events(events)
            action.reasoning["mode"] = why
            if self._dropped_frames or self._policy_errors or self._forced_resets:
                action.reasoning["dropped"] = self._dropped_frames
                action.reasoning["errors"] = self._policy_errors
                action.reasoning["forced_resets"] = self._forced_resets
                action.reasoning["session_reopens"] = self._session_reopens
        return action

    def _record_pending_outcome(
        self,
        state_after: str,
        levels_completed_after: int,
        events: list[_ObjectEvent],
        touching: set[tuple[int, int]],
    ) -> None:
        if self._pending is None:
            return
        state_before, key, color = self._pending
        score_increased = levels_completed_after > self._score_before_pending
        frame_changed = state_after != state_before
        self._model.record(
            state=state_before,
            key=key,
            next_state=state_after,
            frame_changed=frame_changed,
            score_increased=score_increased,
        )
        self._goal.observe(events, touching, score_increased)
        if score_increased:
            self._credit_contact(died=False, scored=True)
        action_id = int(key.split(":", 1)[0])
        self._effect.observe(
            action_id=action_id, color=color, frame_changed=frame_changed
        )
        # Only simple actions have a fixed meaning worth learning; a click's
        # effect depends on where it landed, not on the action id.
        if ":" not in key:
            self._motion.observe(action_id, events)
        self._pending = None
