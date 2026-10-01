"""The frontier map: a mechanical state graph so the model deliberates only at the edge.

O2 measured that persistent memory and factual recall do not move progress; the
bottleneck is within-level deliberation. This module spends none of the model's
deliberation: the harness tracks where the game has been, which actions were
tried from each state, and where the nearest untried thing is — the part of
exploration that is graph traversal, not thinking. Rudakov et al. placed 3rd on
the preview board with state-graph exploration alone (training-free), and our
determinism work supplies the masking that makes state identity sound (raw
frames disagree on 57% of repeats; isolation-masked boards on 10%, on 1%
conditioned on animation).

Design contract (kept deliberately small):

- States are digests of the *masked* settled board. The chrome mask is learned
  online with the isolation rule validated offline: a cell is chrome once it has
  changed in a transition where at most ISOLATED cells changed in total.
- The graph records, per state, the exact actions tried and their outcomes
  (no-op / edge to another state). Levels partition the graph: a level change
  resets it, because boards of different levels share nothing.
- `map_line()` renders one navigational sentence for the prompt.
- `path_to_frontier()` returns a list of action signatures through known edges
  to the nearest state with untried available actions — BFS over the graph the
  game itself built. The caller executes it with per-step verification and
  aborts on divergence; this module never touches the engine.
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Iterable


class FrontierMap:
    """Per-game state graph over masked boards. Everything is O(edges)."""

    def __init__(self, isolated: int = 2, max_states: int = 4096) -> None:
        self._isolated = int(isolated)
        self._max_states = int(max_states)
        self._chrome: set[tuple[int, int]] = set()
        self._level: Any = None
        self._prev_grid: Any = None          # unmasked, for chrome learning
        self._edges: dict[str, dict[str, str | None]] = {}   # state -> action -> next (None = no-op)
        self._avail: dict[str, tuple[str, ...]] = {}         # state -> available action sigs
        self._visits: dict[str, int] = {}
        self.goto_calls = 0
        self.goto_aborts = 0

    # -- state identity ----------------------------------------------------

    def _digest(self, grid: Any) -> str:
        rows = []
        for r, row in enumerate(grid):
            cells = tuple(0 if (r, c) in self._chrome else int(v)
                          for c, v in enumerate(row))
            rows.append(cells)
        return hashlib.blake2b(repr(tuple(rows)).encode(), digest_size=8).hexdigest()

    def _learn_chrome(self, before: Any, after: Any) -> None:
        diff = [(r, c) for r, row in enumerate(before) for c, v in enumerate(row)
                if int(after[r][c]) != int(v)]
        if 0 < len(diff) <= self._isolated:
            self._chrome.update(diff)

    # -- observation -------------------------------------------------------

    def observe(self, *, level: Any, before: Any, after: Any, action_sig: str,
                available_sigs: Iterable[str]) -> None:
        """Record one executed action. `before`/`after` are unmasked grids."""
        if level != self._level:
            # a new level is a new game for graph purposes
            self._level = level
            self._edges.clear()
            self._avail.clear()
            self._visits.clear()
            self._prev_grid = None
        self._learn_chrome(before, after)
        if len(self._edges) >= self._max_states:
            return
        s = self._digest(before)
        t = self._digest(after)
        self._avail.setdefault(s, tuple(str(a) for a in available_sigs))
        self._visits[s] = self._visits.get(s, 0) + 1
        self._edges.setdefault(s, {})[str(action_sig)] = (None if t == s else t)
        self._prev_grid = after

    # -- queries -----------------------------------------------------------

    @staticmethod
    def _base(sig: str) -> str:
        """ACTION6@12,8 -> ACTION6. Availability is per action, not per target."""
        return sig.split("@", 1)[0]

    def _untried(self, state: str) -> list[str]:
        """Available actions whose base was never tried from this state.

        Clicks are deliberately conservative: one tried click marks ACTION6
        "tried" for frontier purposes, because its target space is unbounded
        and a map that always says "clicks untried" would never terminate.
        Click exploration stays the model's job; the map's job is the finite
        action set and the graph.
        """
        tried_bases = {self._base(a) for a in self._edges.get(state, {})}
        return [a for a in self._avail.get(state, ())
                if self._base(a) not in tried_bases]

    def current_state(self, grid: Any) -> str:
        return self._digest(grid)

    def map_line(self, grid: Any, available_sigs: Iterable[str] = ()) -> str:
        """One navigational sentence about the current masked board.

        `available_sigs` covers the common case where the current board has
        never been acted *from* yet, so the graph holds no availability for it.
        """
        s = self._digest(grid)
        if s not in self._avail and available_sigs:
            self._avail[s] = tuple(str(a) for a in available_sigs)
        tried = self._edges.get(s, {})
        untried = self._untried(s)
        noops = [a for a, t in tried.items() if t is None]
        parts = [f"this board seen {self._visits.get(s, 0)}x",
                 f"{len(tried)} actions tried from it"]
        if noops:
            parts.append(f"no-ops here: {', '.join(sorted(noops)[:4])}")
        if untried:
            parts.append(f"untried here: {', '.join(sorted(untried)[:6])}")
        else:
            path = self.path_to_frontier(grid)
            if path:
                parts.append(f"nearest board with untried actions is {len(path)} "
                             f"step(s) away via {', '.join(path[:6])}")
            else:
                parts.append("no reachable board has untried actions on this level")
        return "- Frontier map (mechanical): " + "; ".join(parts)

    def path_to_frontier(self, grid: Any) -> list[str] | None:
        """BFS through known edges to the nearest state with untried actions."""
        start = self._digest(grid)
        if self._untried(start):
            return []
        seen = {start}
        q: deque[tuple[str, list[str]]] = deque([(start, [])])
        while q:
            state, path = q.popleft()
            for action, nxt in self._edges.get(state, {}).items():
                if nxt is None or nxt in seen:
                    continue
                npath = path + [action]
                if self._untried(nxt):
                    return npath
                seen.add(nxt)
                q.append((nxt, npath))
        return None

    def stats(self) -> str:
        return (f"states={len(self._avail)} edges="
                f"{sum(len(v) for v in self._edges.values())} "
                f"chrome={len(self._chrome)} goto={self.goto_calls}"
                f"/{self.goto_aborts} aborted")
