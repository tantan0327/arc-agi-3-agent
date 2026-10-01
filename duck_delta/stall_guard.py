"""Once a level has clearly stalled, stop letting the agent re-enter its own cycle.

**The diagnosis, measured rather than argued.** Against a random policy on the
same 25 official games, at twice each game's own summed `baseline_actions`:

    clears at least one level    our agent 11/25 (44%)   random 2/25 (8%)
    wins outright                our agent  0/25         random 0/25

The agent is 5.5x chance at *getting started* and exactly equal to chance at
*finishing*. So the failure is not that it cannot act sensibly — it plainly can,
by a wide margin — it is that progress does not survive past the opening.

**Where the stall goes.** Per level, on the same runs:

    levels it completed        median  20 actions, p90 127, max 150
    the level it stalled on    median 122 actions, p75 217, max 397

A level that gets completed has never taken more than 150 actions. A level that
stalls burns up to 397. That gap is the whole opportunity, and it also fixes the
threshold: **150 actions without a level advance is the point past which no
observed completion has ever happened**, so arming there cannot cost a level the
agent would otherwise have finished.

**What it does when armed.** It blocks any `(level, board, action)` this level has
already tried. Not only proven no-ops — that is [`noop_guard`], and waste is not
the gap — but any exact repeat. On a board whose settled playfield is deterministic
to 93.4% agreement once the chrome is masked, taking the same action from the same
board yields the same board, so a repeat is provably re-entry into a cycle rather
than exploration.

**Why block rather than tell.** Every modification this project shipped that
*informed* the model lost, the click-effect tally most directly: it cut wasted
clicks from 86% to 15% and scored 0.69. Informing does not work here. Constraining
does.

**Why only when stalled.** Repeating a `(board, action)` is legitimate in normal
play — walking back down a corridor is exactly that. It is pathological only once
nothing has advanced for longer than any completion has ever taken. Below the
threshold this is inert, so ordinary play is untouched.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

# Self-contained on purpose, signature helpers included: the v2 fork's base
# bundle (the anim TAAF) ships its own `noop_guard` with a different API
# (`normalize_action_signature`), so importing signatures from the host bundle
# would tie this module to whichever bundle it happens to land in.
DEFAULT_STALL_ACTIONS = 150


def board_signature(grid) -> str:
    """A short, stable digest of a 2D integer grid."""
    rows = tuple(tuple(int(c) for c in row) for row in (grid or ()))
    return hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8).hexdigest()


def action_signature(name, data) -> str:
    """Identify an action including its click coordinates, when it has them."""
    d = data if isinstance(data, dict) else {}
    x, y = d.get("x"), d.get("y")
    if x is None:
        return str(name)
    return f"{name}@{int(x)},{int(y)}"


class StallGuard:
    """Repeat suppression that arms only after a level has visibly stalled.

    Bounded like `NoopGuard`: oldest boards evicted first, each board keeping only
    its most recent actions, so a 7,920-second game cannot grow it without limit.
    """

    def __init__(self, stall_actions: int = DEFAULT_STALL_ACTIONS,
                 max_boards: int = 2048, max_actions_per_board: int = 32) -> None:
        self._stall_actions = max(1, int(stall_actions))
        self._max_boards = max(1, int(max_boards))
        self._max_actions = max(1, int(max_actions_per_board))
        self._level = 0
        self._since_advance = 0
        self._tried: "OrderedDict[str, OrderedDict[str, None]]" = OrderedDict()
        self.blocked = 0
        self.armed_levels = 0
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._since_advance >= self._stall_actions

    def _bucket(self, board: str) -> "OrderedDict[str, None]":
        if board in self._tried:
            self._tried.move_to_end(board)
        else:
            self._tried[board] = OrderedDict()
            while len(self._tried) > self._max_boards:
                self._tried.popitem(last=False)
        return self._tried[board]

    def observe(self, *, level: int, board: str, action: str) -> None:
        """Record an executed action and track level progress.

        A level advance clears everything: the new level is a new problem and its
        boards are new boards, so carrying the old set over would block moves that
        have never been tried here.
        """
        level = int(level)
        if level != self._level:
            self._level = level
            self._since_advance = 0
            self._tried.clear()
            self._armed = False
            return
        self._since_advance += 1
        if self.armed and not self._armed:
            self._armed = True
            self.armed_levels += 1
        actions = self._bucket(board)
        actions[action] = None
        actions.move_to_end(action)
        while len(actions) > self._max_actions:
            actions.popitem(last=False)

    def is_cycle(self, *, level: int, board: str, action: str) -> bool:
        """True only while stalled, and only for an exact repeat on this level."""
        if int(level) != self._level or not self.armed:
            return False
        actions = self._tried.get(board)
        return bool(actions and action in actions)

    def note_blocked(self) -> None:
        self.blocked += 1
