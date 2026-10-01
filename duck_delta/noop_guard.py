"""Refuse an action already proven to do nothing from this exact board.

Measured, on the agent's own recorded states: **13.9% of its actions leave the
board unchanged**, against 0.6% for a human over the same measure. Of those,
94 of 382 repeat a `(board, action)` pair that had already been shown to do
nothing in that same game.

Two facts decide the shape of this. First, the harness already tells the model
which actions are legal — `_engine_action_names` reads `available_actions` and
`step_env` rejects anything outside it — and the agent already complies, so
there is nothing to enforce there; the dead actions are *legal* ones. Second,
every modification this project has shipped that **told** the model something
lost, the click-effect tally most directly: it cut wasted clicks 86% -> 15% and
scored 0.69. Informing does not work. Constraining does.

So this blocks rather than advises, and it blocks in `step_env` **before**
`_execute_action` runs — which is the whole point, because an action refused
there never reaches `game.execute_action` and so never enters
`actions_per_level`. Since a level scores `min(115, (baseline/actions)^2 * 100)`,
an action that does nothing is pure denominator, and removing it is worth more
than it costs.

Scope is deliberately narrow: same level, same board, same action. A board that
has changed is a different situation and nothing is suppressed there, so this
cannot stop the agent exploring — it can only stop it repeating a move it has
already watched fail from exactly where it is standing.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


def board_signature(grid: Any) -> str:
    """A short, stable digest of a 2D integer grid."""
    rows = tuple(tuple(int(c) for c in row) for row in (grid or ()))
    return hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8).hexdigest()


def action_signature(name: Any, data: Any) -> str:
    """Identify an action including its coordinates.

    A click is only the same action if it lands on the same cell, so the x/y go
    into the key. Without them every click would collapse to one entry and the
    guard would block the whole modality after a single dead click.
    """
    d = data if isinstance(data, dict) else {}
    x, y = d.get("x"), d.get("y")
    return f"{name}" if x is None else f"{name}@{int(x)},{int(y)}"


class NoopGuard:
    """`(level, board, action)` triples known to change nothing.

    Bounded so a long run cannot grow it without limit: oldest boards are evicted
    first, and each board keeps only its most recent dead actions.
    """

    def __init__(self, max_boards_per_level: int = 512,
                 max_actions_per_board: int = 24) -> None:
        self._max_boards = max(1, int(max_boards_per_level))
        self._max_actions = max(1, int(max_actions_per_board))
        self._levels: dict[int, "OrderedDict[str, OrderedDict[str, None]]"] = {}
        self.blocked = 0
        self.recorded = 0

    def _bucket(self, level: int, board: str) -> "OrderedDict[str, None]":
        boards = self._levels.setdefault(int(level), OrderedDict())
        if board in boards:
            boards.move_to_end(board)
        else:
            boards[board] = OrderedDict()
            while len(boards) > self._max_boards:
                boards.popitem(last=False)
        return boards[board]

    def observe(self, *, level: int, board: str, action: str,
                board_changed: bool) -> None:
        """Record the outcome of an executed action.

        An action that *did* change the board is forgotten rather than merely
        left unrecorded: a game whose state depends on something outside the
        grid could make the same move live again, and a stale entry would block
        it forever.
        """
        actions = self._bucket(level, board)
        if board_changed:
            actions.pop(action, None)
            return
        if action not in actions:
            self.recorded += 1
        actions[action] = None
        actions.move_to_end(action)
        while len(actions) > self._max_actions:
            actions.popitem(last=False)

    def is_dead(self, *, level: int, board: str, action: str) -> bool:
        boards = self._levels.get(int(level))
        if not boards:
            return False
        actions = boards.get(board)
        return bool(actions and action in actions)

    def note_blocked(self) -> None:
        self.blocked += 1
