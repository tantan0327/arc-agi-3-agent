"""Is there an avatar, and is it identifiable from one successful move?

Promoted from scratchpad/n5_agency.py (2026-08-31); measurement logic
unchanged from the 2026-08-22 run the paper reports. Usage:

    python scripts/measure_agency.py <intermediate_states.pkl>

The largest behavioural gap left is movement: the agent wastes 16.4% of its
non-click actions against a human's 0.3%. It issues legal moves that change
nothing -- it walks into walls. That is a failure to predict geometry.

Spelke's core knowledge lists agency among the priors ARC is built on, and it is
the only one we have never tested. The concrete question underneath it: when a
direction key works, does a single small object translate by one cell? If so the
avatar is recoverable from **one** successful move, "which blob is me" is cheap
knowledge, and an agent that still walks into walls is failing to hold something
it could have had for free.

Reads the recorded states: for every direction action that changed the board,
diff before and after and ask whether the change is one object moving.
"""
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from load_recorded_states import load  # noqa: E402

DIRS = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}


def aname(st):
    a = getattr(st, "previous_action", None)
    if a is None:
        return None
    ar = getattr(getattr(a, "id", None), "_args", None)
    return ar[1] if ar and len(ar) > 1 else None


def board(st):
    return np.array(st.raw.frame)[-1]


def translation(before, after):
    """If the diff is one blob translating, return (colour, dy, dx, size).

    Vacated cells take the colour that filled them; arrived cells take the
    mover's colour. So the moved object is the set of changed cells that hold
    the same colour in `after`, and the vacated set is its mirror in `before`.
    """
    diff = before != after
    if not diff.any():
        return None
    ys, xs = np.nonzero(diff)
    if len(ys) > 64:                       # a whole-board repaint is not a move
        return None
    after_vals = after[ys, xs]
    before_vals = before[ys, xs]
    arrived_colour = Counter(after_vals.tolist()).most_common(1)[0][0]
    arrived = {(y, x) for y, x, v in zip(ys, xs, after_vals) if v == arrived_colour}
    vacated = {(y, x) for y, x, v in zip(ys, xs, before_vals) if v == arrived_colour}
    if not arrived or len(arrived) != len(vacated):
        return None
    # a pure translation: every arrived cell is a vacated cell plus one offset
    offs = {(a[0] - b[0], a[1] - b[1])
            for a, b in zip(sorted(arrived), sorted(vacated))}
    if len(offs) != 1:
        return None
    dy, dx = offs.pop()
    if (dy, dx) == (0, 0):
        return None
    return int(arrived_colour), dy, dx, len(arrived)


def main(path):
    games = load(path)
    print(f"{'game':<6}{'dir moves':>10}{'read as a translation':>23}"
          f"{'distinct movers':>17}  most common mover")
    ok_games = 0
    for states in games:
        if len(states) < 5:
            continue
        gid = getattr(states[0].raw, "game_id", "?")[:4]
        movers = Counter()
        n_dir = n_trans = 0
        for i in range(1, len(states)):
            if aname(states[i]) not in DIRS:
                continue
            b, a = board(states[i - 1]), board(states[i])
            if np.array_equal(b, a):
                continue
            n_dir += 1
            t = translation(b, a)
            if t:
                n_trans += 1
                movers[(t[0], t[3])] += 1
        if not n_dir:
            continue
        top = movers.most_common(1)[0] if movers else None
        share = f"{top[1]}/{n_trans}" if top else "-"
        desc = (f"colour {top[0][0]}, {top[0][1]} cells ({share})"
                if top else "-")
        if top and n_trans and top[1] / n_trans >= 0.8:
            ok_games += 1
        print(f"{gid:<6}{n_dir:>10}{f'{n_trans} ({n_trans/n_dir:.0%})':>23}"
              f"{len(movers):>17}  {desc}")
    print(f"\n  games where one object accounts for >=80% of the moves: {ok_games}")
    print("  If that is most of them, the avatar is recoverable from a single")
    print("  successful direction press, and 'which blob is me' is free knowledge.")


if __name__ == "__main__":
    main(sys.argv[1])
