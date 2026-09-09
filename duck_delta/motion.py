"""What each directional action actually does, learned from the moves already made.

The harness tells its model that ACTION1 through ACTION5 exist and that "the
exact in-game effect depends on the title". Working that out is left to the
model, from prose descriptions of frames, on every turn. On the games where it
matters most this does not happen: across five identical offline runs, wa30 and
tr87 -- both movement games -- scored zero every single time, and ls20 scored in
three of five.

Our own agent settled this by tallying, and the answer on ls20 was exactly
`{1: (-5,0), 2: (5,0), 3: (0,-5), 4: (0,5)}` -- up, down, left, right, five
cells at a time. That is a fact about the game, obtainable in a handful of moves,
and it is the kind of thing that should be measured once rather than re-inferred
from a rendered grid every turn.

Two things fall out of the same table, and they are opposites:

  the avatar   moves *differently* depending on which action was pressed
  a counter    moves *the same way* whatever was pressed -- ls20's step counter
               advances by (0,1) on literally every action, changing the board
               every frame while carrying no information about the situation

Both are worth saying out loud. The first is the control scheme. The second is
the thing the harness's own prompt spends a paragraph warning about.

Why this is a different kind of addition from the last two tried here: it is a
measurement, not a theory. "ACTION1 moved this object by (-5,0), eleven times
out of eleven" is either true of the run so far or it is not. An earlier delta
suggested what a board might *mean* and the model chased that reading for 1,225
actions on a game the baseline finished in 91.

Stdlib only and self-contained, so the source splices into the Python-tool
sandbox the way `segmentation.py` does.
"""


# An action has to have been seen this many times before its offset is reported.
# Below it a single coincidence -- an object that happened to drift once -- would
# be presented as the control scheme.
MIN_OBSERVATIONS = 3
# And that offset has to be the majority answer. Walls make the blocked case
# common on movement games, so demanding unanimity would report nothing.
MIN_AGREEMENT = 0.6


def _components(grid):
    """4-connected same-value blobs: {(colour, normalised shape): [anchors]}.

    Keyed by shape rather than position so the same object can be found in the
    next frame wherever it has moved to. The anchor is the blob's top-most,
    left-most cell.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    seen = [[False] * width for _ in range(height)]
    out = {}
    for r0 in range(height):
        for c0 in range(width):
            if seen[r0][c0]:
                continue
            colour = grid[r0][c0]
            stack, cells = [(r0, c0)], []
            seen[r0][c0] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and not seen[nr][nc]
                            and grid[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            shape = tuple(sorted((r - top, c - left) for r, c in cells))
            out.setdefault((colour, shape), []).append((top, left))
    return out


def _offsets(before, after):
    """(colour, (dr, dc)) for every object that kept its shape and moved.

    Only shapes appearing exactly once on each side are used. A shape present
    several times -- ls20 has six objects sharing the avatar's colours -- cannot
    be matched without tracking identity, and guessing there is how a previous
    attempt at this ended up reporting the avatar in a different place each frame.
    """
    a, b = _components(before), _components(after)
    moves = []
    for key, starts in a.items():
        ends = b.get(key)
        if not ends or len(starts) != 1 or len(ends) != 1:
            continue
        (r0, c0), (r1, c1) = starts[0], ends[0]
        if (r1 - r0, c1 - c0) != (0, 0):
            moves.append((key[0], (r1 - r0, c1 - c0)))
    return moves


def _grid_of(frame):
    grid = getattr(frame, "grid", None)
    if grid:
        return grid
    art = getattr(frame, "ascii", None)
    return [list(line) for line in str(art).splitlines() if line] if art else None


def _action_name(action):
    if isinstance(action, dict):
        return str(action.get("action", "")).upper() or None
    text = str(action or "").strip()
    return text.split("(")[0].upper() or None


def tally(transitions):
    """{action: {colour: {'offset': (dr, dc), 'seen': n, 'agreement': f}}}."""
    counts = {}
    for transition in transitions or []:
        name = _action_name(getattr(transition, "action", None))
        # MOUSE has coordinates rather than a fixed effect, and RESET restarts the
        # level -- the board jumps back to its opening layout, which looks like a
        # large translation and would be reported as part of the control scheme.
        # Measured: bp35 came back with "RESET moves colour 11 by (0,-30)".
        if not name or name in ("MOUSE", "RESET"):
            continue
        before = _grid_of(getattr(transition, "before_frame", None))
        after = _grid_of(getattr(transition, "after_frame", None))
        if not before or not after:
            continue
        for colour, offset in _offsets(before, after):
            counts.setdefault(name, {}).setdefault(colour, {})
            bucket = counts[name][colour]
            bucket[offset] = bucket.get(offset, 0) + 1

    out = {}
    for name, by_colour in counts.items():
        for colour, offsets in by_colour.items():
            total = sum(offsets.values())
            offset, hits = max(offsets.items(), key=lambda kv: kv[1])
            if total < MIN_OBSERVATIONS or hits / total < MIN_AGREEMENT:
                continue
            out.setdefault(name, {})[colour] = {
                "offset": list(offset), "seen": total,
                "agreement": round(hits / total, 2)}
    return out


def reading(transitions):
    """The control scheme and the counters, in one line, or None if not yet known."""
    table = tally(transitions)
    if not table:
        return None

    # A colour that moves the same way under every action tried is a readout,
    # not something being controlled.
    per_colour = {}
    for name, by_colour in table.items():
        for colour, info in by_colour.items():
            per_colour.setdefault(colour, {})[name] = tuple(info["offset"])
    actions = set(table)
    counters = [c for c, seen in per_colour.items()
                if len(seen) == len(actions) > 1 and len(set(seen.values())) == 1]

    controls = []
    for name in sorted(table):
        moved = {c: v for c, v in table[name].items() if c not in counters}
        if moved:
            controls.append(f"{name} moves " + ", ".join(
                f"colour {c} by ({v['offset'][0]},{v['offset'][1]})"
                f" [{v['seen']}x]" for c, v in sorted(moved.items(), key=str)))
    parts = []
    if controls:
        parts.append("measured control scheme: " + "; ".join(controls))
    if counters:
        parts.append("colours " + ", ".join(str(c) for c in sorted(counters, key=str))
                     + " move identically under every action — readouts, not gameplay")
    return " | ".join(parts) or None
