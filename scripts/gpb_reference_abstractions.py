"""GP-B: hand-written reference abstractions, to calibrate the scorer's ceiling.

Each function reads a raw 64x64 board (list of lists) and returns the
game-specific state a competent human tracks -- the Astra-style "which
variables matter" answer, written by hand so we learn what a GOOD
extract_state scores before asking a model to author one (GP-C).

    python scripts/score_abstraction.py --module scripts/gpb_reference_abstractions.py
"""
import numpy as np


def _blobs(mask):
    """4-connected components of a boolean mask -> list of (size, cy, cx)."""
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
                out.append((len(cells), sum(y for y, _ in cells) // len(cells),
                            sum(x for _, x in cells) // len(cells)))
    return out


def m0r0(board):
    """Twin avatars (colour 10) mirrored across two maze halves. What matters:
    which maze (the level) and where the avatar is. Level identity is the
    WALL MASK: everything in the interior that is not path, avatar, or the
    border counter (colour 0). The path colour is read off the avatar's own
    4-neighbourhood (the avatar always sits on path), because wall/path
    colours change across levels (11/12 early, 15/6/8 later) and a colour
    histogram collides between levels with equal counts."""
    g = np.asarray(board)
    inner = g[1:-1, 1:-1]
    av = _blobs(g == 10)
    left = min(av, key=lambda b: b[2]) if av else None
    pos = (left[1], left[2]) if left else None
    if not av:
        vals, cnts = np.unique(inner[(inner != 0)], return_counts=True)
        return (("nolevel", tuple(zip(vals.tolist(), cnts.tolist()))), None)
    ys, xs = np.nonzero(g == 10)
    neigh = []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < 64) & (nx >= 0) & (nx < 64)
        vals = g[ny[ok], nx[ok]]
        neigh.extend(vals[vals != 10].tolist())
    path = max(set(neigh), key=neigh.count) if neigh else -1
    walls = (inner != path) & (inner != 10) & (inner != 0)
    return (hash(walls.tobytes()), pos)


def tr87(board):
    """A cursor (colour 0) selects a tile column in the bottom band; UP/DOWN
    cycles that tile's 3x3 pattern (7/5) to match the target tiles above.
    What matters: cursor column, the bottom tiles' patterns, the targets."""
    g = np.asarray(board)
    cur = np.argwhere(g == 0)
    cursor = int(cur[:, 1].mean()) // 4 if len(cur) else None
    bottom = g[50:58, 14:52]
    top = g[4:30, 12:52]
    return (cursor, bottom.tobytes(), hash(top.tobytes()))


ABSTRACTIONS = {"m0r0": m0r0, "tr87": tr87}


def s5i5(board):
    """Astra's own example. Pads (static per level) extend bars / move markers
    by 3 cells per click. What matters: the level (pad layout) and the sizes
    and positions of the MOVABLE objects. Excludes the background (5), the
    pads themselves (colour 2/4 blobs that never move), and the border step
    counter. Returned as a flat numeric tuple so that dynamics can be read as
    DELTAS -- these states never repeat within a level (bars only grow), so a
    lookup table is useless and the planner needs 'click pad k -> bar k += 3'."""
    g = np.asarray(board)
    inner = g[1:-1, 1:-1]
    comps = _components_by_colour(inner)
    pads = tuple(sorted((c, s, cy, cx) for c in (2, 4) for s, cy, cx in comps.get(c, [])))
    movable = []
    for c, v in comps.items():
        if c in (5, 2, 4):
            continue
        for s, cy, cx in sorted(v):
            movable.append((c, s, cy, cx))
    return (hash(pads), tuple(movable))


def _components_by_colour(grid):
    """4-connected blobs per colour (excluding nothing): {colour: [(size,cy,cx)]}."""
    out = {}
    seen = np.zeros(grid.shape, dtype=bool)
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if not seen[r, c]:
                colour = int(grid[r, c]); stack = [(r, c)]; seen[r, c] = True; cells = []
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and not seen[ny, nx] and int(grid[ny, nx]) == colour:
                            seen[ny, nx] = True; stack.append((ny, nx))
                # (size, top, left): bbox anchor instead of centroid, so a bar
                # growing by a fixed step yields an exact, repeatable delta
                # (a rounded centroid of a growing bar alternates +1/+2).
                out.setdefault(colour, []).append((len(cells), min(y for y, _ in cells), min(x for _, x in cells)))
    return out


ABSTRACTIONS["s5i5"] = s5i5
