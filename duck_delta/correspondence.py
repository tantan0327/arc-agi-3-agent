"""Rare colours that mark where two objects are meant to meet.

Reading cn04's opening board: a black two-pronged plug at top-left with a pale
blue square on the tip of each prong, and a purple socket on the right with two
pale blue squares on its inner edge. Pale blue appears nowhere else. A person
reads that instantly -- the blue marks are the contacts, and the job is to bring
them together. The segmentation the harness hands its model sees four small blue
components among the rest and says nothing about their relationship.

The pattern generalises past cn04 because it is a Core Knowledge prior, not a
game rule: a colour used sparingly, in more than one place, on more than one
object, is a correspondence marker. Keys and keyholes, plugs and sockets, a
legend and the pieces it names.

What makes it cheap to detect is that the marker is *rare*. A colour covering a
quarter of the board is scenery; one covering a dozen cells in four places, each
sitting on a different large object, is a label. So the test is:

    few cells overall, several separate components,
    and those components sit on at least two different host objects

Everything here is derived from a single frame, so it needs no history and can
run the moment a level opens -- which matters, because the games this is aimed at
score zero and therefore never produce the successful outcome that a learned goal
model would need to bootstrap from.

Stdlib only and self-contained, so the source splices into the Python-tool
sandbox the way `segmentation.py` does.
"""


# A marker is by definition sparing. Above this share of the board a colour is
# scenery or a playfield surface, whatever else is true of it.
MAX_MARKER_SHARE = 0.02
# One blob is a lone object, not a correspondence. Two is the interesting case
# (a pair), and cn04 has four (two pins, two contacts).
MIN_COMPONENTS = 2
# And they have to mark different things: four blue squares all on the same plug
# say "this object has blue bits", not "these two objects belong together".
MIN_HOSTS = 2


def _components(cells):
    """Split a set of cells into 4-connected blobs."""
    remaining = set(cells)
    blobs = []
    while remaining:
        seed = remaining.pop()
        blob = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in remaining:
                    remaining.discard(nb)
                    blob.add(nb)
                    stack.append(nb)
        blobs.append(blob)
    return blobs


def _neighbour_colours(blob, grid, own):
    """Colours 4-adjacent to a blob, excluding its own."""
    height = len(grid)
    width = len(grid[0]) if height else 0
    out = set()
    for r, c in blob:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                value = grid[nr][nc]
                if value != own:
                    out.add(value)
    return out


def _centroid(blob):
    return (sum(r for r, _ in blob) // len(blob), sum(c for _, c in blob) // len(blob))


def find_markers(grid, background=None):
    """Colours that look like correspondence markers, with where they sit.

    Returns a list, rarest first, of dicts:

        colour     the marker colour
        cells      how many cells it covers in total
        points     [{'centroid': [r, c], 'cells': n, 'on': [host colours]}, ...]
        hosts      the distinct host colours it appears on, sorted
        reading    a one-line description meant to be shown as-is

    `background` is excluded from host counting when given; when it is not, the
    most common colour on the board is used, since a marker lying on empty space
    is not telling you about an object.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    area = height * width
    if not area:
        return []

    by_colour = {}
    counts = {}
    for r in range(height):
        for c in range(width):
            value = grid[r][c]
            counts[value] = counts.get(value, 0) + 1
            by_colour.setdefault(value, set()).add((r, c))

    if background is None:
        background = max(counts, key=lambda v: counts[v])

    found = []
    for colour, cells in by_colour.items():
        if colour == background or len(cells) > area * MAX_MARKER_SHARE:
            continue
        blobs = _components(cells)
        if len(blobs) < MIN_COMPONENTS:
            continue

        points = []
        hosts = set()
        for blob in blobs:
            on = sorted(v for v in _neighbour_colours(blob, grid, colour)
                        if v != background)
            hosts.update(on)
            points.append({"centroid": list(_centroid(blob)),
                           "cells": len(blob),
                           "on": on})
        if len(hosts) < MIN_HOSTS:
            continue

        # Identical-sized blobs is what separates a deliberate mark from a colour
        # that happens to be scattered. Measured on the opening boards: cn04's
        # contacts are 9,9,9,9 and sk48's legend/target pairs are 16,16, while
        # r11l's flagged colour is fourteen single cells plus one blob of twenty
        # and ft09's runs 4,4,8,12,8. Reported rather than filtered on, since a
        # marker drawn at two scales is still a marker.
        sizes = {len(b) for b in blobs}
        uniform = len(sizes) == 1

        found.append({
            "colour": colour,
            "cells": len(cells),
            "uniform": uniform,
            "points": sorted(points, key=lambda p: p["centroid"]),
            "hosts": sorted(hosts),
            "reading": (f"colour {colour} appears only {len(cells)} times, in "
                        f"{len(blobs)} places"
                        + (f" of identical size ({sizes.pop()} cells each)"
                           if uniform else "")
                        + f", on {len(hosts)} different objects "
                        f"({', '.join(str(h) for h in sorted(hosts))}) — likely "
                        f"marks points that are meant to be brought together"),
        })

    # Uniform first, then rarest: the strongest reading at the top of the list.
    return sorted(found, key=lambda m: (not m["uniform"], m["cells"]))
