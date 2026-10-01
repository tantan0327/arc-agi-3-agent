"""Which colours respond to being clicked, tallied from what has already happened.

ft09 is Lights Out: sixty-odd objects on screen and only the colour-9 tiles do
anything. In one measured run the harness spent 98 clicks there and 84 of them
changed nothing at all, and the game finished with no level completed.

The obvious fix is not the one that works. Those 84 wasted clicks landed on 79
*distinct* cells — only 6% were repeats — so remembering which cells are dead
prevents almost nothing. By colour it separates cleanly:

    colour  9   9 live / 14 dead    39%
    colour  8   5 live /  9 dead    36%
    colour  5   0 live / 17 dead     0%
    colour  4   0 live / 16 dead     0%
    colour  2   0 live / 13 dead     0%
    colour 12   0 live / 11 dead     0%
    colour  0   0 live /  4 dead     0%

**61 of the 84 wasted clicks, 73%, landed on a colour that had never once
responded.** A handful of samples per colour settles it, and unlike a cell it
generalises to the rest of the board.

It also stays quiet where it should. On cd82 every clicked colour ran 84-100%
live, and on s5i5 all 54 clicks worked; a tally reports that and suppresses
nothing. This matters more than it sounds: the previous delta added a *hypothesis*
about what the board meant, and on sk48 the model chased it for 1,225 actions
instead of 91. Taking options away cannot send anyone down a garden path.

Computed from `transitions`, which the sandbox already keeps, so there is no state
to carry between tool calls.

Stdlib only and self-contained, so the source splices into the Python-tool sandbox
the way `segmentation.py` does.
"""


# Enough tries on one colour to believe a run of failures. The measured dead
# colours on ft09 had 4 to 17 tries and never once worked; live ones showed their
# first success well inside that. Below this a colour is reported but not called
# dead, since "not yet" and "never" are different claims.
MIN_TRIES = 4


def _click_target(action):
    """(row, col) a click was aimed at, or None if this was not a click.

    The action arrives either as the display string the harness logs --
    `MOUSE(row=12, col=14)` -- or as the mapping the sandbox accepts,
    `{'action': 'MOUSE', 'row': 12, 'col': 14}`. Accept both and nothing else.
    """
    if isinstance(action, dict):
        if str(action.get("action", "")).upper() != "MOUSE":
            return None
        try:
            return int(action["row"]), int(action["col"])
        except (KeyError, TypeError, ValueError):
            return None

    text = str(action or "")
    if "MOUSE" not in text.upper():
        return None
    row = col = None
    for part in text.replace("(", " ").replace(")", " ").replace(",", " ").split():
        if part.startswith("row="):
            row = part[4:]
        elif part.startswith("col="):
            col = part[4:]
    try:
        return int(row), int(col)
    except (TypeError, ValueError):
        return None


def _grid_of(frame):
    """A frame's cells as rows of ints, from whichever view the frame offers."""
    grid = getattr(frame, "grid", None)
    if grid:
        return grid
    ascii_art = getattr(frame, "ascii", None)
    if not ascii_art:
        return None
    # The sandbox exposes colours as single characters; keep them as characters,
    # since all this needs is equality and a label.
    return [list(line) for line in str(ascii_art).splitlines() if line]


def tally(transitions):
    """{colour: {'tries', 'live', 'rate', 'dead'}} over every click seen so far.

    `colour` is whatever occupied the clicked cell *before* the click, since that
    is what a decision about where to click next can actually look at.
    """
    counts = {}
    for transition in transitions or []:
        target = _click_target(getattr(transition, "action", None))
        if target is None:
            continue
        before = _grid_of(getattr(transition, "before_frame", None))
        after = _grid_of(getattr(transition, "after_frame", None))
        if not before or not after:
            continue
        row, col = target
        if not (0 <= row < len(before) and 0 <= col < len(before[0])):
            continue
        colour = before[row][col]
        entry = counts.setdefault(colour, {"tries": 0, "live": 0})
        entry["tries"] += 1
        if before != after:
            entry["live"] += 1

    for entry in counts.values():
        entry["rate"] = round(entry["live"] / entry["tries"], 2)
        entry["dead"] = entry["tries"] >= MIN_TRIES and entry["live"] == 0
    return counts


def reading(transitions):
    """One line for the model, or None when there is nothing worth saying."""
    counts = tally(transitions)
    if not counts:
        return None
    dead = sorted(c for c, e in counts.items() if e["dead"])
    live = sorted((c for c, e in counts.items() if e["live"]),
                  key=lambda c: -counts[c]["rate"])
    parts = [f"clicks so far: " + ", ".join(
        f"{c}={e['live']}/{e['tries']}" for c, e in sorted(counts.items(), key=str))]
    if dead:
        parts.append("never once responded: " + ", ".join(str(c) for c in dead)
                     + " — clicking these colours has done nothing every time")
    if live:
        parts.append("responds: " + ", ".join(str(c) for c in live))
    return "; ".join(parts)
