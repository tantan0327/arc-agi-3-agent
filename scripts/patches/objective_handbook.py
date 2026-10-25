# PATCH: objective handbook (CC0, ours) on Franzen's M2 harness.
#
# What it adds: ONE block of text to the system prompt - a prior over what a level's
# objective can be (core-knowledge archetypes: navigate, collect-then-exit, key-and-door,
# push blocks, match a reference, toggle switches, ordered sequence, route a beam, sort or
# align, counter or resource, grow or fill, select by property), how each is recognised on
# the board, its cheapest discriminating probe, its completion signal, and a short protocol
# for stating and testing objective hypotheses at the start of every level.
# Nothing game-specific, no coordinates, no model, prompts otherwise untouched. The block
# sits at the end of the system prompt, so the server's prefix cache covers it.
# Flags: ARC3_HANDBOOK=0 disables; ARC3_HANDBOOK_TURN_LINE=0 drops the one-line per-turn reminder;
# ARC3_HANDBOOK_VERSION=2 adds three archetypes and the cross-game regularities distilled from the 15
# public games the demo never runs (held-out from the 10 demo games).
import os as _os

from inference.agent import tool_agent as _TA

_ENABLED = _os.environ.get("ARC3_HANDBOOK", "1").strip().lower() not in ("0", "false", "off", "")
_TURN_LINE = _os.environ.get("ARC3_HANDBOOK_TURN_LINE", "1").strip().lower() not in ("0", "false", "off", "")
_VERSION = int(_os.environ.get("ARC3_HANDBOOK_VERSION", "1") or 1)

HANDBOOK = """

Objective handbook (how levels in these games are won - read this before probing mechanics):
- Every level has ONE objective, and it is almost always a CHAIN of 2-4 sub-goals, not a single move: a precondition must be made true (collect, unlock, align, toggle, match), and only then does the finishing move (enter the exit, press the final switch, place the last piece) clear the level. Reaching a plausible target location while the precondition is false does nothing - that is the most common way to spend 100 actions on a level without progress.
- The objective is readable from the board's DESIGN before any action: (1) a small framed panel or strip set apart from the play area (often bottom/top/side) is a legend or reference: its icons mirror things on the board and show what the finished state looks like, which items are still missing, or which colour is "on"; (2) the RAREST interior colour is usually a target, key, exit, or goal marker; (3) N identical small objects mean "visit/collect/activate all N"; (4) a gap in a wall filled with a distinct colour is a door; (5) a small block whose cells increment or change colour after your actions is a counter or indicator; (6) the one object that moves when you press a direction key is the avatar - find it with a single probe; (7) two regions with the same shape but different colours usually mean "make this one look like that one".
- Humans discover the objective within the first few actions because the first obvious interaction is the intended one (games are designed to be learnable by a person in minutes). Prefer the hypothesis that explains the most board elements at once.

Archetypes - recognise, chain, cheapest probe, completion signal:
1. NAVIGATE-TO-TARGET: an avatar, walls, one distinctive target cell/object. Chain: reach the target (sometimes it must be entered from a specific side or with a specific colour/shape). Probe: one key press to find the avatar and its step size. Signal: the target vanishes or the board changes to a new layout.
2. COLLECT-THEN-EXIT: several identical items plus an exit/door that is closed or absent at first. Chain: touch every item (they disappear or change colour), then enter the exit (it opens or appears when the last item is taken). Probe: touch one item and check whether the count/legend changes. Signal: items vanish one by one; the exit changes when the set is complete.
3. KEY-AND-DOOR: a rare-coloured object and a same-coloured door/gap. Chain: pick up or stand on the key (it may follow the avatar or vanish), then walk through the door. Probe: touch the key, then try the door. Signal: the door opens or changes colour after the key is taken.
4. PUSH-BLOCKS: movable blocks and target pads of a matching colour/shape (sokoban). Chain: push every block onto a pad; blocks move only when pushed from behind and stop at walls - plan pushes in order, avoid corners. Probe: push one block one cell and see whether it moves. Signal: a block on a pad changes colour; all pads filled clears the level.
5. MATCH-REFERENCE-PATTERN: a small reference picture in a panel and an editable region on the board. Chain: make the editable region identical to the reference (by clicking cells, moving pieces, or rotating). Probe: change one cell/piece and compare the region to the reference in code. Signal: the panel flashes, changes colour, or the level clears the moment the regions match.
6. TOGGLE-SWITCHES: several switches/lights with on/off colours and an indicator panel. Chain: set every switch to the required state (some switches flip their neighbours - treat it as a parity/lights-out puzzle and solve it in code). Probe: toggle one switch and read which cells changed. Signal: the indicator panel turns fully "on".
7. SEQUENCE-ORDER: a strip of icons/colours in a panel and matching objects on the board. Chain: activate the objects in the order shown by the strip. Probe: activate the first icon's match and watch the strip (a cursor or mark advances). Signal: a marker advances along the strip; a wrong order resets the strip.
8. ROUTE-BEAM: a source, mirrors/deflectors, a receiver. Chain: orient or place deflectors so the beam reaches the receiver (and threads through every required piece). Probe: rotate one deflector and trace the beam path in code. Signal: the receiver lights up; the beam animation shows the path.
9. SORT-OR-ALIGN: several objects that differ in size, colour or shape and a row/column/slot structure. Chain: place each object in the slot with its matching property (same colour, same shape, increasing size). Probe: move one object into a slot that matches it and check for a mark. Signal: a mark or colour change per correct slot; all correct clears.
10. COUNTER-OR-RESOURCE: a counter block or a resource bar that is not the step bar. Chain: perform the counted action exactly N times, or bring the counter to the reference value shown elsewhere. Probe: do the action once and read the counter. Signal: the counter reaches the target and the level clears.
11. GROW-OR-FILL: paint/fill mechanics where touched cells change colour. Chain: fill a region/shape completely (or make a path of one colour from A to B). Probe: touch one cell and see whether neighbours change too. Signal: the region is uniform; the level clears when the last cell flips.
12. SELECT-BY-PROPERTY: several candidate objects, one is "correct" by a property shown in a legend (the odd one out, the one matching the sample, the largest). Chain: identify the property from the legend, then click/visit the matching object. Probe: compare candidates in code (hash, pixels, colour) before clicking; a wrong click often costs a life or resets.

Objective protocol (every new level, including level 1 and every level after a clear):
- Before the first action on a level, write 2-3 objective hypotheses from the archetypes above, each with the board evidence for it (panel, rare colour, counts, doors, avatar), and the chain of sub-goals each implies.
- Choose the single cheapest action that DISCRIMINATES between the hypotheses; after it, read the legend/panel/counter/indicator cells (not only the avatar) with `frame_diff()` - sub-goal progress is usually signalled there.
- When a hypothesis is confirmed, encode its sub-goals as code checks (counts of remaining items, door state, region equality, indicator state) and search for the shortest action sequence that satisfies them in order; batch confidently once the chain is verified.
- If 15 or more actions change the board but no sub-goal signal appears, the objective hypothesis is wrong, not the mechanics: re-read the board design, switch to the next hypothesis, and do not keep walking to the same target.
- On the next level of the same game, assume the SAME chain with a new layout: verify its first signal within the first 3 actions, then plan the whole chain.
"""

HANDBOOK_V2_EXTRA = """
13. ASSEMBLE-PIECES: several polyomino pieces with connector marks on their edges. Chain: select a piece, slide or rotate it until its connectors touch another piece's; mated connectors change colour; the level ends when no free connector remains. Probe: select one piece, move it one step, rotate once. Signal: connectors change colour on contact.
14. REDUCE-TO-ONE: pegs on a board; a peg jumps over a neighbour into an empty cell and removes it. Chain: plan the whole jump sequence in code until one peg (sometimes two) remains; later levels add transport between rooms and the view may scroll. Probe: click a peg, then a landing cell. Signal: the jumped peg disappears.
15. MEET-AND-MERGE: two or more agents move at once, sometimes with mirrored controls. Chain: bring the pair onto one cell so they merge; later levels add keys and doors that stay open only while the other agent holds the key. Probe: one key press to see which agents move and in which directions. Signal: merged agents change state and the level ends at once.
Variants seen often: NAVIGATE where you edit the terrain (destroy blocks so a rising or sliding piece lands on the goal); KEY-AND-DOOR where the key is a CONFIGURATION (cycle a HUD glyph's shape, colour and rotation until it matches the goal box, then enter); SEQUENCE-ORDER as a small PROGRAM (set instruction glyphs, then press run; a wrong program just resets the piece); PUSH-BLOCKS as carry-and-deposit (face the box, SPACE grabs, SPACE drops into a slot).

Regularities seen across many games (priors - verify each on the board):
- Level 1 is a one-mechanic tutorial: a person clears it in 4-15 actions in most games. If 30 actions have gone into level 1, the OBJECTIVE hypothesis is wrong, not the mechanics.
- The win condition is the SAME for every level of a game; each later level adds exactly one element (a new colour, a second pad, a moving hazard, mirrored controls, scrolling, flipped gravity). On a new level, re-use the chain and look for the one new element first.
- Among identical sprites, the controlled one carries a marker (a dot, a ring, a distinct frame); clicking another sprite often transfers control to it. Step size per key press is constant within a game (1, 2, 3 or 6 cells): measure it once, then plan in multiples of it.
- SPACE is the commit key: pour, rotate, grab or drop, run the program, rewind. Test it once, early.
- The legend or reference panel sits OUTSIDE the play area (a target picture, a colour-order strip, a palette with a preview, a glyph box, example cards) and defines the objective; a rare colour INSIDE the play area marks a goal, key or receptacle. Trap: the step bar can share a colour with a goal object.
- Sub-goal progress is a LOCAL colour change on the receptacle or legend (a connector turns gray, a legend item gets a white centre, a slot gets a filled centre). A flash of a whole object usually means rejection or failure.
- Step bars: a strip along the top or bottom row (the row can change between levels); one cell is not one action; some games budget only arrow moves, others every action; when it empties the level restarts from its initial state (earlier levels are kept); some games add lives and kill on the first action after the bar empties.
- Completion is checked after EVERY action, so a batch can clear the level mid-way and its remaining actions land on the NEXT level: keep batches short when completion is near, and re-ground as soon as `level_completed` appears.
- One action can span many frames (animations of 6-35 frames); a rejected move can wiggle for one frame and look like a no-op in a before/after diff: read `last_animation_frames` when a move seems to do nothing.
- Deaths besides the budget: resting under a hazard, a moving hazard that eats pieces, fluid touching the floor or ceiling, entering a box with the wrong key configuration. Coordinates can shift with camera scrolling; controls can be mirrored between two avatars; gravity can flip between levels.
- Cheapest probes: one press per arrow key (avatar, step size, walls), one click per object class, one SPACE, and the bar read before and after each probe.
"""
if _VERSION >= 2:
    HANDBOOK = HANDBOOK.rstrip("\n") + "\n" + HANDBOOK_V2_EXTRA

TURN_LINE = (
    "Objective check: name the objective archetype and the sub-goal you are advancing now "
    "(or the hypothesis you are testing) before you act."
)

_orig_system = _TA._build_system_prompt


def _system_with_handbook(*args, **kwargs):
    prompt = _orig_system(*args, **kwargs)
    if _ENABLED and "Objective handbook (" not in prompt:
        prompt = prompt.rstrip("\n") + HANDBOOK
    return prompt


_orig_build = _TA.ToolAgent._build_user_prompt


def _build_with_handbook(self, *args, **kwargs):
    text = _orig_build(self, *args, **kwargs)
    if _ENABLED and _TURN_LINE and TURN_LINE not in text:
        text = text.rstrip() + "\n\n" + TURN_LINE + "\n"
    return text


if _ENABLED:
    _TA._build_system_prompt = _system_with_handbook
    _TA.ToolAgent._build_user_prompt = _build_with_handbook
print("HANDBOOK_PATCH installed: enabled=%s version=%d turn_line=%s handbook=%d chars (~%d words)" % (
    _ENABLED, _VERSION, _TURN_LINE, len(HANDBOOK), len(HANDBOOK.split())), flush=True)
