# Level-objective archetypes for 15 ARC-AGI-3 games

Sources: the 60-minute duck-harness transcripts of 2026-10-09 ("older" = the 25-minute 2026-10-07 run), the obfuscated game sources ("source") and the official human replays.

### bp35
- Mechanics: A blue bubble one 6x6 block wide moves one block LEFT/RIGHT, then rises to the top of its column by itself. CLICK destroys a green/orange block. Purple hazards are indestructible. Purple step bar on the bottom row (top row in level 3).
- Objective chain: carve a column so the rising bubble rests on the pink gem; resting under a hazard or on a spike kills (source).
- Completion signal: a rise/land animation, then the next level; on death the bubble dissolves.
- Human cost: L1 15, L2 40, L3 34, L4 23, L5 31, L6 42, L7 46, L8 52, L9 89 (LEFT/RIGHT/CLICK only).
- Agent result: 3 of 9 (older 1); stuck on level 4 after three deaths under hazards.
- Archetype: NAVIGATE-TO-TARGET (terrain-editing variant).

### cd82
- Mechanics: Click a palette swatch to pick a colour; arrows move a cup around a 10x10 canvas in 8 positions; SPACE pours, covering the half-plane on the cup's side. A 10x10 target panel sits top-left; per-level action cap (source).
- Objective chain: order the pours until the canvas equals the target; the check ignores the two diagonals (source).
- Completion signal: none between pours; the next level loads on the completing pour.
- Human cost: L1 8, L2 6, L3 20, L4 15, L5 16, L6 16.
- Agent result: all 6 in 20 min (older 0), the only full win.
- Archetype: MATCH-REFERENCE-PATTERN.

### cn04
- Mechanics: Polyomino pieces of 3x3 blocks on a 20x20 board. Click selects a piece, arrows slide it one block, SPACE rotates it. Red pins on piece edges mate when they touch and turn gray. Top-row step bar (MaxSteps 75-200, source).
- Objective chain: join every piece until no free pin remains (source: win when no red cell is left).
- Completion signal: mated pins turn gray; the level ends on the last mating, even mid-batch (agent, level 4).
- Human cost: L1 14, L2 30, L3 36, L4 34, L5 53, L6 64.
- Agent result: all 6 in 45 min (older 1); most actions went to clicking pieces to read pins.
- Archetype: ASSEMBLE-PIECES (new: jigsaw with connectors).

### dc22
- Mechanics: A green 2x2 avatar moves 2 cells per press over gray or coloured blocks; checker and charcoal are walls. Coloured buttons in the right panel rotate the same-coloured arm assembly on the left. StepCounter 128-1024 (source).
- Objective chain: press buttons in the order that links walkable blocks, then stand on the yellow 2x2 (source: avatar position equals goal position).
- Completion signal: purple checkers turn solid when their button is pressed; ends on arrival.
- Human cost: L1 22, L2 55, L3 48, L4 64, L5 115, L6 217.
- Agent result: 4 of 6 (older 1); stuck on level 5 with a ring it could not move along its path.
- Archetype: NAVIGATE-TO-TARGET with TOGGLE-SWITCHES.

### g50t
- Mechanics: A 5x5 blue ring moves 6 cells per press; white is wall; no clicks. A 3x3 pad holds its 5x5 gate open only while something stands on it. A gray echo of the avatar spawns at a vacated cell (rule never pinned down; unverified); spawns are budgeted by HUD slots; SPACE rewinds.
- Objective chain: leave an echo on the pad so the gate stays open, then walk through to the goal ring (a blue square ring open on one side).
- Completion signal: the level ends on entering the ring (source: flag checked each step).
- Human cost: L1 17, L2 31, L3 64, L4 31, L5 50, L6 41, L7 43 (SPACE 1-3 per level).
- Agent result: 2 of 7 (older never acted); 100+ actions on the echo rule in level 2.
- Archetype: KEY-AND-DOOR (pressure pad, clone as key-holder).

### ka59
- Mechanics: Several identical green sprites; the one with a white dot is controlled (3 cells per press); clicking another moves the dot to it. Yellow pluses are kicked and slide up to 5 steps. Frame blocks tick periodically and eject a sprite on their red side. StepCounter 100-200.
- Objective chain: every green sprite inside a charcoal square ring and every yellow plus inside a plus-shaped outline (source: two containment checks).
- Completion signal: nothing per placement; the level ends on the last one.
- Human cost: L1 13, L2 43, L3 38, L4 39, L5 21, L6 69, L7 93.
- Agent result: 6 of 7 (older 1); level 7 reached at step 395 with no time left.
- Archetype: SORT-OR-ALIGN (shape-to-socket, with pushing).

### lf52
- Mechanics: Peg solitaire in white rooms: click a peg (dark ring), then a landing cell (preview ring) to jump a neighbour and remove it. Black wires carry orange ferry blocks that arrows move, ferrying one peg to another room; the camera pans with the ferry. UNDO exists.
- Objective chain: reduce the pegs to one (levels 6-7: two) (source).
- Completion signal: a short animation, then the next level. Arrow moves are capped per level (64/320/640, source).
- Human cost: L1 8, L2 41, L3 47, L4 52, L5 102, L6 92, L7 157, L8 71, L9 113, L10 99.
- Agent result: 3 of 10 (older 1); stuck on level 4 remapping rooms after each camera scroll.
- Archetype: REDUCE-TO-ONE (new: peg solitaire).

### ls20
- Mechanics: Arrows only. The avatar is a block (orange cap, blue body) in a maze. A white cross pad cycles the HUD glyph's shape; a central pinwheel cycles its colour; a target box in the maze shows the goal glyph. Yellow rings refill the 42-step bar; 3 red lives.
- Objective chain: make the HUD glyph match the box glyph in shape, colour and rotation (source: three-attribute equality), then push the block into the box.
- Completion signal: the HUD box flashes white on a rejected entry; a ring turns gray when consumed; the level ends on entry.
- Human cost: no replay file; official baseline 22 / 123 / 73 / 84 / 96 / 192 / 186.
- Agent result: 4 of 7 (older never acted); level 5 added a second pad and probes cost lives.
- Archetype: KEY-AND-DOOR (configure the key, then enter).

### m0r0
- Mechanics: Two sky-blue agents move at once with mirrored controls (LEFT for one is RIGHT for the other) through mazes with checker hazards; levels 5-6 add coloured keys and matching doors that open while the other agent holds the key. 150 actions per level (source).
- Objective chain: bring the pair onto one cell so they merge; the source counts unmerged pairs, so later levels may need every coloured pair merged (unverified).
- Completion signal: merged items turn intangible; the level ends instantly.
- Human cost: L1 15, L2 29, L3 79, L4 13, L5 48, L6 50.
- Agent result: 5 of 6 (older 0); level 6's key/door pairing took the rest of the hour.
- Archetype: MEET-AND-MERGE (new: mirrored twins).

### s5i5
- Mechanics: Click only. Coloured bars with a dot; bottom panels (two jaw symbols per colour) grow or shrink that colour's bar; bars push other bars; dark-red crosses are the targets. StepCounter 50-200.
- Objective chain: extend each bar until its dot sits on its cross (source: every target has a dot sprite at its position).
- Completion signal: the cross's centre fills; the level ends on the last activation.
- Human cost: L1 13, L2 26, L3 46, L4 32, L5 34, L6 25, L7 48, L8 44 (all clicks).
- Agent result: 3 of 8 (older 0); level 4's merging bars broke its object model.
- Archetype: ROUTE-BEAM (extensible bars).

### sk48
- Mechanics: A head with an extensible tail inside a panel; arrows move the head on the panel lattice and extend or retract the tail; the tail tip touching a block collects it onto the body. A bottom legend gives the colour order and marks the current target with a white centre. The row-53 bar counts arrow moves only (source: not clicks or UNDO).
- Objective chain: collect every block in the legend's order.
- Completion signal: a 35-frame colour flash of the blocks (source), then the next level.
- Human cost: L1 14, L2 30, L3 33, L4 36, L5 63, L6 40, L7 63, L8 46 (clicks only from L6).
- Agent result: 2 of 8 (older 0); level 3's vertical chain defeated its search within the 30-second Python limit.
- Archetype: SEQUENCE-ORDER.

### sp80
- Mechanics: A goo source (magenta block, charcoal spout); arrows move a blue paddle; red bars are fixed; SPACE pours. The stream spreads 3 cells past an obstacle and drips on. Gravity flips in some levels (source). One step per action; 4 pours per level, the fifth is fatal (source).
- Objective chain: place the paddle so one pour fills every yellow cup without goo touching floor or ceiling (source: all receptacles filled -> next level).
- Completion signal: failure flashes the cups for 6 frames and resets the level; success loads the next level.
- Human cost: L1 4, L2 10, L3 16, L4 50, L5 57, L6 56.
- Agent result: 4 of 6 (older 1); level 5's new shapes changed the spread rule.
- Archetype: ROUTE-BEAM (fluid routing).

### su15
- Mechanics: Click only. Level 1: a token hops along a dotted trail toward the click. From level 2: clicking a pair of light-blue dots spawns a magenta 2x2; two magentas merge into a purple 3x3, two purples into a yellow 4x4 (palette and target preview on the top row). A pink comet eats dots and blocks (level 4). Bar 32-48 steps; losing all dots restarts the level (source).
- Objective chain: build a block of the previewed colour and size and drop it on each blue circle (source: every colour/size spec has a matching block on a target).
- Completion signal: a 10-frame colour wash (source), then the next level.
- Human cost: L1 12, L2 15, L3 19, L4 12, L5 18, L6 10, L7 7, L8 10, L9 12.
- Agent result: 3 of 9 (older never acted); six game overs on level 4, mostly to the comet.
- Archetype: GROW-OR-FILL (merge to size, then deliver).

### tn36
- Mechanics: Click only. A panel of glyph columns (bar and stem toggles) is the program; clicking the blue ball runs it on the yellow plug in the checkerboard panel (animated, then the plug snaps back); reference cards on the left demonstrate instructions.
- Objective chain: set the glyphs so the executed program moves and rotates the plug into the socket (source: win flag after execution; level data lists Programs, Positions, Rotations).
- Completion signal: the level ends after the execution animation; a wrong program just resets the plug.
- Human cost: L1 7, L2 9, L3 13, L4 18, L5 21, L6 29, L7 53 (all clicks).
- Agent result: 1 of 7 (older 1); the whole hour on level 2's instruction semantics.
- Archetype: SEQUENCE-ORDER (program synthesis).

### wa30
- Mechanics: A green avatar with a white strip showing its facing moves 1 cell; SPACE grabs the faced box (its frame turns white) or drops it. Boxes belong in a blue tray with gray slots; an orange block makes a touched box home into the tray by itself. StepCounter 70-200 (source).
- Objective chain: every box in a tray slot (source: all box sprites at tray positions).
- Completion signal: a filled slot shows a blue centre; the level ends on the last drop.
- Human cost: L1 35, L2 48, L3 76, L4 61, L5 109, L6 49, L7 53, L8 118, L9 61.
- Agent result: 2 of 9 (older 1); three budget deaths on level 3.
- Archetype: PUSH-BLOCKS (carry-and-deposit).

## Cross-game regularities

- The avatar is the one small object that moves on an arrow press, coloured unlike the terrain. Among identical sprites the controlled one carries a marker (white dot in ka59, own colour in cn04, dark ring in lf52). Step size per press: 1 cell (wa30), 2 (dc22), 3 (ka59), 6 (g50t, bp35).
- Click-only games (s5i5, su15, tn36) have no avatar; clicks on nothing still cost a step.
- SPACE is the commit key: pour (cd82, sp80), rotate (cn04), grab or drop (wa30), rewind (g50t).
- Legends sit outside the play area and define the objective: a 10x10 target top-left (cd82), a colour-order legend at the bottom (sk48), a palette plus preview on the top row (su15), a HUD glyph box bottom-left (ls20), reference cards on the left (tn36).
- Rare colours mark goals and receptacles: a pink gem (bp35), a yellow 2x2 (dc22), charcoal outlines (ka59), red pins (cn04). The trap: in bp35 the gem, the hazards and the HUD bar share one colour.
- A step bar is a 64-cell strip on the top or bottom row (sp80 and bp35 switch rows between levels; sk48 row 53; ls20 adds 3 red lives). One pixel is not one action (sp80 about 2 px per action, wa30 about 1 px per 3). Budgets run from 30 to 1024. When it empties the level restarts from its initial state (earlier levels kept), so every probe is paid from the current attempt.
- Which actions are budgeted differs: sk48 and lf52 count only arrow moves (source), cd82 and m0r0 every action; su15's UNDO looks budget-free (unverified). ls20's empty bar kills on the next action.
- Sub-goals are signalled by a local colour change on the receptacle or legend: pins turn gray (cn04), a legend item gets a white centre (sk48), a slot gets a blue centre (wa30). A flash means rejection or failure (ls20, sp80).
- Completion is checked every step, so it fires mid-batch and the rest of the batch lands on the next level (cn04, level 4); the post-win frame is already the next level, so the win animation is never observable (sp80, tn36).
- One action can span many frames (cd82 15, sk48 35, su15 10, sp80 6, tn36's program run); dc22's rejected moves wiggle one frame and look like no-ops in a before/after diff.
- Cheapest probes: one press per arrow (avatar, step size, walls), one click per object class, one SPACE, reading the bar before and after. Humans need 4-15 actions on level 1 in 12 of 15 games: level 1 is a one-mechanic tutorial.
- Deaths other than the budget: resting under a hazard (bp35), the comet (su15), goo on the floor or ceiling (sp80). Coordinates shift with camera scroll (lf52, bp35); controls are mirrored (m0r0); gravity flips between levels (sp80).
- Later levels keep the same win function (one condition per game in source) and add one element each (a new colour, a second pad, a moving hazard, scrolling); the goal type never changed within a game.
