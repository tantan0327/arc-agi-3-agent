#!/bin/zsh
# Draw the unmodified duck baseline once per UTC day, unattended.
#
# Why this exists. Resubmitting one kernel version re-runs it rather than
# returning a cached score -- version 14 went 1.11 then 1.23 from identical
# bytes -- so a draw costs no GPU and no push. The board keeps only the best
# score, so a draw has no downside either. What it does have is an expiry: the
# allowance is one submission per UTC day and an unused day does not roll over.
# Fifty days of drawing is worth about +0.23 on the best score; a day missed
# because nobody was at the keyboard at 09:00 JST is simply gone.
#
# The schedule is every three hours rather than once a day, and each run does
# two things in this order:
#
#   1. reconcile -- if the last submission has finished, write its score into
#      SUBMISSIONS.md. This has to come first, because `watch` reconciles
#      against the sha in .cycle-state.json and submitting overwrites it.
#   2. draw -- if a slot is free, submit. `cycle.py submit` refuses when the
#      day's allowance is spent, so running eight times a day still submits
#      once; the extra runs exist so that a machine asleep at 09:00 still
#      draws when it wakes, and so a score is picked up the same day.
#
# Deliberately no `caffeinate`: a scored run takes up to eight hours and
# holding the machine awake for it would be a worse bug than a late score.
# Nothing here waits -- each run is a few API calls and exits.
#
# Scope: this submits the *unmodified* baseline only, which is the standing
# approval. It never builds or pushes, so it cannot submit a modified build
# even if a delta gate is turned on in the working tree.

set -u

ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1

PY="$ROOT/.venv/bin/python"
# Standing approval 2026-08-21: the default draw is the Qwen3.8 build that
# banked 1.87 (arc3-duck-qwen3-8-27b v1, drawn 2026-08-18), as a variance play
# while the throughput/banking work (T1-T3) is built. The board keeps the max,
# 5/25 games flip between identical runs, and four draws of the previous
# baseline spanned 0.98-1.23 -- a redraw cannot lose and can win. v1 exactly:
# v2+ of this kernel carry other experiments, one of which scored 0.00.
# Retargeted 2026-08-26 with the user's approval: the anim-bundle configuration
# measures +25% levels offline (25 vs 20) and the same public recipe verifiably
# drew 2.23 on the board (LB-9 kernel's best score), so nightly samples now come
# from the better-evidenced distribution. Our 1.50 single draw does not
# contradict it -- single draws are noise. Bank 1.87 is kept by the board.
#
# Alternating since 2026-08-27 (q38/anim by UTC day parity); superseded
# 2026-08-30 with the user's approval: dflash v7 drew 2.27 on its first board
# run -- the first of fourteen measured modifications ever to beat the
# baseline families, +0.40 over the old bank -- and the same-day ablation
# attributes the effect to the nightly vLLM engine, so this is a different
# and better distribution, not another sample of the old one. Every night now
# redraws the scored artifact itself. v7 exactly: earlier versions of this
# kernel are the boot-trap iterations and must never be drawn.
# Retargeted 2026-09-04 with the user's approval: the flash-next as-is fork
# drew 2.80 on its first board run, the licence chain is verified end to end
# (official vLLM image digest-matched; Qwen Community 1.0 weights), and the
# 89-team band at 2.5+ prices the family's center at population scale. The
# dflash family (center ~1.73 over five draws) retires as the vehicle.
# Retargeted 2026-09-18 with the user's approval to CANDIDATE B: the same
# flash-next vehicle (same bundle, model, runtime, serving profile, clock) with
# ONE change -- the agent's carried world-model note is backfilled from its own
# retained reasoning when it ends a turn with a tool call and no prose. The
# harness only updated the note from assistant TEXT, and this model thinks
# then calls the tool, so the note was empty for whole games (bp35 55/55
# turns, sk48 53/53). In the production-clock run of this exact kernel version
# the patch verifiably fired and three games that had never cleared their
# level in six runs did (sk48 L1, m0r0 L2, sb26 L2). Thirteen floor draws
# give mean 2.90 sd 0.48, so a floor draw beats the banked 3.92 ~2% of the
# time. Read the mean after three draws; revert to arc3-flashnext-asis v1 if
# it is below 2.5. v1 exactly: it is the validated run.
KERNEL="tantan0327/arc3-probe-notefull"
FAMILY="flash-next candidate B (world-model note backfilled from reasoning)"
VERSION=1
LOG="$ROOT/scratchpad/daily-draw.log"
mkdir -p "$ROOT/scratchpad"

say() { print -r -- "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >>"$LOG" }

say "--- run start"

# Open-sourcing is the one thing that is worth nothing early and everything on
# time. Publishing sooner has no upside -- all three Milestone 1 winners
# published within 46 minutes of each other on the deadline evening, 2026-06-30
# -- and it hands a competitor the human-replay analysis and the action gate,
# neither of which is public. The only risk is forgetting, so the job that
# already runs every three hours carries the countdown.
#
# There are two dates because there are two prizes, and the first is conditional:
# publishing for Milestone 2 exposes the method for the 33 days that follow,
# which is only worth it if we are actually in the top three by then.
# Corrected 2026-08-20: open sourcing is a PRECONDITION for a milestone prize,
# not a reward for placing -- a solution not published by the deadline cannot
# place at all. Publish by 09-30 regardless of rank; participation is decided.
for deadline in "2026-09-30:Milestone 2 (publish BY this date -- eligibility requires it)" \
                "2026-11-02:Top Score Award (publish -- this one is unconditional)"; do
    when="${deadline%%:*}"; what="${deadline#*:}"
    left=$(( ( $(date -j -f "%Y-%m-%d" "$when" "+%s") - $(date "+%s") ) / 86400 ))
    if (( left <= 21 && left >= 0 )); then
        say "OSS DEADLINE: $left days to $when — $what"
        say "              gh repo edit tantan0327/arc-agi-3-agent --visibility public"
        say "              and set is_private=false in the kernel metadata, then push."
    fi
done

# 0. A hold, for the days when the slot is wanted for something else. The draw is
#    the default use of a slot, not the only one -- a configuration with real
#    evidence behind it should outrank another sample of a known one. `touch
#    .draw-hold` before 09:00 JST and the day is yours; delete it to resume.
if [[ -f "$ROOT/.draw-hold" ]]; then
    "$PY" scripts/cycle.py watch --timeout 1 --poll 1 >>"$LOG" 2>&1
    say "HELD by .draw-hold — reconciled only, no draw. Delete the file to resume."
    say "--- run end"
    exit 0
fi

# 1. reconcile whatever is outstanding (one poll, no waiting)
"$PY" scripts/cycle.py watch --timeout 1 --poll 1 >>"$LOG" 2>&1

# 2. what to submit. The default is the baseline draw, which carries a standing
#    approval. `.draw-next` overrides it for one submission and is written only
#    when the user has approved that specific configuration -- the standing
#    approval covers the unmodified baseline and nothing else. It is deleted on
#    success, so the next day reverts to drawing.
#
#    Why this rather than waiting in a loop for the slot to open: the allowance
#    resets at 00:00 UTC, which is 09:00 JST, and a shell loop started the
#    evening before dies with its terminal or stalls while the machine sleeps.
#    The launchd job already runs every three hours and survives both.
logged=$(grep -c '^| 20' SUBMISSIONS.md 2>/dev/null || print 0)
draw=$(( logged + 1 ))
note="duck: candidate B, automated daily draw (#${draw}). Family: ${FAMILY}, ${KERNEL} v${VERSION}. ONE change against the floor vehicle, everything else byte-identical (same bundle, same model, same runtime, same serving profile, same clock): the agent's carried world-model note is backfilled from its own retained reasoning whenever it ends a turn with a tool call and no prose. Diagnosis from six full runs read against the human replays: the harness updated the note only from assistant TEXT, and this model thinks then calls the tool, so cross-turn memory was silently off for whole games. In the production-clock run of this kernel version the patch verifiably fired (bp35 0->31, m0r0 0->19 prose-less turns carrying a note) and three games that had never cleared their level in six runs did: sk48 L1, m0r0 L2, sb26 L2. Drawn instead of the floor because thirteen floor draws give mean 2.90 sd 0.48, so a floor draw beats the banked 3.92 about 2% of the time. Approved by the user 2026-09-18."
oneshot=0

if [[ -f "$ROOT/.draw-next" ]]; then
    KERNEL=""; VERSION=""; NOTE=""
    source "$ROOT/.draw-next"
    if [[ -z "$KERNEL" || -z "$VERSION" ]]; then
        say "REFUSING: .draw-next is missing KERNEL or VERSION — not falling back to the"
        say "          baseline, because a malformed override means someone meant something else."
        say "--- run end"
        exit 1
    fi
    note="$NOTE"
    oneshot=1
    say "override: .draw-next -> $KERNEL v$VERSION"
fi

"$PY" scripts/cycle.py submit \
    --kernel "$KERNEL" --version "$VERSION" --note "$note" --yes >>"$LOG" 2>&1
# `status` is read-only in zsh; naming it that silently aborts the script here.
rc=$?

if [[ $rc -eq 0 ]]; then
    if (( oneshot )); then
        rm -f "$ROOT/.draw-next"
        say "submitted the approved override $KERNEL v$VERSION; .draw-next consumed"
    else
        say "submitted version $VERSION as draw #$draw"
    fi
else
    say "no submission this run (slot spent, or the API refused) — exit $rc"
fi

say "--- run end"
