"""Paper figures for §5.1 (the abstraction ladder) and §6.1 (authoring).

Every number is one already stated in docs/paper/draft.md and reproducible
from the scripts named there. Style matches the existing figures (matplotlib,
a single blue sequential hue for magnitude, direct labels, recessive axes).

    python scripts/make_paper_figures_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper"
BLUE = ["#104281", "#256abf", "#5598e7", "#9ec5f4"]   # sequential steps 650/500/350/200
NEUTRAL = "#b8b7b2"
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 11})


def recess(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_ladder():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.05, 1]})
    # Panel A: novelty of the current state by representation (6,512 human decisions)
    rungs = ["masked\nboard (A0)", "object set\n(A1)", "object\nlayout (A2)", "census +\navatar (A3)"]
    novel = [98.3, 37.9, 92.1, 89.4]
    cols = [NEUTRAL, BLUE[0], BLUE[2], BLUE[3]]
    bars = a.bar(rungs, novel, color=cols, width=0.62)
    for r, v in zip(bars, novel):
        a.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", color=INK)
    a.set_ylim(0, 108); a.set_ylabel("human decisions at a never-seen state  [%]")
    a.set_title("Novelty is a property of the representation\n(6,512 human decisions, 22 games)", color=INK)
    a.annotate("49% of decisions land on an\nalready-tried (state, action)", (1, 37.9), xytext=(1, 62),
               textcoords="data", ha="center", va="bottom", fontsize=8.5, color=INK2,
               arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    recess(a)
    # Panel B: the planner-grade corner -- determinism vs responsiveness
    generic = {"A0": (98.4, 20.7), "A1": (52.5, 71.7), "A2": (93.9, 18.8), "A3": (89.8, 26.4)}
    hand = {"tr87": (87.4, 100.0), "m0r0": (64.5, 95.2), "s5i5*": (97.0, 66.7)}
    b.axhspan(90, 102, xmin=0.5, xmax=1.0, color="#e8f1fc", zorder=0)
    b.text(101, 88.5, "planner-grade corner", ha="right", va="top", fontsize=8.5, color=INK2)
    goff = {"A0": (6, 6), "A1": (7, -3), "A2": (-4, -14), "A3": (7, -3)}
    for k, (x, y) in generic.items():
        b.scatter(x, y, s=64, color=SERIES_ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
        b.annotate(k, (x, y), xytext=goff[k], textcoords="offset points", fontsize=9, color=INK)
    for k, (x, y) in hand.items():
        b.scatter(x, y, s=70, color=SERIES_BLUE, edgecolor="white", linewidth=1.2, zorder=4)
        dx, dy = {"tr87": (6, -13), "m0r0": (-34, -4), "s5i5*": (-38, -12)}[k]
        b.annotate(k, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK)
    b.set_xlim(0, 102); b.set_ylim(0, 106)
    b.set_xlabel("responsiveness  (state changed when the board changed)  [%]")
    b.set_ylabel("next-state determinism  [%]")
    b.set_title("A good abstraction is deterministic AND responsive\n(generic rungs vs hand-written, human replays)", color=INK)
    b.scatter([], [], color=SERIES_ORANGE, label="generic rung (all games)")
    b.scatter([], [], color=SERIES_BLUE, label="hand-written, per game")
    b.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False, fontsize=8.5)
    b.text(1, 3, "*s5i5 is a progression game: no state recurs;\n its per-control delta consistency is 79%", fontsize=7.5, color=INK2, va="bottom")
    recess(b)
    fig.tight_layout()
    fig.savefig(OUT / "fig_abstraction_ladder.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_authoring():
    hist = json.load(open(ROOT / "scratchpad" / "gpc" / "grade_histograms.json"))
    order = ["v3", "v4", "v5", "v7"]
    labels = {"v3": "v3  base loop,\n3 rounds", "v4": "v4  + element\nfeedback", "v5": "v5  best-of-6,\nscorer selects", "v7": "v7  + isolation\nhints"}
    grades = ["PASS", "LOW-COV", "NONDET", "NO-REVISIT"]
    gcol = {"PASS": BLUE[0], "LOW-COV": BLUE[1], "NONDET": BLUE[2], "NO-REVISIT": BLUE[3], "BLIND": NEUTRAL, "ERROR": NEUTRAL}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.38})
    y = list(range(len(order)))[::-1]
    for yi, run in zip(y, order):
        left = 0
        h = hist[run][0]
        for g in grades + ["BLIND", "ERROR"]:
            n = h.get(g, 0)
            if not n:
                continue
            a.barh(yi, n, left=left, color=gcol[g], height=0.6, edgecolor="white", linewidth=1.5)
            if n >= 1:
                a.text(left + n / 2, yi, f"{n}", ha="center", va="center", fontsize=9,
                       color="white" if g in ("PASS", "LOW-COV") else INK)
            left += n
        a.text(22.4, yi, "PASS: " + ", ".join(hist[run][1]), va="center", fontsize=8.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels([labels[r] for r in order])
    a.set_xlim(0, 30); a.set_xticks([0, 5, 10, 15, 20, 22]); a.set_xlabel("games (of 22), best grade per run")
    a.set_title("An open model authors a planner-grade abstraction\non ~2 of 22 games per run (Qwen3.8-Flash-Next, 437 authorings)", color=INK)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=gcol[g], label=g) for g in grades], loc="lower right",
             frameon=False, fontsize=8, ncol=4, bbox_to_anchor=(1.0, -0.34))
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.grid(axis="x", color="#e6e5e1", linewidth=0.8); a.set_axisbelow(True)
    # Panel B: the closed loop
    names = ["fork baseline\n(gates off)", "stall-breaker\nrun 1", "stall-breaker\nrun 2", "share bundle\n(other harness)"]
    levels = [19, 21, 20, 27]
    cols = [NEUTRAL, SERIES_BLUE, SERIES_BLUE, "#e6e5e1"]
    bars = b.bar(names, levels, color=cols, width=0.6)
    for r, v in zip(bars, levels):
        b.text(r.get_x() + r.get_width() / 2, v + 0.6, str(v), ha="center", va="bottom", color=INK)
    b.set_ylim(0, 37); b.set_ylabel("levels completed")
    b.set_title("Closing the loop: exploring an authored abstraction\nis a null against its own baseline (25 games, one clock)", color=INK)
    b.text(-0.35, 36.5, "same engine, model and clock; the share-bundle\ncontrol (27) is a different harness, shown for scale", ha="left", va="top", fontsize=8, color=INK2)
    recess(b)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.2, wspace=0.38)
    fig.savefig(OUT / "fig_authoring.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_ladder(); fig_authoring()
    print("wrote", OUT / "fig_abstraction_ladder.png", "and", OUT / "fig_authoring.png")
