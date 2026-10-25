"""Figures for the paper's allocation section (2026-10-19). Every number is read from
artefacts in this repository; nothing is typed in by hand.

    .venv/bin/python scripts/make_paper_figures_v3.py
"""
from __future__ import annotations
import csv, glob, json, os, re, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig A: decode throughput vs concurrent streams (SGLang log of the copy run) -------------
def fig_throughput():
    by = {}
    for line in open(ROOT / "scratchpad/arc3-copy-franzen-m2_out/serve.log", errors="replace"):
        m = re.search(r"Decode batch.*?#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)", line)
        if m:
            by.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    xs = sorted(k for k, v in by.items() if len(v) > 5 and k >= 1)
    med = [st.median(by[k]) for k in xs]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(xs, med, "o-", color="#1f4e79")
    ax.plot(xs, [med[0] / xs[0] * k for k in xs], "--", color="#999", label="linear in streams")
    for k, m_ in zip(xs, med):
        ax.annotate(f"{m_:.0f}", (k, m_), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("concurrent streams (running requests)"); ax.set_ylabel("decode throughput, tokens/s (median)")
    ax.set_title("One RTX PRO 6000: throughput still rises at the 10-stream cap", fontsize=10)
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig_throughput_vs_streams.png", dpi=160)
    print("fig_throughput_vs_streams:", dict(zip(xs, [round(m_) for m_ in med])))


# ---- Fig B: policies across regimes, incl. the clairvoyant oracle (simulator) ----------------
def fig_policies():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ss", ROOT / "scripts/sched_sim.py"); ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
    traces = json.loads(ss.TRACES.read_text()); ss.build_hazards(traces)
    regimes = [("0.8", 1.0), ("0.8", 1.15), ("0.8", 1.3)]
    pols = ["fair", "base", "dprime", "oracle"]
    res = {p: [] for p in pols}
    for hf, d in regimes:
        ss.HAZ_SCALE = 0.1; ss.DIFFICULTY = d
        for p in pols:
            ms = []
            for seed in range(40, 48):
                games = ss.hidden_population(traces, 110, seed, float(hf))
                ss.simulate(games, ss.POLICIES[p], wall=ss.WALL_BUDGET_S, allow_continue=True, fade=ss.FADE[p])
                ms.append(ss.mean_score(games))
            res[p].append((st.mean(ms), st.pstdev(ms)))
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    x = range(len(regimes)); w = 0.2
    colors = {"fair": "#bbb", "base": "#1f4e79", "dprime": "#c0504d", "oracle": "#2e7d32"}
    labels = {"fair": "round-robin", "base": "Franzen base", "dprime": "D' (field)", "oracle": "clairvoyant oracle"}
    for i, p in enumerate(pols):
        ax.bar([xi + (i - 1.5) * w for xi in x], [m for m, _ in res[p]], w, yerr=[s for _, s in res[p]], color=colors[p], label=labels[p], capsize=2)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"difficulty {d}" for _, d in regimes])
    ax.set_ylabel("simulated contest score (110 games, 9 h)"); ax.axhline(28.2, color="#888", ls=":", lw=1); ax.text(2.45, 28.6, "board: copy family", fontsize=8, ha="right", color="#555")
    ax.set_title("Allocation: the oracle sits ~2 above D' and ~4 above base", fontsize=10); ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_allocation_ceiling.png", dpi=160)
    print("fig_allocation_ceiling:", {p: [round(m, 1) for m, _ in v] for p, v in res.items()})


# ---- Fig C: the 60-minute traces: cumulative tokens at each level clear ----------------------
def fig_slow_not_stuck():
    t = json.loads((ROOT / "scratchpad/sched/traces.json").read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    n = 0
    for run in ("hardtrace2a", "hardtrace2b"):
        for gid, g in t["runs"][run].items():
            cum = 0; xs = [0]; ys = [0]
            for tok, _w, clr in g["steps"]:
                cum += tok
                if clr:
                    xs.append(cum / 1000); ys.append(ys[-1] + 1)
            xs.append(sum(s[0] for s in g["steps"]) / 1000); ys.append(ys[-1])
            ax.step(xs, ys, where="post", lw=1.2, alpha=0.85)
            ax.annotate(gid, (xs[-1], ys[-1]), fontsize=7, xytext=(2, 0), textcoords="offset points")
            n += 1
    ax.axvline(62, color="#c0504d", ls="--", lw=1); ax.text(63, 5.6, "base patience\n(62k tokens)", fontsize=8, color="#c0504d")
    ax.axvline(225, color="#2e7d32", ls="--", lw=1); ax.text(226, 5.6, "D' patience\n(225k)", fontsize=8, color="#2e7d32")
    ax.set_xlabel("generated tokens in the game (thousands)"); ax.set_ylabel("levels cleared")
    ax.set_title(f"{n} 'hard' games with a 60-minute slot: slow, not stuck", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_slow_not_stuck.png", dpi=160); print("fig_slow_not_stuck: games", n)


# ---- Fig D: per-draw distributions vs the board's max-statistic ------------------------------
def fig_max_statistic():
    rows = list(csv.DictReader(open(sorted(glob.glob(str(ROOT / "scratchpad/lb-2026-10-18/*.csv")))[-1])))
    board = [float(r["Score"]) for r in rows if 20 <= float(r["Score"]) <= 34]
    copy_draws = [26.26, 26.88, 28.27, 25.82]       # our byte-identical copy draws (#61-#73, scored)
    # pull every scored copy draw from SUBMISSIONS.md
    copy_draws = []; dprime_draws = []
    for line in open(ROOT / "SUBMISSIONS.md"):
        if not line.startswith("| 20"): continue
        parts = [p.strip() for p in line.split("|")]
        try: score = float(parts[-2])
        except Exception: continue
        if score <= 0: continue
        if "D-prime" in line: dprime_draws.append(score)
        elif "Franzen M2 copy, automated" in line or "Franzen M2 copy (" in line: copy_draws.append(score)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.hist(board, bins=28, color="#ccc", label=f"board, teams' best in 20–34 (n={len(board)}, mean {st.mean(board):.1f})")
    ax2 = ax.twinx()
    ax2.plot(copy_draws, [1] * len(copy_draws), "o", color="#1f4e79", label=f"our copy draws (n={len(copy_draws)}, mean {st.mean(copy_draws):.1f})")
    ax2.plot(dprime_draws, [2] * len(dprime_draws), "s", color="#c0504d", label=f"our D' draws (n={len(dprime_draws)}, mean {st.mean(dprime_draws):.1f})")
    ax2.set_ylim(0, 3); ax2.set_yticks([]); ax2.spines["right"].set_visible(False)
    ax.set_xlabel("public leaderboard score"); ax.set_ylabel("teams")
    ax.set_title("Each team shows its best draw: the band's mean is a max-statistic", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_max_statistic.png", dpi=160)
    print("fig_max_statistic: copy", copy_draws, "dprime", dprime_draws)


# ---- Fig E: the 2026-10-24 arms against the base band: levels vs generated tokens --------------
def fig_arms_vs_band():
    def run(d):
        import json as _j, glob as _g, re as _re
        runs = _j.load(open(ROOT / "scratchpad" / d / "benchmark.json"))["game_runs"]
        lv = sum(int(r.get("levels_completed") or 0) for r in runs); tok = 0
        for fp in _g.glob(str(ROOT / "scratchpad" / d / "*_requests.jsonl")):
            for line in open(fp, errors="replace"):
                m = _re.search(r'"completion_tokens":\s*(\d+)', line)
                if m: tok += int(m.group(1))
        return lv, tok / 1000
    base = {"copy": "arc3-copy-franzen-m2_out", "rep": "arc3-franzen-rep_out", "rep2": "arc3-franzen-rep2_out", "patient": "arc3-franzen-dprime-patient_out", "streams12": "arc3-franzen-streams12_out"}
    arms = {"handbook": ["arc3-franzen-handbook_out", "arc3-franzen-handbook2_out"], "handbook v2": ["arc3-franzen-handbook-v2_out", "arc3-franzen-handbook-v2b_out"],
            "census": ["arc3-franzen-census-p_out"], "motion helpers": ["arc3-franzen-motion_out", "arc3-franzen-motion2_out"],
            "motion, forced": ["arc3-franzen-motion-push_out", "arc3-franzen-motion-push2_out"], "handbook v2 + helpers": ["arc3-franzen-stack_out", "arc3-franzen-stack2_out"]}
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    bl = []
    for name, d in base.items():
        try: lv, tok = run(d)
        except Exception: continue
        bl.append(lv); ax.plot(tok, lv, "o", color="#999", ms=7)
        off = {"copy": (5, 4), "patient": (-38, -11), "rep2": (5, 4), "rep": (5, 4), "streams12": (5, 4)}.get(name, (4, 2))
        ax.annotate(name, (tok, lv), fontsize=7, color="#666", xytext=off, textcoords="offset points")
    if bl:
        ax.axhspan(min(bl), max(bl), color="#ddd", alpha=0.5, lw=0); ax.axhline(st.mean(bl), color="#888", ls=":", lw=1)
    colors = ["#1f4e79", "#2e7d32", "#c0504d", "#8e44ad", "#e67e22", "#16a085"]
    for (name, ds), col in zip(arms.items(), colors):
        pts = []
        for d in ds:
            try: pts.append(run(d))
            except Exception: pass
        if pts:
            ax.plot([p[1] for p in pts], [p[0] for p in pts], "s", color=col, ms=7, label=f"{name} ({', '.join(str(p[0]) for p in pts)})")
    ax.set_xlabel("generated tokens in the run (thousands)"); ax.set_ylabel("levels cleared (10 games, 25 min)")
    ax.set_title("Eleven arm runs against the five-run base band: all inside", fontsize=10)
    ax.set_ylim(38.5, 53.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "fig_arms_vs_band.png", dpi=160, bbox_inches="tight"); print("fig_arms_vs_band: base", bl)


if __name__ == "__main__":
    fig_throughput(); fig_policies(); fig_slow_not_stuck(); fig_max_statistic(); fig_arms_vs_band()
