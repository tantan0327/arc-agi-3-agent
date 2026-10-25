"""Append (or complete) a row in `SUBMISSIONS.md`.

Over a two-month competition the leaderboard alone can't tell you *which*
change helped — by the time a score lands you've usually moved on. This keeps
a one-line-per-submission record tying each leaderboard score back to the exact
commit that produced it.

Typical two-step use, matching the two-phase Kaggle flow:

    make submit                                  # Phase A
    make log-submission NOTE="predicate goals"   # record what went up
    # ...click "Submit to Competition", wait for the leaderboard...
    make log-submission SCORE=0.043              # fill in the result

The second call finds the most recent row for the current commit that has no
score yet and completes it, rather than appending a duplicate.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "SUBMISSIONS.md"

HEADER = """# Submission log

One row per Kaggle submission. `make log-submission NOTE="..."` adds a row
after `make submit`; `make log-submission SCORE=...` fills in the leaderboard
score once Phase B finishes.

| Date | Commit | Change | Local | LB score |
|------|--------|--------|-------|----------|
"""

PENDING = "_pending_"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--note", default=None, help="What changed in this submission.")
    p.add_argument(
        "--score",
        default=None,
        help="Leaderboard score. Completes the newest unscored row for this commit.",
    )
    p.add_argument("--local", default=None, help="Local benchmark result, if measured.")
    args = p.parse_args()

    sha = git("rev-parse", "--short", "HEAD")
    if not LOG.exists():
        LOG.write_text(HEADER)

    lines = LOG.read_text().splitlines()

    if args.score:
        # Complete the newest unscored row for this commit, newest last.
        for i in range(len(lines) - 1, -1, -1):
            cells = [c.strip() for c in lines[i].split("|")]
            if len(cells) >= 6 and cells[2] == sha and cells[5] == PENDING:
                cells[5] = args.score
                if args.local:
                    cells[4] = args.local
                lines[i] = "| " + " | ".join(cells[1:6]) + " |"
                LOG.write_text("\n".join(lines) + "\n")
                print(f"[log_submission] Scored {sha}: {args.score}")
                return
        print(
            f"[log_submission] No unscored row for {sha}. "
            f"Add one first:  make log-submission NOTE=\"...\""
        )
        raise SystemExit(1)

    note = args.note or git("log", "-1", "--format=%s")
    row = (
        f"| {date.today().isoformat()} | {sha} | {note} | "
        f"{args.local or '—'} | {PENDING} |"
    )
    LOG.write_text("\n".join(lines + [row]) + "\n")
    print(f"[log_submission] Logged {sha}: {note}")


if __name__ == "__main__":
    main()
