"""Drive one submission cycle end to end, stopping where judgement is needed.

The loop this project runs is: push a change, wait for Kaggle to build it,
spend the day's one submission, wait hours for a leaderboard score, write the
score down, decide what to try next. Only the last step needs a person. The
rest was being done by hand — including a click on kaggle.com that turns out to
have an API behind it (`competition_submit_code`), so nothing here requires a
browser.

Three stages, deliberately separate:

    python scripts/cycle.py push                 # build + upload, wait for the run
    python scripts/cycle.py submit --note "..."  # spends the day's slot
    python scripts/cycle.py watch                # poll for the score, record it

`push` is free and repeatable. `submit` is not: Kaggle allows one per day, and
a version that measures well locally can still lose on the leaderboard (v7 gained
3x locally and lost 3.6x on the board), so it prints what it is about to spend
the slot on and requires `--yes` to go through. `watch` writes the score into
SUBMISSIONS.md against the commit that was submitted, which is why `submit`
records that commit to a state file rather than reading HEAD later — by the time
a score lands, HEAD has usually moved.

    python scripts/cycle.py status               # slots left + recent scores
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cycle-state.json"
LOG = ROOT / "SUBMISSIONS.md"
COMPETITION = "arc-prize-2026-arc-agi-3"
OUTPUT_FILE = "submission.parquet"
PENDING = "_pending_"
# Kaggle allows one competition submission per day, per its own message.
DAILY_ALLOWANCE = 1

# Kernel directories this script knows how to push, and what has to run first.
# The agent target mirrors `make submit`: splice agent/my_agent.py into the
# notebook, then run preflight, which exits non-zero if the agent does not parse,
# cannot play, or the notebook has drifted out of sync with the source. Skipping
# either would upload a stale notebook and burn a day finding out.
TARGETS = {
    "agent": ("notebooks", ["scripts/build_notebook.py", "scripts/preflight.py"]),
    "duck": ("notebooks-duck", ["scripts/build_duck_notebook.py"]),
    # Identical notebook on a fresh slug, for when the primary kernel starts
    # erroring on every scored rerun regardless of what it contains.
    "duck-b": ("notebooks-duck-b", ["scripts/build_duck_notebook.py"]),
}


def api():
    """Authenticate using the project-local token, the way the Makefile does."""
    import os

    token = ROOT / ".kaggle" / "access_token"
    if token.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi

    client = KaggleApi()
    client.authenticate()
    return client


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()


def retry(call, attempts: int = 4, pause: float = 20.0, what: str = ""):
    """Call something flaky, and say so rather than dying on the first failure.

    Kaggle's endpoints fail independently: during one outage `kernels_list_files`
    and `competition_submissions` answered normally while `kernels_status`,
    `kernels_logs` and `GetKernel` all returned HTML that the client tried to
    parse as JSON, and the submission-limits endpoint returned 400. A push that
    has already landed should not be lost because the version lookup after it
    hit a bad minute.
    """
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — any transport failure is worth retrying
            last = exc
            if attempt + 1 < attempts:
                print(f"[cycle] {what or 'call'} failed ({type(exc).__name__}), "
                      f"retrying in {pause:.0f}s")
                time.sleep(pause)
    raise last


def kernel_id(directory: str) -> str:
    meta = json.loads((ROOT / directory / "kernel-metadata.json").read_text())
    return meta["id"]


def slots_left(client) -> tuple[int, int]:
    """(available now, used today), preferring the limits endpoint.

    That endpoint was seen returning 400 while submission listing still worked,
    so fall back to counting today's submissions. The daily allowance is 1, which
    is what makes the fallback usable at all.
    """
    # Counted from the submission list, which carries timestamps that can be
    # checked, with the limits endpoint used to corroborate. The endpoint has
    # returned 400 before, so something has to work without it.
    #
    # The day boundary is 00:00 **UTC**, and this is worth being explicit about:
    # the two were confused once here, JST having already rolled over while UTC
    # had not, which produced a confident report that the endpoint was wrong when
    # it was right.
    today = datetime.now(timezone.utc).date()
    subs = recent(client, 10)
    used = sum(1 for s in subs
               if getattr(s, "date", None) and s.date.date() == today)
    try:
        limits = client.competition_get_submission_limits(COMPETITION)
        if limits.num_today != used:
            print(f"[cycle] limits endpoint says {limits.num_today} used today, "
                  f"submission list says {used} — going with the list")
    except Exception as exc:  # noqa: BLE001
        print(f"[cycle] submission-limits endpoint unavailable ({type(exc).__name__})")
    return max(0, DAILY_ALLOWANCE - used), used


def recent(client, n: int = 6):
    return retry(lambda: client.competition_submissions(COMPETITION, page_size=n),
                 what="submission listing") or []


def _submission_landed(client, note: str, started) -> bool:
    """Did this submission reach Kaggle despite the client raising?

    Conservative on purpose: a false positive here would delete `.draw-next` for
    a draw that never happened, silently skipping an approved arm. So it returns
    True only for a submission that both carries this note and is newer than the
    attempt — and False whenever the listing itself cannot be read, since "we do
    not know" must not be reported as "it worked".
    """
    try:
        subs = recent(client, 5)
    except Exception:  # noqa: BLE001
        return False
    want = (note or "").strip()[:200]
    for s in subs:
        when = getattr(s, "date", None)
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when is None or when < started:
            continue
        if (getattr(s, "description", "") or "").strip()[:200] == want:
            return True
    return False


# --------------------------------------------------------------------------- push


def stage_push(args) -> None:
    directory, prepare = TARGETS[args.target]
    for step in prepare:
        print(f"[cycle] {step}")
        subprocess.run([sys.executable, str(ROOT / step)], check=True, cwd=ROOT)

    client = api()
    kid = kernel_id(directory)
    print(f"[cycle] pushing {directory}/ -> {kid}")
    client.kernels_push(str(ROOT / directory))

    # Kaggle assigns the version number server-side; read it back rather than
    # trusting the push response, which has changed shape between CLI releases.
    # Record what was pushed before waiting on anything else. The version lookup
    # and the status poll both go through endpoints that have failed while the
    # push itself succeeded, and a state file that still names the previous
    # version is how a stale configuration gets resubmitted.
    try:
        version = retry(lambda: current_version(client, kid), what="version lookup")
        save_state({"kernel": kid, "version": version, "target": args.target})
        print(f"[cycle] pushed version {version}")
    except Exception as exc:  # noqa: BLE001
        version = None
        save_state({"kernel": kid, "version": None, "target": args.target})
        print(f"[cycle] pushed, but the version lookup is down ({type(exc).__name__}). "
              f"Read it off the kernel page and pass --version when submitting.")

    print("[cycle] waiting for the Kaggle run to finish", flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            status = client.kernels_status(kid)
        except Exception as exc:  # noqa: BLE001
            print(f"[cycle] status unavailable ({type(exc).__name__}); still waiting")
            time.sleep(args.poll)
            continue
        name = str(getattr(status, "status", ""))
        if "COMPLETE" in name:
            print(f"[cycle] complete — version {version if version else '(unknown)'}")
            return
        if "ERROR" in name or "CANCEL" in name:
            raise SystemExit(

                
            )
        time.sleep(args.poll)
    raise SystemExit(f"[cycle] still running after {args.timeout}s — check by hand")


def current_version(client, kid: str) -> int:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    owner, slug = kid.split("/", 1)
    with client.build_kaggle_client() as kaggle:
        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = owner, slug
        response = kaggle.kernels.kernels_api_client.get_kernel(request)
    return int(response.metadata.current_version_number)


# ------------------------------------------------------------------------- submit


def stage_submit(args) -> None:
    client = api()
    state = load_state()
    kid = args.kernel or state.get("kernel")
    version = args.version or state.get("version")
    if not kid or not version:
        raise SystemExit("[cycle] no kernel/version — run `cycle.py push` first")

    allowed, used = slots_left(client)
    print(f"[cycle] submission slots: {allowed} available, {used} used today")
    if allowed < 1:
        raise SystemExit("[cycle] no slot available today — try again after 00:00 UTC")

    # The state file only advances when the push went through this script, so a push
    # made any other way leaves it behind, and submitting a stale version silently
    # re-runs a configuration that already has a score -- a whole day for no
    # information, and invisible, because the submission succeeds and looks right.
    #
    # But an older version is often exactly what is wanted. Diagnostics and
    # submissions share one kernel, so the moment a diagnostic is pushed the version
    # carrying the configuration worth submitting is no longer the latest. Naming it
    # with --version is a deliberate act; refusing that was the guard working against
    # the thing it was written to protect.
    live = retry(lambda: current_version(client, kid), what="version lookup")
    if int(version) != live:
        if args.version is None:
            raise SystemExit(
                f"[cycle] state says version {version} but {kid} is on {live}.\n"
                f"        Pass --version {version} to submit that one on purpose, or "
                f"--version {live} for the newest."
            )
        print(f"[cycle] submitting version {version} while {kid} is on {live} "
              f"— deliberate, since --version was given")

    sha = git("rev-parse", "--short", "HEAD")
    note = args.note or git("log", "-1", "--format=%s")
    print("\n  kernel   :", kid)
    print("  version  :", version)
    print("  file     :", OUTPUT_FILE)
    print("  commit   :", sha)
    print("  note     :", note)
    if not args.yes:
        print("\n[cycle] this spends today's only submission. Re-run with --yes.")
        return

    # `competition_submit_code` can raise *after* Kaggle has accepted the
    # submission -- a DNS failure or a dropped connection on the response leaves
    # the caller believing it failed while the submission is live. That is not
    # hypothetical: on 2026-08-16 the null-control replicate landed at 01:06:55
    # UTC and this call raised `RemoteDisconnected` anyway.
    #
    # Treating that as a failure costs three things, and all three were observed:
    # `daily_draw.sh` deletes `.draw-next` only on a zero exit, so the same
    # approved arm is submitted again at the next slot -- a duplicate draw, and a
    # second submission of a build that was signed off for one; `save_state` never
    # records the commit, so the score cannot be tied back to it; and `add_row`
    # never writes the SUBMISSIONS.md row, which is why the reconciler later says
    # "no unscored row for <sha>" and drops the score on the floor.
    #
    # So on failure, ask Kaggle whether it actually landed before believing the
    # exception. Match on the note *and* on the submission being newer than this
    # attempt, since notes repeat across replicate draws.
    started = datetime.now(timezone.utc)
    try:
        client.competition_submit_code(
            file_name=OUTPUT_FILE,
            message=note,
            competition=COMPETITION,
            kernel=kid,
            kernel_version=int(version),
        )
    except Exception as exc:  # noqa: BLE001
        if not _submission_landed(client, note, started):
            raise
        print(f"[cycle] the submit call failed ({type(exc).__name__}), but Kaggle "
              f"has the submission — continuing with the bookkeeping rather than "
              f"leaving it half-recorded.")
    save_state({
        "kernel": kid, "version": version, "target": state.get("target"),
        "sha": sha, "note": note, "submitted_at": datetime.now().isoformat(timespec="seconds"),
    })
    add_row(sha, note, args.local)
    print(f"[cycle] submitted. Next:  python scripts/cycle.py watch")


# -------------------------------------------------------------------------- watch


def stage_watch(args) -> None:
    client = api()
    state = load_state()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        subs = recent(client, 3)
        if subs:
            newest = subs[0]
            status = str(getattr(newest, "status", ""))
            score = getattr(newest, "public_score", None)
            stamp = datetime.now().strftime("%H:%M:%S")
            if "COMPLETE" in status and score:
                print(f"[{stamp}] score {score}")
                if state.get("sha"):
                    set_score(state["sha"], score)
                return
            if "ERROR" in status:
                # A failed submission does not spend the day's allowance. One
                # participant reported seven errored submissions and one
                # successful one on the same calendar day, and was blocked only
                # after the one that completed. So this is a prompt to resubmit,
                # not a lost day — and saying so matters, because the error text
                # is generic and the instinct is to assume the slot is gone.
                allowed, used = slots_left(client)
                detail = getattr(newest, "error_description", "") or "(no detail given)"
                print(f"\n[cycle] the submission ERRORED: {detail}")
                print(f"[cycle] slots: {allowed} available, {used} used — an errored "
                      f"submission does not count against the daily allowance, so "
                      f"resubmit rather than waiting for tomorrow.")
                if allowed:
                    print(f"[cycle] if it errors repeatedly on this kernel, the kernel "
                          f"itself may be the problem: a kernel iterated over many "
                          f"versions has been reported to fail every scored rerun "
                          f"regardless of its code, while the same code on a fresh "
                          f"slug ran first time. Try:  make cycle-push TARGET=duck-b")
                raise SystemExit(1)
            print(f"[{stamp}] {status.split('.')[-1]}", flush=True)
        time.sleep(args.poll)
    raise SystemExit(f"[cycle] no score after {args.timeout}s")


# ------------------------------------------------------------------- SUBMISSIONS.md


def add_row(sha: str, note: str, local: str | None) -> None:
    lines = LOG.read_text().splitlines()
    lines.append(f"| {date.today().isoformat()} | {sha} | {note} | {local or '—'} | {PENDING} |")
    LOG.write_text("\n".join(lines) + "\n")
    print(f"[cycle] logged row for {sha}")


def set_score(sha: str, score: str) -> None:
    """Complete the newest unscored row for this commit, matching log_submission."""
    lines = LOG.read_text().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        cells = [c.strip() for c in lines[i].split("|")]
        if len(cells) >= 6 and cells[2] == sha and cells[5] == PENDING:
            cells[5] = str(score)
            lines[i] = "| " + " | ".join(cells[1:6]) + " |"
            LOG.write_text("\n".join(lines) + "\n")
            print(f"[cycle] SUBMISSIONS.md: {sha} -> {score}")
            return
    print(f"[cycle] no unscored row for {sha}; score {score} not recorded")


# -------------------------------------------------------------------------- status


def stage_status(args) -> None:
    client = api()
    allowed, used = slots_left(client)
    print(f"slots: {allowed} available now, {used} used today")
    state = load_state()
    if state:
        print(f"last cycle: {json.dumps(state, indent=2)}")
    print("\nrecent submissions:")
    for sub in recent(client, args.n):
        status = str(getattr(sub, "status", "")).split(".")[-1]
        score = getattr(sub, "public_score", None) or "—"
        when = getattr(sub, "date", "")
        desc = (getattr(sub, "description", "") or "")[:44]
        print(f"  {str(when)[:19]}  {status:<10} {score:>6}  {desc}")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("push", help="build + upload, wait for the Kaggle run")
    p.add_argument("--target", choices=sorted(TARGETS), default="agent")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--timeout", type=int, default=7200)
    p.set_defaults(func=stage_push)

    p = sub.add_parser("submit", help="spend the day's submission (needs --yes)")
    p.add_argument("--note", default=None)
    p.add_argument("--local", default=None, help="local benchmark result, if measured")
    p.add_argument("--kernel", default=None)
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--yes", action="store_true", help="actually spend the slot")
    p.set_defaults(func=stage_submit)

    p = sub.add_parser("watch", help="poll until the leaderboard score lands")
    # The rerun takes hours -- the duck harness alone budgets 2.2 h per game, and
    # our own agent runs an eight-hour wall clock -- so a short interval just
    # spends API calls on an answer that cannot have changed yet.
    p.add_argument("--poll", type=int, default=1800)
    p.add_argument("--timeout", type=int, default=43200)
    p.set_defaults(func=stage_watch)

    p = sub.add_parser("status", help="slots left, last cycle, recent scores")
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(func=stage_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
