"""Write a sheet for checking the DFL event sync against public match video, by hand.

    python external/dfl_video_check.py          # writes external/dfl_video_check.md

For every goal, the last touches before it, with the game clock to find the moment in the
highlights and the gaps between touches as the raw log and the snap give them. Highlights show no
tracking clock, so the check uses gaps between touches, which the video shows directly. It only
measures: the sync method was frozen (commit b79114c) before this sheet existed.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl_sync as ds  # noqa: E402
import dfl_tracking as dt  # noqa: E402

# Params
TOUCHES_BEFORE = 4          # touches listed before each goal, the shot included
LOOKBACK_S = 15.0           # never further back than this
OUT = os.path.join(HERE, "dfl_video_check.md")
ACTION = {"Play": "pass", "ShotAtGoal": "shot", "OtherBallAction": "touch",
          "BallClaiming": "keeper/claim"}


def names(match):
    root = ET.parse(dt.fetch(match, "matchinformation")).getroot()
    general = root.find(".//General")
    people = {p.get("PersonId"): p.get("Shortname") or p.get("LastName")
              for p in root.iter("Player")}
    teams = {t.get("TeamId"): t.get("TeamName") for t in root.iter("Team")}
    return general, people, teams


def clock(game, row):
    """Game clock as the broadcast shows it: mm:ss.s, second half from 45:00."""
    period = game["period"][row]
    start = np.flatnonzero(game["period"] == period)[0]
    seconds = (row - start) / ds.FPS + (45 * 60 if period == 2 else 0)
    return "{:02d}:{:04.1f}".format(int(seconds // 60), seconds % 60), period


def action(event):
    if event["tag"] in ("ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "Penalty"):
        return event["tag"]
    for tag in event["tags"]:
        if tag in ACTION:
            return ACTION[tag] if tag != "Play" or "Cross" not in event["tags"] else "cross"
    return event["tag"]


def match_section(match):
    game, extras = dt.load_game(match), dt.load_extras(match)
    events = dt.read_events(match)
    ds.snap(game, extras, events)
    general, people, teams = names(match)
    touches = [e for e in events if ds.touch_player(e) in people]
    goals = [i for i, e in enumerate(touches) if "SuccessfulShot" in e["tags"]]

    lines = ["## {} – {} ({}, {}, {})".format(
        *general.get("MatchTitle").split(":"), general.get("CompetitionName"),
        general.get("KickoffTime")[:10], general.get("Result")), ""]
    lines.append("Match {}. Logged goals: {}; final score {}.".format(
        match, len(goals), general.get("Result")))
    lines.append("")
    for n, gi in enumerate(goals, 1):
        shot = touches[gi]
        chain = [e for e in touches[max(0, gi - TOUCHES_BEFORE + 1):gi + 1]
                 if (shot["row"] - e["row"]) / ds.FPS <= LOOKBACK_S]
        when, period = clock(game, shot["row"])
        lines.append("### Goal {}: {} ({}), shot at about {} on the clock (half {})".format(
            n, people[ds.touch_player(shot)], teams.get(shot["attrs"].get("Team"), "?"),
            when, period))
        lines.append("")
        lines.append("| # | Player | Action | Clock (snapped) | Gap, raw log | Gap, snapped "
                     "| Gap, your timing | Snapped? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k, e in enumerate(chain):
            if k == 0:
                raw_gap = snap_gap = "–"
            else:
                raw_gap = "{:.2f} s".format((e["raw_row"] - chain[k - 1]["raw_row"]) / ds.FPS)
                snap_gap = "{:.2f} s".format((e["row"] - chain[k - 1]["row"]) / ds.FPS)
            lines.append("| {} | {} | {} | {} | {} | {} |  | {} |".format(
                k + 1, people[ds.touch_player(e)], action(e), clock(game, e["row"])[0],
                raw_gap, snap_gap, "yes" if e["snapped"] else "no (lag only)"))
        lines.append("")
    return lines


def main():
    lines = [
        "# DFL sync: video check sheet",
        "",
        "Checks whether the snapped event times match what really happened, using public match "
        "video. The sync method was frozen (commit b79114c) before this sheet was made, so this "
        "only measures it. Nothing gets tuned on the result.",
        "",
        "## How to use it",
        "",
        "1. Find the match's highlights (Bundesliga / 2. Bundesliga YouTube channel, DAZN or "
        "Sky). Each goal below gives the game clock of the shot; highlights show it on screen.",
        "2. For each goal, step through the build-up frame by frame (VLC: press E) and time the "
        "gap between each listed touch and the one before it. Write it in **Gap, your timing**.",
        "3. Skip a touch if the clip doesn't show it, or write `?`.",
        "",
        "We compare your gaps with the **raw log** and **snapped** columns. Snapping is working if "
        "the snapped gaps are clearly closer to yours, typically within about 0.2 s.",
        "",
        "Notes:",
        "",
        "* Clock times come from the tracking (seconds since that half's kick-off), so they may "
        "differ from the broadcast clock by a second or so. Use them to find the moment, not to "
        "check it.",
        "* Second-half times run from 45:00. Stoppage time shows as 45:xx or 90:xx and beyond, "
        "where the broadcast shows 45+x or 90+x.",
        "* A touch marked \"no (lag only)\" couldn't be snapped and keeps the lag-corrected log "
        "time; its gaps are expected to be worse.",
        "* Penalties and own goals may be logged differently; if a goal from the final score is "
        "missing below, that's why.",
        "",
    ]
    for match in dt.MATCHES:
        lines += match_section(match)
    with open(OUT, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("written", os.path.relpath(OUT, dt.ROOT))


if __name__ == "__main__":
    main()
