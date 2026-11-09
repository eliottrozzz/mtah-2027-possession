"""Answer key for the external games: per-frame H/A/D labels built from Metrica's open event data.

We may not train or tune on the provided match, so every parameter is tuned on other games
instead. Metrica's three open sample games (github.com/metrica-sports/sample-data) have tracking
and event data, but no possession labels. This script derives those labels from the events,
following the rules in POSSESSION_RULES.md, so the official metric can score a model on them.

    python external/metrica_labels.py                     # restart pass ends D at its reception
    python external/metrica_labels.py --restart kick      # ... or at the kick

The event files are downloaded on first run into data/external/metrica/, which is gitignored like
the rest of data/. Labels are written next to them, one frame,label CSV per game.

How the rules map onto Metrica events
-------------------------------------
Metrica's own definition of possession (events-definitions.pdf in their repo) is the one the
public ground truth shows: a player is in possession as soon as they touch the ball, and an
opponent's touch that changes nothing is not a loss. So:

  PASS, SHOT, RECOVERY, CARRY, SET PIECE   that team has the ball from the event's start frame
  CHALLENGE, BALL LOST, CARD                no change on their own; the switch comes with the
                                            other team's next RECOVERY or PASS
  BALL OUT, FAULT RECEIVED, CHALLENGE with  D: play has stopped
  a FAULT subtype, shot GOAL/OUT, OFFSIDE,
  FORCED, HAND BALL, END HALF
  SET PIECE after D                         D until the restart pass is received (--restart
                                            reception) or from the kick (--restart kick)
  RECOVERY after D (keeper rolling the ball alive from the RECOVERY, before the goal kick,
  out, Metrica's goal-kick marker)          as the public ground truth does at 12:15

Deflections are mostly not events in Metrica's data, so they are ignored without any rule. The
exceptions are blocks and saves, which Metrica logs as RECOVERY: a block never switches
possession, and a save only does when the defending team plays the ball next.

Frame numbering
---------------
Label row i is Metrica frame i, the same counter the sample tracking uses (it starts at 1, so
row 0 is always X). Frames before the first event of a half, between halves and after the last
event are X, so the scorer's valid_runs() reads the halves straight off the labels.

This answer key is frozen before any model is scored on it. Change a rule for a football reason,
never because a model scores better under it.
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
OUT_DIR = os.path.join(ROOT, "data", "external", "metrica")
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawEventsData.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawEventsData.csv"],
    3: ["Sample_Game_3/Sample_Game_3_events.json", "Sample_Game_3/Sample_Game_3_metadata.xml"],
}
FPS = 25

# Events that put a team on the ball. Everything else (CHALLENGE, BALL LOST, CARD) leaves
# possession where it was: a challenge is followed by a RECOVERY if it actually won the ball.
POSSESSION_TYPES = {"PASS", "SHOT", "RECOVERY", "CARRY", "SET PIECE"}

# Subtypes that stop play. Matched as whole subtype names, so "OUT" never matches "THROUGH BALL".
STOPPING_SUBTYPES = {"GOAL", "OUT", "OFFSIDE", "FORCED", "HAND BALL", "END HALF"}


def fetch(game):
    """Download a game's event files unless they are already there. Returns the local paths."""
    paths = []
    for rel in FILES[game]:
        local = os.path.join(OUT_DIR, os.path.basename(rel))
        if not os.path.exists(local):
            os.makedirs(OUT_DIR, exist_ok=True)
            print("downloading", rel)
            urllib.request.urlretrieve(BASE_URL + "/" + rel, local)
        paths.append(local)
    return paths


def subtype_names(subtypes):
    """Game 3 stores subtypes as None, a dict or a list of dicts. Games 1-2 join them with '-'."""
    if not subtypes:
        return set()
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return {s["name"].upper() for s in subtypes}


def home_team_game3(meta_path):
    """Name of the home team: the first Team in the EPTS metadata, as in starter/baseline.py."""
    root = ET.parse(meta_path).getroot()
    return root.find(".//Teams/Team").findtext("Name")


def load_events(game):
    """Events as dicts with team ('H'/'A'/None), type, subtypes (a set), period, start, end."""
    paths = fetch(game)
    events = []
    if game in (1, 2):
        with open(paths[0]) as handle:
            for row in csv.DictReader(handle):
                events.append({
                    "team": {"Home": "H", "Away": "A"}.get(row["Team"]),
                    "type": row["Type"].strip().upper(),
                    "sub": {s.strip().upper() for s in row["Subtype"].split("-") if s.strip()},
                    "period": int(row["Period"]),
                    "start": int(row["Start Frame"]),
                    "end": int(row["End Frame"]),
                })
    else:
        home = home_team_game3(paths[1])
        with open(paths[0]) as handle:
            for row in json.load(handle)["data"]:
                team = (row.get("team") or {}).get("name")
                events.append({
                    "team": None if team is None else ("H" if team == home else "A"),
                    "type": row["type"]["name"].strip().upper(),
                    "sub": subtype_names(row.get("subtypes")),
                    "period": int(row["period"]),
                    "start": int(row["start"]["frame"]),
                    "end": int(row["end"]["frame"] or 0),
                })
    # Stable sort: events sharing a start frame keep the order Metrica logged them in, which is
    # the order they happened (SET PIECE before its restart PASS, BALL LOST before the RECOVERY).
    events.sort(key=lambda e: (e["period"], e["start"]))
    return events


def end_frame(event):
    """End frame, falling back to the start where Metrica leaves it empty (0) or earlier."""
    return max(event["start"], event["end"])


def stop_frame(event):
    """Frame at which an event stops play, or None if it does not.

    A ball crossing a line, a goal or a shot going out stops play at the event's end (the ball
    arriving); a foul, an offside flag or the referee stepping in stops it where it is logged.
    """
    if event["type"] == "BALL OUT":
        return end_frame(event)
    if event["type"] == "FAULT RECEIVED":
        return event["start"]
    if event["type"] == "CHALLENGE" and "FAULT" in event["sub"]:
        # Metrica's FAULT subtype means the referee awarded a free kick, and many fouls are
        # logged only this way, with no FAULT RECEIVED. Advantage is a separate subtype.
        return event["start"]
    hits = event["sub"] & STOPPING_SUBTYPES
    if not hits:
        return None
    if hits & {"GOAL", "OUT", "OFFSIDE"}:
        return end_frame(event)
    return event["start"]


def label_period(events, labels, restart):
    """Write H/A/D into labels for one half, as a sequence of 'from this frame on' changes.

    Each change overwrites everything after it, and changes are made in event order, so a later
    event always wins. The half starts dead: the kick-off is a restart like any other.
    """
    first, last = events[0]["start"], max(end_frame(e) for e in events)

    def change(frame, label, until=None):
        labels[max(frame, first):last + 1 if until is None else until] = label

    change(first, "D")
    state = "D"
    holder = None  # team that last had the ball, kept through stoppages
    awaiting = None  # restarting team whose restart kick has been taken but not yet received
    stopped_at = first  # frame the current stoppage began

    for event in events:
        team, kind = event["team"], event["type"]

        if kind in POSSESSION_TYPES and team is not None:
            if state == "D" and event["start"] < stopped_at:
                # Logged after the stoppage but started before it, like the carry that took the
                # ball out in Game 3. It can change who had the ball, never restart play.
                if team != holder:
                    change(event["start"], team, until=stopped_at)
                    holder = team
                continue

            if state != "D":
                # Live ball: the first deliberate touch by the other team switches possession.
                if kind == "SET PIECE":
                    # A restart with no logged stoppage before it: play must have stopped.
                    change(event["start"], "D")
                    state, awaiting, stopped_at = "D", None, event["start"]
                elif team != state:
                    change(event["start"], team)
                    state = team

            if state == "D" and awaiting is None:
                if kind == "SET PIECE":
                    # The kick itself; the ball comes alive at the restart pass below.
                    awaiting = team
                    if restart == "kick":
                        change(event["start"], team)
                        state, awaiting = team, None
                elif kind == "PASS":
                    # A restart pass with no SET PIECE logged, e.g. a goal kick.
                    change(end_frame(event) if restart == "reception" else event["start"], team)
                    state = team
                else:
                    # A RECOVERY in a stoppage is Metrica's marker for the ball being brought
                    # back into play other than by a SET PIECE: usually the keeper rolling it out
                    # before a goal kick, a median 2.4 s before the kick. The public ground truth
                    # agrees: at 12:15 it ends D when the keeper rolls the ball to a defender,
                    # before the kick. A SHOT or CARRY means play is already running.
                    change(event["start"], team)
                    state = team
            elif state == "D":
                if kind == "SET PIECE":
                    awaiting = team  # retaken: still waiting for the restart pass
                    continue
                if kind == "PASS" and team == awaiting:
                    # The restart pass: alive at its reception, or at the kick.
                    change(end_frame(event) if restart == "reception" else event["start"], team)
                else:
                    change(event["start"], team)  # intercepted, or a direct shot
                state, awaiting = team, None

        if state != "D":
            holder = state

        frame = stop_frame(event)
        if frame is not None:
            change(frame, "D")
            state, awaiting, stopped_at = "D", None, frame


def demote_blocks_and_saves(events):
    """Turn blocks, and saves the shooting team recovers, into deflections that switch nothing.

    Metrica logs a block or save as a RECOVERY by the defending player. A block is a deflection,
    so it never switches on its own: if the defenders then play the ball, that touch switches it.
    A save switches at the save only when the defenders play the ball next (a catch, or a parry
    to a teammate); when the shooting team gets the rebound, possession never left them.
    """
    for i, event in enumerate(events):
        if event["type"] != "RECOVERY" or not event["sub"] & {"BLOCKED", "SAVED"}:
            continue
        following = next((e for e in events[i + 1:]
                          if e["type"] in POSSESSION_TYPES and e["team"] is not None), None)
        keeps = following is not None and following["team"] == event["team"]
        if "BLOCKED" in event["sub"] or not keeps:
            event["type"] = "DEFLECTION"


def label_game(game, restart):
    """Per-frame labels for one game, as a string array indexed by Metrica frame."""
    events = load_events(game)
    demote_blocks_and_saves(events)
    n_frames = max(end_frame(e) for e in events) + 1
    labels = np.full(n_frames, "X", dtype="<U1")
    for period in sorted({e["period"] for e in events}):
        label_period([e for e in events if e["period"] == period], labels, restart)
    return labels


def write_labels(labels, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))


def describe(game, labels):
    """One line of sanity numbers per game: if these look wrong, a rule is wrong."""
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    stats = pm.possession_stats(codes, runs)
    graded = codes != pm.X
    dead = float(np.mean(codes[graded] == pm.D))
    print("game {}  {:6.1f} min graded  {:4d} possessions  mean {:5.1f} s  median {:4.1f} s  "
          "under 1 s {:3d}  home share {:.2f}  dead {:.2f}".format(
              game, graded.sum() / FPS / 60.0, len(spells), stats["mean_duration_s"],
              float(np.median(lengths)), int(np.sum(lengths < 1.0)), stats["home_share"], dead))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--restart", choices=("reception", "kick"), default="reception",
                        help="when a restart brings the ball back to life (POSSESSION_RULES.md)")
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    args = parser.parse_args()

    for game in args.games:
        labels = label_game(game, args.restart)
        path = os.path.join(OUT_DIR, "game{}_labels_{}.csv".format(game, args.restart))
        write_labels(labels, path)
        describe(game, labels)
    print("labels written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
