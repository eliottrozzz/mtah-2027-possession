"""Tracking for the DFL sealed test games, in the shared format of metrica_tracking.py.

The DFL / Bundesliga open dataset (Bassek et al. 2025, CC BY 4.0, see external/README.md) has 7
matches of official tracking at 25 fps plus official events. We use it ONLY as a sealed test set:
nothing is fitted or tuned on it, and it is scored only on the final candidates.

    python external/dfl_tracking.py              # download, convert and check all 7 matches

The shared format
-----------------
One compressed .npz per match in data/external/dfl/ (gitignored). load_game() returns the same
dict as metrica_tracking.load_game():

    ball      (n_rows, 2)             x, y in 0-1 pitch units, NaN where the ball is missing
    players   (n_rows, n_players, 2)  x, y in 0-1, NaN where a player is not on the pitch
    team      (n_players,)            'H' or 'A'
    player    (n_players,)            DFL person id
    period    (n_rows,)               1 or 2, 0 where there is no tracking

DFL numbers frames from 10000 in the first half and 100000 in the second, so rows are not DFL
frame numbers. Row 0 is empty (as in Metrica), then every first-half frame, one empty row, then
every second-half frame. load_extras() gives, per row, the DFL frame number, the UTC time (used
to put the events on the same clock) and the provider's own per-frame BallStatus and
BallPossession, kept for comparison only and never handed to a model.

Coordinates: DFL gives metres from the centre spot; x runs along the pitch, y across it. We map
them to 0-1 with y flipped, so y = 0 is the same touchline as in the ATD-style 0-1 data. Referees
are dropped.

The check
---------
As in metrica_tracking.py: ball coverage, and how far the ball is from the passer when a pass is
logged. DFL events carry wall-clock times, not frames, so that distance also tests the clock.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
OUT_DIR = os.path.join(ROOT, "data", "external", "dfl")
RAW_DIR = os.path.join(OUT_DIR, "raw")
ARTICLE = "https://api.figshare.com/v2/articles/28196177"   # version 1, CC BY 4.0
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FPS = 25
SECTIONS = {"firstHalf": 1, "secondHalf": 2}

FRAMESET = re.compile(r'<FrameSet\b([^>]*)>')
ATTR = re.compile(r'([\w-]+)="([^"]*)"')
FRAME = re.compile(r'<Frame N="(\d+)" T="([^"]+)" X="([-\d.]+)" Y="([-\d.]+)"')
BALL_EXTRA = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


# -- files -------------------------------------------------------------------------------------

def manifest():
    """{file name: (download url, md5)} from Figshare, cached next to the data."""
    path = os.path.join(OUT_DIR, "figshare_metadata.json")
    if not os.path.exists(path):
        os.makedirs(OUT_DIR, exist_ok=True)
        urllib.request.urlretrieve(ARTICLE, path)
    with open(path) as handle:
        meta = json.load(handle)
    return {f["name"]: (f["download_url"], f["computed_md5"]) for f in meta["files"]}


def fetch(match, kind):
    """Local path of a match's 'matchinformation', 'events_raw' or 'positions_raw_observed' file,
    downloading and checking it the first time."""
    files = manifest()
    name = next(n for n in files if kind in n and n.endswith("DFL-MAT-{}.xml".format(match)))
    local = os.path.join(RAW_DIR, name)
    if not os.path.exists(local):
        os.makedirs(RAW_DIR, exist_ok=True)
        print("downloading", name)
        urllib.request.urlretrieve(files[name][0], local)
    digest = hashlib.md5()
    with open(local, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != files[name][1]:
        raise ValueError("{}: checksum does not match Figshare's".format(name))
    return local


def parse_time(text):
    """ISO time with offset -> UTC seconds. Tracking is in UTC, events in local time."""
    return datetime.datetime.fromisoformat(text).timestamp()


def match_info(match):
    """Home and guest team ids, pitch size in metres, and each player's team ('H'/'A')."""
    root = ET.parse(fetch(match, "matchinformation")).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    home = general.get("HomeTeamId")
    team_of = {}
    for team in root.iter("Team"):
        side = "H" if team.get("TeamId") == home else "A"
        for player in team.iter("Player"):
            team_of[player.get("PersonId")] = side
    return {"home": home, "guest": general.get("GuestTeamId"),
            "title": general.get("MatchTitle"), "competition": general.get("CompetitionName"),
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))), "team_of": team_of}


def read_events(match):
    """Every event as a dict: time (UTC s), tag (outer element), tags (all element names inside),
    attrs (merged attributes of the elements inside, outer ones winning), and the raw element.
    Sorted by time; the raw file is in delivery order, not time order. 'Delete' markers are
    dropped: they reference nothing and the events they removed are already gone from the file.
    """
    root = ET.parse(fetch(match, "events_raw")).getroot()
    events = []
    for event in root.iter("Event"):
        if not len(event) or event[0].tag == "Delete":
            continue
        inner = list(event[0].iter())
        attrs = {}
        for element in reversed(inner):
            attrs.update(element.attrib)
        events.append({"time": parse_time(event.get("EventTime")), "tag": event[0].tag,
                       "tags": [e.tag for e in inner], "attrs": attrs, "id": event.get("EventId"),
                       "element": event[0]})
    events.sort(key=lambda e: e["time"])
    return events


# -- conversion --------------------------------------------------------------------------------

def read_positions(match):
    """{(period, team id, person id): (frame N, x, y) arrays}, plus the ball's times and
    provider columns. Line based: the files are 350-420 MB with one element per line."""
    sets, ball_extra = {}, {}
    key, rows = None, None
    with open(fetch(match, "positions_raw_observed")) as handle:
        for line in handle:
            frame = FRAME.search(line)
            if frame:
                rows.append(frame.groups())
                if key[1] == "BALL":
                    ball_extra[key[0]].append(BALL_EXTRA.search(line).groups())
                continue
            head = FRAMESET.search(line)
            if head:
                attrs = dict(ATTR.findall(head.group(1)))
                key = (SECTIONS[attrs["GameSection"]], attrs["TeamId"], attrs["PersonId"])
                rows = sets.setdefault(key, [])
                if attrs["TeamId"] == "BALL":
                    ball_extra.setdefault(key[0], [])
    out = {}
    for key, rows in sets.items():
        n = np.array([int(r[0]) for r in rows])
        xy = np.array([[float(r[2]), float(r[3])] for r in rows])
        times = (np.array([parse_time(r[1]) for r in rows]) if key[1] == "BALL" else None)
        out[key] = (n, xy, times)
    return out, ball_extra


def convert(match):
    info = match_info(match)
    sets, ball_extra = read_positions(match)
    length, width = info["pitch"]

    # Row layout: row 0 empty, half 1, one empty row, half 2.
    offsets, start = {}, 1
    for period in (1, 2):
        frames = np.concatenate([n for (p, _, _), (n, _, _) in sets.items() if p == period])
        first, last = frames.min(), frames.max()
        offsets[period] = (first, start)
        start += last - first + 1 + (1 if period == 1 else 0)
    n_rows = start

    people = sorted({pid for (_, team, pid) in sets if team not in ("BALL", "referee")},
                    key=lambda pid: (info["team_of"][pid] != "H", pid))
    column = {pid: i for i, pid in enumerate(people)}

    out = {
        "ball": np.full((n_rows, 2), np.nan, dtype=np.float32),
        "players": np.full((n_rows, len(people), 2), np.nan, dtype=np.float32),
        "team": np.array([info["team_of"][pid] for pid in people]),
        "player": np.array(people),
        "period": np.zeros(n_rows, dtype=np.int8),
        "dfl_frame": np.full(n_rows, -1, dtype=np.int64),
        "time": np.full(n_rows, np.nan),
        "dfl_ball_status": np.full(n_rows, -1, dtype=np.int8),
        "dfl_ball_possession": np.full(n_rows, "", dtype="<U1"),
    }
    for period, (first, row0) in offsets.items():
        rows = np.arange(row0, row0 + max(n.max() for (p, _, _), (n, _, _) in sets.items()
                                          if p == period) - first + 1)
        out["period"][rows] = period
        out["dfl_frame"][rows] = first + rows - row0

    def to_unit(xy):
        return np.stack([xy[:, 0] / length + 0.5, 0.5 - xy[:, 1] / width], axis=1)

    for (period, team, pid), (n, xy, times) in sets.items():
        first, row0 = offsets[period]
        rows = n - first + row0
        if team == "referee":
            continue
        if team == "BALL":
            out["ball"][rows] = to_unit(xy)
            out["time"][rows] = times
            extra = np.array(ball_extra[period])
            out["dfl_ball_status"][rows] = extra[:, 1].astype(int)
            # BallPossession 1 = home, 2 = guest (checked against the kick-off in check()).
            out["dfl_ball_possession"][rows] = np.where(extra[:, 0] == "1", "H", "A")
        else:
            out["players"][rows, column[pid]] = to_unit(xy)

    # Fill any row the ball skipped with a time from the 25 Hz clock, so events can be placed.
    for period in (1, 2):
        rows = np.flatnonzero(out["period"] == period)
        known = ~np.isnan(out["time"][rows])
        out["time"][rows] = np.interp(rows, rows[known], out["time"][rows][known])
    return out


SHARED = ("ball", "players", "team", "player", "period")


def cached(match):
    path = os.path.join(OUT_DIR, "{}_tracking.npz".format(match))
    if not os.path.exists(path):
        np.savez_compressed(path, **convert(match))
    return path


def load_game(match):
    """Shared-format tracking for one match: exactly the keys a model may see."""
    with np.load(cached(match)) as data:
        return {key: data[key] for key in SHARED}


def load_extras(match):
    """Per-row DFL frame number, UTC time and provider columns. Never a model input."""
    with np.load(cached(match)) as data:
        return {key: data[key] for key in data.files if key not in SHARED}


def time_to_row(times, extras):
    """Rows nearest to the given UTC times (s), within the half they fall in."""
    t, period = extras["time"], None
    rows = np.flatnonzero(~np.isnan(t))
    index = np.clip(np.searchsorted(t[rows], times), 1, len(rows) - 1)
    before, after = rows[index - 1], rows[index]
    return np.where(np.abs(t[before] - times) <= np.abs(t[after] - times), before, after)


# -- check -------------------------------------------------------------------------------------

def check(match, data, extras):
    """Ball coverage, passer-to-ball distance at logged passes, and the provider columns' sense."""
    in_play = data["period"] > 0
    alive = extras["dfl_ball_status"] == 1
    ball_seen = ~np.isnan(data["ball"][:, 0])
    column = {pid: i for i, pid in enumerate(data["player"])}
    scale = np.array(load_info_pitch(match))

    passes = [(e["time"], e["attrs"]["Player"]) for e in read_events(match)
              if "Play" in e["tags"] and e["attrs"].get("Player") in column]
    rows = time_to_row(np.array([t for t, _ in passes]), extras)
    gap = np.array([(data["players"][r, column[p]] - data["ball"][r]) * scale
                    for r, (_, p) in zip(rows, passes)])
    distances = np.hypot(gap[:, 0], gap[:, 1])
    distances = distances[~np.isnan(distances)]

    print("{}  {:6d} frames  {:2d} players  ball seen {:5.1f}% of frames, {:5.1f}% of alive  "
          "provider alive {:4.1f}%  passer to ball at pass: median {:.1f} m, {:.0f}% within 2 m "
          "({} passes)".format(
              match, int(in_play.sum()), len(data["team"]), 100 * ball_seen[in_play].mean(),
              100 * ball_seen[alive].mean(), 100 * alive[in_play].mean(),
              float(np.median(distances)), 100 * float(np.mean(distances < 2.0)), len(distances)))
    return distances


def load_info_pitch(match):
    return match_info(match)["pitch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matches", nargs="+", default=list(MATCHES), choices=MATCHES)
    args = parser.parse_args()
    for match in args.matches:
        check(match, load_game(match), load_extras(match))
    print("tracking written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
