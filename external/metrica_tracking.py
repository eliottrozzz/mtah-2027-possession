"""Tracking for the external games, in one shared format, checked against their events.

Metrica's three open sample games come in two tracking formats: games 1 and 2 as one CSV per
team, game 3 in the same FIFA/EPTS text format as the provided match. This script reads both into
the same arrays, so everything downstream (features, noise, models) is written once.

    python external/metrica_tracking.py           # download, convert and check all three games

The shared format
-----------------
One compressed .npz per game in data/external/metrica/ (gitignored), indexed by Metrica frame so
row i lines up with row i of the answer key from metrica_labels.py:

    ball      (n_frames, 2)             x, y in 0-1 pitch units, NaN where the ball is missing
    players   (n_frames, n_players, 2)  x, y in 0-1, NaN where a player is not on the pitch
    team      (n_players,)              'H' or 'A'
    player    (n_players,)              Metrica's player id within the game
    period    (n_frames,)               1 or 2, 0 where there is no tracking

Coordinates stay normalised like the provided ATD; multiply by PITCH for metres. Row 0 is empty
because Metrica frames count from 1.

The check
---------
The answer key is only usable if the tracking and the events share a clock. For every pass we
measure how far the ball is from the passer at the pass's start frame: with aligned frames that
is about a metre. We also report how often the ball is missing, which noise has to match later.
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

# Params
OUT_DIR = os.path.join(ROOT, "data", "external", "metrica")
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawEventsData.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawEventsData.csv"],
    3: ["Sample_Game_3/Sample_Game_3_tracking.txt",
        "Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_events.json"],
}
PITCH = (105.0, 68.0)  # metres, as Metrica's README states for all three games


def fetch(game):
    """Download a game's files unless they are already there. Returns the local paths."""
    paths = []
    for rel in FILES[game]:
        local = os.path.join(OUT_DIR, os.path.basename(rel))
        if not os.path.exists(local):
            os.makedirs(OUT_DIR, exist_ok=True)
            print("downloading", rel)
            urllib.request.urlretrieve(BASE_URL + "/" + rel, local)
        paths.append(local)
    return paths


def read_team_csv(path):
    """One team's CSV: (frames, periods, player names, positions (n, players, 2), ball (n, 2)).

    Three header rows; the third names the columns, with each player's name over their x column
    and an empty name over their y column. The ball is the last pair, in both teams' files.
    """
    with open(path) as handle:
        rows = csv.reader(handle)
        next(rows)
        next(rows)
        header = next(rows)
        data = np.array([[float(v) for v in row] for row in rows], dtype=np.float64)
    names = [header[i] for i in range(3, len(header) - 2, 2)]
    xy = data[:, 3:3 + 2 * len(names)].reshape(len(data), len(names), 2)
    ball = data[:, -2:]
    return data[:, 1].astype(int), data[:, 0].astype(int), names, xy, ball


def load_csv_game(game):
    """Games 1 and 2: merge the home and away CSVs, which share frame numbers."""
    home_path, away_path, _ = fetch(game)
    frames, periods, home_names, home_xy, ball = read_team_csv(home_path)
    away_frames, _, away_names, away_xy, _ = read_team_csv(away_path)
    if not np.array_equal(frames, away_frames):
        raise ValueError("game {}: home and away CSVs have different frames".format(game))

    n = frames.max() + 1
    out = empty(n, len(home_names) + len(away_names))
    out["ball"][frames] = ball
    out["players"][frames] = np.concatenate([home_xy, away_xy], axis=1)
    out["period"][frames] = periods
    out["team"] = np.array(["H"] * len(home_names) + ["A"] * len(away_names))
    out["player"] = np.array(home_names + away_names)
    return out


def load_epts_game(game):
    """Game 3: the FIFA/EPTS text format, with the column layout read from the metadata.

    Each line is "frame:x,y;x,y;...:ball_x,ball_y". Which player each x,y pair belongs to changes
    with substitutions, so the metadata gives one layout per frame range.
    """
    raw_path, meta_path, _ = fetch(game)
    root = ET.parse(meta_path).getroot()

    home_id = root.find(".//Teams/Team").get("id")
    team_of = {p.get("id"): ("H" if p.get("teamId") == home_id else "A")
               for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): int(p.findtext("Value"))
              for p in root.iter("ProviderParameter") if p.findtext("Value", "").isdigit()}

    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[ref.get("playerChannelId")]
                 for ref in spec.iter("PlayerChannelRef")
                 if ref.get("playerChannelId").endswith("_x")]
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    player_ids = sorted({pid for _, _, order in layouts for pid in order})
    column = {pid: i for i, pid in enumerate(player_ids)}

    n = max(end for _, end, _ in layouts) + 1
    out = empty(n, len(player_ids))
    out["team"] = np.array([team_of[pid] for pid in player_ids])
    out["player"] = np.array(player_ids)

    layout_index = 0
    with open(raw_path) as handle:
        for line in handle:
            frame_text, players_text, ball_text = line.strip().split(":")
            frame = int(frame_text)
            while frame > layouts[layout_index][1]:
                layout_index += 1
            cols = [column[pid] for pid in layouts[layout_index][2]]
            pairs = [group.split(",") for group in players_text.split(";")]
            out["players"][frame, cols] = np.array(pairs, dtype=np.float64)
            out["ball"][frame] = np.array(ball_text.split(","), dtype=np.float64)

    frames = np.arange(n)
    out["period"][(frames >= params["first_half_start"]) & (frames <= params["first_half_end"])] = 1
    out["period"][(frames >= params["second_half_start"])
                  & (frames <= params["second_half_end"])] = 2
    return out


def empty(n_frames, n_players):
    return {
        "ball": np.full((n_frames, 2), np.nan, dtype=np.float32),
        "players": np.full((n_frames, n_players, 2), np.nan, dtype=np.float32),
        "period": np.zeros(n_frames, dtype=np.int8),
    }


def load_game(game):
    """Shared-format tracking for one game, converting and caching it on first use."""
    path = os.path.join(OUT_DIR, "game{}_tracking.npz".format(game))
    if not os.path.exists(path):
        data = load_csv_game(game) if game in (1, 2) else load_epts_game(game)
        np.savez_compressed(path, **data)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def pass_starts(game):
    """(frame, player) for every pass, with player in the tracking's naming for that game."""
    events_path = fetch(game)[-1]
    if game in (1, 2):
        with open(events_path) as handle:
            return [(int(r["Start Frame"]), r["From"].replace(" ", ""))
                    for r in csv.DictReader(handle) if r["Type"] == "PASS"]
    with open(events_path) as handle:
        return [(e["start"]["frame"], e["from"]["id"])
                for e in json.load(handle)["data"] if e["type"]["name"] == "PASS" and e["from"]]


def check(game, data):
    """Print ball coverage and passer-to-ball distance: the two numbers that say it is usable."""
    in_play = data["period"] > 0
    ball_seen = ~np.isnan(data["ball"][:, 0])
    scale = np.array(PITCH)
    column = {name: i for i, name in enumerate(data["player"])}

    distances = []
    for frame, player in pass_starts(game):
        if player in column and frame < len(data["ball"]):
            gap = (data["players"][frame, column[player]] - data["ball"][frame]) * scale
            distances.append(np.hypot(*gap))
    distances = np.array(distances)
    distances = distances[~np.isnan(distances)]

    print("game {}  {:6d} frames  {:2d} players  ball seen {:5.1f}% of in-play frames  "
          "passer to ball at pass start: median {:.1f} m, {:.0f}% within 2 m ({} passes)".format(
              game, int(in_play.sum()), len(data["team"]), 100 * ball_seen[in_play].mean(),
              float(np.median(distances)), 100 * float(np.mean(distances < 2.0)), len(distances)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    args = parser.parse_args()
    for game in args.games:
        check(game, load_game(game))
    print("tracking written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
