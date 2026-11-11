"""Tracking for the PFF FC World Cup 2022 games, in the shared format of metrica_tracking.py.

STATUS: written from kloppy's reader (kloppy/infra/serializers/tracking/pff.py), before we had
the files. Untested. Nothing may be run on PFF data until its licence terms are recorded in
external/README.md (TEAM_SETUP.md: check the licence of any dataset before using it).

PFF FC released broadcast tracking and events for all 64 games of the 2022 men's World Cup, on
request from PFF FC. Files go in data/external/pff/ (gitignored):

    {game_id}.jsonl.bz2    tracking, one JSON object per frame, about 29.97 fps
    metadata {game_id}.json   home/away team, pitch size, fps
    rosters  {game_id}.json   team sheets: team id, player id, shirt number
    events.json            events for all games

    python external/pff_tracking.py --games 10517 3812

The shared format
-----------------
load_game() returns the dict of metrica_tracking.load_game(): ball, players, team, player,
period. PFF runs at about 29.97 fps; we resample to 25 fps (each row takes the nearest PFF frame
in time, always within 0.02 s), so every model and FPS constant stays the same. Row 0 is empty,
then the first half, one empty row, then the second half. Extra time is dropped, like the other
datasets, which only have two halves.

Coordinates: PFF gives metres from the centre spot (x along the pitch). We map them to 0-1 with
the pitch size from the metadata and y flipped, as dfl_tracking.py does.

Players are matched to the roster by shirt number within their team. Both ball and players come
from the smoothed fields, as kloppy reads them. Being broadcast tracking, the ball and some
players go missing when they are off camera; that is kept as NaN.

load_extras() gives per row: the PFF frame number, and the provider's ball state and owning team
(from the per-frame game_event, as kloppy derives them), for comparison only, never for a model.
"""

import argparse
import bz2
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
OUT_DIR = os.path.join(ROOT, "data", "external", "pff")
FPS = 25
PERIODS = (1, 2)                          # extra time (3, 4) is dropped
ALIVE_EVENTS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"}
DEAD_EVENTS = {"OUT", "END"}
SHARED = ("ball", "players", "team", "player", "period")
# Raw, not smoothed: PFF's ballsSmoothed puts the ball exactly on the player PFF's analysts logged
# as on the ball (98-99% of those frames, 0.1% in DFL's optical tracking), so it carries the
# answer key. The raw 'balls' field is genuine tracking (never on that player; median 1.0-1.2 m).
SOURCE = "raw"


def game_files(game_id):
    """(tracking, metadata, roster, events) paths, in the Drive folder's layout. Event Data's
    top level is the latest release (June 2025); its dated subfolders are older ones."""
    paths = (os.path.join(OUT_DIR, "Tracking Data", "{}.jsonl.bz2".format(game_id)),
             os.path.join(OUT_DIR, "Metadata", "{}.json".format(game_id)),
             os.path.join(OUT_DIR, "Rosters", "{}.json".format(game_id)),
             os.path.join(OUT_DIR, "Event Data", "{}.json".format(game_id)))
    missing = [os.path.relpath(p, ROOT) for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("game {}: missing {}".format(game_id, ", ".join(missing)))
    return paths


def available_games():
    """Game ids with all four files."""
    names = os.listdir(os.path.join(OUT_DIR, "Tracking Data"))
    ids = [n.split(".")[0] for n in names if n.endswith(".jsonl.bz2")]
    out = []
    for game_id in sorted(ids, key=int):
        try:
            game_files(game_id)
            out.append(game_id)
        except FileNotFoundError:
            pass
    return out


def read_meta(game_id):
    _, meta_path, roster_path, _ = game_files(game_id)
    with open(meta_path) as handle:
        meta = json.load(handle)[0]
    with open(roster_path) as handle:
        roster = json.load(handle)
    home, away = str(meta["homeTeam"]["id"]), str(meta["awayTeam"]["id"])
    pitch = meta["stadium"]["pitches"][0]
    shirt = {}  # (side, shirt number) -> player id
    for entry in roster:
        side = "H" if str(entry["team"]["id"]) == home else "A"
        shirt[side, int(entry["shirtNumber"])] = str(entry["player"]["id"])
    return {"home": home, "away": away, "start_left": meta.get("homeTeamStartLeft"),
            "pitch": (float(pitch["length"]), float(pitch["width"])), "shirt": shirt}


def read_frames(game_id):
    """The raw lines we need, in file order, for the kept periods."""
    tracking = game_files(game_id)[0]
    frames = []
    with bz2.open(tracking, "rt") as handle:
        for line in handle:
            frame = json.loads(line)
            if frame.get("period") in PERIODS:
                frames.append(frame)
    return frames


def convert(game_id, source=SOURCE):
    ball_key = "balls" if source == "raw" else "ballsSmoothed"
    player_keys = (("H", "homePlayers"), ("A", "awayPlayers")) if source == "raw" else (
        ("H", "homePlayersSmoothed"), ("A", "awayPlayersSmoothed"))
    meta = read_meta(game_id)
    frames = read_frames(game_id)
    length, width = meta["pitch"]
    players = sorted(meta["shirt"].items(), key=lambda kv: (kv[0][0] != "H", kv[1]))
    column = {key: i for i, (key, _) in enumerate(players)}

    # Resample each period to 25 fps on periodElapsedTime.
    plan, start = [], 1
    for period in PERIODS:
        rows = [f for f in frames if f["period"] == period]
        times = np.array([f["periodElapsedTime"] for f in rows], dtype=float)
        targets = np.arange(0.0, times.max() + 1e-9, 1.0 / FPS)
        nearest = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
        nearest = np.where(np.abs(times[nearest - 1] - targets) <= np.abs(times[nearest] - targets),
                           nearest - 1, nearest)
        plan.append((period, start, [rows[i] for i in nearest]))
        start += len(targets) + (1 if period == PERIODS[0] else 0)
    n_rows = start

    out = {
        "ball": np.full((n_rows, 2), np.nan, dtype=np.float32),
        "players": np.full((n_rows, len(players), 2), np.nan, dtype=np.float32),
        "team": np.array([side for (side, _), _ in players]),
        "player": np.array([pid for _, pid in players]),
        "period": np.zeros(n_rows, dtype=np.int8),
        "pff_frame": np.full(n_rows, -1, dtype=np.int64),
        "pff_ball_state": np.full(n_rows, "", dtype="<U1"),       # 'L' alive, 'D' dead
        "pff_ball_owner": np.full(n_rows, "", dtype="<U1"),       # 'H' / 'A'
        "pff_ball_visible": np.zeros(n_rows, dtype=bool),         # seen, not estimated
    }

    def unit(x, y):
        return x / length + 0.5, 0.5 - y / width

    # Provider state carried forward over every source frame, as kloppy does, then sampled.
    state, owner = "D", ""
    carried = {}
    for frame in frames:
        event = frame.get("game_event") or {}
        kind = event.get("game_event_type")
        if kind in ALIVE_EVENTS:
            state = "L"
        elif kind in DEAD_EVENTS:
            state = "D"
        if event.get("home_ball") is not None:
            owner = "H" if event["home_ball"] else "A"
        carried[frame["frameNum"]] = (state, owner)

    for period, row0, sampled in plan:
        for k, frame in enumerate(sampled):
            row = row0 + k
            out["period"][row] = period
            out["pff_frame"][row] = frame["frameNum"]
            out["pff_ball_state"][row], out["pff_ball_owner"][row] = carried[frame["frameNum"]]
            ball = frame.get(ball_key) or {}
            if isinstance(ball, list):
                ball = ball[0] if ball else {}
            if ball.get("x") is not None and ball.get("y") is not None:
                out["ball"][row] = unit(ball["x"], ball["y"])
                out["pff_ball_visible"][row] = ball.get("visibility") == "VISIBLE"
            for side, key in player_keys:
                for p in frame.get(key) or []:
                    col = column.get((side, int(p["jerseyNum"])))
                    if col is not None and p.get("x") is not None and p.get("y") is not None:
                        out["players"][row, col] = unit(p["x"], p["y"])
    return out


def cached(game_id, source=SOURCE):
    path = os.path.join(OUT_DIR, "{}_tracking_{}.npz".format(game_id, source))
    if not os.path.exists(path):
        np.savez_compressed(path, **convert(game_id, source))
    return path


def load_game(game_id, source=SOURCE):
    """Shared-format tracking for one game: exactly the keys a model may see."""
    with np.load(cached(game_id, source)) as data:
        return {key: data[key] for key in SHARED}


def load_extras(game_id, source=SOURCE):
    """Per-row PFF frame number and provider ball state / owner. Never a model input."""
    with np.load(cached(game_id, source)) as data:
        return {key: data[key] for key in data.files if key not in SHARED}


def check(game_id, data, extras):
    """Ball coverage overall and on provider-alive frames, players per frame. The passer-to-ball
    check needs events.json and is added with the answer key once its format is known."""
    in_play = data["period"] > 0
    alive = extras["pff_ball_state"] == "L"
    seen = ~np.isnan(data["ball"][:, 0])
    count = (~np.isnan(data["players"][:, :, 0])).sum(axis=1)
    print("game {}  {:6d} rows  {:2d} players  ball seen {:5.1f}% of rows, {:5.1f}% of provider-"
          "alive  provider alive {:4.1f}%  players per row {:.1f}".format(
              game_id, int(in_play.sum()), len(data["team"]), 100 * seen[in_play].mean(),
              100 * seen[alive].mean(), 100 * alive[in_play].mean(), count[in_play].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", nargs="+", help="PFF game ids (default: every complete game)")
    args = parser.parse_args()
    for game_id in args.games or available_games():
        check(game_id, load_game(game_id), load_extras(game_id))
    print("tracking written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
