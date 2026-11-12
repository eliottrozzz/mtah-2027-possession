"""Make the submission file: fit the model on the external games, predict the provided match.

    python predict_match.py                                   # default model, writes and validates
    python predict_match.py --model models.touch_dead:TouchDead --out evaluation/results/v2.csv

What it does:
  1. Fits the model on all three Metrica games (noisy copies live20 and live35, seed 0, reception
     answer key). Nothing from the provided match is used for fitting or tuning.
  2. Reads the provided tracking (the ATD) into the same format as the Metrica games.
  3. Predicts one label per video frame and writes frame,label for all 199319 frames. Frames
     outside the two halves get X (they are never graded).
  4. Runs the official format check.

It prints summary numbers only (label shares, number of possessions), never frames or positions,
so its output is safe to share. The output file itself is labels for the provided match: it stays
in evaluation/results/ (gitignored) and is never committed or shared.
"""

import argparse
import importlib
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, half_video_blocks, id_to_video, require, source_frames  # noqa: E402
import score_external as se  # noqa: E402

# Params
DEFAULT_MODEL = "models.touch_dead:TouchDead"
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "submission.csv")
FIT_LEVELS = se.TUNING_LEVELS
FIT_SEED = 0
FPS = 25
MIN_BALL_SEEN = 0.2      # stop if the ball is seen on fewer graded frames: the frames didn't map
MIN_PLAYERS_SEEN = 0.9   # ... or if fewer graded frames have any player tracked


# -- reading EPTS tracking (the provided ATD, and Metrica game 3, use the same format) -----------

def read_epts(meta_path, raw_path, n_frames, to_row):
    """{ball, players, team, player} from a FIFA/EPTS pair. Row = to_row(frame_id); lines where
    it returns None or a row outside 0..n_frames-1 are skipped.

    Home ('H') is the first <Team> in the metadata. If the metadata gives per-frame-range channel
    layouts (DataFormatSpecification), they are followed; otherwise the columns are taken in the
    order of the <Player> elements.
    """
    root = ET.parse(meta_path).getroot()
    teams = root.findall(".//Teams/Team")
    if not teams:
        sys.exit("tracking metadata has no <Teams>: can't tell home from away")
    home_id = teams[0].get("id")
    players_meta = list(root.iter("Player"))
    team_of = {p.get("id"): ("H" if p.get("teamId") == home_id else "A") for p in players_meta}
    if any(p.get("teamId") is None for p in players_meta):
        sys.exit("some <Player> elements have no teamId: can't assign tracks to teams")

    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        order = [channel_player[ref.get("playerChannelId")]
                 for ref in spec.iter("PlayerChannelRef")
                 if ref.get("playerChannelId", "").endswith("_x")
                 and ref.get("playerChannelId") in channel_player]
        if order:
            layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    if not layouts:
        layouts = [(0, 10 ** 12, [p.get("id") for p in players_meta])]
    player_ids = sorted({pid for _, _, order in layouts for pid in order})
    column = {pid: i for i, pid in enumerate(player_ids)}

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, len(player_ids), 2), np.nan, dtype=np.float32)
    layout_index = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            row = to_row(frame_id)
            if row is None or not 0 <= row < n_frames:
                continue
            while layout_index < len(layouts) - 1 and frame_id > layouts[layout_index][1]:
                layout_index += 1
            cols = [column[pid] for pid in layouts[layout_index][2]]
            for col, group in zip(cols, parts[1].split(";")):
                values = group.split(",")
                if values[0] not in ("", "NaN", "N") and not values[0].startswith("N"):
                    players[row, col] = [float(values[0]), float(values[1])]
            values = parts[2].split(",")
            if values[0] and not values[0].startswith("N"):
                ball[row] = [float(values[0]), float(values[1])]
    return {"ball": ball, "players": players,
            "team": np.array([team_of[pid] for pid in player_ids]),
            "player": np.array(player_ids)}


def read_match(game=GAME, data_dir=None):
    """The provided match's tracking, one row per video frame, in the shared format."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    # The ATD counts IdFrame (about 1.7 million), with a different offset to the video in each
    # half (games.py), so map every line through id_to_video; halftime ids give None.
    tracking = read_epts(meta_path, raw_path, n, lambda frame_id: id_to_video(game, frame_id))
    period = np.zeros(n, dtype=np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half
    tracking["period"] = period
    return tracking


# -- fitting and predicting --------------------------------------------------------------------

def fit_on_external(model):
    train = []
    for game in se.GAMES:
        for level in FIT_LEVELS:
            tracking = se.load_tracking(game, level, FIT_SEED)
            train.append((tracking, se.load_answer_key(game, se.HEADLINE_KEY, tracking["period"])))
    model.fit(train)
    return model


def summary(labels, period):
    graded = period > 0
    shares = {k: float((labels[graded] == k).mean()) for k in "HADX"}
    spells = 0
    for half in (1, 2):
        lab = labels[period == half]
        poss = np.isin(lab, ["H", "A"])
        spells += int(np.sum(poss & np.concatenate(([True], lab[1:] != lab[:-1]))))
    return shares, spells


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    module, _, attr = args.model.partition(":")
    model = getattr(importlib.import_module(module), attr)()
    print("fitting {} on the external games (this takes a few minutes)".format(model.name))
    fit_on_external(model)
    print("settings:", model.settings())

    print("reading the provided tracking")
    tracking = read_match()
    print("  {} frames, {} track columns ({} H, {} A); ball seen on {:.0%} of graded frames".format(
        len(tracking["ball"]), len(tracking["team"]), int((tracking["team"] == "H").sum()),
        int((tracking["team"] == "A").sum()),
        float(np.mean(~np.isnan(tracking["ball"][tracking["period"] > 0, 0])))))

    graded = tracking["period"] > 0
    ball_seen = float(np.mean(~np.isnan(tracking["ball"][graded, 0])))
    players_seen = float(np.mean(np.any(~np.isnan(tracking["players"][graded, :, 0]), axis=1)))
    if ball_seen < MIN_BALL_SEEN or players_seen < MIN_PLAYERS_SEEN:
        sys.exit("only {:.0%} of graded frames have the ball and {:.0%} have players: the tracking "
                 "frames did not map onto the video, not writing a file".format(ball_seen,
                                                                                  players_seen))

    labels = np.asarray(model.predict(tracking), dtype="<U1")
    labels = np.where(tracking["period"] > 0, labels, "X")
    shares, spells = summary(labels, tracking["period"])
    print("  graded frames: H {H:.1%}  A {A:.1%}  D {D:.1%}  X {X:.1%}".format(**shares))
    print("  possessions predicted: {} ({:.1f} s on average)".format(
        spells, (shares["H"] + shares["A"]) * (tracking["period"] > 0).sum() / FPS / max(spells, 1)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    print("wrote", os.path.relpath(args.out, ROOT))
    subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                     args.out])


if __name__ == "__main__":
    main()
