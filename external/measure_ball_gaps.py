"""Measure how the provided tracking loses the ball, so the noise on the external games can copy it.

Run this yourself, on your own machine. It reads the provided ATD (not the ground truth, not the
video) and prints summary numbers only: how often the ball is missing, how long the gaps are, and
how much more often they start on a fast ball, in a crowd or near a goal. No frame, position or
label is printed or saved, so the output is safe to paste into a chat or a document.

    python external/measure_ball_gaps.py          # prints the summary, saves data/ball_gap_stats.json

metrica_noise.py --calibration data/ball_gap_stats.json then copies these numbers onto the
Metrica games. Nothing here is a label or a model parameter: it only describes the tracking.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import load_smart_tagging  # noqa: E402
from models.touch import touch_strength  # noqa: E402

# Params
OUT_PATH = os.path.join(ROOT, "data", "ball_gap_stats.json")
FPS = 25
GAP_QUANTILES = (10, 25, 50, 75, 90, 95, 99)
SHORT_GAP_S = 15.0                                        # longer gaps are mostly stoppages
# Bins shared with metrica_noise.py, which reads the hazard ratios back by these names.
SPEED_BINS = ((0, 5), (5, 15), (15, 25), (25, np.inf))   # ball speed, m/s
CROWD_BINS = ((0, 1), (1, 2), (2, 3), (3, np.inf))       # players within CROWD_RADIUS
CROWD_RADIUS = 3.0                                        # metres
NEAR_GOAL = 0.17                                          # x within ~18 m of either goal line
NEAR_TOUCHLINE = 0.10                                     # y within ~7 m of either touchline
SET_PIECE_MARGIN_S = 5.0                                  # a gap "ends at" a set piece this close
STRENGTH_QUANTILES = (50, 75, 90, 95, 99)


def read_atd(raw_path, n_frames, n_slots):
    """Ball (n, 2) and track slots (n, slots, 2) in 0-1 units, by video frame. NaN = missing."""
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if parts[2][0] != "N":
                ball[frame] = [float(v) for v in parts[2].split(",")[:2]]
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] != "N":
                    players[frame, slot] = [float(v) for v in group.split(",")[:2]]
    return ball, players


def runs(mask):
    """[(start, stop)] of the True stretches in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def ball_speed(ball, pitch):
    """Ball speed in m/s from positions two frames apart; NaN where either is missing."""
    speed = np.full(len(ball), np.nan)
    step = (ball[2:] - ball[:-2]) * np.array(pitch)
    speed[2:] = np.hypot(step[:, 0], step[:, 1]) * FPS / 2.0
    return speed


def hazard_ratios(values, bins, seen, gap_starts):
    """How much more often a gap starts in each bin than overall, per frame with the ball seen.

    A ratio of 3 means the ball is lost three times as often per frame in that situation. The
    ratios, not the raw rates, carry over to other games: the overall rate is set separately.
    """
    overall = gap_starts.sum() / float(seen.sum())
    out = {}
    for low, high in bins:
        in_bin = seen & (values >= low) & (values < high)
        frames = int(in_bin.sum())
        rate = gap_starts[in_bin].sum() / float(frames) if frames else 0.0
        out["{}-{}".format(low, high)] = {"ratio": rate / overall if overall else 0.0,
                                          "share_of_frames": frames / float(seen.sum())}
    return out


def wobble_m(xy, pitch):
    """Typical frame-to-frame position error in metres (one axis), from the second difference.

    For a smoothly moving object x[t+1] - 2x[t] + x[t-1] is close to zero, so what's left is
    tracking error: with independent errors of size s per frame its spread is sqrt(6) * s. The
    median keeps real kicks and bounces, which are rare, from counting.
    """
    pos = xy * np.array(pitch, dtype=np.float32)
    second = (pos[2:] - 2 * pos[1:-1] + pos[:-2]).reshape(-1)
    second = np.abs(second[np.isfinite(second)])
    return float(np.median(second) / 0.6745 / np.sqrt(6)) if len(second) else float("nan")


def stoppage_share(gaps, graded, game, data_dir):
    """How many gaps end at a Smart Tagging set piece, against what chance would give.

    A gap that ends as a restart is taken was very likely a stoppage, so this estimates how much
    of the missing ball is dead-ball time. Smart Tagging is automatic and misses some restarts
    (fewer throw-ins are tagged than a match has), so this is a lower bound. No ground truth.
    """
    margin = int(SET_PIECE_MARGIN_S * FPS)
    set_pieces = [t for t in load_smart_tagging.load(game, data_dir)
                  if t["code"].strip().lower() == "set pieces"]
    near = np.zeros(len(graded), dtype=bool)
    for tag in set_pieces:
        near[max(0, tag["start_frame"] - margin):tag["end_frame"] + margin] = True
    ends_near = np.array([near[min(b, len(near) - 1)] for _, b in gaps])
    lengths = np.array([b - a for a, b in gaps])
    return {
        "set_pieces_tagged": len(set_pieces),
        "gaps_ending_at_set_piece": float(ends_near.mean()),
        "missing_in_gaps_ending_at_set_piece": float(lengths[ends_near].sum() / lengths.sum()),
        "chance_share_of_graded_frames": float(near[graded].mean()),
    }


def measure(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_slots = len(list(ET.parse(meta_path).getroot().iter("Player")))
    pitch = GAMES[game]["pitch"]
    ball, players = read_atd(raw_path, source_frames("tactical", game), n_slots)

    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    seen = ~np.isnan(ball[:, 0])
    missing = graded & ~seen

    # Gaps inside the graded halves, and the frame just before each one.
    gaps = [(a, b) for a, b in runs(missing) if a > 0]
    lengths = np.array([b - a for a, b in gaps]) / float(FPS)
    gap_starts = np.zeros(len(ball), dtype=bool)
    gap_starts[[a - 1 for a, _ in gaps]] = True
    gap_starts &= seen & graded

    scale = np.array(pitch, dtype=np.float32)
    near = np.hypot(*((players - ball[:, None, :]) * scale).transpose(2, 0, 1))
    crowd = np.sum(near < CROWD_RADIUS, axis=1).astype(float)
    speed = ball_speed(ball, pitch)
    near_goal = (np.minimum(ball[:, 0], 1 - ball[:, 0]) < NEAR_GOAL).astype(float)
    near_touch = (np.minimum(ball[:, 1], 1 - ball[:, 1]) < NEAR_TOUCHLINE).astype(float)
    on_ball = seen & graded

    present = ~np.isnan(players[:, :, 0]) & graded[:, None]
    fragments = np.array([b - a for slot in range(n_slots) for a, b in runs(present[:, slot])])

    # Jitter: how much positions wobble, and the touch signal the model sees (same formula).
    ball_graded = np.where(graded[:, None], ball, np.nan)
    players_graded = np.where(graded[:, None, None], players, np.nan)
    strength = touch_strength(ball_graded)
    strength = strength[np.isfinite(strength)]

    return {
        "game": game,
        "graded_frames": int(graded.sum()),
        "ball_missing_rate": float(missing.sum() / graded.sum()),
        "n_gaps": len(gaps),
        "gap_quantiles_s": {str(q): float(np.percentile(lengths, q)) for q in GAP_QUANTILES},
        # Long gaps are mostly stoppages; the noise copies only the short, live-play ones.
        "short_gap_quantiles_s": {str(q): float(np.percentile(lengths[lengths <= SHORT_GAP_S], q))
                                  for q in GAP_QUANTILES},
        "missing_in_gaps_over_10s": float(lengths[lengths > 10].sum() * FPS / missing.sum()),
        "stoppages": stoppage_share(gaps, graded, game, data_dir),
        "hazard_speed": hazard_ratios(speed, SPEED_BINS, on_ball, gap_starts),
        "hazard_crowd": hazard_ratios(crowd, CROWD_BINS, on_ball, gap_starts),
        "hazard_near_goal": hazard_ratios(near_goal, ((0, 1), (1, 2)), on_ball, gap_starts),
        "hazard_near_touchline": hazard_ratios(near_touch, ((0, 1), (1, 2)), on_ball, gap_starts),
        "objects_per_frame": float(present.sum() / graded.sum()),
        "track_slots": n_slots,
        "track_fragments": int(len(fragments)),
        "fragment_quantiles_s": {str(q): float(np.percentile(fragments, q) / FPS)
                                 for q in GAP_QUANTILES},
        "ball_jitter_m": wobble_m(ball_graded, pitch),
        "player_jitter_m": wobble_m(players_graded, pitch),
        "touch_strength_quantiles": {str(q): float(np.percentile(strength, q))
                                     for q in STRENGTH_QUANTILES},
    }


def show(stats):
    print("ball missing on {:.1%} of {} graded frames, in {} gaps".format(
        stats["ball_missing_rate"], stats["graded_frames"], stats["n_gaps"]))
    print("gap length (s), percentiles: " + "  ".join(
        "p{} {:.1f}".format(q, v) for q, v in stats["gap_quantiles_s"].items()))
    print("gaps up to {:.0f} s, percentiles:  ".format(SHORT_GAP_S) + "  ".join(
        "p{} {:.1f}".format(q, v) for q, v in stats["short_gap_quantiles_s"].items()))
    print("share of missing frames in gaps over 10 s: {:.1%}".format(
        stats["missing_in_gaps_over_10s"]))
    stop = stats["stoppages"]
    print("gaps ending at a tagged set piece: {:.1%} of gaps, {:.1%} of missing frames "
          "({} set pieces tagged; chance level {:.1%})".format(
              stop["gaps_ending_at_set_piece"], stop["missing_in_gaps_ending_at_set_piece"],
              stop["set_pieces_tagged"], stop["chance_share_of_graded_frames"]))
    for name in ("hazard_speed", "hazard_crowd", "hazard_near_goal", "hazard_near_touchline"):
        print("{:<22}".format(name) + "  ".join(
            "{} x{:.2f} ({:.0%})".format(k, v["ratio"], v["share_of_frames"])
            for k, v in stats[name].items()))
    print("tracked objects per frame {:.1f}, {} track slots, {} fragments".format(
        stats["objects_per_frame"], stats["track_slots"], stats["track_fragments"]))
    print("fragment length (s), percentiles: " + "  ".join(
        "p{} {:.1f}".format(q, v) for q, v in stats["fragment_quantiles_s"].items()))
    print("jitter (m per frame): ball {:.3f}, players {:.3f}".format(
        stats["ball_jitter_m"], stats["player_jitter_m"]))
    print("velocity change over 0.16 s (m/s), percentiles: " + "  ".join(
        "p{} {:.1f}".format(q, v) for q, v in stats["touch_strength_quantiles"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=OUT_PATH, help="where to save the summary as JSON")
    args = parser.parse_args()
    stats = measure()
    show(stats)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(stats, handle, indent=2)
    print("saved", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
