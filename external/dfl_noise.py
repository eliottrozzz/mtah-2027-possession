"""Noisy copies of the DFL sealed test games, made exactly like the Metrica ones.

Reuses metrica_noise.py (same calibration, smoothing, wobble, random gaps, ball shown in some
stoppages and carried to the restart, broken player tracks). make_noisy() repeats metrica_noise's
steps in the same order: when a step is added there, add it here and rebuild. Only what the DFL data needs is different:

  * DFL tracks the ball during stoppages too; Metrica's clean data mostly doesn't. So the ball is
    first deleted on every dead frame of the answer key (agreed with Jordan, 2026-11-11), then the
    Metrica noise runs unchanged, and puts a still ball back in some stoppages as it does there.
  * Flights (balls in the air, where the camera loses the ball) come from DFL's own event fields:
    a pass with Height "high", a cross, a shot, a goal kick, a keeper throw or punt, or a pass
    longer than metrica_noise.LONG_PASS_M. A flight runs from the snapped event to the next touch.

    python external/dfl_noise.py --calibration data/ball_gap_stats.json

Output: data/external/dfl/noisy/{match}_live{level}_seed0.npz, the shared format plus ball_reason
(analysis only, never a model input). Seed 0 only: these games are a sealed test, not tuning data.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dfl_sync as ds  # noqa: E402
import dfl_tracking as dt  # noqa: E402
import metrica_noise as mn  # noqa: E402

# Params
OUT_DIR = os.path.join(dt.OUT_DIR, "noisy")
LEVELS = mn.LEVELS
SEED = 0
LABEL_PATH = os.path.join(dt.OUT_DIR, "{}_labels_reception.csv")


def load_labels(match, n):
    path = LABEL_PATH.format(match)
    if not os.path.exists(path):
        sys.exit("missing {}: run external/dfl_labels.py".format(os.path.relpath(path, ROOT)))
    labels = np.loadtxt(path, delimiter=",", skiprows=1, dtype=str, usecols=1)
    return np.concatenate([labels, np.full(max(0, n - len(labels)), "X")])[:n]


def flights(match, game, rng):
    """Frames where the ball is in the air and the camera loses it (DFL events)."""
    extras = dt.load_extras(match)
    events = dt.read_events(match)
    ds.snap(game, extras, events)
    touch_rows = np.array(sorted(e["row"] for e in events if ds.touch_player(e)))
    ball = mn.metres(game["ball"])
    drop = np.zeros(len(ball), dtype=bool)
    for e in events:
        if not ({"Play", "ShotAtGoal"} & set(e["tags"])):
            continue
        start = e["row"]
        later = touch_rows[touch_rows > start]
        if not len(later):
            continue
        end = int(later[0])
        if end - start < 10:
            continue
        length = np.hypot(*(ball[end] - ball[start]))
        airborne = (e["attrs"].get("Height") == "high" or "Cross" in e["tags"]
                    or "ShotAtGoal" in e["tags"] or e["tag"] == "GoalKick"
                    or e["attrs"].get("GoalKeeperAction") or length > mn.LONG_PASS_M)
        if airborne and rng.random() < mn.AIR_DROP_PROB:
            first = start + rng.integers(2, 7)
            last = end - rng.integers(0, 9)
            drop[first:max(first, last)] = True
    return drop


def make_noisy(match, level, seed=SEED):
    rng = np.random.default_rng([int(match[-3:], 36), level, seed])
    game = dt.load_game(match)
    n = len(game["ball"])
    labels = load_labels(match, n)
    live = np.isin(labels, ["H", "A"])
    clean = dict(game, ball=game["ball"].copy())
    clean["ball"][~live] = np.nan          # no ball in stoppages, as in Metrica's clean data
    seen = ~np.isnan(clean["ball"][:, 0])

    air = flights(match, clean, rng)
    gaps = mn.random_gaps(clean, live, seen, level / 100.0, air, rng)

    ball = mn.smooth(clean["ball"], mn.BALL_SMOOTH_FRAMES)
    ball = ball + mn.slow_wobble(n, mn.BALL_SMOOTH_FRAMES, mn.BALL_WOBBLE_M, rng)
    reason = np.where(seen, 0, 1).astype(np.int8)
    reason[seen & gaps] = 3
    reason[seen & air] = 2
    ball[reason > 1] = np.nan
    ball = mn.show_ball_in_stoppages(ball, labels, rng)
    reason[(reason == 1) & ~np.isnan(ball[:, 0])] = 4
    ball = mn.carry_dead_ball(ball, labels, rng)       # as in metrica_noise: carried, not still
    ball = mn.jitter(ball, mn.BALL_JITTER_M, rng)

    players, teams = mn.break_tracks(clean, rng)
    players = mn.jitter(players, mn.PLAYER_JITTER_M, rng)
    return {
        "ball": ball, "players": players, "team": teams,
        "player": np.array(["slot{}".format(i + 1) for i in range(len(teams))]),
        "period": game["period"], "ball_reason": reason,
    }, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--matches", nargs="+", default=list(dt.MATCHES), choices=dt.MATCHES)
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--calibration", help="ball_gap_stats.json from measure_ball_gaps.py")
    args = parser.parse_args()
    if args.calibration:
        stats = mn.calibrate(args.calibration)
        print("calibrated on {}: real match missing {:.0%} of graded frames".format(
            stats["game"], stats["ball_missing_rate"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    for match in args.matches:
        for level in args.levels:
            data, labels = make_noisy(match, level)
            path = os.path.join(OUT_DIR, "{}_live{}_seed{}.npz".format(match, level, SEED))
            np.savez_compressed(path, **data)
            mn.describe(match, level, SEED, data, labels)
    print("written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
