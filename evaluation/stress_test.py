"""Stress test: does a model hold up if our noise imitates the provided tracking wrongly?

The noisy copies copy summary statistics of the provided tracking. This script asks what happens
if one of them is off. It fits the model per round exactly as score_external.py does (two games,
standard live20/35 seed-0 copies, reception key), then predicts each test game on a VARIANT copy
(live35, seed 0) with one noise setting changed, built in memory.

    python evaluation/stress_test.py --model models.touch_dead:TouchDead
    python evaluation/stress_test.py --model models.touch_dead:TouchDead --variants calibrated blur_longer wobble

Most variants change one noise setting. The two "moving ball" variants instead edit the copy after
it is made: the ball shown during stoppages is replaced by a ball being carried to the restart
(see carried_ball).

The variants are a check, never a target: nothing is tuned on them and nothing is logged to
results/runs/. Nothing is written to data/. External data only; the calibration file is the
aggregate summary the standard copies were made with. The "calibrated" variant must reproduce the
stored copies exactly; the script stops if it doesn't, since then the comparison means nothing.
"""

import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "external")]
import metrica_noise as mn  # noqa: E402
import possession_metrics as pm  # noqa: E402
import score_external as se  # noqa: E402

# Params
LEVEL = 35
SEED = 0
CALIBRATION = os.path.join(ROOT, "data", "ball_gap_stats.json")
CARRY_SPEED_MS = (1.0, 3.0)    # carried ball: walking to jogging pace, drawn per stoppage
CARRY_TURN_PER_S = 1.0         # how fast it steers toward the restart spot
CARRY_WANDER = 1.0             # random turning, radians per sqrt(second)
SETTINGS = ("CUTS_PER_MINUTE", "BALL_SMOOTH_FRAMES", "GAP_QUANTILES_S", "MAX_LIVE_GAP_S",
            "DEAD_VISIBLE_PROB", "PLAYER_JITTER_M", "AIR_DROP_PROB")


def variants(base):
    """name -> (description, settings to change, live level[, edit of the made copy])."""
    odd = lambda frames: max(1, int(frames) | 1)    # smoothing windows are odd
    return {
        "calibrated": ("as calibrated (the copies we score on)", {}, LEVEL),
        "breaks_x2": ("track breaks x2", {"CUTS_PER_MINUTE": base["CUTS_PER_MINUTE"] * 2}, LEVEL),
        "breaks_x0.5": ("track breaks x0.5",
                        {"CUTS_PER_MINUTE": base["CUTS_PER_MINUTE"] * 0.5}, LEVEL),
        "blur_shorter": ("ball blur x0.5",
                         {"BALL_SMOOTH_FRAMES": odd(base["BALL_SMOOTH_FRAMES"] // 2)}, LEVEL),
        "blur_longer": ("ball blur x1.5",
                        {"BALL_SMOOTH_FRAMES": odd(base["BALL_SMOOTH_FRAMES"] * 1.5)}, LEVEL),
        "gaps_x2": ("random ball gaps twice as long",
                    {"GAP_QUANTILES_S": {k: v * 2 for k, v in base["GAP_QUANTILES_S"].items()},
                     "MAX_LIVE_GAP_S": base["MAX_LIVE_GAP_S"] * 2}, LEVEL),
        "missing65": ("ball missing on 65% of live frames", {}, 65),
        "stoppage_ball_never": ("still ball never shown in stoppages",
                                {"DEAD_VISIBLE_PROB": 0.0}, LEVEL),
        "stoppage_ball_always": ("still ball always shown in stoppages",
                                 {"DEAD_VISIBLE_PROB": 1.0}, LEVEL),
        "wobble": ("player position wobble 0.1 m per frame", {"PLAYER_JITTER_M": 0.1}, LEVEL),
        "high_balls_lost": ("every high ball lost", {"AIR_DROP_PROB": 1.0}, LEVEL),
        "moving_ball_stoppages": ("ball shown moving in stoppages (same share)", {}, LEVEL,
                                  lambda data, labels, rng: move_dead_ball(data, labels, rng, False)),
        "moving_ball_all_stoppages": ("ball shown moving on every stoppage frame", {}, LEVEL,
                                      lambda data, labels, rng: move_dead_ball(data, labels, rng, True)),
    }


def carried_ball(n, start, target, rng):
    """(n, 2) path in metres: a ball carried from start toward target at a steady walking or
    jogging pace, steering toward the target with some random wandering. Once there it keeps
    circling near the spot (radius about speed / turn rate), so it never sits still."""
    speed = rng.uniform(*CARRY_SPEED_MS)
    pos, path = np.array(start, dtype=np.float64), np.empty((n, 2))
    heading = np.arctan2(*(target - pos)[::-1]) + rng.normal(0.0, 0.5)
    for i in range(n):
        wanted = np.arctan2(*(target - pos)[::-1])
        turn = (wanted - heading + np.pi) % (2 * np.pi) - np.pi
        heading += CARRY_TURN_PER_S * turn / mn.FPS + rng.normal(0.0, CARRY_WANDER / np.sqrt(mn.FPS))
        pos = pos + speed / mn.FPS * np.array([np.cos(heading), np.sin(heading)])
        path[i] = pos
    return path


def move_dead_ball(data, labels, rng, every_frame):
    """Replace the ball shown in stoppages (D frames) with a carried ball. It runs from the last
    ball seen before the stoppage to the first seen after it (where play restarts), then gets
    the copies' 0.84 s blur and usual jitter. every_frame=False keeps the frames on which the
    copy showed a ball; True shows it on every stoppage frame."""
    ball, pitch = data["ball"].copy(), np.array(mn.mt.PITCH, dtype=np.float64)
    seen = np.flatnonzero(np.isfinite(ball[:, 0]))
    for start, stop in mn.runs(labels == "D"):
        before, after = seen[seen < start], seen[seen >= stop]
        if not len(before) and not len(after):
            continue
        a = ball[before[-1]] if len(before) else ball[after[0]]
        b = ball[after[0]] if len(after) else a
        n = stop - start
        path = carried_ball(n, a * pitch, b * pitch, rng) / pitch
        blur = min(mn.BALL_SMOOTH_FRAMES, n if n % 2 else n - 1)   # odd, no longer than the stoppage
        path = mn.jitter(mn.smooth(path, blur), mn.BALL_JITTER_M, rng)
        shown = np.ones(n, bool) if every_frame else np.isfinite(ball[start:stop, 0])
        ball[start:stop][shown] = path[shown]
    data["ball"] = ball


def same_tracking(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in b)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="module:Class")
    parser.add_argument("--variants", nargs="+", help="names to run (default: all)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    mn.calibrate(CALIBRATION)
    base = {k: copy.deepcopy(getattr(mn, k)) for k in SETTINGS}
    table = variants(base)
    names = args.variants or list(table)
    unknown = [n for n in names if n not in table]
    if unknown:
        parser.error("unknown variants {}; have {}".format(unknown, list(table)))

    # 1. Fit once per round, on the standard copies
    keys, standard = {}, {}
    for g in se.GAMES:
        for level in se.TUNING_LEVELS:
            standard[g, level] = se.load_tracking(g, level, SEED)
        keys[g] = se.load_answer_key(g, se.HEADLINE_KEY, standard[g, se.TUNING_LEVELS[0]]["period"])
    models = {}
    for test in se.GAMES:
        start = time.time()
        models[test] = se.load_model(args.model)
        models[test].fit([(standard[g, level], keys[g]) for g in se.GAMES if g != test
                          for level in se.TUNING_LEVELS])
        print("fitted round {} ({:.0f} s)".format(test, time.time() - start), flush=True)

    # 2. Each variant: rebuild each test game's copy, predict, score
    print("\n{:<42}{:>8}  {:<20}{:>8}".format("variant (live{})".format(LEVEL), "spellF1",
                                                "game 1 / 2 / 3", "macroF1"))
    reference = None
    for name in names:
        description, change, level, *edit = table[name]
        for key, value in base.items():
            setattr(mn, key, copy.deepcopy(value))
        for key, value in change.items():
            setattr(mn, key, value)
        spell, macro = [], []
        for test in se.GAMES:
            data, labels = mn.make_noisy(test, level, SEED)
            data.pop("ball_reason")
            if edit:
                edit[0](data, labels, np.random.default_rng([test, level, SEED, 1]))
            if name == "calibrated" and not same_tracking(data, se.load_tracking(test, LEVEL, SEED)):
                sys.exit("the calibrated variant doesn't reproduce the stored copy of game {}: "
                         "re-make the noisy copies first".format(test))
            result = pm.evaluate(keys[test], pm.label_codes(models[test].predict(data)))
            spell.append(result["spell_f1_mean"])
            macro.append(result["macro_f1"])
        mean = float(np.mean(spell))
        if name == "calibrated":
            reference = mean
        change_text = "" if reference is None or name == "calibrated" else "{:+.3f}".format(
            mean - reference)
        print("{:<42}{:>8.3f}  {:<20}{:>8.3f}  {}".format(
            description, mean, " / ".join("{:.2f}".format(s) for s in spell),
            float(np.mean(macro)), change_text), flush=True)


if __name__ == "__main__":
    main()
