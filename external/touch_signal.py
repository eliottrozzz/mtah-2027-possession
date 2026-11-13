"""Touch-signal percentiles at several windows: a check that the noisy copies' ball blur matches the
real tracking's from more than one angle.

The blur in metrica_noise.py (smoothing over BALL_SMOOTH_FRAMES, plus a slow wobble) is fitted to
one signal: the touch-strength percentiles at window 4 (models.touch.touch_strength). Windows 8
and 12 look at the same velocity changes over a longer stretch, so a blur of the wrong length
should show up there even when window 4 matches.

    python external/touch_signal.py                                    # the copies only
    python external/touch_signal.py --real data/ball_gap_stats.json    # real vs copies

The copies are the calibrated noisy Metrica games (live20 and live35, seed 0, games 1-3); their
percentiles are averaged over the 6. The real numbers come from measure_ball_gaps.py, which uses
touch_signal() below on the provided tracking, so both sides are computed the same way. Prints
summary numbers only. This only measures: it changes no calibration and writes no files.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from models.touch import touch_strength  # noqa: E402

# Params
FPS = 25
WINDOWS = (4, 8, 12)                      # frames each side; 4 is the one calibrate() reads
QUANTILES = (50, 75, 90, 95, 99)
COPY_PATH = os.path.join(ROOT, "data", "external", "metrica", "noisy", "game{}_live{}_seed{}.npz")
COPY_GAMES = (1, 2, 3)
COPY_LEVELS = (20, 35)
COPY_SEED = 0


def key(window):
    """JSON key in ball_gap_stats.json. Window 4 keeps its original name: calibrate() reads it."""
    return "touch_strength_quantiles" if window == 4 else "touch_strength_quantiles_w{}".format(
        window)


def touch_signal(ball_graded):
    """{key(window): {percentile: m/s}} for every window in WINDOWS.

    ball_graded is the ball with every frame outside the graded halves set to NaN. touch_strength
    is NaN wherever a frame in its window has no ball, and those frames are left out.
    """
    out = {}
    for window in WINDOWS:
        strength = touch_strength(ball_graded, window)
        strength = strength[np.isfinite(strength)]
        out[key(window)] = {str(q): float(np.percentile(strength, q)) for q in QUANTILES}
    return out


def show(stats, label=""):
    for window in WINDOWS:
        if key(window) in stats:
            print("{}velocity change over {:.2f} s (m/s, window {:2d}), percentiles: ".format(
                label, window / FPS, window) + "  ".join(
                    "p{} {:.1f}".format(q, v) for q, v in stats[key(window)].items()))


def copies():
    """touch_signal() averaged over the calibrated noisy copies, graded frames = both halves."""
    found = []
    for game in COPY_GAMES:
        for level in COPY_LEVELS:
            path = COPY_PATH.format(game, level, COPY_SEED)
            if not os.path.exists(path):
                sys.exit("missing {}: run external/metrica_noise.py --calibration "
                         "data/ball_gap_stats.json".format(os.path.relpath(path, ROOT)))
            data = np.load(path)
            ball = np.where((data["period"] > 0)[:, None], data["ball"], np.nan)
            found.append(touch_signal(ball))
    return {k: {q: float(np.mean([f[k][q] for f in found])) for q in found[0][k]}
            for k in found[0]}


def compare(real, copy):
    print("\n{:>8} {:>5} {:>7} {:>7} {:>7} {:>7}".format(
        "window", "pct", "real", "copies", "diff", "diff %"))
    for window in WINDOWS:
        if key(window) not in real:
            print("{:>8}  not in the real stats: rerun external/measure_ball_gaps.py".format(window))
            continue
        for q in map(str, QUANTILES):
            r, c = real[key(window)][q], copy[key(window)][q]
            print("{:>8} {:>5} {:7.2f} {:7.2f} {:+7.2f} {:+6.0f}%".format(
                window, "p" + q, r, c, c - r, 100.0 * (c - r) / r))
        print()
    print("diff = copies - real. Negative: the copies' ball is smoother (more blurred) than the "
          "real one at that window.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--real", help="ball_gap_stats.json from measure_ball_gaps.py")
    args = parser.parse_args()
    copy = copies()
    print("noisy copies (live{} seed {}, games {}, averaged):".format(
        "/".join(map(str, COPY_LEVELS)), COPY_SEED, ", ".join(map(str, COPY_GAMES))))
    show(copy, "  ")
    if args.real:
        with open(args.real) as handle:
            real = json.load(handle)
        print("real ({}):".format(real.get("game", os.path.basename(args.real))))
        show(real, "  ")
        compare(real, copy)


if __name__ == "__main__":
    main()
