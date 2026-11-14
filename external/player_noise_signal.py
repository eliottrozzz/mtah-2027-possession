"""Player-tracking noise in summary numbers: how much do player positions jitter and drift?

The noisy copies give the players almost no position error. The stress test (stress_test.py,
players_* variants) shows v3 loses a lot under slow player drift, so we need to know how much
drift the real tracking has. This prints the same aggregate numbers for:
  * Metrica's clean tracking (games 1-3),
  * the noisy copies (live20/35, seed 0, averaged),
  * the stress test's player-drift variants (live35, seed 0),
  * and, only when you pass --match, the provided match (run that yourself; agents never do).

    python external/player_noise_signal.py             # Metrica and the variants only
    python external/player_noise_signal.py --match     # also the provided match: Jordan only

With --match it reads the provided ATD exactly as measure_ball_gaps.py does (read_atd, graded
halves only) and prints one summary line. No frame, position or label is printed or saved, so the
line is safe to paste.

Numbers, per tracked player-frame inside the halves:
  speed_*      speed (m/s) over 1 frame, 0.4 s and 1 s, percentiles p10 / p50 / p90 / p99. Slow
               drift lifts the low percentiles at 0.4-1 s (a standing player seems to move).
  accel_*      change of the 0.4 s velocity over 0.4 s (m/s^2), p50 / p90 / p99.
  jitter_m     frame-to-frame wobble (one axis), the same formula as measure_ball_gaps.wobble_m.
  jumps        share of 1-frame moves faster than 12 m/s (a swap or a glitch, not running).
This only measures: it changes no calibration and writes no files.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE, os.path.join(ROOT, "evaluation")]

# Params
FPS = 25
PITCH = (105.0, 68.0)
STEPS_S = (1.0 / FPS, 0.4, 1.0)
SPEED_Q = (10, 50, 90, 99)
ACCEL_Q = (50, 90, 99)
ACCEL_STEP_S = 0.4
JUMP_SPEED = 12.0                         # m/s, as dead_ball.MAX_SPEED
COPY_GAMES = (1, 2, 3)
COPY_LEVELS = (20, 35)
VARIANTS = ("players_drift0.3", "players_drift0.5", "players_all", "players_harsh")
VARIANT_LEVEL = 35
SEED = 0


def velocity(pos, step_frames):
    """(frames, players, 2) m/s from step_frames/2 before to after; NaN at the ends."""
    half = max(1, step_frames // 2)
    out = np.full(pos.shape, np.nan, dtype=np.float32)
    out[half:-half] = (pos[2 * half:] - pos[:-2 * half]) * FPS / (2 * half)
    return out


def player_signal(players_graded, pitch=PITCH):
    """{name: value} for players (frames, slots, 2) in 0-1 units, NaN outside the graded halves."""
    pos = players_graded * np.array(pitch, dtype=np.float32)
    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        one = np.hypot(*(pos[1:] - pos[:-1]).transpose(2, 0, 1)) * FPS
        one = one[np.isfinite(one)]
        stats["jumps"] = float(np.mean(one > JUMP_SPEED))
        for step in STEPS_S:
            frames = max(1, int(round(step * FPS)))
            speed = one if frames == 1 else np.hypot(*velocity(pos, frames).transpose(2, 0, 1))
            speed = speed[np.isfinite(speed) & (speed <= JUMP_SPEED)]
            stats["speed_{:.2g}s".format(step)] = {str(q): float(np.percentile(speed, q))
                                                   for q in SPEED_Q}
        k = int(round(ACCEL_STEP_S * FPS))
        v = velocity(pos, k)
        accel = np.hypot(*((v[k:] - v[:-k]) / ACCEL_STEP_S).transpose(2, 0, 1))
        accel = accel[np.isfinite(accel)]
        stats["accel"] = {str(q): float(np.percentile(accel, q)) for q in ACCEL_Q}
        second = (pos[2:] - 2 * pos[1:-1] + pos[:-2]).reshape(-1)
        second = np.abs(second[np.isfinite(second)])
        stats["jitter_m"] = float(np.median(second) / 0.6745 / np.sqrt(6))
    return stats


def graded_players(tracking):
    return np.where((tracking["period"] > 0)[:, None, None], tracking["players"], np.nan)


def average(found):
    out = {}
    for key, value in found[0].items():
        out[key] = ({q: float(np.mean([f[key][q] for f in found])) for q in value}
                    if isinstance(value, dict) else float(np.mean([f[key] for f in found])))
    return out


def line(label, stats):
    parts = ["{:<26}".format(label), "jitter {:.3f} m".format(stats["jitter_m"]),
             "jumps {:.4f}".format(stats["jumps"])]
    for step in STEPS_S:
        key = "speed_{:.2g}s".format(step)
        parts.append("speed {}s p{} {}".format(
            "{:.2g}".format(step), "/".join(map(str, SPEED_Q)),
            "/".join("{:.2f}".format(stats[key][str(q)]) for q in SPEED_Q)))
    parts.append("accel p{} {}".format("/".join(map(str, ACCEL_Q)),
                                       "/".join("{:.1f}".format(stats["accel"][str(q)])
                                                for q in ACCEL_Q)))
    return "  ".join(parts)


def metrica_rows():
    import metrica_tracking as mt
    import score_external as se
    import stress_test as st
    rows = [("metrica clean", average([player_signal(graded_players(mt.load_game(g)))
                                       for g in COPY_GAMES]))]
    rows.append(("copies live{} seed{}".format("/".join(map(str, COPY_LEVELS)), SEED),
                 average([player_signal(graded_players(se.load_tracking(g, lv, SEED)))
                          for g in COPY_GAMES for lv in COPY_LEVELS])))
    for name in VARIANTS:
        rows.append((name.replace("players_", "") + " (live{})".format(VARIANT_LEVEL),
                     average([player_signal(graded_players(
                         st.player_variant_copy(name, g, VARIANT_LEVEL))) for g in COPY_GAMES])))
    return rows


def match_row():
    """The provided match, read exactly as measure_ball_gaps.py reads it. Jordan only."""
    import xml.etree.ElementTree as ET
    import measure_ball_gaps as mb
    meta_path, raw_path = mb.require(["atd_meta", "atd_raw"], mb.GAME, None)
    n_slots = len(list(ET.parse(meta_path).getroot().iter("Player")))
    _, players = mb.read_atd(raw_path, mb.source_frames("tactical", mb.GAME), n_slots)
    graded = np.zeros(len(players), dtype=bool)
    for _, first, last in mb.half_video_blocks(mb.GAME):
        graded[first:last + 1] = True
    return player_signal(np.where(graded[:, None, None], players, np.nan),
                         mb.GAMES[mb.GAME]["pitch"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--match", action="store_true",
                        help="also measure the provided match (run it yourself; prints one line)")
    args = parser.parse_args()
    for label, stats in metrica_rows():
        print(line(label, stats))
    if args.match:
        print(line("provided match", match_row()))


if __name__ == "__main__":
    main()
