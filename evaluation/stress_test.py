"""Stress test: does a model hold up if our noise imitates the provided tracking wrongly?

The noisy copies copy summary statistics of the provided tracking. This script asks what happens
if one of them is off. It fits the model per round exactly as score_external.py does (two games,
standard live20/35 seed-0 copies, reception key), then predicts each test game on a VARIANT copy
(live35, seed 0) with one noise setting changed, built in memory.

    python evaluation/stress_test.py --model models.touch_dead:TouchDead
    python evaluation/stress_test.py --model models.touch_dead:TouchDead --variants calibrated blur_longer wobble

Most variants change one noise setting. In the copies, a ball shown in a stoppage is being carried
to the restart (metrica_noise.carry_dead_ball); the stoppage_ball_* variants change that: never
shown, shown in more stoppages, shown on every stoppage frame, lying still as in the old copies,
or carried more slowly.

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
SETTINGS = ("CUTS_PER_MINUTE", "BALL_SMOOTH_FRAMES", "GAP_QUANTILES_S", "MAX_LIVE_GAP_S",
            "DEAD_VISIBLE_PROB", "PLAYER_JITTER_M", "AIR_DROP_PROB", "CARRY_SPEED_MS",
            "carry_dead_ball")


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
        "stoppage_ball_never": ("ball never shown in stoppages", {"DEAD_VISIBLE_PROB": 0.0}, LEVEL),
        "stoppage_ball_more": ("ball shown in every stoppage (for part of it)",
                               {"DEAD_VISIBLE_PROB": 1.0}, LEVEL),
        "stoppage_ball_every_frame": ("ball shown on every stoppage frame", {}, LEVEL,
                                      carry_every_stoppage_frame),
        "stoppage_ball_still": ("stoppage ball lying still (the old copies)",
                                {"carry_dead_ball": lambda ball, labels, rng: ball}, LEVEL),
        "stoppage_ball_slow": ("stoppage ball carried slowly (1-3 m/s, not 4-6)",
                               {"CARRY_SPEED_MS": (1.0, 3.0)}, LEVEL),
        "wobble": ("player position wobble 0.1 m per frame", {"PLAYER_JITTER_M": 0.1}, LEVEL),
        "high_balls_lost": ("every high ball lost", {"AIR_DROP_PROB": 1.0}, LEVEL),
    }


def carry_every_stoppage_frame(data, labels, rng):
    """Show the carried stoppage ball on every stoppage frame, not just where the copy shows one:
    the same carry as the copies (metrica_noise.carried_ball), blurred and jittered like them."""
    ball, pitch = data["ball"].copy(), np.array(mn.mt.PITCH, dtype=np.float64)
    seen = np.flatnonzero(np.isfinite(ball[:, 0]))
    for start, stop in mn.runs(labels == "D"):
        before, after = seen[seen < start], seen[seen >= stop]
        if not len(before) and not len(after):
            continue
        a = ball[before[-1]] if len(before) else ball[after[0]]
        b = ball[after[0]] if len(after) else a
        n = stop - start
        path = mn.carried_ball(n, a * pitch, b * pitch, rng) / pitch
        blur = min(mn.BALL_SMOOTH_FRAMES, n if n % 2 else n - 1)   # odd, no longer than the stoppage
        path = mn.jitter(mn.smooth(path, max(blur, 1)), mn.BALL_JITTER_M, rng)
        ball[start:stop] = path
    data["ball"] = ball


# -- player-tracking error, added on top of the STORED copies (no calibration file needed) --------
#
# The copies cut tracks, drop players briefly and swap close players within a team, with a tiny
# per-frame jitter. Real optical tracking also has slow position error and identity swaps in duels,
# which can cross teams. Sizes from public validation work, not from the provided match:
#   * Modern multi-camera optical systems measure about 0.1 m RMSE against a reference system
#     (e.g. Linke, Link & Lames 2020, PLoS ONE, on TRACAB); single-camera or broadcast-based
#     tracking is markedly worse, around a metre, and worse far from the camera. Drift 0.3 and
#     0.5 m stand for a mid-range system, 1.0 m for broadcast-like tracking. The error is slow
#     (smoothed over about a second), as a tracker smooths its own output.
#   * Dropouts: players lost for 1-3 s in crowds or at the image edge, once a minute on top of the
#     copies' own (twice a minute in the harsh variant).
#   * Swaps across teams: when an opponent comes within 1.5 m, a fifth of such encounters swap the
#     two players' positions for 0.5-2 s (two fifths in the harsh variant).
PLAYER_SMOOTH_FRAMES = 25          # 1 s
SWAP_CLOSE_M = 1.5
SWAP_CHECK_FRAMES = 5


def player_drift(data, rng, metres_sd):
    """Slow position error, about metres_sd, independent per slot."""
    players = data["players"]
    for slot in range(players.shape[1]):
        players[:, slot] += mn.slow_wobble(len(players), PLAYER_SMOOTH_FRAMES, metres_sd, rng)


def player_dropouts(data, rng, per_minute, seconds=(1.0, 3.0)):
    """Each slot goes missing for 1-3 s, per_minute times a minute while it is tracked."""
    players = data["players"]
    n = len(players)
    for slot in range(players.shape[1]):
        present = np.isfinite(players[:, slot, 0])
        for frame in np.flatnonzero(present & (rng.random(n) < per_minute / (60.0 * mn.FPS))):
            players[frame:frame + int(rng.uniform(*seconds) * mn.FPS), slot] = np.nan


def cross_team_swaps(data, rng, prob, seconds=(0.5, 2.0)):
    """When a home and an away slot come within SWAP_CLOSE_M, swap their positions for 0.5-2 s
    with probability prob (once per encounter)."""
    players, team = data["players"], data["team"]
    home, away = np.flatnonzero(team == "H"), np.flatnonzero(team == "A")
    pitch = np.array(mn.mt.PITCH, dtype=np.float32)
    was_close, busy_until = set(), 0
    for frame in range(0, len(players), SWAP_CHECK_FRAMES):
        h, a = players[frame, home] * pitch, players[frame, away] * pitch
        dist = np.hypot(h[:, None, 0] - a[None, :, 0], h[:, None, 1] - a[None, :, 1])
        close = set(zip(*np.nonzero(dist < SWAP_CLOSE_M)))              # NaN compares False
        for i, j in close - was_close:
            if frame >= busy_until and rng.random() < prob:
                stop = frame + int(rng.uniform(*seconds) * mn.FPS)
                si, sj = home[i], away[j]
                players[frame:stop, [si, sj]] = players[frame:stop, [sj, si]]
                busy_until = stop
        was_close = close


def player_noise(drift_m=0.0, dropouts_per_minute=0.0, swap_prob=0.0):
    def edit(data, rng):
        if drift_m:
            player_drift(data, rng, drift_m)
        if dropouts_per_minute:
            player_dropouts(data, rng, dropouts_per_minute)
        if swap_prob:
            cross_team_swaps(data, rng, swap_prob)
    return edit


def player_variants():
    """name -> (description, edit of a stored copy). Applied on top of the stored copies."""
    return {
        "players_as_scored": ("stored copy, no extra player error", player_noise()),
        "players_drift0.3": ("player drift 0.3 m (slow)", player_noise(drift_m=0.3)),
        "players_drift0.5": ("player drift 0.5 m (slow)", player_noise(drift_m=0.5)),
        "players_dropouts": ("players lost 1-3 s, once a minute", player_noise(dropouts_per_minute=1)),
        "players_swaps": ("cross-team swaps in duels (20%)", player_noise(swap_prob=0.2)),
        "players_all": ("drift 0.5 m + dropouts + swaps", player_noise(0.5, 1, 0.2)),
        "players_harsh": ("drift 1 m + dropouts x2 + swaps 40%", player_noise(1.0, 2, 0.4)),
    }


def player_variant_copy(name, game, level, seed=SEED):
    """A stored copy with the named player error added (reproducible per game, level, seed)."""
    data = {k: v.copy() for k, v in se.load_tracking(game, level, seed).items()}
    names = list(player_variants())
    rng = np.random.default_rng([game, level, seed, 2, names.index(name)])
    player_variants()[name][1](data, rng)
    return data


def same_tracking(a, b):
    return all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in b)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="module:Class")
    parser.add_argument("--variants", nargs="+",
                        help="names to run (default: all rebuilt variants; players_* variants "
                             "add player error to the stored copies)")
    parser.add_argument("--levels", nargs="+", type=int, default=[LEVEL],
                        help="live levels for the players_* variants (default 35)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)

    players_table = player_variants()
    rebuilt = [n for n in (args.variants or []) if n not in players_table]
    if args.variants is None or rebuilt:
        mn.calibrate(CALIBRATION)
    base = {k: copy.deepcopy(getattr(mn, k)) for k in SETTINGS}
    table = variants(base)
    names = args.variants or list(table)
    unknown = [n for n in names if n not in table and n not in players_table]
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
    for name in [n for n in names if n in players_table]:
        for level in args.levels:
            spell, macro = [], []
            for test in se.GAMES:
                data = player_variant_copy(name, test, level)
                result = pm.evaluate(keys[test], pm.label_codes(models[test].predict(data)))
                spell.append(result["spell_f1_mean"])
                macro.append(result["macro_f1"])
            print("{:<42}{:>8.3f}  {:<20}{:>8.3f}".format(
                "{} (live{})".format(players_table[name][0], level), float(np.mean(spell)),
                " / ".join("{:.2f}".format(s) for s in spell), float(np.mean(macro))), flush=True)
    for name in [n for n in names if n not in players_table]:
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
