"""Noisy copies of the external games' tracking, so they look like the provided match's.

Metrica's sample tracking is clean: the ball is tracked on 98-99% of live frames and every player
keeps one column all game. The provided match loses the ball on 41% of graded frames and breaks
22 players into 46 anonymous track slots. A model trained on the clean version learns shortcuts
that fail on the real match ("no ball means dead ball", "follow player 7"), so every training,
tuning and scoring run uses these noisy copies instead. The clean originals are never changed.

    python external/metrica_noise.py                                     # default noise
    python external/metrica_noise.py --calibration data/ball_gap_stats.json

The calibration file comes from measure_ball_gaps.py, run on the provided tracking. Without it the
defaults below are educated guesses plus the published 41%.

What gets deleted
-----------------
The ball, during live play, where a camera tends to lose it:
  * in the air: long passes, crosses, deep balls, clearances, goal kicks, keeper throws and kicks,
    and shots, dropped for most of their flight
  * in crowds and on fast balls: random gaps that start more often there (hazard ratios)
  * anywhere else: the same random gaps at the base rate
The overall amount is set per level, as the share of live frames without a ball, and the random
rate is solved for to hit it exactly.

During some stoppages the ball is put back, still, where play stopped, so a visible ball does not
mean live play either. Small position jitter goes on everything.

Players: each track is cut at random and when two players come close, short stretches are
dropped, and the pieces go into anonymous slots per team, like the provided ATD.

Output: data/external/metrica/noisy/game{g}_live{level}_seed{s}.npz, the shared format of
metrica_tracking.py, plus ball_reason (why the ball is missing on each frame, for analysis only).
Use different seeds for tuning and for testing.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrica_labels as ml  # noqa: E402
import metrica_tracking as mt  # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from models.touch import touch_strength  # noqa: E402  (the same touch signal the model reads)

# Params
OUT_DIR = os.path.join(mt.OUT_DIR, "noisy")
FPS = 25
LEVELS = (20, 35, 50)          # % of live frames with no ball
SEEDS = (0, 1)                 # 0 for tuning, 1 for testing

# Ball in the air. A flight is a pass longer than LONG_PASS_M, a shot, or one of AIR_SUBTYPES.
LONG_PASS_M = 30.0
AIR_SUBTYPES = {"CROSS", "DEEP BALL", "CLEARANCE", "GOAL KICK", "THROUGH BALL"}
AIR_DROP_PROB = 0.6            # chance a flight loses the ball
KEEPER_RADIUS_M = 2.0          # a pass starting this close to a keeper is a keeper kick or throw

# Random gaps. Lengths in seconds as percentiles, multipliers per situation. Replaced by the
# measured values when a calibration file is given; bins match measure_ball_gaps.py.
GAP_QUANTILES_S = {"10": 0.2, "25": 0.4, "50": 1.0, "75": 2.0, "90": 4.0, "95": 6.0, "99": 12.0}
MAX_LIVE_GAP_S = 15.0          # longer gaps in the real match are mostly stoppages
SPEED_RATIOS = {"0-5": 0.8, "5-15": 1.0, "15-25": 2.0, "25-inf": 3.0}
CROWD_RATIOS = {"0-1": 0.8, "1-2": 1.0, "2-3": 1.5, "3-inf": 2.5}
GOAL_RATIOS = {"0-1": 1.0, "1-2": 1.0}          # "1-2" is near a goal line
TOUCHLINE_RATIOS = {"0-1": 1.0, "1-2": 1.0}     # "1-2" is near a touchline
CROWD_RADIUS_M = 3.0
NEAR_GOAL = 0.17               # as in measure_ball_gaps.py
NEAR_TOUCHLINE = 0.10

# Ball during stoppages.
DEAD_VISIBLE_PROB = 0.5        # chance a stoppage shows a ball for part of it
# Where a ball is shown in a stoppage it is being carried to the restart, not lying still. The
# provided tracking's touch signal at 0.32 s and 0.48 s windows (median about 2.5 m/s at 0.48 s)
# matches our live play alone, so its stoppage ball is moving or missing, not still; a still ball
# made the copies' median half the real one (checked 2026-11-13, summary numbers only). Speed and
# steering follow Track B's stress test; CARRY_SPEED_MS is fitted with --fit-stoppages.
# Fitted 2026-11-13 (miss over the 15 percentiles): 0.5-1.5 m/s 3.70, 1-3 2.46, 2-4 2.08, 3-5 1.59,
# 4-6 1.41, 5-7 1.67, 2-6 1.52; the still ball it replaces scored about 2.4. A ball rolled or passed
# back to the restart. Trade-off: the 4-frame p99 rises to about 7.7 m/s (real 5.9).
CARRY_SPEED_MS = (4.0, 6.0)    # drawn per stoppage
CARRY_SPEED_CANDIDATES = ((0.5, 1.5), (1.0, 3.0), (2.0, 4.0), (3.0, 5.0), (4.0, 6.0), (5.0, 7.0),
                          (2.0, 6.0))
CARRY_TURN_PER_S = 1.0         # how fast it steers toward the restart spot
CARRY_WANDER = 1.0             # random turning, radians per sqrt(second)

# Jitter, metres: independent wobble on every frame.
BALL_JITTER_M = 0.3
PLAYER_JITTER_M = 0.15

# Smoothing. The provided tracking barely wobbles (about 0.003 m per frame) but is smoothed over
# time, which blurs every kick: its velocity change over 0.16 s is under 6 m/s 99% of the time.
# So, when calibrated, the ball is smoothed over BALL_SMOOTH_FRAMES and gets a slow wobble of
# BALL_WOBBLE_M, both picked so Metrica's touch signal matches the measured percentiles.
BALL_SMOOTH_FRAMES = 1         # 1 = no smoothing (uncalibrated default)
BALL_WOBBLE_M = 0.0
SMOOTH_CANDIDATES = (1, 5, 9, 11, 13, 15, 17, 19, 21, 25, 29, 33)
WOBBLE_CANDIDATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)

# Players.
CUTS_PER_MINUTE = 2.0          # random breaks per player
CLOSE_M = 1.0                  # two players this close may swap or lose a track
CLOSE_CUT_PROB = 0.3
DROPOUT_PROB = 0.8             # chance a break also loses the player for a moment
DROPOUT_S = (0.2, 4.0)
SLOTS_PER_TEAM = 23            # 46 slots for 22 players, as in the provided ATD


def metres(xy):
    return xy * np.array(mt.PITCH, dtype=np.float32)


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def calibrate(path):
    """Replace the default gap lengths and ratios with ones measured on the provided match."""
    global GAP_QUANTILES_S, SPEED_RATIOS, CROWD_RATIOS, GOAL_RATIOS, TOUCHLINE_RATIOS
    global AIR_DROP_PROB, CUTS_PER_MINUTE
    with open(path) as handle:
        stats = json.load(handle)
    GAP_QUANTILES_S = stats["short_gap_quantiles_s"]
    SPEED_RATIOS = {k: v["ratio"] for k, v in stats["hazard_speed"].items()}
    CROWD_RATIOS = {k: v["ratio"] for k, v in stats["hazard_crowd"].items()}
    GOAL_RATIOS = {k: v["ratio"] for k, v in stats["hazard_near_goal"].items()}
    TOUCHLINE_RATIOS = {k: v["ratio"] for k, v in stats["hazard_near_touchline"].items()}
    # Flights lose the ball as often as the real tracker loses a fast ball (over 15 m/s).
    fast = [v for k, v in stats["hazard_speed"].items() if float(k.split("-")[0]) >= 15]
    share = sum(v["share_of_frames"] for v in fast)
    if share:
        AIR_DROP_PROB = min(1.0, AIR_DROP_PROB * sum(v["ratio"] * v["share_of_frames"]
                                                     for v in fast) / share)
    # A median fragment of m seconds means a break roughly every m seconds.
    CUTS_PER_MINUTE = 60.0 / max(stats["fragment_quantiles_s"]["50"], 1.0)
    # Position wobble, when measured. Metrica's own tracking is much smoother, so adding the
    # measured amount on top is close to the real total.
    global BALL_JITTER_M, PLAYER_JITTER_M, BALL_SMOOTH_FRAMES, BALL_WOBBLE_M
    BALL_JITTER_M = stats.get("ball_jitter_m", BALL_JITTER_M)
    PLAYER_JITTER_M = stats.get("player_jitter_m", PLAYER_JITTER_M)
    if "touch_strength_quantiles" in stats:
        BALL_SMOOTH_FRAMES, BALL_WOBBLE_M = match_touch_signal(stats["touch_strength_quantiles"])
    return stats


def smooth(xy, frames):
    """Moving average over `frames` frames, ignoring missing ones; NaN where most are missing."""
    if frames <= 1:
        return xy
    kernel = np.ones(frames) / frames
    out = np.full(xy.shape, np.nan, dtype=np.float64)
    for axis in range(xy.shape[1]):
        values = xy[:, axis].astype(np.float64)
        ok = np.isfinite(values)
        total = np.convolve(np.where(ok, values, 0.0), kernel, "same")
        weight = np.convolve(ok.astype(np.float64), kernel, "same")
        out[:, axis] = np.where(ok & (weight > 0.5), total / np.maximum(weight, 1e-9), np.nan)
    return out.astype(np.float32)


def slow_wobble(n, frames, metres_sd, rng):
    """Smoothed random wobble with a spread of about metres_sd, in 0-1 pitch units."""
    if metres_sd <= 0:
        return np.zeros((n, 2), dtype=np.float32)
    white = rng.normal(0.0, 1.0, (n, 2)) * metres_sd * np.sqrt(max(frames, 1))
    return (smooth(white, frames) / np.array(mt.PITCH)).astype(np.float32)


def touch_quantiles(ball):
    strength = touch_strength(ball)
    strength = strength[np.isfinite(strength)]
    return np.percentile(strength, [float(q) for q in sorted(QUANTILE_KEYS, key=float)])


QUANTILE_KEYS = ("50", "75", "90", "95", "99")


def match_touch_signal(target):
    """(smoothing frames, wobble m) whose touch signal on Metrica game 1 best matches target.

    Smoothing is matched on the upper percentiles (the kicks), then the wobble on all of them
    (it mostly lifts the median). Distances are on a log scale so each percentile counts equally.
    """
    goal = np.array([target[q] for q in sorted(QUANTILE_KEYS, key=float)])
    clean = mt.load_game(1)["ball"]

    def miss(ball, upper_only):
        got = touch_quantiles(ball)
        err = np.abs(np.log(np.maximum(got, 1e-3) / goal))
        return float(err[1:].sum() if upper_only else err.sum())

    frames = min(SMOOTH_CANDIDATES, key=lambda f: miss(smooth(clean, f), True))
    rng = np.random.default_rng(0)
    wobble = {m: slow_wobble(len(clean), frames, m, rng) for m in WOBBLE_CANDIDATES}
    smoothed = smooth(clean, frames)
    metres_sd = min(WOBBLE_CANDIDATES, key=lambda m: miss(smoothed + wobble[m], False))
    final = touch_quantiles(smoothed + wobble[metres_sd])
    print("touch signal: smoothing {} frames ({:.2f} s), slow wobble {} m -> p50-p99 {} "
          "(real {})".format(frames, frames / FPS, metres_sd, np.round(final, 1).tolist(),
                             np.round(goal, 1).tolist()))
    return frames, metres_sd


def bin_ratio(values, ratios):
    """Per-frame multiplier from bins named 'low-high', as measure_ball_gaps.py writes them."""
    out = np.ones(len(values))
    for name, ratio in ratios.items():
        low, high = (float(v) for v in name.split("-"))
        out[(values >= low) & (values < high)] = ratio
    return out


def sample_gap_lengths(rng, n):
    """Gap lengths in frames, drawn from the percentile table by linear interpolation."""
    qs = np.array(sorted(float(q) for q in GAP_QUANTILES_S)) / 100.0
    vals = np.array([GAP_QUANTILES_S[k] for k in sorted(GAP_QUANTILES_S, key=float)])
    qs, vals = np.concatenate(([0.0], qs, [1.0])), np.concatenate(([1.0 / FPS], vals, [vals[-1]]))
    seconds = np.interp(rng.random(n), qs, vals)
    return np.clip(np.round(seconds * FPS), 1, MAX_LIVE_GAP_S * FPS).astype(int)


def keepers(game, labels_live):
    """Column of each team's keeper per period: the player deepest towards their own goal."""
    out = {}
    for period in (1, 2):
        rows = (game["period"] == period) & labels_live
        x = np.nanmedian(game["players"][rows, :, 0], axis=0)
        for team in ("H", "A"):
            cols = np.flatnonzero((game["team"] == team) & ~np.isnan(x))
            own_goal_low = np.nanmean(x[cols]) < 0.5
            out[(period, team)] = cols[np.argmin(x[cols]) if own_goal_low else np.argmax(x[cols])]
    return out


def flights(game_no, game, live, rng):
    """Frames where the ball is in the air and the camera loses it."""
    drop = np.zeros(len(game["ball"]), dtype=bool)
    keeper = keepers(game, live)
    ball = metres(game["ball"])
    for event in ml.load_events(game_no):
        start, end = event["start"], ml.end_frame(event)
        if event["type"] not in ("PASS", "SHOT") or end - start < 10 or end >= len(ball):
            continue
        length = np.hypot(*(ball[end] - ball[start]))
        gk = keeper.get((event["period"], event["team"]))
        from_keeper = gk is not None and np.hypot(
            *(metres(game["players"][start, gk]) - ball[start])) < KEEPER_RADIUS_M
        airborne = (event["type"] == "SHOT" or event["sub"] & AIR_SUBTYPES or from_keeper
                    or length > LONG_PASS_M)
        if airborne and rng.random() < AIR_DROP_PROB:
            # Lost a few frames after the kick, found again a few frames before it lands.
            first = start + rng.integers(2, 7)
            last = end - rng.integers(0, 9)
            drop[first:max(first, last)] = True
    return drop


def random_gaps(game, live, seen, target_missing, fixed_drop, rng):
    """Random gaps, solved so live frames end up target_missing without a ball.

    Draws are fixed up front (one uniform number and one gap length per frame), so changing the
    base rate only changes which draws fire: the result is monotonic and bisection finds it.
    """
    n = len(seen)
    ball = metres(game["ball"])
    speed = np.full(n, np.nan)
    speed[2:] = np.hypot(*(ball[2:] - ball[:-2]).T) * FPS / 2.0
    near = np.hypot(*(metres(game["players"]) - ball[:, None, :]).transpose(2, 0, 1))
    crowd = np.sum(near < CROWD_RADIUS_M, axis=1).astype(float)
    xy = game["ball"]
    near_goal = (np.minimum(xy[:, 0], 1 - xy[:, 0]) < NEAR_GOAL).astype(float)
    near_touch = (np.minimum(xy[:, 1], 1 - xy[:, 1]) < NEAR_TOUCHLINE).astype(float)
    weight = (bin_ratio(speed, SPEED_RATIOS) * bin_ratio(crowd, CROWD_RATIOS)
              * bin_ratio(near_goal, GOAL_RATIOS) * bin_ratio(near_touch, TOUCHLINE_RATIOS))
    weight[~(live & seen)] = 0.0

    draws = rng.random(n)
    lengths = sample_gap_lengths(rng, n)
    index = np.arange(n)

    def gaps(rate):
        starts = draws < rate * weight
        reach = np.where(starts, index + lengths, 0)
        return np.maximum.accumulate(reach) > index

    def missing_share(rate):
        missing = ~seen | fixed_drop | gaps(rate)
        return missing[live].mean()

    low, high = 0.0, 1.0
    if missing_share(high) < target_missing:
        raise ValueError("cannot reach {:.0%} missing".format(target_missing))
    for _ in range(40):
        mid = (low + high) / 2.0
        low, high = (mid, high) if missing_share(mid) < target_missing else (low, mid)
    return gaps(high)


def show_ball_in_stoppages(ball, labels, rng):
    """Put a still ball back during part of some stoppages, where play stopped."""
    out = ball.copy()
    for start, stop in runs(labels == "D"):
        if rng.random() >= DEAD_VISIBLE_PROB:
            continue
        before = np.flatnonzero(~np.isnan(ball[:start, 0]))
        if not len(before):
            continue
        spot = np.clip(ball[before[-1]], 0.0, 1.0)
        first = start + int(rng.random() * 0.7 * (stop - start))
        hole = np.isnan(out[first:stop, 0])
        out[first:stop][hole] = spot
    return out


def carried_ball(n, start, target, rng):
    """(n, 2) path in metres: carried from start toward target at a walking or jogging pace,
    steering with some random wandering, then circling near the spot so it never sits still."""
    speed = rng.uniform(*CARRY_SPEED_MS)
    pos, path = np.array(start, dtype=np.float64), np.empty((n, 2))
    heading = np.arctan2(*(target - pos)[::-1]) + rng.normal(0.0, 0.5)
    for i in range(n):
        wanted = np.arctan2(*(target - pos)[::-1])
        turn = (wanted - heading + np.pi) % (2 * np.pi) - np.pi
        heading += CARRY_TURN_PER_S * turn / FPS + rng.normal(0.0, CARRY_WANDER / np.sqrt(FPS))
        pos = pos + speed / FPS * np.array([np.cos(heading), np.sin(heading)])
        path[i] = pos
    return path


def carry_dead_ball(ball, labels, rng):
    """Replace every ball shown in a stoppage with one carried from the last ball seen before it
    to the first seen after it (where play restarts), blurred like the rest of the ball."""
    out, pitch = ball.copy(), np.array(mt.PITCH, dtype=np.float64)
    seen = np.flatnonzero(np.isfinite(ball[:, 0]))
    for start, stop in runs(labels == "D"):
        shown = np.isfinite(ball[start:stop, 0])
        if not shown.any():
            continue
        before, after = seen[seen < start], seen[seen >= stop]
        a = ball[before[-1]] if len(before) else ball[start + np.argmax(shown)]
        b = ball[after[0]] if len(after) else a
        n = stop - start
        path = carried_ball(n, a * pitch, b * pitch, rng)
        blur = min(BALL_SMOOTH_FRAMES, n if n % 2 else n - 1)
        out[start:stop][shown] = (smooth(path, max(blur, 1)) / pitch)[shown]
    return out.astype(np.float32)


def break_tracks(game, rng):
    """Cut every player's track into pieces and deal the pieces into anonymous team slots."""
    players = game["players"]
    n, n_players, _ = players.shape
    pos = metres(players)
    present = ~np.isnan(players[:, :, 0])

    pieces = []
    for p in range(n_players):
        cut = rng.random(n) < CUTS_PER_MINUTE / (60.0 * FPS)
        others = np.delete(pos, p, axis=1)
        close = np.nanmin(np.hypot(*(others - pos[:, p:p + 1]).transpose(2, 0, 1)), axis=1) < CLOSE_M
        close_start = close & ~np.concatenate(([False], close[:-1]))
        cut |= close_start & (rng.random(n) < CLOSE_CUT_PROB)
        alive = present[:, p].copy()
        for frame in np.flatnonzero(cut & alive):
            if rng.random() < DROPOUT_PROB:
                alive[frame:frame + int(rng.uniform(*DROPOUT_S) * FPS)] = False
        cut_points = set(np.flatnonzero(cut).tolist())
        for start, stop in runs(alive):
            bounds = [start] + sorted(f for f in cut_points if start < f < stop) + [stop]
            pieces += [(a, b, p) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    out = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    teams = np.array(["H"] * SLOTS_PER_TEAM + ["A"] * SLOTS_PER_TEAM)
    free_from = np.zeros(2 * SLOTS_PER_TEAM, dtype=int)
    for start, stop, p in sorted(pieces):
        slots = np.flatnonzero((teams == game["team"][p]) & (free_from <= start))
        if not len(slots):
            continue  # more pieces at once than slots: drop it, the real tracker does too
        slot = rng.choice(slots)
        out[start:stop, slot] = players[start:stop, p]
        free_from[slot] = stop + 1
    return out, teams


def jitter(xy, metres_sd, rng):
    noise = rng.normal(0.0, 1.0, xy.shape) * (metres_sd / np.array(mt.PITCH, dtype=np.float32))
    return (xy + noise).astype(np.float32)


def make_noisy(game_no, level, seed):
    rng = np.random.default_rng([game_no, level, seed])
    game = mt.load_game(game_no)
    labels = ml.label_game(game_no, "reception")
    n = len(game["ball"])
    labels = np.concatenate([labels, np.full(max(0, n - len(labels)), "X")])[:n]
    live = np.isin(labels, ["H", "A"])
    seen = ~np.isnan(game["ball"][:, 0])

    air = flights(game_no, game, live, rng)
    gaps = random_gaps(game, live, seen, level / 100.0, air, rng)

    # Smooth first, like a tracker does on the path it sees, then cut the gaps out of it.
    ball = smooth(game["ball"], BALL_SMOOTH_FRAMES)
    ball = ball + slow_wobble(n, BALL_SMOOTH_FRAMES, BALL_WOBBLE_M, rng)
    reason = np.where(seen, 0, 1).astype(np.int8)    # 0 seen, 1 missing in the clean data
    reason[seen & gaps] = 3                            # 3 random gap
    reason[seen & air] = 2                             # 2 ball in the air
    ball[reason > 1] = np.nan
    ball = show_ball_in_stoppages(ball, labels, rng)
    reason[(reason == 1) & ~np.isnan(ball[:, 0])] = 4  # 4 ball shown in a stoppage, put back
    ball = carry_dead_ball(ball, labels, rng)          # every stoppage ball shown is carried
    ball = jitter(ball, BALL_JITTER_M, rng)

    players, teams = break_tracks(game, rng)
    players = jitter(players, PLAYER_JITTER_M, rng)
    return {
        "ball": ball, "players": players, "team": teams,
        "player": np.array(["slot{}".format(i + 1) for i in range(len(teams))]),
        "period": game["period"], "ball_reason": reason,
    }, labels


def describe(game_no, level, seed, data, labels):
    live, dead = np.isin(labels, ["H", "A"]), labels == "D"
    graded = live | dead
    missing = np.isnan(data["ball"][:, 0])
    reason = data["ball_reason"]
    objects = (~np.isnan(data["players"][:, :, 0])).sum(axis=1)
    print("game {} live{} seed{}  no ball: {:.0%} of live, {:.0%} of dead, {:.0%} overall  "
          "(air {:.0%}, random {:.0%} of live)  players per frame {:.1f}".format(
              game_no, level, seed, missing[live].mean(), missing[dead].mean(),
              missing[graded].mean(), (reason[live] == 2).mean(), (reason[live] == 3).mean(),
              objects[graded].mean()))


def fit_stoppage_ball(stats):
    """Print, for each CARRY_SPEED_CANDIDATES range, how well the finished copies' touch signal
    at windows 4, 8 and 12 matches the real one (summed |log ratio| over the percentiles), using
    live20/35 seed-0 copies of all three games built in memory. Writes nothing."""
    from models.touch import touch_strength
    keys = {4: "touch_strength_quantiles", 8: "touch_strength_quantiles_w8",
            12: "touch_strength_quantiles_w12"}
    qs = sorted(QUANTILE_KEYS, key=float)
    goal = {w: np.array([stats[k][q] for q in qs]) for w, k in keys.items() if k in stats}
    global CARRY_SPEED_MS
    keep, results = CARRY_SPEED_MS, []
    for speeds in CARRY_SPEED_CANDIDATES:
        CARRY_SPEED_MS = speeds
        got = {w: [] for w in goal}
        for game_no in (1, 2, 3):
            for level in (20, 35):
                data, _ = make_noisy(game_no, level, 0)
                for w in goal:
                    st = touch_strength(data["ball"], w)
                    st = st[(data["period"] > 0) & np.isfinite(st)]
                    got[w].append(np.percentile(st, [float(q) for q in qs]))
        got = {w: np.mean(v, axis=0) for w, v in got.items()}
        err = sum(np.abs(np.log(np.maximum(got[w], 1e-3) / goal[w])).sum() for w in goal)
        results.append((err, speeds))
        print("carry {}-{} m/s: miss {:.2f}   ".format(*speeds, err) + "  ".join(
            "w{} {}".format(w, np.round(got[w], 1).tolist()) for w in goal))
    CARRY_SPEED_MS = keep
    print("real: " + "  ".join("w{} {}".format(w, np.round(goal[w], 1).tolist()) for w in goal))
    print("best:", min(results)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--calibration", help="ball_gap_stats.json from measure_ball_gaps.py")
    parser.add_argument("--fit-stoppages", action="store_true",
                        help="print how well each carry speed matches the real touch signal "
                             "(needs --calibration); writes nothing")
    args = parser.parse_args()

    if args.calibration:
        stats = calibrate(args.calibration)
        print("calibrated on {}: real match missing {:.0%} of graded frames".format(
            stats["game"], stats["ball_missing_rate"]))
        if args.fit_stoppages:
            fit_stoppage_ball(stats)
            return
    os.makedirs(OUT_DIR, exist_ok=True)
    for game_no in args.games:
        for level in args.levels:
            for seed in args.seeds:
                data, labels = make_noisy(game_no, level, seed)
                path = os.path.join(OUT_DIR, "game{}_live{}_seed{}.npz".format(
                    game_no, level, seed))
                np.savez_compressed(path, **data)
                describe(game_no, level, seed, data, labels)
    print("written to", os.path.relpath(OUT_DIR, mt.ROOT))


if __name__ == "__main__":
    main()
