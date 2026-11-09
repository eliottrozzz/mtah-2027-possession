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
CROWD_RADIUS_M = 3.0

# Ball during stoppages.
DEAD_VISIBLE_PROB = 0.5        # chance a stoppage shows a still ball for part of it

# Jitter, metres.
BALL_JITTER_M = 0.3
PLAYER_JITTER_M = 0.15

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
    global GAP_QUANTILES_S, SPEED_RATIOS, CROWD_RATIOS, CUTS_PER_MINUTE
    with open(path) as handle:
        stats = json.load(handle)
    GAP_QUANTILES_S = stats["short_gap_quantiles_s"]
    SPEED_RATIOS = {k: v["ratio"] for k, v in stats["hazard_speed"].items()}
    CROWD_RATIOS = {k: v["ratio"] for k, v in stats["hazard_crowd"].items()}
    # A median fragment of m seconds means a break roughly every m seconds.
    CUTS_PER_MINUTE = 60.0 / max(stats["fragment_quantiles_s"]["50"], 1.0)
    return stats


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
    weight = bin_ratio(speed, SPEED_RATIOS) * bin_ratio(crowd, CROWD_RATIOS)
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

    ball = game["ball"].copy()
    reason = np.where(seen, 0, 1).astype(np.int8)    # 0 seen, 1 missing in the clean data
    reason[seen & gaps] = 3                            # 3 random gap
    reason[seen & air] = 2                             # 2 ball in the air
    ball[reason > 1] = np.nan
    ball = show_ball_in_stoppages(ball, labels, rng)
    reason[(reason == 1) & ~np.isnan(ball[:, 0])] = 4  # 4 still ball shown in a stoppage
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--calibration", help="ball_gap_stats.json from measure_ball_gaps.py")
    args = parser.parse_args()

    if args.calibration:
        stats = calibrate(args.calibration)
        print("calibrated on {}: real match missing {:.0%} of graded frames".format(
            stats["game"], stats["ball_missing_rate"]))
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
