"""Baseline B plus a dead-ball detector that looks mainly at the players.

In our match the ball is missing on 41% of frames, much of it during live play, so "no ball" must
not mean "dead ball". The detector decides dead or alive from what the players do (everyone
slowing down, nobody sprinting, more players walking, the team shapes freezing, players gathering
for a wall or a corner, someone standing at the touchline), with the ball as a supporting clue only
while it is visible (off the pitch, or lying still). Whether the ball is missing is never an input.

How it works:
  1. Per-frame signals from the players and the visible ball, each averaged over several windows
     (centred, just before and just after the frame: play slows after a whistle and speeds up
     after a restart). Player speeds are divided by that half's typical speed, so a slower game
     or noisier tracking doesn't shift everything.
  2. A logistic regression turns them into P(dead), fitted on the training games.
  3. P(dead) is smoothed, thresholded, and short dead or live runs are removed. The threshold and
     minimum lengths are picked in fit() by spellF1 on the training games, never on test scores.
  4. Baseline B (nearest team, switch after 0.5 s, never dead) gives the team; dead frames get D.
  Before all this, short ball gaps are filled where one player clearly carries the ball through
  them (fill_ball_gaps). The longest gap to fill is learned in fit(); longer gaps stay missing.

    python evaluation/score_external.py --model models.dead_ball:DeadBallB --who agentB
"""

import itertools
import os
import sys
import warnings

import numpy as np

from models.baselines import FPS, PITCH, NeverDead

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
MAX_SPEED = 12.0              # m/s; a player "moving" faster than this is a track swap
SPEED_STEP_S = 0.2            # measure player movement over this long, so per-frame wobble in
                              # the tracking cancels out instead of looking like running
WALK_SPEED = 2.0              # m/s, relative units after normalising (see frame_features)
SPRINT_SPEED = 3.5
BALL_STILL_SPEED = 1.0        # m/s
GROUP_RADIUS_M = 4.0          # players this close together count as one group (a wall, a corner)
TOUCHLINE_M = 1.0             # a player within this of a touchline, or beyond it, is "at" it
STANDING_SPEED = 1.0          # m/s
WINDOWS_S = (2, 8, 20)        # centred smoothing windows
SIDE_WINDOW_S = 6             # windows just before and just after the frame
FIT_STEP = 5                  # use every 5th frame to fit the regression (neighbours are alike)
L2 = 1.0                      # ridge penalty on the regression weights
CARRY_RADIUS_M = 2.0          # a player this close to the ball at both ends of a gap is carrying it
GAP_CANDIDATES_S = (1, 2, 3, 4, 6)   # longest gap to fill, picked in fit() (see learn_max_gap)
GAP_SAMPLES = 2000            # hidden stretches per candidate length when learning it
# Post-processing candidates, chosen in fit() by spellF1 on the training games
THRESHOLDS = (0.35, 0.45, 0.55, 0.65)
MIN_DEAD_S = (2, 4, 8, 12)
MIN_LIVE_S = (2, 4, 8, 12)


# -- features -----------------------------------------------------------------------------------

def rolling_mean(x, before, after):
    """Mean of x over [i - before, i + after], ignoring NaN (NaN only if the whole window is)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    ccnt = np.concatenate([[0.0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo = np.clip(i - before, 0, len(x))
    hi = np.clip(i + after + 1, 0, len(x))
    count = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[hi] - csum[lo]) / count, np.nan)


def step_rate(x, step_s=SPEED_STEP_S):
    """Change of x per second, measured from step_s/2 before to step_s/2 after each frame
    (axis 0). NaN at the ends and wherever either end is missing."""
    half = max(1, int(round(step_s * FPS / 2)))
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[half:-half] = (x[2 * half:] - x[:-2 * half]) * FPS / (2 * half)
    return out


def player_speeds(players):
    """(frames, players) speed in m/s over SPEED_STEP_S; NaN where a track starts, ends or
    jumps (a swap)."""
    rate = step_rate(players * np.array(PITCH, dtype=np.float32))
    speed = np.hypot(rate[..., 0], rate[..., 1])
    speed[speed > MAX_SPEED] = np.nan
    return speed


def per_half_scale(x, period):
    """Divide x by its median within each half, so the level of each game doesn't matter."""
    out = np.full(len(x), np.nan)
    for half in (1, 2):
        rows = period == half
        if rows.any() and np.isfinite(x[rows]).any():
            out[rows] = x[rows] / max(np.nanmedian(x[rows]), 1e-6)
    return out


def frame_features(tracking):
    """Raw per-frame signals, before smoothing. NaN where a signal can't be measured."""
    players, ball, period = tracking["players"], tracking["ball"], tracking["period"]
    speed = player_speeds(players)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(speed, axis=1)
        fast = np.nanpercentile(speed, 90, axis=1)
        tracked = np.sum(np.isfinite(speed), axis=1)
    # Normalise speeds by the half's typical median speed, then count walkers and sprinters
    level = np.nanmedian(median[period > 0]) / 1.5 if np.isfinite(median[period > 0]).any() else 1.0
    rel = speed / max(level, 1e-6)
    share = lambda mask: np.where(tracked > 0, mask.sum(axis=1) / np.maximum(tracked, 1), np.nan)
    walking = share(rel < WALK_SPEED)
    sprinting = share(rel > SPRINT_SPEED)

    # Ball: only when visible. A missing ball gives NaN, never a value of its own.
    bpos = ball * np.array(PITCH, dtype=np.float32)
    bspeed = np.full(len(ball), np.nan, dtype=np.float32)
    bspeed[1:] = np.hypot(bpos[1:, 0] - bpos[:-1, 0], bpos[1:, 1] - bpos[:-1, 1]) * FPS
    bspeed[bspeed > 40] = np.nan
    seen = np.isfinite(ball[:, 0])
    off = (ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1)
    ball_out = np.where(seen, off, np.nan)
    ball_still = np.where(np.isfinite(bspeed), bspeed < BALL_STILL_SPEED, np.nan)

    return {"median_speed": per_half_scale(median, period),
            "fast_speed": per_half_scale(fast, period),
            "walking": walking, "sprinting": sprinting,
            "shape_moving": per_half_scale(shape_movement(players, tracking["team"]), period),
            "gathered": largest_group(players),
            "at_touchline": standing_at_touchline(players, speed),
            "ball_out": ball_out, "ball_still": ball_still}


def shape_movement(players, team):
    """How fast the team shapes change: speed of each team's centre plus the rate its spread
    changes, averaged over both teams. Low when both teams stand and wait."""
    pos = players * np.array(PITCH, dtype=np.float32)
    total = np.zeros(len(players))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in ("H", "A"):
            p = pos[:, team == side]
            centre = np.nanmean(p, axis=1)
            spread = np.nanmean(np.hypot(*(p - centre[:, None, :]).transpose(2, 0, 1)), axis=1)
            move = np.hypot(*step_rate(centre).T) + np.abs(step_rate(spread))
            move[move > MAX_SPEED] = np.nan           # a track dropping in or out, not movement
            total += move
    return total / 2


def largest_group(players, radius=GROUP_RADIUS_M, chunk=20000):
    """Most players within radius metres of any one player (a wall, a corner, a crowd at a
    free kick), divided by the number tracked."""
    pos = players * np.array(PITCH, dtype=np.float32)
    out = np.full(len(players), np.nan)
    for start in range(0, len(players), chunk):
        p = pos[start:start + chunk]
        dist = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
        near = np.sum(dist < radius, axis=2)                  # NaN compares False
        tracked = np.sum(np.isfinite(p[:, :, 0]), axis=1)
        out[start:start + chunk] = np.where(tracked > 0, near.max(axis=1) / np.maximum(tracked, 1),
                                            np.nan)
    return out


def standing_at_touchline(players, speed):
    """1 if a slow player stands on or just beyond a touchline (taking or waiting for a throw-in)."""
    y = players[:, :, 1] * PITCH[1]
    at_line = (y < TOUCHLINE_M) | (y > PITCH[1] - TOUCHLINE_M)
    return np.any(at_line & (speed < STANDING_SPEED), axis=1).astype(np.float64)


def features(tracking):
    """(frames, features) matrix: each signal smoothed over centred, before and after windows."""
    cols = []
    for x in frame_features(tracking).values():
        for w in WINDOWS_S:
            half = w * FPS // 2
            cols.append(rolling_mean(x, half, half))
        side = SIDE_WINDOW_S * FPS
        cols.append(rolling_mean(x, side, 0))
        cols.append(rolling_mean(x, 0, side))
    return np.column_stack(cols)


# -- filling short ball gaps -------------------------------------------------------------------

def carrier(players, ball, frame):
    """Index of the player within CARRY_RADIUS_M of the ball at frame, or None."""
    dist = np.hypot(*((players[frame] - ball[frame]) * np.array(PITCH)).T)
    if not np.isfinite(dist).any() or np.nanmin(dist) > CARRY_RADIUS_M:
        return None
    return int(np.nanargmin(dist))


def fill_gap(players, ball, start, end):
    """The carrier's positions over ball[start:end] if one player is clearly carrying the ball
    (near it just before and just after the gap, tracked throughout), else None."""
    if start == 0 or end >= len(ball):
        return None
    who = carrier(players, ball, start - 1)
    if who is None or carrier(players, ball, end) != who:
        return None
    path = players[start:end, who]
    return path if np.isfinite(path).all() else None


def fill_ball_gaps(tracking, max_gap_frames):
    """Copy of tracking with gaps up to max_gap_frames filled where one player carries the ball.
    Longer gaps, and gaps nobody clearly carries the ball through, stay missing."""
    ball = tracking["ball"].copy()
    for start, end in runs(np.isnan(ball[:, 0]) & (tracking["period"] > 0)):
        if end - start <= max_gap_frames:
            path = fill_gap(tracking["players"], tracking["ball"], start, end)
            if path is not None:
                ball[start:end] = path
    return dict(tracking, ball=ball)


def learn_max_gap(trackings, seed=0):
    """Longest gap length (from GAP_CANDIDATES_S) where filling is still as accurate as "the ball is
    at the carrier's feet": hide stretches of visible ball in the training games, fill them, and
    keep a length only if the filled positions are typically within CARRY_RADIUS_M of the truth."""
    rng = np.random.default_rng(seed)
    best = 0
    for seconds in GAP_CANDIDATES_S:
        length = int(seconds * FPS)
        errors = []
        for tracking in trackings:
            ball, players = tracking["ball"], tracking["players"]
            seen = np.isfinite(ball[:, 0])
            # stretches where the ball is visible for the hidden part and one frame either side
            window = np.convolve(seen, np.ones(length + 2), mode="valid") == length + 2
            starts = np.flatnonzero(window) + 1
            for start in rng.choice(starts, min(GAP_SAMPLES, len(starts)), replace=False):
                path = fill_gap(players, ball, start, start + length)
                if path is not None:
                    errors.append(np.hypot(*((path - ball[start:start + length]) * np.array(PITCH)).T))
        if errors and np.median(np.concatenate(errors)) <= CARRY_RADIUS_M:
            best = length
        else:
            break
    return best


# -- logistic regression ------------------------------------------------------------------------

def fit_logistic(X, y, l2=L2, iters=25):
    """Newton's method with a ridge penalty. X already standardised; returns (weights, bias)."""
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    penalty = np.full(Xb.shape[1], l2)
    penalty[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) + penalty * w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w[:-1], w[-1]


# -- post-processing ----------------------------------------------------------------------------

def runs(mask):
    """(start, end) of each run of True, end exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def clean_runs(dead, min_dead, min_live):
    """Fill live gaps shorter than min_live inside dead play, then drop dead runs shorter than
    min_dead."""
    dead = dead.copy()
    for start, end in runs(~dead):
        if start > 0 and end < len(dead) and end - start < min_live:
            dead[start:end] = True
    for start, end in runs(dead):
        if end - start < min_dead:
            dead[start:end] = False
    return dead


def to_dead(prob, threshold, min_dead_s, min_live_s):
    return clean_runs(prob > threshold, int(min_dead_s * FPS), int(min_live_s * FPS))


# -- the detector, usable from any model ------------------------------------------------------

class DeadBallDetector:
    """Dead or alive per frame, from the players (and the ball when visible).

        detector = DeadBallDetector()
        detector.fit(train)                 # train: list of (tracking, labels), as in fit()
        dead = detector.dead_mask(tracking)  # bool per frame; True = dead ball

    fit() learns the regression, the threshold and minimum run lengths, and the longest ball gap
    to fill. The threshold and run lengths are picked by spellF1 of base_predict's labels with
    the dead frames set to D, so pass your own model's predict to tune them for it (default:
    baseline B). Short ball gaps are filled inside the detector only; the caller's tracking is
    never changed.
    """

    def __init__(self):
        self.mean = self.std = self.weights = self.bias = None
        self.post = {"threshold": 0.5, "min_dead_s": 3, "min_live_s": 3}
        self.max_gap = 0

    def settings(self):
        return {"windows_s": WINDOWS_S, "side_window_s": SIDE_WINDOW_S, "l2": L2,
                "max_gap_frames": self.max_gap, "carry_radius_m": CARRY_RADIUS_M,
                "inputs": list(frame_features_names()), "post": self.post,
                "post_grid": {"threshold": THRESHOLDS, "min_dead_s": MIN_DEAD_S,
                              "min_live_s": MIN_LIVE_S}}

    def dead_probability(self, tracking):
        X = features(fill_ball_gaps(tracking, self.max_gap))
        Z = (X - self.mean) / self.std
        Z = np.where(np.isfinite(Z), Z, 0.0)       # unmeasured = the training average
        return 1.0 / (1.0 + np.exp(-(Z @ self.weights + self.bias)))

    def dead_mask(self, tracking):
        return to_dead(self.dead_probability(tracking), **self.post) | (tracking["period"] == 0)

    def fit(self, train, base_predict=None):
        base_predict = base_predict or NeverDead().predict
        # 0. How long a gap can be filled (dead_probability fills gaps the same way)
        self.max_gap = learn_max_gap([tracking for tracking, _ in train])

        # 1. Regression on every FIT_STEP-th frame inside a half
        feats = [features(fill_ball_gaps(tracking, self.max_gap)) for tracking, _ in train]
        rows = [(np.arange(len(labels)) % FIT_STEP == 0) & (tracking["period"] > 0)
                & (labels != pm.X) for (tracking, labels) in train]
        X = np.concatenate([f[r] for f, r in zip(feats, rows)])
        y = np.concatenate([labels[r] == pm.D for (_, labels), r in zip(train, rows)])
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-6
        Z = np.where(np.isfinite(X), (X - self.mean) / self.std, 0.0)
        self.weights, self.bias = fit_logistic(Z, y.astype(np.float64))

        # 2. Threshold and minimum run lengths: best mean spellF1 on the training games
        probs = [self.dead_probability(tracking) for tracking, _ in train]
        bases = [base_predict(tracking) for tracking, _ in train]
        best = -1.0
        for threshold, min_dead, min_live in itertools.product(THRESHOLDS, MIN_DEAD_S, MIN_LIVE_S):
            post = {"threshold": threshold, "min_dead_s": min_dead, "min_live_s": min_live}
            score = np.mean([
                pm.evaluate(labels, pm.label_codes(with_dead(base, to_dead(prob, **post), tracking)))
                ["spell_f1_mean"]
                for (tracking, labels), base, prob in zip(train, bases, probs)])
            if score > best:
                best, self.post = score, post


def with_dead(labels, dead, tracking):
    """labels with D on dead frames and outside the halves."""
    labels = labels.copy()
    labels[dead | (tracking["period"] == 0)] = "D"
    return labels


# -- the model: baseline B plus the detector ---------------------------------------------------

class DeadBallB:
    name = "dead_ball_players"

    def __init__(self):
        self.base = NeverDead()
        self.detector = DeadBallDetector()

    def settings(self):
        return dict(self.detector.settings(), base=self.base.name)

    def fit(self, train):
        self.detector.fit(train, self.base.predict)

    def predict(self, tracking):
        return with_dead(self.base.predict(tracking), self.detector.dead_mask(tracking), tracking)


def frame_features_names():
    return ("median_speed", "fast_speed", "walking", "sprinting", "shape_moving", "gathered",
            "at_touchline", "ball_out", "ball_still")
