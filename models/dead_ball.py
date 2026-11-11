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
  3. P(dead) is thresholded with hysteresis (a high threshold to start a dead run, a lower one to
     keep it going), and short dead or live runs are removed. The thresholds and minimum lengths are picked in fit() by spellF1 on the training games, never on test scores.
  4. Baseline B (nearest team, switch after 0.5 s, never dead) gives the team; dead frames get D.

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
# Post-processing candidates, chosen in fit() by spellF1 on the training games
POST_GRID = {
    "enter": (0.5, 0.6, 0.7, 0.8),    # P(dead) must pass this for a dead run to start...
    "exit": (0.3, 0.4, 0.5),          # ...and the run lasts while P(dead) stays above this
    "min_dead_s": (1, 2, 4, 8),
    "min_live_s": (2, 4, 8),
}


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


def player_speeds(players):
    """(frames, players) speed in m/s; NaN where a track starts, ends or jumps (a swap)."""
    pos = players * np.array(PITCH, dtype=np.float32)
    speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    speed[1:] = np.hypot(pos[1:, :, 0] - pos[:-1, :, 0], pos[1:, :, 1] - pos[:-1, :, 1]) * FPS
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
            move = np.full(len(players), np.nan)
            move[1:] = (np.hypot(*(centre[1:] - centre[:-1]).T)
                        + np.abs(spread[1:] - spread[:-1])) * FPS
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


def to_dead(prob, enter, exit, min_dead_s, min_live_s):
    """Dead runs: stretches where P(dead) > exit that somewhere reach P(dead) > enter."""
    dead = np.zeros(len(prob), dtype=bool)
    for start, end in runs(prob > exit):
        if prob[start:end].max() > enter:
            dead[start:end] = True
    return clean_runs(dead, int(min_dead_s * FPS), int(min_live_s * FPS))


# -- the model ----------------------------------------------------------------------------------

class DeadBallB:
    name = "dead_ball_players"

    def __init__(self):
        self.base = NeverDead()
        self.mean = self.std = self.weights = self.bias = None
        self.post = {"enter": 0.6, "exit": 0.4, "min_dead_s": 2, "min_live_s": 4}

    def settings(self):
        return {"base": self.base.name, "windows_s": WINDOWS_S, "side_window_s": SIDE_WINDOW_S,
                "l2": L2, "inputs": list(frame_features_names()), "post": self.post,
                "post_grid": POST_GRID}

    def dead_probability(self, X):
        Z = (X - self.mean) / self.std
        Z = np.where(np.isfinite(Z), Z, 0.0)       # unmeasured = the training average
        return 1.0 / (1.0 + np.exp(-(Z @ self.weights + self.bias)))

    def fit(self, train):
        # 1. Regression on every FIT_STEP-th frame inside a half
        feats = [features(tracking) for tracking, _ in train]
        rows = [(np.arange(len(labels)) % FIT_STEP == 0) & (tracking["period"] > 0)
                & (labels != pm.X) for (tracking, labels) in train]
        X = np.concatenate([f[r] for f, r in zip(feats, rows)])
        y = np.concatenate([labels[r] == pm.D for (_, labels), r in zip(train, rows)])
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0) + 1e-6
        Z = np.where(np.isfinite(X), (X - self.mean) / self.std, 0.0)
        self.weights, self.bias = fit_logistic(Z, y.astype(np.float64))

        # 2. Thresholds and minimum run lengths: best mean spellF1 on the training games
        probs = [self.dead_probability(f) for f in feats]
        bases = [self.base.predict(tracking) for tracking, _ in train]
        best = -1.0
        for values in itertools.product(*POST_GRID.values()):
            post = dict(zip(POST_GRID, values))
            if post["exit"] > post["enter"]:
                continue
            score = np.mean([
                pm.evaluate(labels, pm.label_codes(self.combine(base, prob, tracking, **post)))
                ["spell_f1_mean"]
                for (tracking, labels), base, prob in zip(train, bases, probs)])
            if score > best:
                best, self.post = score, post

    @staticmethod
    def combine(base, prob, tracking, **post):
        labels = base.copy()
        labels[to_dead(prob, **post)] = "D"
        labels[tracking["period"] == 0] = "D"
        return labels

    def predict(self, tracking):
        prob = self.dead_probability(features(tracking))
        return self.combine(self.base.predict(tracking), prob, tracking, **self.post)


def frame_features_names():
    return ("median_speed", "fast_speed", "walking", "sprinting", "shape_moving", "gathered",
            "at_touchline", "ball_out", "ball_still")
