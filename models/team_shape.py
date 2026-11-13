"""Which team has the ball, from the team shapes alone (no ball input).

The team in possession spreads out and pushes forward; the team without it narrows and drops
back. This prior reads that from the players only, so it works the same whether or not the ball is
tracked, and gives a second opinion where touches are unclear (close calls, long ball gaps).

Built to read behaviour, never identity:
  * every feature is measured per team in that team's own attacking direction (per half), and
  * P(H) = sigmoid(w . (f(H) - f(A))): the same weights for both teams and no intercept, so
    swapping the two teams' labels and players swaps the prediction exactly. It can't learn
    "the home team usually has the ball" or "team X plays wide".

Per-team signals (each averaged over centred 1, 4 and 12 s windows and the 6 s before and after):
    spread_x, spread_y   spread of the team's players along and across the pitch
    depth                the team's centre, toward the goal it attacks
    forward              speed of the team's centre toward the goal it attacks (over 1 s)

Fitted on the training games' live frames (logistic regression, ridge penalty), external data only.
"""

import os
import sys
import warnings

import numpy as np

from models.dead_ball import fit_logistic, rolling_mean, step_rate

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
PITCH = np.array((105.0, 68.0))
WINDOWS_S = (1, 4, 12)        # centred smoothing windows
SIDE_WINDOW_S = 6             # windows just before and just after the frame
FORWARD_STEP_S = 1.0          # team centre speed measured over this long
FIT_STEP = 10                 # every 10th live frame (neighbours are alike)


def attack_sign(centre_x, period):
    """+1 where the team attacks toward x = 105 m, -1 toward 0, per half (from its mean x)."""
    sign = np.zeros(len(period))
    for half in (1, 2):
        rows = period == half
        if np.isfinite(centre_x[rows]).any():
            sign[rows] = 1.0 if np.nanmean(centre_x[rows]) < PITCH[0] / 2 else -1.0
    return sign


def team_signals(tracking, team):
    """Per-frame raw signals for one team, in its own attacking direction."""
    pos = tracking["players"][:, tracking["team"] == team] * PITCH
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(pos, axis=1)
        spread_x = np.nanstd(pos[:, :, 0], axis=1)
        spread_y = np.nanstd(pos[:, :, 1], axis=1)
    sign = attack_sign(centre[:, 0], tracking["period"])
    depth = np.where(sign > 0, centre[:, 0], PITCH[0] - centre[:, 0])
    forward = sign * step_rate(centre[:, 0], FORWARD_STEP_S)
    return (spread_x, spread_y, depth, forward)


def team_features(tracking, team):
    """(frames, features) for one team: each signal over every smoothing window."""
    cols = []
    for x in team_signals(tracking, team):
        for w in WINDOWS_S:
            half = w * FPS // 2
            cols.append(rolling_mean(x, half, half))
        side = SIDE_WINDOW_S * FPS
        cols.append(rolling_mean(x, side, 0))
        cols.append(rolling_mean(x, 0, side))
    return np.column_stack(cols)


def features(tracking):
    """f(H) - f(A): antisymmetric in the two teams by construction."""
    return team_features(tracking, "H") - team_features(tracking, "A")


class TeamShapePrior:
    """P(home team has the ball) per frame, from the team shapes.

        prior = TeamShapePrior()
        prior.fit(train)                      # list of (tracking, answer-key codes)
        p_home = prior.predict_proba(tracking)
    """

    def __init__(self):
        self.scale = self.weights = None

    def settings(self):
        return {"windows_s": WINDOWS_S, "side_window_s": SIDE_WINDOW_S,
                "forward_step_s": FORWARD_STEP_S,
                "signals": ["spread_x", "spread_y", "depth", "forward"]}

    def fit(self, train):
        X, y = [], []
        for tracking, labels in train:
            live = np.flatnonzero((labels == pm.H) | (labels == pm.A))[::FIT_STEP]
            X.append(features(tracking)[live])
            y.append((labels[live] == pm.H).astype(np.float64))
        X, y = np.concatenate(X), np.concatenate(y)
        # Scale only, no centring: centring would add an offset and break the H/A symmetry.
        self.scale = np.sqrt(np.nanmean(X ** 2, axis=0)) + 1e-6
        Z = np.nan_to_num(X / self.scale)
        # Fit with mirrored copies so the intercept is exactly zero, then drop it.
        weights, bias = fit_logistic(np.concatenate([Z, -Z]), np.concatenate([y, 1 - y]))
        self.weights = weights

    def predict_proba(self, tracking):
        Z = np.nan_to_num(features(tracking) / self.scale)
        return 1.0 / (1.0 + np.exp(-(Z @ self.weights)))
