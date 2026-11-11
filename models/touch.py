"""Model v1: possession goes to the team that touched the ball last.

A touch is a sudden change in the ball's velocity with a player close by. Being near the ball is
not enough (a high ball passes "through" players in 2D tracking), so a touch needs both.

    1. Velocity before and after each frame, each measured over WINDOW frames, which averages out
       the tracking jitter. The size of the change is the touch strength.
    2. A touch is a local peak of that strength above min_strength, with a player within
       radius_m; it belongs to that player's team.
    3. Possession is the team of the last touch, except a deflection: a touch by the other team
       followed by the original team touching again within deflection_s doesn't count.
    4. Before the first touch, and through gaps, possession carries forward. v1 never predicts D;
       dead balls come from the dead-ball detector (Track B), merged in separately.

min_strength, radius_m and deflection_s are chosen in fit() by trying a small grid on the
training games (external data only) and keeping the best spellF1.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
PITCH = np.array((105.0, 68.0), dtype=np.float32)
WINDOW = 4                       # frames each side used to measure velocity (0.16 s)
GRID = {
    "min_strength": (5.0, 7.0, 9.0, 12.0),     # m/s change in velocity
    "radius_m": (1.5, 2.5, 3.5),
    "deflection_s": (0.0, 0.4, 0.8),
}
DEFAULTS = {"min_strength": 7.0, "radius_m": 2.5, "deflection_s": 0.4}


def touch_strength(ball, window=WINDOW):
    """Change in ball velocity (m/s) at each frame, comparing the window before and after."""
    n = len(ball)
    strength = np.full(n, np.nan, dtype=np.float32)
    if n <= 2 * window:
        return strength
    pos = ball * PITCH
    before = (pos[window:n - window] - pos[:n - 2 * window]) * FPS / window
    after = (pos[2 * window:] - pos[window:n - window]) * FPS / window
    strength[window:n - window] = np.hypot(*(after - before).T)
    return strength


def find_touches(tracking, min_strength, radius_m, window=WINDOW):
    """(frames, teams) of touches: peaks of touch strength with a player within radius_m."""
    strength = touch_strength(tracking["ball"], window)
    filled = np.nan_to_num(strength, nan=0.0)
    # A peak is the largest value within +-window frames.
    padded = np.pad(filled, window)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * window + 1)
    peaks = np.flatnonzero((filled >= min_strength) & (filled == windows.max(axis=1)))
    if not len(peaks):
        return peaks, np.array([], dtype="<U1")

    ball = tracking["ball"][peaks] * PITCH
    players = tracking["players"][peaks] * PITCH
    dist = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    close = dist[np.arange(len(peaks)), nearest] <= radius_m
    return peaks[close], tracking["team"][nearest[close]]


def drop_deflections(frames, teams, deflection_frames):
    """Remove touches by the other team that the original team gets back within the window."""
    if deflection_frames <= 0 or len(frames) < 3:
        return frames, teams
    keep = np.ones(len(frames), dtype=bool)
    holder, i = teams[0], 1
    while i < len(frames):
        if teams[i] == holder:
            i += 1
            continue
        # The other team touched it. If the holder touches it again within the window, every
        # touch in between was a deflection: drop them and carry on from the holder's touch.
        j = i + 1
        while j < len(frames) and teams[j] != holder:
            j += 1
        if j < len(frames) and frames[j] - frames[i] <= deflection_frames:
            keep[i:j] = False
            i = j
        else:
            holder = teams[i]
            i += 1
    return frames[keep], teams[keep]


def possession_from_touches(n, frames, teams):
    """Per-frame team: the last touch's team, carried forward; the first touch's team before it."""
    labels = np.full(n, "D", dtype="<U1")
    if not len(frames):
        return labels
    index = np.searchsorted(frames, np.arange(n), side="right") - 1
    labels[:] = teams[np.clip(index, 0, None)]
    return labels


class LastTouch:
    name = "touch_v1"

    def __init__(self, **params):
        self.params = dict(DEFAULTS, **params)

    def settings(self):
        return dict(self.params, window=WINDOW)

    def predict_with(self, tracking, params):
        frames, teams = find_touches(tracking, params["min_strength"], params["radius_m"])
        frames, teams = drop_deflections(frames, teams, int(params["deflection_s"] * FPS))
        labels = possession_from_touches(len(tracking["ball"]), frames, teams)
        labels[tracking["period"] == 0] = "D"
        return labels

    def predict(self, tracking):
        return self.predict_with(tracking, self.params)

    def fit(self, train):
        """Try every grid combination on the training games; keep the best mean spellF1."""
        best, best_score = None, -1.0
        for values in itertools.product(*GRID.values()):
            params = dict(zip(GRID.keys(), values))
            scores = [pm.evaluate(gt, pm.label_codes(self.predict_with(tracking, params)))
                      ["spell_f1_mean"] for tracking, gt in train]
            score = float(np.mean(scores))
            if score > best_score:
                best, best_score = params, score
        self.params = best
        print("    fit: {} (train spellF1 {:.3f})".format(best, best_score))
