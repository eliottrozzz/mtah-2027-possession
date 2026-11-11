"""Two deliberately simple baselines: the floor every real model has to beat.

Both give the ball to the team of the player nearest to it, and only switch team once the other
team has been nearest for SWITCH_S in a row, so a crowd doesn't create hundreds of tiny fake
possessions. They differ only on frames with no ball:

    NoBallDead   "no ball = dead": the shortcut we expect to fail on the noisy copies
    NeverDead    never predicts D: carries the last team forward through every gap

Nothing here is tuned. SWITCH_S is a fixed, round value chosen before any scoring.

A model is any class with:
    name                       short name for the results log
    settings()                 dict of its parameters, written to the log
    fit(train)                 optional; train is a list of (tracking, labels) from the training games
    predict(tracking)          array of 'H' / 'A' / 'D', one per tracking frame
tracking is the shared dict from external/metrica_tracking.py (ball, players, team, period).
"""

import numpy as np

# Params
FPS = 25
SWITCH_S = 0.5            # the other team must be nearest this long before possession switches
PITCH = (105.0, 68.0)     # metres, to measure distances on the real pitch shape


def nearest_team(tracking):
    """'H', 'A' or '' per frame: the team of the player nearest the ball ('' if no ball or no players)."""
    ball, players, team = tracking["ball"], tracking["players"], tracking["team"]
    scale = np.array(PITCH, dtype=np.float32)
    offset = (players - ball[:, None, :]) * scale
    dist = np.hypot(offset[..., 0], offset[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = np.argmin(dist, axis=1)
    found = np.isfinite(dist[np.arange(len(dist)), nearest])
    return np.where(found, team[nearest], "")


def hold_switches(raw, switch_frames):
    """Carry the current team until the other team is nearest for switch_frames frames in a row.

    Frames without a nearest team ('') keep the current team and don't break or extend a streak.
    """
    out = np.empty(len(raw), dtype="<U1")
    current, challenger, streak = "", "", 0
    for i, team in enumerate(raw):
        if team and team != current:
            if not current:
                current = team
            elif team == challenger:
                streak += 1
                if streak >= switch_frames:
                    current, challenger, streak = team, "", 0
            else:
                challenger, streak = team, 1
        elif team == current:
            challenger, streak = "", 0
        out[i] = current or "D"
    return out


class NoBallDead:
    name = "baseline_noball_dead"

    def settings(self):
        return {"switch_s": SWITCH_S, "no_ball": "D"}

    def fit(self, train):
        pass

    def predict(self, tracking):
        raw = nearest_team(tracking)
        labels = hold_switches(raw, int(round(SWITCH_S * FPS)))
        labels[raw == ""] = "D"
        labels[tracking["period"] == 0] = "D"
        return labels


class NeverDead:
    name = "baseline_never_dead"

    def settings(self):
        return {"switch_s": SWITCH_S, "no_ball": "carry last team"}

    def fit(self, train):
        pass

    def predict(self, tracking):
        labels = hold_switches(nearest_team(tracking), int(round(SWITCH_S * FPS)))
        labels[tracking["period"] == 0] = "D"
        return labels
