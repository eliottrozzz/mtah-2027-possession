"""Model v1: possession goes to the team that touched the ball last.

A touch is a sudden change in the ball's velocity with a player close by. Being near the ball is
not enough (a high ball passes "through" players in 2D tracking), so a touch needs both.

    1. Velocity before and after each frame, each measured over WINDOW frames, which averages out
       the tracking jitter. The size of the change is the touch strength.
    2. A touch is a local peak of that strength above min_strength, with a player within
       radius_m; it belongs to that player's team.
    3. Possession is the team of the last touch, except a deflection: a touch by the other team
       followed by the original team touching again within deflection_s doesn't count.
    4. Optionally, the ball reappearing after a tracking gap of at least reappear_s counts as a
       touch by the nearest player within radius_m: the team often changed while it was unseen.
    5. Before the first touch, and through gaps, possession carries forward. v1 never predicts D;
       dead balls come from the dead-ball detector (Track B), merged in separately.

min_strength, radius_m, window and reappear_s are chosen in fit() by trying a small grid on the
training games (external data only) and keeping the best spellF1.
"""

import itertools
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
PITCH = np.array((105.0, 68.0), dtype=np.float32)
WINDOW = 4                       # default frames each side (0.16 s); the model tunes its own
GRID = {
    # The provided tracking is smoothed over about 0.8 s, which blurs kicks into small velocity
    # changes (under 6 m/s 99% of the time), so the ranges cover weak touches, longer measuring
    # windows and a lagging ball.
    # Touch threshold as a share of the match's own 90th-percentile touch strength, so it adapts
    # if the real ball is blurrier or sharper than our noise (Track B's stress test, 2026-11-12:
    # an absolute 2.0 m/s threshold lost 0.084 when the blur went 0.84 s -> 1.24 s). Under our
    # noise the 90th percentile is about 7.1 m/s, so these match 1, 1.5, 2, 3, 4 m/s.
    "min_strength_rel": (0.14, 0.21, 0.28, 0.42, 0.56),
    "radius_m": (1.0, 1.5, 2.5, 3.5),
    "window": (4, 8, 12),                        # frames each side (0.16, 0.32, 0.48 s)
    # Whose touch it is (see assign_players). Tried 2026-11-11: with "at", "after" and "closest"
    # in the grid, fit picked "closest" in every round and v1 alone rose 0.358 -> 0.365, but v2
    # only rose 0.464 -> 0.467 and game 3 dropped 0.026 at live35, over the 0.02 limit: not kept.
    # The other rules stay available for later tests.
    "assign": ("at",),
    # Fixed at 0.5 s, a rule decision, not tuned: the public ground truth counts single touches of
    # about 0.8 s as real possessions, so the window must stay well below that (touch timing is
    # only known to about +-0.16 s). Metrica prefers 0.8 s, but its events log fewer quick touches
    # than the organisers do. Scores for 0.4-0.6 s were within 0.003. Decided 2026-11-11.
    "deflection_s": (0.5,),
    # Close calls between players: cost = distance - leave_m * cos(angle between the ball's new
    # direction and the player-to-ball line). Tried 2026-11-12 with (0, 0.5, 1, 2): v2 rose
    # 0.479 -> 0.486, but clean data fell sharply (v2 game 2 clean 0.773 -> 0.615), so the gain
    # only made up for our simulated blur, not real physics. Not kept; off.
    "leave_m": (0.0,),
    # Ball reappearing after a gap of at least this long counts as a touch (0 = off). Added
    # 2026-11-12: on game 1, 12-21% of missed switches happened while the ball was unseen.
    "reappear_s": (0.0, 0.2, 0.5, 1.0),
}
STRENGTH_QUANTILE = 0.9
DEFAULTS = {"min_strength": 2.0, "min_strength_rel": None, "radius_m": 2.5, "window": 8, "assign": "at",
            "deflection_s": 0.5, "reappear_s": 0.0,
            "leave_m": 0.0}
AFTER_FRAMES = 12               # "after": how long the toucher should stay near the ball (0.48 s)


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


def find_touches(tracking, min_strength, radius_m, window=WINDOW, assign="at", leave_m=0.0):
    """(frames, teams) of touches: peaks of touch strength with a player within radius_m.

    With leave_m > 0 (and assign "at"), the toucher is the player with the lowest
    distance - leave_m * cos(angle between the ball's direction after the touch and the line from
    the player to the ball): the ball moves away from whoever kicked it. The radius still applies
    to that player's real distance.
    """
    strength = touch_strength(tracking["ball"], window)
    filled = np.nan_to_num(strength, nan=0.0)
    # A peak is the largest value within +-window frames.
    padded = np.pad(filled, window)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * window + 1)
    peaks = np.flatnonzero((filled >= min_strength) & (filled == windows.max(axis=1)))
    if not len(peaks):
        return peaks, np.array([], dtype="<U1")

    if leave_m > 0 and assign == "at":
        nearest, dist = leaving_player(tracking, peaks, window, leave_m)
    else:
        nearest, dist = assign_players(tracking, peaks, assign, window)
    close = dist <= radius_m
    return peaks[close], tracking["team"][nearest[close]]


def assign_players(tracking, peaks, assign, window):
    """(player index, distance in m) for each touch frame, by one of three rules.

        at       nearest player at the touch frame
        after    nearest on average over the AFTER_FRAMES frames from the touch: whoever keeps
                 the ball close (a receiver) rather than whoever it passed on the way
        closest  the player who came closest to the ball at any frame within +-window of the
                 touch, which cancels the lag of a smoothed ball
    """
    n = len(tracking["ball"])
    if assign == "at":
        offsets = np.array([0])
    elif assign == "after":
        offsets = np.arange(0, AFTER_FRAMES + 1)
    elif assign == "closest":
        offsets = np.arange(-window, window + 1)
    else:
        raise ValueError("unknown assign rule: " + str(assign))
    frames = np.clip(peaks[:, None] + offsets[None, :], 0, n - 1)          # (touches, offsets)
    ball = tracking["ball"][frames] * PITCH                                 # (t, o, 2)
    players = tracking["players"][frames] * PITCH                           # (t, o, p, 2)
    dist = np.hypot(*(players - ball[:, :, None, :]).transpose(3, 0, 1, 2))   # (t, o, p)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_player = np.nanmean(dist, axis=1) if assign == "after" else np.nanmin(dist, axis=1)
    per_player = np.where(np.isnan(per_player), np.inf, per_player)
    nearest = np.argmin(per_player, axis=1)
    return nearest, per_player[np.arange(len(peaks)), nearest]


def leaving_player(tracking, peaks, window, leave_m):
    """(player index, distance in m) by the 'ball moves away from the kicker' cost."""
    n = len(tracking["ball"])
    ball = tracking["ball"][peaks] * PITCH                                        # (t, 2)
    later = tracking["ball"][np.clip(peaks + window, 0, n - 1)] * PITCH
    move = later - ball
    move = move / np.maximum(np.hypot(*move.T), 1e-6)[:, None]                    # unit direction
    rel = ball[:, None, :] - tracking["players"][peaks] * PITCH                  # player -> ball
    dist = np.hypot(*rel.transpose(2, 0, 1))                                      # (t, p)
    with np.errstate(all="ignore"):
        cos = np.einsum("tpk,tk->tp", rel, move) / np.maximum(dist, 1e-6)
    cost = dist - leave_m * np.nan_to_num(cos, nan=0.0)
    cost = np.where(np.isnan(dist), np.inf, cost)
    nearest = np.argmin(cost, axis=1)
    dist = np.where(np.isnan(dist), np.inf, dist)
    return nearest, dist[np.arange(len(peaks)), nearest]


def find_reappearances(tracking, min_gap_frames, radius_m):
    """(frames, teams): the first frame the ball is seen after a gap of at least min_gap_frames,
    given to the nearest player at that frame if within radius_m."""
    if min_gap_frames <= 0:
        return np.array([], dtype=int), np.array([], dtype="<U1")
    missing = np.isnan(tracking["ball"][:, 0])
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    frames = stops[(stops - starts >= min_gap_frames) & (starts > 0) & (stops < len(missing))]
    if not len(frames):
        return frames, np.array([], dtype="<U1")
    nearest, dist = assign_players(tracking, frames, "at", 0)
    close = dist <= radius_m
    return frames[close], tracking["team"][nearest[close]]


def strength_threshold(tracking, params):
    """min_strength in m/s: absolute, or min_strength_rel times this match's 90th percentile."""
    rel = params.get("min_strength_rel")
    if rel is None:
        return params["min_strength"]
    strength = touch_strength(tracking["ball"], params["window"])
    strength = strength[(tracking["period"] > 0) & np.isfinite(strength)]
    return rel * float(np.quantile(strength, STRENGTH_QUANTILE)) if len(strength) else np.inf


def all_touches(tracking, params):
    """Touches, plus reappearances if switched on, in time order, with deflections dropped."""
    frames, teams = find_touches(tracking, strength_threshold(tracking, params),
                                 params["radius_m"],
                                 params["window"], params.get("assign", "at"),
                                 params.get("leave_m", 0.0))
    re_frames, re_teams = find_reappearances(
        tracking, int(round(params.get("reappear_s", 0.0) * FPS)), params["radius_m"])
    if len(re_frames):
        frames = np.concatenate((frames, re_frames))
        teams = np.concatenate((teams, re_teams))
        order = np.argsort(frames, kind="stable")
        frames, teams = frames[order], teams[order]
    return drop_deflections(frames, teams, int(params["deflection_s"] * FPS))


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
        return dict(self.params)

    def predict_with(self, tracking, params):
        frames, teams = all_touches(tracking, params)
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
