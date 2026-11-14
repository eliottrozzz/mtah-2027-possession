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
    5. Optionally (gap_rule), when the ball reappears after a gap of at least gap_min_s near the
       other team, the switch is moved back inside the gap (see gap_switches).
    6. Before the first touch, and through gaps, possession carries forward. v1 never predicts D;
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
    "min_strength": (1.0, 1.5, 2.0, 3.0, 4.0),  # m/s change in velocity
    # A threshold relative to the match's own 90th-percentile touch strength (min_strength_rel,
    # see strength_threshold) was tried 2026-11-12 for robustness to a blurrier ball: no change
    # (0.478 vs 0.479; at a 1.24 s blur 0.385 vs 0.387). Even a model fitted on 1.24 s copies
    # only reaches 0.404, so most of that loss is information the blur destroys. Not kept.
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
# Long ball gaps (step 5), fitted after GRID with the rest fixed. On the external games, of 129
# gaps of 3 s or more where the team switches, the switch was within 1 s of the reappearance
# 13% of the time, the gap's middle 22%, the "line" rule 12-15% and the "advance" rule with a
# 1 s shift 38% (checked 2026-11-13 before building).
GAP_CANDIDATES = [{"gap_rule": "off"}] + [
    {"gap_rule": "line", "gap_min_s": g, "gap_line_m": 3.0} for g in (2.0, 3.0, 5.0)] + [
    {"gap_rule": "advance", "gap_min_s": g, "gap_shift_s": k}
    for g in (2.0, 3.0, 5.0) for k in (0.0, 1.0, 2.0)]
# Deflection window, fitted last with everything else fixed (step 3 of the module docstring).
# (0.5,) keeps the 2026-11-11 rule decision. Longer windows were tried 2026-11-13 (Track C, false
# switches) as an information run only: Metrica's key logs fewer quick touches than the
# organisers, so it prefers long windows, and the public ground truth counts single touches of
# about 0.8 s, so nothing above 0.75 s may be kept (Jordan, 2026-11-13).
DEFLECTION_CANDIDATES = (0.5,)
# Confirmed switches (Track C, 2026-11-13): a touch by the other team only switches possession
# if the ball then either moves away from where the toucher was, by confirm_away_m and steadily
# (a pass or clearance), or stays with the toucher (a dribble), within confirm_s. Otherwise it is
# treated as a deflection. Physics only, never "who touches next", so a single deliberate touch
# such as a clearance still counts (public ground truth). Unseen ball in the window: kept.
# Fitted last, with everything else fixed; "off" is a candidate.
CONFIRM_CANDIDATES = ({"confirm": "off"},)
CONFIRM_STEADY = 0.7            # share of frames the distance must grow on
# Long deflection window only where the ball is doubtful (Track C, 2026-11-14, Jordan's middle
# ground): a touch by the other team whose ball is unseen anywhere within +-long_unseen_w s of it,
# or that is a reappearance (or gap switch) touch, gets long_deflection_s instead of deflection_s.
# Clearly tracked touches keep the 0.5 s rule and the 0.75 s cap. Physics and visibility only,
# never "who touches next". Fitted after the deflection stage; "off" is a candidate.
# long_where "live" (Jordan, 2026-11-14): count unseen frames only where the model itself predicts
# live play (its dead-ball detector, never the key), so a stoppage gap doesn't make the restart
# touch doubtful. Needs a dead mask: TouchDead fits this stage after its detector (defer_long).
LONG_DEFLECTION_CANDIDATES = ({"long_deflection_s": 0.0},)
ADVANCE_STEP_S = 1.0            # team centre velocity measured over this long
ADVANCE_SMOOTH_S = 2.0          # then averaged over this long
STRENGTH_QUANTILE = 0.9
DEFAULTS = {"min_strength": 2.0, "min_strength_rel": None, "radius_m": 2.5, "window": 8, "assign": "at",
            "deflection_s": 0.5, "reappear_s": 0.0, "long_deflection_s": 0.0, "long_unseen_w": 0.5, "long_where": "all",
            "leave_m": 0.0, "gap_rule": "off"}
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


def ball_gaps(tracking, min_frames):
    """(starts, stops) of ball gaps inside the data at least min_frames long; stop = reappearance."""
    missing = np.isnan(tracking["ball"][:, 0])
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (stops - starts >= min_frames) & (starts > 0) & (stops < len(missing))
    return starts[keep], stops[keep]


def team_advance(tracking):
    """team -> per-frame speed (m/s) of the team's centre toward the other goal, smoothed. The
    team with the ball pushes forward; attacking direction per half from the team's mean x."""
    pos = tracking["players"][..., 0] * PITCH[0]
    period, out = tracking["period"], {}
    step, smooth = int(ADVANCE_STEP_S * FPS) // 2, int(ADVANCE_SMOOTH_S * FPS) // 2
    for team in np.unique(tracking["team"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            centre = np.nanmean(pos[:, tracking["team"] == team], axis=1)
        speed = np.full(len(centre), np.nan)
        speed[step:-step] = (centre[2 * step:] - centre[:-2 * step]) * FPS / (2 * step)
        sign = np.zeros(len(centre))
        for half in (1, 2):
            rows = period == half
            if np.isfinite(centre[rows]).any():
                sign[rows] = 1.0 if np.nanmean(centre[rows]) < PITCH[0] / 2 else -1.0
        speed = np.nan_to_num(speed * sign)
        kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        out[team] = np.convolve(speed, kernel, "same")
    return out


def gap_switches(tracking, params, frames, teams):
    """(frames, teams) of extra touches: a switch placed inside each long gap after which the
    ball reappears near the other team (nearest player within radius_m).

        line     first gap frame where a player of the new team is within gap_line_m of the
                 straight line from the last-seen ball to the reappearance, at the matching point
        advance  start of the last stretch of the gap in which the new team's centre moves toward
                 the other goal faster than the old team's, moved gap_shift_s earlier (the
                 smoothing lags); never before the gap starts

    With no qualifying frame the switch stays at the reappearance."""
    starts, stops = ball_gaps(tracking, int(round(params["gap_min_s"] * FPS)))
    if not len(starts):
        return np.array([], dtype=int), np.array([], dtype="<U1")
    nearest, dist = assign_players(tracking, stops, "at", 0)
    new_teams = tracking["team"][nearest]
    advance = team_advance(tracking) if params["gap_rule"] == "advance" else None
    out_frames, out_teams = [], []
    for start, stop, new, d in zip(starts, stops, new_teams, dist):
        before = np.flatnonzero(frames < start)
        if d > params["radius_m"] or not len(before) or teams[before[-1]] == new:
            continue
        old = teams[before[-1]]
        switch = stop
        if params["gap_rule"] == "line":
            gap = np.arange(start, stop)
            last, seen = tracking["ball"][start - 1] * PITCH, tracking["ball"][stop] * PITCH
            frac = (gap - start + 1) / (stop - start + 1)
            line = last + frac[:, None] * (seen - last)
            mine = tracking["players"][gap][:, tracking["team"] == new] * PITCH
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                close = np.nanmin(np.hypot(*(mine - line[:, None]).transpose(2, 0, 1)), axis=1)
            hits = np.flatnonzero(close <= params["gap_line_m"])
            if len(hits):
                switch = start + hits[0]
        else:
            ahead = advance[new][start:stop] > advance[old][start:stop]
            if ahead[-1]:
                behind = np.flatnonzero(~ahead)
                switch = start + (behind[-1] + 1 if len(behind) else 0)
                switch = max(start, switch - int(round(params["gap_shift_s"] * FPS)))
        out_frames.append(switch)
        out_teams.append(new)
    return np.array(out_frames, dtype=int), np.array(out_teams, dtype="<U1")


def strength_threshold(tracking, params):
    """min_strength in m/s: absolute, or min_strength_rel times this match's 90th percentile."""
    rel = params.get("min_strength_rel")
    if rel is None:
        return params["min_strength"]
    strength = touch_strength(tracking["ball"], params["window"])
    strength = strength[(tracking["period"] > 0) & np.isfinite(strength)]
    return rel * float(np.quantile(strength, STRENGTH_QUANTILE)) if len(strength) else np.inf


def all_touches(tracking, params, dead=None):
    """Touches, plus reappearances if switched on, in time order, with deflections dropped.
    dead: the model's own dead-ball mask, used only by long_where "live"."""
    frames, teams = find_touches(tracking, strength_threshold(tracking, params),
                                 params["radius_m"],
                                 params["window"], params.get("assign", "at"),
                                 params.get("leave_m", 0.0))
    # Only real touches need confirming; reappearances and gap switches have no ball to check.
    frames, teams = confirm_switches(tracking, frames, teams, params)
    re_frames, re_teams = find_reappearances(
        tracking, int(round(params.get("reappear_s", 0.0) * FPS)), params["radius_m"])
    if params.get("gap_rule", "off") != "off":
        gap_frames, gap_teams = gap_switches(tracking, params, frames, teams)
        re_frames = np.concatenate((re_frames, gap_frames)).astype(int)
        re_teams = np.concatenate((re_teams, gap_teams))
    from_gap = np.zeros(len(frames), dtype=bool)
    if len(re_frames):
        frames = np.concatenate((frames, re_frames))
        teams = np.concatenate((teams, re_teams))
        from_gap = np.concatenate((from_gap, np.ones(len(re_frames), dtype=bool)))
        order = np.argsort(frames, kind="stable")
        frames, teams, from_gap = frames[order], teams[order], from_gap[order]
    return drop_deflections(frames, teams,
                            deflection_windows(tracking, frames, from_gap, params, dead))


def doubtful_touches(tracking, frames, from_gap, w_s, dead=None):
    """True for touches whose ball is unseen anywhere within +-w_s s, or that come from a gap.
    With a dead mask, unseen frames inside predicted stoppages don't count."""
    missing = np.isnan(tracking["ball"][:, 0])
    if dead is not None:
        missing = missing & ~dead
    missing = missing.astype(np.int32)
    w = int(round(w_s * FPS))
    cum = np.concatenate(([0], np.cumsum(missing)))
    lo = np.clip(frames - w, 0, len(missing))
    hi = np.clip(frames + w + 1, 0, len(missing))
    return from_gap | (cum[hi] - cum[lo] > 0)


def deflection_windows(tracking, frames, from_gap, params, dead=None):
    """Per-touch deflection window in frames (see LONG_DEFLECTION_CANDIDATES)."""
    short = int(params["deflection_s"] * FPS)
    windows = np.full(len(frames), short, dtype=int)
    long_s = params.get("long_deflection_s", 0.0)
    if long_s > 0 and len(frames):
        if params.get("long_where", "all") == "live":
            if dead is None:
                raise ValueError('long_where "live" needs the dead mask (see LastTouch.dead_fn)')
        else:
            dead = None
        doubt = doubtful_touches(tracking, np.asarray(frames, dtype=int), from_gap,
                                 params["long_unseen_w"], dead)
        windows[doubt] = max(short, int(long_s * FPS))
    return windows


def confirmed(tracking, frame, team, params):
    """True if the touch at `frame` by `team` looks like control: see CONFIRM_CANDIDATES."""
    k = int(round(params["confirm_s"] * FPS))
    ball = tracking["ball"]
    if frame + k >= len(ball):
        return True
    path = ball[frame:frame + k + 1] * PITCH
    if np.isnan(path).any():
        return True                                   # can't tell: keep the touch
    cols = np.flatnonzero(tracking["team"] == team)
    players = tracking["players"][frame, cols] * PITCH
    d0 = np.hypot(*(players - path[0]).T)
    if not np.isfinite(d0).any():
        return True
    p = int(np.nanargmin(d0))
    start = players[p]
    away = np.hypot(*(path - start).T)
    growing = np.mean(np.diff(away) > 0)
    if away[-1] - away[0] >= params["confirm_away_m"] and growing >= CONFIRM_STEADY:
        return True                                   # passed or cleared away from the toucher
    follow = tracking["players"][frame:frame + k + 1, cols[p]] * PITCH
    near = np.hypot(*(follow - path).T)
    return bool(np.all(np.nan_to_num(near, nan=np.inf) <= params["radius_m"]))  # dribbling


def confirm_switches(tracking, frames, teams, params):
    """Drop touches that would switch possession but aren't confirmed (CONFIRM_CANDIDATES)."""
    if params.get("confirm", "off") == "off" or len(frames) < 2:
        return frames, teams
    keep = np.ones(len(frames), dtype=bool)
    holder = teams[0]
    for i in range(1, len(frames)):
        if teams[i] == holder:
            continue
        if confirmed(tracking, int(frames[i]), teams[i], params):
            holder = teams[i]
        else:
            keep[i] = False
    return frames[keep], teams[keep]


def drop_deflections(frames, teams, deflection_frames):
    """Remove touches by the other team that the original team gets back within the window.
    deflection_frames is one window, or one per touch (the window of the other team's first touch
    applies)."""
    windows = np.broadcast_to(np.asarray(deflection_frames, dtype=int), (len(frames),))
    if not len(frames) or windows.max() <= 0 or len(frames) < 3:
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
        if j < len(frames) and frames[j] - frames[i] <= windows[i]:
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

    def __init__(self, deflection_candidates=DEFLECTION_CANDIDATES,
                 confirm_candidates=CONFIRM_CANDIDATES,
                 long_deflection_candidates=LONG_DEFLECTION_CANDIDATES, **params):
        self.params = dict(DEFAULTS, **params)
        self.deflection_candidates = tuple(deflection_candidates)
        self.long_deflection_candidates = tuple(long_deflection_candidates)
        self.defer_long = False     # True: fit() skips the long stage; the owner calls fit_long
        self.dead_fn = None         # tracking -> the owner's dead-ball mask (long_where "live")
        self.confirm_candidates = tuple(confirm_candidates)

    def settings(self):
        return dict(self.params)

    def touches(self, tracking, params=None, dead=None):
        params = self.params if params is None else params
        if dead is None and params.get("long_where", "all") == "live" \
                and params.get("long_deflection_s", 0.0) > 0:
            dead = self.dead_fn(tracking)
        return all_touches(tracking, params, dead)

    def predict_with(self, tracking, params, dead=None):
        frames, teams = self.touches(tracking, params, dead)
        labels = possession_from_touches(len(tracking["ball"]), frames, teams)
        labels[tracking["period"] == 0] = "D"
        return labels

    def predict(self, tracking):
        return self.predict_with(tracking, self.params)

    def fit(self, train):
        """Try every GRID combination on the training games and keep the best mean spellF1; then,
        with those fixed, the best of GAP_CANDIDATES."""
        def score(params):
            return float(np.mean([pm.evaluate(gt, pm.label_codes(self.predict_with(tracking, params)))
                                  ["spell_f1_mean"] for tracking, gt in train]))
        best, best_score = None, -1.0
        for values in itertools.product(*GRID.values()):
            params = dict(DEFAULTS, **dict(zip(GRID.keys(), values)))
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        base = best
        for gap in GAP_CANDIDATES:
            params = dict(base, **gap)
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        base = best
        for deflection_s in self.deflection_candidates:
            if deflection_s == base["deflection_s"]:
                continue
            params = dict(base, deflection_s=deflection_s)
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        base = best
        if not self.defer_long:
            for long in self.long_deflection_candidates:
                if long.get("long_deflection_s", 0.0) <= 0:
                    continue                          # base is already "off"
                params = dict(base, **long)
                value = score(params)
                if value > best_score:
                    best, best_score = params, value
        base = best
        for confirm in self.confirm_candidates:
            if confirm.get("confirm", "off") == "off":
                continue                              # base is already "off"
            params = dict(base, **confirm)
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        self.params = best
        print("    fit: {} (train spellF1 {:.3f})".format(best, best_score))

    def fit_long(self, train):
        """The long-deflection stage on its own, everything else fixed (see defer_long)."""
        deads = [self.dead_fn(tracking) for tracking, _ in train]

        def score(params):
            return float(np.mean([pm.evaluate(gt, pm.label_codes(
                self.predict_with(tracking, params, dead)))["spell_f1_mean"]
                for (tracking, gt), dead in zip(train, deads)]))
        best, best_score = self.params, score(self.params)
        for long in self.long_deflection_candidates:
            if long.get("long_deflection_s", 0.0) <= 0:
                continue
            params = dict(self.params, **long)
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        self.params = best
        print("    long deflection: {} (train spellF1 {:.3f})".format(
            {k: best[k] for k in ("long_deflection_s", "long_unseen_w", "long_where")}, best_score))
