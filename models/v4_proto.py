"""v4 prototype: a per-frame H/A/D scorer on top of v3.1's signals, decoded with Viterbi.

v3.1 decides with hand-built rules: last touch, a dead-ball threshold, restart snapping, and a
team-shape override where the ball is unseen. v4 keeps all of v3.1's signals but lets a fitted
model weigh them frame by frame, then picks the most likely label sequence.

    1. Signals per frame (frame_features):
         v3.1's own label; the dead-ball probability; the team-shape prior WHERE THE BALL IS
         UNSEEN (with it everywhere, clean scores fell up to 0.105: v3.1's rule, Jordan
         2026-11-14); whether the ball is
         seen, for how long it has been missing, and how long the gap is; the last and next touch
         (team, time since or until, and how clear it was: the gap between the nearest home and
         the nearest away player); the players' typical speed over 1 s.
    2. Scorer: multinomial logistic regression (numpy, ridge penalty) giving P(H), P(A), P(D).
       Fitted on the training games' frames, plus a mirrored copy with home and away swapped, so
       it can't learn "the home team has the ball more".
    3. Decoder: Viterbi over 0.2 s blocks with a penalty per switch (H<->A, and into or out of D)
       and minimum durations for live and dead runs. Penalties and minimums are picked by train
       spellF1.

Fitted parts (v3.1 itself, the dead-ball detector, the shape prior) are applied OUT OF FOLD on the
training games: each training game's signals come from a v3.1 fitted on the other training game
only, so the scorer learns how much v3.1 is worth on a game it hasn't seen (Jordan, 2026-11-13).
The test game gets the v3.1 fitted on both training games, as v3.1 itself does.

External data only. Name: v4_proto (a version number is given only if it is merged).
"""

import itertools
import os
import sys
import warnings

import numpy as np

from models.dead_ball import MAX_SPEED, per_half_scale, rolling_mean, step_rate
from models.touch import FPS, PITCH, touch_strength
from models.touch_dead import TouchDeadShapeUnseen

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FIT_STEP = 5                  # every 5th frame to fit the scorer (neighbours are alike)
L2 = 1.0                      # ridge penalty
CLIP_LOGIT = 6.0
CAP_S = 10.0                  # times since / until are capped here
BLOCK = 5                     # decoder block, frames (0.2 s)
DECODE_GRID = {
    "switch": (1.0, 2.0, 4.0, 8.0),       # H <-> A (stage 1: the same seen or unseen)
    "dead": (2.0, 4.0, 8.0),              # into or out of D
    "min_live": (1, 3, 5),                # blocks (0.2, 0.6, 1.0 s)
    "min_dead": (5, 10, 20),              # blocks (1, 2, 4 s)
}
# Stage 2: separate H <-> A penalties where the ball is seen and where it isn't, the rest fixed.
# One penalty for both made the fit pick 4, which removed short possessions: noise in the copies,
# real on clean tracking (clean down 0.112). A switch at a visible touch needs less evidence than
# one hidden in a ball gap (Jordan, 2026-11-14).
SWITCH_GRID = {"switch_cheap": (0.5, 1.0, 2.0, 4.0), "switch_other": (1.0, 2.0, 4.0, 8.0)}
# Tried 2026-11-14: "cheap" = ball seen didn't fix clean (the fit still wanted 4 where seen: a
# blurred ball makes short touches unreliable even when visible). V4ProtoTouch makes switches cheap
# only near a CLEAR touch instead; V4ProtoCap caps the single penalty (Jordan: try both).
TOUCH_SWITCH_GRID = {"clear_m": (1.0, 2.0), "switch_cheap": (0.5, 1.0, 2.0),
                     "switch_other": (2.0, 4.0, 8.0)}
CAP_DECODE_GRID = dict(DECODE_GRID, switch=(0.5, 1.0))
SIGNED = ("v31_team", "prior_unseen", "v31_team_seen", "last_team", "last_team_recent",
          "next_team", "next_team_soon", "last_margin", "next_margin")
EVEN = ("v31_dead", "dead_logit", "dead_logit_2s", "dead_logit_8s", "seen", "since_seen",
        "gap_len", "since_touch", "until_touch", "last_clear", "next_clear",
        "touch_strength", "speed_1s")
LABELS = np.array(["H", "A", "D"])


# -- signals ------------------------------------------------------------------------------------

def base_outputs(model, tracking):
    """What a fitted v3.1 (TouchDeadShapeUnseen) says about one tracking: its label and the
    signals it is built from."""
    return {"label": model.predict(tracking),
            "dead_prob": model.detector.dead_probability(tracking),
            "p_home": model.prior.predict_proba(tracking),
            "touches": model.touches(tracking),
            "window": int(model.touch.params["window"])}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -CLIP_LOGIT, CLIP_LOGIT)


def team_distances(tracking, frames):
    """(nearest home, nearest away) player distance in metres from the ball at each frame."""
    ball = tracking["ball"][frames] * PITCH
    players = tracking["players"][frames] * PITCH
    dist = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for side in ("H", "A"):
            d = np.nanmin(np.where(tracking["team"] == side, dist, np.nan), axis=1)
            out.append(np.where(np.isfinite(d), d, 10.0))
    return out


def since_and_until(mask_frames, n):
    """(seconds since the last, seconds until the next) frame in mask_frames, capped at CAP_S, and
    the index of that last / next one (-1 if none)."""
    idx = np.arange(n)
    last = np.searchsorted(mask_frames, idx, side="right") - 1
    nxt = np.searchsorted(mask_frames, idx, side="left")
    since = np.where(last >= 0, (idx - mask_frames[np.clip(last, 0, None)]) / FPS, CAP_S)
    has_next = nxt < len(mask_frames)
    until = np.where(has_next, (mask_frames[np.clip(nxt, 0, len(mask_frames) - 1)] - idx) / FPS,
                     CAP_S)
    return np.minimum(since, CAP_S), np.minimum(until, CAP_S), last, np.where(has_next, nxt, -1)


def frame_features(tracking, base):
    """(signed, even) feature matrices. Signed features flip sign when home and away swap; even
    features don't. Home-positive convention."""
    n = len(tracking["ball"])
    label = base["label"]
    v31_team = (label == "H").astype(float) - (label == "A").astype(float)
    seen = np.isfinite(tracking["ball"][:, 0])
    prior = logit(base["p_home"])
    dead = logit(base["dead_prob"])

    # ball gaps: time since seen, and the length of the gap a frame sits in
    since_seen, _, _, _ = since_and_until(np.flatnonzero(seen), n)
    gap_len = np.zeros(n)
    edges = np.diff(np.concatenate(([0], (~seen).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        gap_len[a:b] = min((b - a) / FPS, 60.0)

    # touches: team, timing, and how clear it was
    frames, teams = base["touches"]
    frames = np.asarray(frames, dtype=int)
    sign = np.where(teams == "H", 1.0, -1.0)
    if len(frames):
        d_home, d_away = team_distances(tracking, frames)
        margin = np.clip(d_away - d_home, -5, 5) / 5.0          # + = clearly a home player
        strength = touch_strength(tracking["ball"], base["window"])[frames]
        strength = np.log1p(np.nan_to_num(strength, nan=0.0))
    else:
        margin = strength = np.zeros(0)
    since_t, until_t, last, nxt = since_and_until(frames, n)
    pick = lambda values, which: np.where(which >= 0, values[np.clip(which, 0, None)], 0.0) \
        if len(values) else np.zeros(n)
    last_team, next_team = pick(sign, last), pick(sign, nxt)
    last_margin, next_margin = pick(margin, last), pick(margin, nxt)

    speed = player_speeds_1s(tracking)

    signed = np.column_stack([
        v31_team, prior * ~seen, v31_team * seen,
        last_team, last_team * (since_t < 2.0), next_team, next_team * (until_t < 2.0),
        last_margin, next_margin])
    even = np.column_stack([
        (label == "D").astype(float), dead,
        rolling_mean(dead, FPS, FPS), rolling_mean(dead, 4 * FPS, 4 * FPS),
        seen.astype(float), np.log1p(since_seen), np.log1p(gap_len),
        np.log1p(since_t), np.log1p(until_t),
        np.abs(last_margin), np.abs(next_margin), pick(strength, last),
        np.nan_to_num(speed, nan=1.0)])
    return signed, even


def player_speeds_1s(tracking):
    """Median player speed over 1 s, divided by the half's median: robust to frame wobble."""
    rate = step_rate(tracking["players"] * PITCH, 1.0)
    speed = np.hypot(rate[..., 0], rate[..., 1])
    speed[speed > MAX_SPEED] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return per_half_scale(np.nanmedian(speed, axis=1), tracking["period"])


def clear_touch_frames(tracking, base, clear_m):
    """Per-frame flags: within BLOCK frames of a touch where one team's nearest player is at least
    clear_m closer to the ball than the other team's."""
    n = len(tracking["ball"])
    flags = np.zeros(n, dtype=bool)
    frames = np.asarray(base["touches"][0], dtype=int)
    if not len(frames):
        return flags
    d_home, d_away = team_distances(tracking, frames)
    for f in frames[np.abs(d_away - d_home) >= clear_m]:
        flags[max(0, f - BLOCK):f + BLOCK + 1] = True
    return flags


# -- scorer -------------------------------------------------------------------------------------

def design(signed, even, mirror=False):
    """Rows for the multinomial model. Mirroring swaps home and away (signed features flip)."""
    return np.column_stack([-signed if mirror else signed, even, np.ones(len(even))])


def fit_multinomial(X, y, l2=L2, iters=30):
    """Softmax regression over 3 classes (H, A, D), D as the reference. Newton's method with a
    ridge penalty (not on the intercepts). Returns W of shape (2, features)."""
    n, k = X.shape
    W = np.zeros((2, k))
    penalty = np.full(k, l2)
    penalty[-1] = 0.0
    Y = np.stack([y == 0, y == 1], axis=1).astype(float)
    for _ in range(iters):
        z = X @ W.T                                             # (n, 2)
        z = np.column_stack([z, np.zeros(n)])
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p = p[:, :2]
        grad = ((p - Y).T @ X) + penalty * W                    # (2, k)
        # Hessian blocks: X^T diag(p_a (delta_ab - p_b)) X
        H = np.zeros((2 * k, 2 * k))
        for a in range(2):
            for b in range(2):
                w = p[:, a] * ((a == b) - p[:, b])
                H[a * k:(a + 1) * k, b * k:(b + 1) * k] = (X * w[:, None]).T @ X
        H += np.diag(np.concatenate([penalty, penalty]))
        step = np.linalg.solve(H, grad.reshape(-1)).reshape(2, k)
        W -= step
        if np.abs(step).max() < 1e-6:
            break
    return W


def log_proba(W, X):
    z = np.column_stack([X @ W.T, np.zeros(len(X))])
    z -= z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))   # (n, 3): log P(H, A, D)


# -- decoder ------------------------------------------------------------------------------------

def transition_matrix(params, switch=None):
    """Dense log-transition matrix over chained states: each label L has min_L sub-states that
    must be passed in order before L can be left. Returns (T, label of each state, entry state of
    each label)."""
    sizes = {"H": params["min_live"], "A": params["min_live"], "D": params["min_dead"]}
    states, entry, last = [], {}, {}
    for lab in "HAD":
        entry[lab] = len(states)
        states += [lab] * sizes[lab]
        last[lab] = len(states) - 1
    S = len(states)
    T = np.full((S, S), -np.inf)
    for lab in "HAD":
        for i in range(entry[lab], last[lab]):
            T[i, i + 1] = 0.0
        T[last[lab], last[lab]] = 0.0
        for other in "HAD":
            if other == lab:
                continue
            cost = (params["switch"] if switch is None else switch) if {lab, other} == {"H", "A"} \
                else params["dead"]
            T[last[lab], entry[other]] = -cost
    return T, np.array(states), entry


def viterbi(block_logp, params, block_cheap=None):
    """Most likely block labels given (blocks, 3) log-probabilities of H, A, D. With block_cheap,
    an H <-> A switch into a block costs switch_cheap where the flag is set and switch_other where
    it isn't (if those are set; otherwise "switch" everywhere)."""
    T, states, entry = transition_matrix(params)
    split = block_cheap is not None and "switch_cheap" in params
    if split:
        T_seen = transition_matrix(params, params["switch_cheap"])[0]
        T_unseen = transition_matrix(params, params["switch_other"])[0]
    col = {"H": 0, "A": 1, "D": 2}
    emit = block_logp[:, [col[s] for s in states]]              # (blocks, S)
    m = len(block_logp)
    S = len(states)
    score = np.full(S, -np.inf)
    for lab in "HAD":
        score[entry[lab]] = 0.0
    score = score + emit[0]
    back = np.zeros((m, S), dtype=np.int32)
    for t in range(1, m):
        cand = score[:, None] + ((T_seen if block_cheap[t] else T_unseen) if split else T)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(S)] + emit[t]
    path = np.empty(m, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(m - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return states[path]


def decode(logp, period, params, cheap=None):
    """Per-frame labels: Viterbi within each half on BLOCK-frame blocks; D outside the halves.
    cheap: per-frame flags where a switch is cheap (a block is cheap if any of its frames is)."""
    n = len(logp)
    out = np.full(n, "D", dtype="<U1")
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        a, b = rows[0], rows[-1] + 1
        seg = logp[a:b]
        m = int(np.ceil(len(seg) / BLOCK))
        pad = np.full((m * BLOCK - len(seg), 3), np.nan)
        blocks = np.nanmean(np.concatenate([seg, pad]).reshape(m, BLOCK, 3), axis=1)
        block_cheap = None
        if cheap is not None:
            cv = np.concatenate([cheap[a:b], np.zeros(m * BLOCK - (b - a), dtype=bool)])
            block_cheap = cv.reshape(m, BLOCK).any(axis=1)
        out[a:b] = np.repeat(viterbi(blocks, params, block_cheap), BLOCK)[:b - a]
    return out


# -- the model ----------------------------------------------------------------------------------

def group_by_game(train):
    """Indices of train items that are the same game (same frame count and halves)."""
    groups = {}
    for i, (tracking, _) in enumerate(train):
        key = (len(tracking["period"]), tracking["period"].tobytes().__hash__())
        groups.setdefault(key, []).append(i)
    return list(groups.values())


class V4Proto:
    """v4 with a separate switch penalty where the ball is seen (switch_mode "seen")."""
    name = "v4_proto"
    switch_mode = "seen"
    decode_grid = DECODE_GRID
    switch_grid = SWITCH_GRID

    def __init__(self):
        self.v31 = TouchDeadShapeUnseen()
        self.W = None
        self.decode_params = None

    def settings(self):
        return {"v31": self.v31.settings(), "scorer": {"signed": SIGNED, "even": EVEN, "l2": L2,
                "fit_step": FIT_STEP, "weights": None if self.W is None else self.W.tolist()},
                "decode": dict(self.decode_params or {}), "decode_grid": self.decode_grid,
                "switch_grid": self.switch_grid, "switch_mode": self.switch_mode,
                "block": BLOCK, "out_of_fold": True}

    def oof_bases(self, train):
        """Each training item's v3.1 signals from a v3.1 fitted on the OTHER training game(s)."""
        groups = group_by_game(train)
        bases = [None] * len(train)
        if len(groups) < 2:                     # nothing to hold out: fall back to in-sample
            return [base_outputs(self.v31, t) for t, _ in train]
        for group in groups:
            other = TouchDeadShapeUnseen()
            other.fit([train[i] for i in range(len(train)) if i not in group])
            for i in group:
                bases[i] = base_outputs(other, train[i][0])
        return bases

    def fit(self, train, bases=None):
        """bases: precomputed out-of-fold signals (development only); computed if None."""
        self.v31.fit(train)
        bases = bases if bases is not None else self.oof_bases(train)
        feats = [frame_features(t, b) for (t, _), b in zip(train, bases)]
        self.fit_scorer(train, feats)
        self.fit_decoder(train, feats, bases)

    def fit_scorer(self, train, feats):
        X, y = [], []
        for (tracking, gt), (signed, even) in zip(train, feats):
            rows = (np.arange(len(gt)) % FIT_STEP == 0) & (tracking["period"] > 0) & (gt != pm.X)
            target = np.select([gt == pm.H, gt == pm.A], [0, 1], 2)[rows]
            X.append(design(signed[rows], even[rows]))
            y.append(target)
            X.append(design(signed[rows], even[rows], mirror=True))
            y.append(np.select([target == 0, target == 1], [1, 0], 2))
        self.W = fit_multinomial(np.concatenate(X), np.concatenate(y))

    def cheap_flags(self, tracking, base, params):
        """Where a switch is cheap, for this model's switch_mode."""
        if self.switch_mode == "seen":
            return np.isfinite(tracking["ball"][:, 0])
        if self.switch_mode == "touch" and "clear_m" in params:
            return clear_touch_frames(tracking, base, params["clear_m"])
        return None

    def fit_decoder(self, train, feats, bases):
        logps = [log_proba(self.W, design(*f)) for f in feats]

        def score(params):
            return float(np.mean([
                pm.evaluate(gt, pm.label_codes(decode(lp, t["period"], params,
                                                      self.cheap_flags(t, b, params))))
                ["spell_f1_mean"] for (t, gt), lp, b in zip(train, logps, bases)]))
        best, best_score = None, -1.0
        for values in itertools.product(*self.decode_grid.values()):
            params = dict(zip(self.decode_grid.keys(), values))
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        for values in itertools.product(*self.switch_grid.values()):
            params = dict(best, **dict(zip(self.switch_grid.keys(), values)))
            value = score(params)
            if value > best_score:
                best, best_score = params, value
        self.decode_params = best
        print("    v4 decode: {} (train spellF1 {:.3f})".format(best, best_score))

    def predict(self, tracking, base=None):
        base = base if base is not None else base_outputs(self.v31, tracking)
        logp = log_proba(self.W, design(*frame_features(tracking, base)))
        return decode(logp, tracking["period"], self.decode_params,
                      self.cheap_flags(tracking, base, self.decode_params))


class V4ProtoTouch(V4Proto):
    """Switches cheap only within 0.2 s of a clear touch (one team clearly nearest)."""
    name = "v4_proto_touch"
    switch_mode = "touch"
    switch_grid = TOUCH_SWITCH_GRID


class V4ProtoCap(V4Proto):
    """One switch penalty, capped at 1 (no separate cheap/other penalties)."""
    name = "v4_proto_cap"
    switch_mode = "single"
    decode_grid = CAP_DECODE_GRID
    switch_grid = {}
