"""Model v3: v1's last-touch possession plus Track B's dead-ball detector.

Versions (headline on the copies of the time):
    v2  2026-11-11  last touch + dead-ball detector + team from the first touch after a stoppage
    v3  2026-11-13  + the ball reappearing after a gap counts as a touch (reappear_s), + the
                    detector measures player speed over 0.4 s, trained on copies whose stoppage
                    ball is carried to the restart: 0.460. Runs before 2026-11-13 are logged
                    under the name touch_dead_v2.

    1. LastTouch (models/touch.py) says which team has the ball, from touches.
    2. DeadBallDetector (models/dead_ball.py) says when the ball is dead, mainly from the players.
       Its threshold and minimum run lengths are tuned on top of LastTouch's own labels.
    3. Dead frames become D.
    4. Optionally (restart_from_next_touch), after each dead stretch the team is taken from the
       first touch after it, not carried over from before the stoppage. A throw-in, goal kick or
       free kick often goes to the other team, so carrying the old team over would start the next
       possession with the wrong team.

Everything is fitted on the training games only (external data).
"""

import itertools
import os
import sys

import numpy as np

from models.dead_ball import DeadBallDetector
from models.team_shape import TeamShapePrior
from models.touch import FPS, LastTouch, all_touches

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "evaluation"))
import possession_metrics as pm  # noqa: E402

# Restart snapping: move each predicted end of a stoppage to a nearby touch, because a restart is
# a kick and then a reception, which the touches time far more sharply than the players do.
# "first" = the first touch within snap_s of the predicted end (the restart kick); "second" = the
# touch after it (the reception, where the reception answer key ends the stoppage).
SNAP_GRID = {"snap": ("none", "first", "second"), "snap_s": (1.0, 2.0, 3.0, 5.0)}


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


class TouchDead:
    name = "touch_dead_v3"
    restart_from_next_touch = True

    def __init__(self):
        self.touch = LastTouch()
        self.detector = DeadBallDetector()
        self.snap = {"snap": "none", "snap_s": 2.0}

    def settings(self):
        return {"touch": self.touch.settings(), "dead": self.detector.settings(),
                "restart_from_next_touch": self.restart_from_next_touch, "snap": dict(self.snap)}

    def touches(self, tracking):
        return all_touches(tracking, self.touch.params)

    def fit(self, train):
        self.touch.fit(train)
        self.detector.fit(train, base_predict=self.touch.predict)
        # The snap is tuned last, on top of everything else; the expensive parts run once.
        parts = [(self.touch.predict(tr), self.detector.dead_mask(tr), self.touches(tr), gt)
                 for tr, gt in train]
        best, best_score = None, -1.0
        for snap, snap_s in itertools.product(*SNAP_GRID.values()):
            if snap == "none" and snap_s != SNAP_GRID["snap_s"][0]:
                continue
            params = {"snap": snap, "snap_s": snap_s}
            score = float(np.mean([
                pm.evaluate(gt, pm.label_codes(self.combine(lab, dead, ft, params)))
                ["spell_f1_mean"] for lab, dead, ft, gt in parts]))
            if score > best_score:
                best, best_score = params, score
        self.snap = best
        print("    snap: {} (train spellF1 {:.3f})".format(best, best_score))

    def predict(self, tracking):
        return self.combine(self.touch.predict(tracking), self.detector.dead_mask(tracking),
                            self.touches(tracking), self.snap)

    @staticmethod
    def snap_restarts(dead, frames, params):
        """Move each end of a predicted stoppage to a touch near it (see SNAP_GRID)."""
        if params["snap"] == "none" or not len(frames):
            return dead
        dead, width = dead.copy(), int(params["snap_s"] * FPS)
        stoppages = runs(dead)
        for i, (start, stop) in enumerate(stoppages):
            if stop >= len(dead):
                continue
            near = np.flatnonzero((frames >= stop - width) & (frames <= stop + width))
            if not len(near):
                continue
            k = near[0] if params["snap"] == "first" else near[0] + 1
            if k >= len(frames) or frames[k] - frames[near[0]] > width:
                continue
            new_stop = int(frames[k])
            limit = stoppages[i + 1][0] if i + 1 < len(stoppages) else len(dead)
            if not start < new_stop < limit:
                continue
            dead[stop:new_stop] = True          # restart later than predicted
            dead[new_stop:stop] = False         # ... or earlier
        return dead

    def combine(self, labels, dead, touches, params):
        labels = labels.copy()
        frames, teams = touches
        dead = self.snap_restarts(dead, frames, params)
        if self.restart_from_next_touch:
            live = ~dead
            for start, stop in runs(live):
                if start == 0:
                    continue
                inside = np.flatnonzero((frames >= start) & (frames < stop))
                if len(inside):
                    first = frames[inside[0]]
                    labels[start:first] = teams[inside[0]]
        labels[dead] = "D"
        return labels


class TouchDeadCarry(TouchDead):
    """The same, but the team before a stoppage carries over to after it (for comparison)."""
    name = "touch_dead_v3_carry"
    restart_from_next_touch = False


# Team-shape override (Track D, 2026-11-13). The team-shape prior (models/team_shape.py) reads
# who has the ball from the players alone. Where it is very sure and disagrees with the touches
# for long enough, it wins: a touch given to the wrong player in a crowd, or a switch hidden in a
# ball gap. "off" stays in the grid, so fit() can turn it off.
OVERRIDE_GRID = [{"sure": None}] + [{"sure": s, "min_s": m}
                                    for s in (0.3, 0.35, 0.4, 0.45) for m in (0.5, 1.0, 2.0, 4.0)]


def shape_override(labels, p_home, params, allowed=None):
    """labels with live frames switched to the prior's team where it is more than 0.5 + sure
    confident, disagrees, and does so for at least min_s in a row. D frames are never touched,
    and with allowed given, only those frames can change."""
    if params["sure"] is None:
        return labels
    labels = labels.copy()
    prior = np.where(p_home > 0.5, "H", "A")
    flip = (np.isin(labels, ["H", "A"]) & (prior != labels)
            & (np.abs(p_home - 0.5) > params["sure"]))
    if allowed is not None:
        flip &= allowed
    for start, stop in runs(flip):
        if stop - start >= params["min_s"] * FPS:
            labels[start:stop] = prior[start:stop]
    return labels


class TouchDeadShape(TouchDead):
    """v3 plus the team-shape override."""
    name = "touch_dead_v3_shape"
    only_unseen = False         # override only where the ball isn't tracked (see below)

    def __init__(self):
        super().__init__()
        self.prior = TeamShapePrior()
        self.override = {"sure": None}

    def settings(self):
        return dict(super().settings(), shape=self.prior.settings(), override=dict(self.override),
                    only_unseen=self.only_unseen)

    def fit(self, train):
        super().fit(train)
        self.prior.fit(train)
        # Tuned last, on top of everything else; the expensive parts run once.
        parts = [(self.predict_v3(tr), self.prior.predict_proba(tr), self.allowed(tr), gt)
                 for tr, gt in train]
        best, best_score = None, -1.0
        for params in OVERRIDE_GRID:
            score = float(np.mean([
                pm.evaluate(gt, pm.label_codes(shape_override(lab, p, params, ok)))
                ["spell_f1_mean"] for lab, p, ok, gt in parts]))
            if score > best_score:
                best, best_score = params, score
        self.override = best
        print("    override: {} (train spellF1 {:.3f})".format(best, best_score))

    def predict_v3(self, tracking):
        return super().predict(tracking)

    def allowed(self, tracking):
        return np.isnan(tracking["ball"][:, 0]) if self.only_unseen else None

    def predict(self, tracking):
        return shape_override(self.predict_v3(tracking), self.prior.predict_proba(tracking),
                              self.override, self.allowed(tracking))


class TouchDeadShapeUnseen(TouchDeadShape):
    """The override only on frames where the ball isn't tracked: when the ball is seen, its
    touches are trusted; when it isn't, the team shapes decide (Jordan, 2026-11-13, after the
    everywhere version failed the clean guard)."""
    name = "v3.1"  # was touch_dead_v3_shape_unseen (runs before 2026-11-13 21:30 use that name)
    only_unseen = True
