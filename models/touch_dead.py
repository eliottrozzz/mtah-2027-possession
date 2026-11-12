"""Model v2: v1's last-touch possession plus Track B's dead-ball detector.

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

import numpy as np

from models.dead_ball import DeadBallDetector
from models.touch import LastTouch, all_touches


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


class TouchDead:
    name = "touch_dead_v2"
    restart_from_next_touch = True

    def __init__(self):
        self.touch = LastTouch()
        self.detector = DeadBallDetector()

    def settings(self):
        return {"touch": self.touch.settings(), "dead": self.detector.settings(),
                "restart_from_next_touch": self.restart_from_next_touch}

    def touches(self, tracking):
        return all_touches(tracking, self.touch.params)

    def fit(self, train):
        self.touch.fit(train)
        self.detector.fit(train, base_predict=self.touch.predict)

    def predict(self, tracking):
        labels = self.touch.predict(tracking)
        dead = self.detector.dead_mask(tracking)
        if self.restart_from_next_touch:
            frames, teams = self.touches(tracking)
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
    name = "touch_dead_v2_carry"
    restart_from_next_touch = False
