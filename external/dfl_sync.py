"""Put DFL's events on the tracking clock, and measure how well that worked.

DFL's event times come from a live manual log: they lag the tracking by a different amount per
match (-1.7 to +1.3 s) and scatter by about +-0.7 s per event, too loose for a frame-level answer
key. The tracking is precise and every touch event names its player, so each touch is SNAPPED to
the moment it physically happens.

    python external/dfl_sync.py                 # development matches only (the default)
    python external/dfl_sync.py --holdout       # the 5 holdout matches: run ONCE, at the end

THE RULES BELOW WERE FIXED BEFORE THE METHOD WAS RUN (2026-11-11, agreed with Jordan)
-------------------------------------------------------------------------------------
Development matches: DEV. The method is built and looked at on these only.
Holdout matches: everything else. Run once, after the method is frozen. If the holdout disagrees,
investigate and report; never tweak the method in response.

Main test, RECIPIENT (the snap never reads it): for each completed pass with a logged Recipient,
the first player other than the passer to come within RECEIVE_M of the ball, between 3 frames and
RECEIVE_S after the snapped pass frame, should be that Recipient. Metrica's pass frames, scored
with the same test, are the reference for synced data.
Secondary: share of touches that could not be placed; agreement of the event's team with DFL's
own per-frame BallPossession just after the touch (their column, a second opinion, not truth).
The kick check is still printed but is NOT independent: the snap itself uses ball acceleration.

Pass mark for the holdout: recipient rate within 5 points of the development rate, no holdout
match more than 10 points below it, and under 15% of touches unplaced.

The snap
--------
A touch is a sudden change in the ball's velocity next to the named player. So for every touch:
  1. Rough lag per half: within ROUGH_S of the logged time, the frame with the largest velocity
     change while the player is within NEAR_M of the ball. The median offset is that half's lag.
  2. Candidates: within WINDOW_S of logged time + lag, the frames where the player is within
     NEAR_M of the ball. Each scores its velocity change (m/s, over +-DV_FRAMES).
  3. Order: one dynamic-programming pass picks a candidate per touch, strictly increasing in time,
     maximising the total score.
  4. Second chance (added once, on the development matches, before the holdout): a touch still
     unplaced searches up to SECOND_S around logged time + lag, but only strictly between its
     placed neighbours, so the order still bounds it. Touches with no candidate even then stay
     unplaced at logged time + lag and are counted.
Non-touch events (fouls, offsides, whistles) get logged time + lag.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl_tracking as dt  # noqa: E402

# Params (fixed before the first run on any match; see above)
FPS = 25
DEV = ("J03WMX", "J03WOH")              # one Bundesliga, one 2. Bundesliga match
ROUGH_S = 3.0                           # rough-lag search, +- around logged time
WINDOW_S = 2.0                          # snap window, +- around logged time + lag
SECOND_S = 4.0                          # second chance for unplaced touches, between neighbours
NEAR_M = 1.5                            # player must be this close for a touch
DV_FRAMES = 2                           # velocity before/after, in frames
RECEIVE_M = 1.0                         # recipient test: "reaches the ball"
RECEIVE_S = 4.0
KICK_FRAMES = 4                         # kick check (not independent)
KICK_DV = 3.0
PITCH = (105.0, 68.0)

# Events whose named player touches the ball on purpose. Duels are left out: they never switch
# possession on their own (as in Metrica), and name two players.
TOUCH_TAGS = {"Play", "ShotAtGoal", "OtherBallAction", "BallClaiming"}


def touch_player(event):
    """The player who touches the ball in this event, or None."""
    if not TOUCH_TAGS & set(event["tags"]):
        return None
    return event["attrs"].get("Player")


def metres(xy):
    return xy * np.array(PITCH, dtype=np.float32)


def velocity_change(ball_m):
    """|v_after - v_before| in m/s per row, over +-DV_FRAMES."""
    k = DV_FRAMES
    out = np.zeros(len(ball_m))
    v_before = (ball_m[k:-k] - ball_m[:-2 * k]) * FPS / k
    v_after = (ball_m[2 * k:] - ball_m[k:-k]) * FPS / k
    out[k:-k] = np.hypot(*(v_after - v_before).T)
    return np.nan_to_num(out)


class Fenwick:
    """Prefix maximum over rows, with the id of the state that holds it."""

    def __init__(self, n):
        self.value = np.full(n + 1, -np.inf)
        self.state = np.full(n + 1, -1, dtype=np.int64)

    def update(self, i, value, state):
        i += 1
        while i < len(self.value):
            if value > self.value[i]:
                self.value[i], self.state[i] = value, state
            i += i & -i

    def query(self, i):
        """Best (value, state) over rows 0..i; (0, -1) if none."""
        best, state = 0.0, -1
        i += 1
        while i > 0:
            if self.value[i] > best:
                best, state = self.value[i], self.state[i]
            i -= i & -i
        return best, state


def snap(game, extras, events):
    """Add 'row' (placed), 'raw_row' (logged), 'lag_row' (logged + lag) and 'snapped' (bool) to
    every event. Returns the lag per half, in seconds."""
    column = {pid: i for i, pid in enumerate(game["player"])}
    period = game["period"]
    n = len(period)
    ball_m = metres(game["ball"])
    dv = velocity_change(ball_m)
    times = np.array([e["time"] for e in events])
    for event, row in zip(events, dt.time_to_row(times, extras)):
        event["raw_row"] = int(row)

    def candidates(event, centre, half_width):
        player = touch_player(event)
        if player not in column:
            return np.array([], dtype=int)
        w = int(round(half_width * FPS))
        rows = np.arange(max(centre - w, 0), min(centre + w, n - 1) + 1)
        rows = rows[period[rows] == period[event["raw_row"]]]
        gap = metres(game["players"][rows, column[player]]) - ball_m[rows]
        return rows[np.hypot(gap[:, 0], gap[:, 1]) < NEAR_M]

    # 1. Rough lag per half.
    lag = {}
    for half in (1, 2):
        offsets = []
        for event in events:
            if period[event["raw_row"]] != half:
                continue
            rows = candidates(event, event["raw_row"], ROUGH_S)
            if len(rows):
                offsets.append(rows[np.argmax(dv[rows])] - event["raw_row"])
        lag[half] = int(np.median(offsets)) if offsets else 0
    for event in events:
        event["lag_row"] = int(np.clip(event["raw_row"] + lag.get(period[event["raw_row"]], 0),
                                       0, n - 1))
        event["row"], event["snapped"] = event["lag_row"], False

    # 2-3. Candidates, then the best strictly increasing choice.
    tree = Fenwick(n)
    states = []  # (event index, row, previous state)
    for i, event in enumerate(events):
        rows = candidates(event, event["lag_row"], WINDOW_S)
        new = []
        for row in rows:
            best, prev = tree.query(row - 1)
            new.append((best + dv[row], row, prev))
        for total, row, prev in new:  # insert after querying, so one event takes one row
            states.append((i, row, prev))
            tree.update(row, total, len(states) - 1)
    _, state = tree.query(n - 1)
    while state >= 0:
        i, row, state = states[state]
        events[i]["row"], events[i]["snapped"] = int(row), True

    # 4. Second chance, bounded by the placed neighbours.
    placed = [i for i, e in enumerate(events) if e["snapped"]]
    for i, event in enumerate(events):
        if event["snapped"] or touch_player(event) not in column:
            continue
        before = [events[j]["row"] for j in placed if j < i]
        after = [events[j]["row"] for j in placed if j > i]
        low, high = (before[-1] if before else -1), (after[0] if after else n)
        rows = candidates(event, event["lag_row"], SECOND_S)
        rows = rows[(rows > low) & (rows < high)]
        if len(rows):
            event["row"], event["snapped"] = int(rows[np.argmax(dv[rows])]), True
            placed = sorted(placed + [i])
    return {half: rows / FPS for half, rows in lag.items()}


# -- the tests ---------------------------------------------------------------------------------

def first_receiver(game, row, passer_col, limit_rows):
    """Column of the first player other than the passer within RECEIVE_M of the ball, from 3
    frames after row up to limit_rows later; -1 if nobody."""
    rows = np.arange(row + 3, min(row + limit_rows, len(game["period"]) - 1) + 1)
    if not len(rows):
        return -1
    gap = metres(game["players"][rows]) - metres(game["ball"][rows])[:, None, :]
    d = np.hypot(gap[..., 0], gap[..., 1])
    d[:, passer_col] = np.inf
    d = np.where(np.isnan(d), np.inf, d)
    hit = np.flatnonzero(d.min(axis=1) < RECEIVE_M)
    return int(np.argmin(d[hit[0]])) if len(hit) else -1


def recipient_rate(game, rows, passer_cols, recipient_cols):
    hits = [first_receiver(game, r, p, int(RECEIVE_S * FPS)) == q
            for r, p, q in zip(rows, passer_cols, recipient_cols)]
    return float(np.mean(hits)) if hits else float("nan")


def kick_rate(game, rows, cols):
    """Share of rows with a kick signature next to the player (NOT independent of the snap)."""
    ball_m = metres(game["ball"])
    k = KICK_FRAMES
    rows = np.clip(rows, k, len(ball_m) - k - 1)
    player_m = metres(game["players"][rows, cols])
    v_before = (ball_m[rows] - ball_m[rows - k]) * FPS / k
    v_after = (ball_m[rows + k] - ball_m[rows]) * FPS / k
    kick = np.hypot(*(v_after - v_before).T) >= KICK_DV
    away = (np.hypot(*(ball_m[rows + k] - player_m).T)
            > np.hypot(*(ball_m[rows] - player_m).T) + 0.5)
    return float(np.mean(kick & away))


def report_match(match):
    game, extras = dt.load_game(match), dt.load_extras(match)
    events = dt.read_events(match)
    lag = snap(game, extras, events)
    column = {pid: i for i, pid in enumerate(game["player"])}
    team_of = dict(zip(game["player"], game["team"]))

    passes = [e for e in events if "Play" in e["tags"] and e["attrs"].get("Player") in column
              and e["attrs"].get("Evaluation") == "successfullyCompleted"
              and e["attrs"].get("Recipient") in column]
    p_cols = np.array([column[e["attrs"]["Player"]] for e in passes])
    r_cols = np.array([column[e["attrs"]["Recipient"]] for e in passes])
    touches = [e for e in events if touch_player(e) in column]

    out = {"match": match, "lag": lag, "passes": len(passes),
           "unplaced": 1.0 - float(np.mean([e["snapped"] for e in touches]))}
    for name, key in (("raw", "raw_row"), ("lag", "lag_row"), ("snap", "row")):
        rows = np.array([e[key] for e in passes])
        out["recipient_" + name] = recipient_rate(game, rows, p_cols, r_cols)
        out["kick_" + name] = kick_rate(game, rows, p_cols)
        hits = []
        for e in touches:
            window = np.arange(e[key] + 5, min(e[key] + 11, len(game["period"])))
            if np.all(extras["dfl_ball_status"][window] == 1):
                hits.append(np.mean(extras["dfl_ball_possession"][window]
                                    == team_of[e["attrs"]["Player"]]))
        out["provider_" + name] = float(np.mean(hits))
    moved = np.array([(e["row"] - e["lag_row"]) / FPS for e in touches if e["snapped"]])
    out["moved_iqr"] = np.percentile(moved, [25, 75])
    out["at_edge"] = float(np.mean(np.abs(moved) >= WINDOW_S - 1.0 / FPS))

    print("{match}  lag H1 {h1:+.2f} s H2 {h2:+.2f} s  passes {passes:4d}  "
          "RECIPIENT raw/lag/snap {recipient_raw:.0%}/{recipient_lag:.0%}/{recipient_snap:.0%}  "
          "provider {provider_raw:.0%}/{provider_lag:.0%}/{provider_snap:.0%}  "
          "unplaced {unplaced:.1%}  moved IQR {lo:+.2f}..{hi:+.2f} s, at edge {at_edge:.1%}  "
          "(kick, not independent: {kick_raw:.0%}/{kick_lag:.0%}/{kick_snap:.0%})".format(
              h1=lag[1], h2=lag[2], lo=out["moved_iqr"][0], hi=out["moved_iqr"][1], **out))
    return out


def report_metrica():
    """The recipient test on Metrica's pass frames: the level synced data reaches."""
    import csv
    import metrica_tracking as mt
    for game_no in (1, 2):  # games 1-2 name the receiver in the CSV ('To')
        game = mt.load_game(game_no)
        column = {name: i for i, name in enumerate(game["player"])}
        rows, p_cols, r_cols = [], [], []
        with open(mt.fetch(game_no)[-1]) as handle:
            for r in csv.DictReader(handle):
                passer, receiver = r["From"].replace(" ", ""), r["To"].replace(" ", "")
                if r["Type"] == "PASS" and passer in column and receiver in column:
                    rows.append(int(r["Start Frame"]))
                    p_cols.append(column[passer])
                    r_cols.append(column[receiver])
        print("metrica game {}  passes {:4d}  RECIPIENT {:.0%}  (kick {:.0%})".format(
            game_no, len(rows), recipient_rate(game, rows, p_cols, r_cols),
            kick_rate(game, np.array(rows), np.array(p_cols))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--holdout", action="store_true",
                        help="the 5 holdout matches: run once, after the method is frozen")
    args = parser.parse_args()
    matches = [m for m in dt.MATCHES if (m in DEV) != args.holdout]
    print("{} matches; columns raw logged time / + lag per half / snapped".format(
        "HOLDOUT" if args.holdout else "development"))
    results = [report_match(m) for m in matches]
    print("mean RECIPIENT snapped {:.1%}, unplaced {:.1%}".format(
        np.mean([r["recipient_snap"] for r in results]), np.mean([r["unplaced"] for r in results])))
    report_metrica()


if __name__ == "__main__":
    main()
