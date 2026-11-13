"""Answer key for the DFL sealed test games: per-row H/A/D labels from DFL's official events.

Same rules as metrica_labels.py (POSSESSION_RULES.md), so the sealed key means the same thing as
the key models are tuned on. Event times are first put on the tracking clock by dfl_sync.py
(frozen at commit b79114c), because DFL's live log scatters by about +-0.7 s.

    python external/dfl_labels.py                     # restart pass ends D at its reception
    python external/dfl_labels.py --restart kick      # ... or at the kick

Writes data/external/dfl/{match}_labels_{reception|kick}.csv, plus {match}_labels_dfl_provider.csv
(DFL's own per-frame BallStatus and BallPossession, for comparison only; never a key we score on).
Rows are tracking rows (dfl_tracking.py), so row i of the labels is row i of the tracking.

How the rules map onto DFL events (decisions agreed with Jordan, 2026-11-11)
--------------------------------------------------------------------------
Possession, switching at the snapped event row:
  Play (pass, cross), ShotAtGoal, OtherBallAction (incl. clearances), BallClaiming (keeper
  catches, interceptions), and the kick of every restart: that team has the ball.
  TacklingGame (duels), BallDeflection: no change on their own, as Metrica's CHALLENGE.
  BallClaiming of Type Block: a keeper block is a deflection, no change.
  A touch by the defending team within BLOCK_S after a BlockedShot is the block: no change.
  TacklingGame with WinnerResult ballClaimed: the winner's team has the ball (as Metrica's won
  CHALLENGE + RECOVERY). Other duel results stay ignored.
  Unlogged receptions: when one team's touch is followed by the other team's with no restart in
  between, the switch moves to the new team's first touch in the tracking after the old team's
  last one (player within ds.NEAR_M, ball speed change over ds.KICK_DV). The log decides who,
  the tracking only when. Duels and receptions never move a stoppage.
Dead ball, from:
  Throw-in, goal kick, corner: the frame the tracked ball crosses the line (a ball crossing in
  the air is out too; a ball stopping exactly on the line counts, as J03WQQ's tracking does).
  Fallback, when no crossing is found: the decision time, lag-corrected, minus the median
  crossing-to-decision gap for that restart type over all crossings found in the 7 matches.
  Goal: the frame the ball crosses the line after the shot. Fallback: the shot + 1 s.
  Free kick, penalty: the Foul / Offside event before it, lag-corrected. If touches are logged
  after that event (advantage), from the last one instead. Fallback: the decision time.
  Drop ball: DROP_BALL_S after the last touch (nothing marks the stoppage itself).
  End of half: the final whistle, lag-corrected; the start of each half is dead until kick-off.
  Then every stoppage start above moves to where DFL's own BallStatus says play stopped, unless
  the logged touches contradict it (a touch after DFL's stop; or DFL stopping later than us with
  no touch in between). This covers what the log can't: fouls noted late, video-referee reviews,
  a half ending before the restart, and out balls the tracking doesn't follow to the line.
Until:
  --restart reception: the restart pass is received: the first frame another player is within
  RECEIVE_M of the ball, at most until the next logged touch. Goal kicks wait for their
  reception like any other restart (answer-key-v1: the keeper rolling the ball out first is the
  model's job, not the key's). Restarts taken as a shot (penalties, direct free kicks) are alive
  at the kick.
  --restart kick: the kick.

Known limits (see external/README.md): touches DFL doesn't log can't switch possession (the switch
comes at the next logged touch, as in Metrica); about 5% of touches keep a lag-corrected time
(+-1 s); foul stoppages keep the log's +-0.7 s scatter; drop-ball stoppage starts are a guess.

This answer key is frozen before any model is scored on it. Change a rule for a football reason,
never because a model scores better under it.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import dfl_sync as ds  # noqa: E402
import dfl_tracking as dt  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
OUT_DIR = dt.OUT_DIR
RESTARTS = ("KickOff", "ThrowIn", "GoalKick", "CornerKick", "FreeKick", "Penalty", "RefereeBall")
OUT_RESTARTS = ("ThrowIn", "GoalKick", "CornerKick")
STOP_TAGS = ("Foul", "Offside")
BLOCK_S = 1.0               # a defender's touch this soon after a blocked shot is the block
GOAL_CROSS_S = 5.0          # look this long after a goal's shot for the ball crossing the line
FOUL_LOOKBACK_S = 20.0      # a free kick's foul/offside is at most this long before its decision
DROP_BALL_S = 1.0
RECEIVE_M = 1.0
AT_LINE_M = 0.05           # the ball at or past a line counts as out (see J03WQQ)
RECEIVE_MIN_FRAMES = 3      # the kicker's own follow-through is not a reception
TOUCH_SLACK_S = 0.7         # DFL's log scatter: a stoppage starts at least this long after the
                            # last logged touch, even where DFL's ball status says earlier
METRICA_DEAD = (0.33, 0.43)  # dead share in the Metrica keys; flag anything outside


def team_side(info, club):
    return "H" if club == info["home"] else ("A" if club == info["guest"] else None)


def label_match(match, restart, out_gap=None):
    """(labels as a string array per tracking row, stats about how stoppages were placed)."""
    game, extras = dt.load_game(match), dt.load_extras(match)
    info = dt.match_info(match)
    events = dt.read_events(match)
    ds.snap(game, extras, events)
    period = game["period"]
    n = len(period)
    ball = game["ball"]
    # Metres inside the nearest line. "Reaching" the line counts as crossing it: in J03WQQ the
    # tracked ball often stops exactly on the line instead of following the ball out.
    ball_m = ball * np.array(ds.PITCH)
    inside_m = np.minimum.reduce([ball_m[:, 0], ds.PITCH[0] - ball_m[:, 0],
                                  ball_m[:, 1], ds.PITCH[1] - ball_m[:, 1]])
    outside = inside_m <= AT_LINE_M
    crossing = np.flatnonzero(outside[1:] & ~outside[:-1]) + 1
    column = {pid: i for i, pid in enumerate(game["player"])}
    how = {"crossing": 0, "fallback": 0, "foul": 0, "advantage": 0, "drop": 0,
           "dfl_status_earlier": 0, "dfl_status_later": 0, "duel_claimed": 0, "reception": 0}
    gaps = []  # (restart type, decision row - crossing row) where a crossing was found

    def lag_row(time_text):
        """Row of a logged time (e.g. a DecisionTimestamp), corrected by the half's lag."""
        raw = int(dt.time_to_row(np.array([dt.parse_time(time_text)]), extras)[0])
        shifts = [e["lag_row"] - e["raw_row"] for e in events
                  if period[e["raw_row"]] == period[raw]]
        return int(np.clip(raw + (int(np.median(shifts)) if shifts else 0), 0, n - 1))

    # -- possession timeline from deliberate touches ---------------------------------------
    touches = []
    for i, e in enumerate(events):
        player = ds.touch_player(e)
        side = team_side(info, e["attrs"].get("Team"))
        if player is None or side is None:
            continue
        if e["tag"] == "BallClaiming" and e["attrs"].get("Type") == "Block":
            continue
        touches.append((i, e["row"], side))
    # Blocks: the defending team's touch right after a blocked shot.
    blocked = set()
    for k, (i, row, side) in enumerate(touches):
        if "BlockedShot" in events[i]["tags"] and k + 1 < len(touches):
            j, next_row, next_side = touches[k + 1]
            if next_side != side and next_row - row <= BLOCK_S * FPS:
                blocked.add(j)
    touches = [t for t in touches if t[0] not in blocked]
    touches.sort(key=lambda t: t[1])
    # Stoppages and restarts only trust touches that are logged and placed by the sync.
    touch_rows = np.array([t[1] for t in touches])

    # The two rules below only change who has the ball, never where a stoppage starts.
    # A duel the winner comes out of with the ball ("ballClaimed") is a deliberate touch by the
    # winner, as Metrica logs a won challenge followed by a RECOVERY. Other duel results (a poke
    # at the ball, a dribble past) stay ignored, like a tackle that only deflects the ball.
    duels = []
    for i, e in enumerate(events):
        if e["tag"] == "TacklingGame" and e["attrs"].get("WinnerResult") == "ballClaimed":
            side = team_side(info, e["attrs"].get("WinnerTeam"))
            if side is not None:
                duels.append((i, e["row"], side))
    how["duel_claimed"] += len(duels)
    timeline = sorted(touches + duels, key=lambda t: t[1])

    # Unlogged receptions. The log notes passes but often not the opponent collecting a
    # misplaced pass or a clearance, so the switch would only come at that player's next logged
    # action, seconds late (Metrica logs the collection as a RECOVERY). When one team's touch is
    # followed by the other team's, with no restart in between, the tracking places the switch:
    # the first touch by the new team after the old team's last touch, where a touch is a player
    # within ds.NEAR_M of the ball as it changes speed by more than ds.KICK_DV (the thresholds the
    # frozen sync already uses), and a moment with players of both teams that close counts as
    # the old team's. The log still decides which team gets the ball; the tracking only says when.
    dv = ds.velocity_change(ds.metres(ball))
    restart_rows = np.array(sorted(e["row"] for e in events if e["tag"] in RESTARTS))
    team_cols = {s: np.flatnonzero(game["team"] == s) for s in ("H", "A")}

    def near_ball(rows, side):
        gap = ds.metres(game["players"][rows][:, team_cols[side]]) \
            - ds.metres(ball[rows])[:, None, :]
        d = np.hypot(gap[..., 0], gap[..., 1])
        return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < ds.NEAR_M

    received = []
    for (_, row, side), (_, next_row, next_side) in zip(timeline, timeline[1:]):
        if next_side == side or next_row - row <= RECEIVE_MIN_FRAMES:
            continue
        if np.any((restart_rows > row) & (restart_rows <= next_row)):
            continue
        rows = np.arange(row + RECEIVE_MIN_FRAMES, next_row)
        rows = rows[dv[rows] > ds.KICK_DV]
        if not len(rows):
            continue
        old, new = near_ball(rows, side), near_ball(rows, next_side)
        if old.any():
            after = rows > rows[old][-1]
            rows, new = rows[after], new[after]
        if new.any():
            received.append((-1, int(rows[np.argmax(new)]), next_side))
    how["reception"] += len(received)
    timeline = sorted(timeline + received, key=lambda t: t[1])

    labels = np.full(n, "X", dtype="<U1")
    labels[period > 0] = "D"
    for k, (_, row, side) in enumerate(timeline):
        end = timeline[k + 1][1] if k + 1 < len(timeline) else n
        labels[row:end] = side
    labels[period == 0] = "X"
    for half in (1, 2):  # nothing before the first touch of a half is live
        rows = np.flatnonzero(period == half)
        first = touch_rows[(touch_rows >= rows[0]) & (touch_rows <= rows[-1])]
        labels[rows[0]:first[0] if len(first) else rows[-1] + 1] = "D"

    # -- stoppages --------------------------------------------------------------------------
    def last_touch_before(row):
        before = touch_rows[touch_rows < row]
        return int(before[-1]) if len(before) else None

    def stop_of(index, kick_row):
        e = events[index]
        decision = e["element"].get("DecisionTimestamp")
        decision_row = lag_row(decision) if decision else kick_row
        tag = e["tag"]
        if tag == "KickOff":
            previous = [x for x in events[:index] if "SuccessfulShot" in x["tags"]
                        and period[x["row"]] == period[kick_row]]
            if not previous:
                return None  # start of a half: already dead until the kick-off
            shot = previous[-1]["row"]
            after = crossing[(crossing > shot) & (crossing <= shot + GOAL_CROSS_S * FPS)]
            how["crossing" if len(after) else "fallback"] += 1
            return int(after[0]) if len(after) else shot + FPS
        if tag in OUT_RESTARTS:
            last = last_touch_before(min(decision_row, kick_row))
            low = last if last is not None else kick_row - 30 * FPS
            found = crossing[(crossing > low) & (crossing < kick_row)]
            how["crossing" if len(found) else "fallback"] += 1
            if len(found):
                if decision:
                    gaps.append((tag, decision_row - int(found[0])))
                return int(found[0])
            # No crossing in the tracking: the decision time, minus the typical crossing-to-
            # decision gap for this restart type, measured on the crossings that were found.
            return decision_row - int((out_gap or {}).get(tag, 0))
        if tag in ("FreeKick", "Penalty"):
            fouls = [x for x in events[:index] if x["tag"] in STOP_TAGS
                     and 0 <= decision_row - x["lag_row"] <= FOUL_LOOKBACK_S * FPS]
            if not fouls:
                how["fallback"] += 1
                return decision_row
            foul = fouls[-1]["lag_row"]
            later = touch_rows[(touch_rows > foul) & (touch_rows < min(decision_row, kick_row))]
            how["advantage" if len(later) else "foul"] += 1
            return int(later[-1]) + 1 if len(later) else foul
        if tag == "RefereeBall":
            how["drop"] += 1
            last = last_touch_before(kick_row)
            return (last + int(DROP_BALL_S * FPS)) if last is not None else kick_row
        return None

    def alive_of(event, kick_row):
        if restart == "kick" or "ShotAtGoal" in event["tags"]:
            return kick_row
        kicker = column.get(ds.touch_player(event))
        later = touch_rows[touch_rows > kick_row]
        limit = int(later[0]) if len(later) else min(kick_row + 4 * FPS, n - 1)
        rows = np.arange(kick_row + RECEIVE_MIN_FRAMES, limit + 1)
        if len(rows):
            gap = (game["players"][rows] - ball[rows][:, None, :]) * np.array(ds.PITCH)
            d = np.hypot(gap[..., 0], gap[..., 1])
            if kicker is not None:
                d[:, kicker] = np.inf
            hit = np.flatnonzero(np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < RECEIVE_M)
            if len(hit):
                return int(rows[hit[0]])
        return limit

    for index, event in enumerate(events):
        if event["tag"] not in RESTARTS:
            continue
        kick_row = event["row"]
        stop = stop_of(index, kick_row)
        alive = alive_of(event, kick_row)
        if stop is None:
            stop = np.flatnonzero(period == period[kick_row])[0]
        stop = min(stop, kick_row)
        labels[stop:alive] = "D"

    for event in events:  # end of each half
        if event["tag"] == "FinalWhistle":
            half = period[event["lag_row"]]
            last = np.flatnonzero(period == half)[-1]
            labels[max(event["lag_row"], (last_touch_before(last) or 0) + 1):last + 1] = "D"

    # -- stoppage starts from DFL's own ball status -----------------------------------------
    # DFL's per-frame BallStatus is the league's record of when play stopped; ours is rebuilt
    # from a live log that can note a foul or an out ball many seconds late, and has no event for
    # a half ending before the restart or a video-referee review. So each stoppage starts where
    # DFL's matching dead stretch starts, unless the logged touches contradict it: a touch after
    # DFL's stop means the ball was still in play (the stoppage starts TOUCH_SLACK_S after the
    # last touch), and DFL stopping later than us counts only if a touch is logged in between.
    # Restarts keep our own rule (DFL marks the ball alive at the kick, the public ground truth
    # at the reception). Decided with Eliott, 2026-11-12, from 21 cases where the two disagreed
    # for over 10 s.
    status_dead = np.concatenate([[False], (extras["dfl_ball_status"] == 0) & (period > 0),
                                  [False]]).astype(int)
    dfl_dead = np.flatnonzero(np.diff(status_dead) == 1)
    dfl_dead_end = np.flatnonzero(np.diff(status_dead) == -1)
    half_starts = {int(np.flatnonzero(period == h)[0]) for h in (1, 2)}
    dead = np.concatenate([[False], labels == "D", [False]]).astype(int)
    for a, b in zip(np.flatnonzero(np.diff(dead) == 1), np.flatnonzero(np.diff(dead) == -1)):
        if a in half_starts:
            continue
        # DFL's dead stretch overlapping ours the most; none means DFL has no stoppage here.
        overlap = [(min(b, e) - max(a, s), s) for s in dfl_dead
                   for e in dfl_dead_end[dfl_dead_end > s][:1] if min(b, e) > max(a, s)]
        if not overlap:
            continue
        dfl_start = max(overlap)[1]
        if period[dfl_start] != period[a]:
            continue
        if dfl_start >= a:
            # DFL stops later than our rebuilt time. Only believe it when a touch is logged in
            # between: after a foul or an out ball DFL sometimes stays "alive" for half a minute
            # with nobody touching the ball (e.g. J03WN1 47:11, before a throw-in at 47:12).
            between = touch_rows[(touch_rows >= a) & (touch_rows < dfl_start)]
            new = dfl_start if len(between) else a
        else:
            last = last_touch_before(a)
            floor = last + int(TOUCH_SLACK_S * FPS) if last is not None else dfl_start
            new = min(a, max(dfl_start, floor))
        new = min(new, b - 1)
        if new == a:
            continue
        if new < a:
            labels[new:a] = "D"
            how["dfl_status_earlier"] += 1
        else:
            labels[a:new] = labels[a - 1]  # play went on: the team before keeps the ball
            how["dfl_status_later"] += 1
    how["gaps"] = gaps
    return labels, how


def typical_out_gaps(matches):
    """Median rows from the ball crossing the line to DFL's (lag-corrected) decision, per restart
    type, over every crossing found in these matches. The fallback when none is found."""
    found = {}
    for match in matches:
        for tag, gap in label_match(match, "reception")[1]["gaps"]:
            found.setdefault(tag, []).append(gap)
    return {tag: float(np.median(g)) for tag, g in found.items()}


def provider_labels(match):
    """DFL's own per-frame columns as a key: dead where BallStatus is 0, else BallPossession."""
    game, extras = dt.load_game(match), dt.load_extras(match)
    labels = np.where(extras["dfl_ball_status"] == 1, extras["dfl_ball_possession"], "D")
    labels[game["period"] == 0] = "X"
    return labels.astype("<U1")


def write_labels(labels, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))


def describe(match, labels, how=None, provider=None):
    """The summary numbers of metrica_labels.py, plus flags for anything far from Metrica's."""
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, _ in spells]) / float(FPS)
    stats = pm.possession_stats(codes, runs)
    graded = codes != pm.X
    dead = float(np.mean(codes[graded] == pm.D))
    flag = "" if METRICA_DEAD[0] <= dead <= METRICA_DEAD[1] else "  <-- dead share outside Metrica's"
    extra = ""
    if provider is not None:
        extra += "  agrees with dfl_provider {:.0%}".format(float(np.mean(labels[graded]
                                                                           == provider[graded])))
    if how:
        extra += "  stops: " + ", ".join("{} {}".format(k, v) for k, v in how.items()
                                         if v and k != "gaps")
    print("{}  {:6.1f} min graded  {:4d} possessions  mean {:5.1f} s  median {:4.1f} s  "
          "under 1 s {:3d}  home share {:.2f}  dead {:.2f}{}{}".format(
              match, graded.sum() / FPS / 60.0, len(spells), stats["mean_duration_s"],
              float(np.median(lengths)), int(np.sum(lengths < 1.0)), stats["home_share"], dead,
              extra, flag))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--restart", choices=("reception", "kick"), default="reception",
                        help="when a restart brings the ball back to life (POSSESSION_RULES.md)")
    parser.add_argument("--matches", nargs="+", default=list(dt.MATCHES), choices=dt.MATCHES)
    args = parser.parse_args()
    out_gap = typical_out_gaps(dt.MATCHES)
    print("typical crossing -> decision gap (fallback when no crossing is found): " + ", ".join(
        "{} {:.1f} s".format(tag, gap / FPS) for tag, gap in sorted(out_gap.items())))
    for match in args.matches:
        labels, how = label_match(match, args.restart, out_gap)
        provider = provider_labels(match)
        write_labels(labels, os.path.join(OUT_DIR, "{}_labels_{}.csv".format(match, args.restart)))
        write_labels(provider, os.path.join(OUT_DIR, "{}_labels_dfl_provider.csv".format(match)))
        describe(match, labels, how, provider)
    print("labels written to", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()
