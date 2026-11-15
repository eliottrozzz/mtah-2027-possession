"""Pretend Smart Tagging set pieces for the external training games (Metrica 1-3, DFL).

The provided match comes with Metrica's automatic Smart Tagging: SET PIECES intervals with a team
and a type (read by external/smart_tagging.py, Track B). The organisers allow it as a model input.
To fit a model that uses it, the training games need tags too. They have none, so this makes them
from each game's event log: one tag per logged restart, then corrupted so the tags are about as
unreliable as the real ones.

    python external/pretend_tags.py                 # summary counts per game (seed 0)
    python external/pretend_tags.py --seed 1        # ... another corruption draw

    import pretend_tags as pt
    tags = pt.pretend_tags("metrica", 1, seed=0)    # or pt.pretend_tags("dfl", "J03WMX", seed=0)

Each tag is a dict in smart_tagging.read_set_pieces' format:
    {"start_frame": int, "end_frame": int, "team": "H" | "A" | None, "type": str, "type_raw": str}
Frames are rows of the model's timeline: the tracking dict's row index (for the provided match
that is the video frame, as smart_tagging gives it). type is one of TYPES. type_raw is "pretend"
for a corrupted tag and "pretend_fp" for an invented one. Everything is computed on the fly; no
file is written.

LEAK RISK. The answer keys' stoppages end at the same restart events, so clean tags would hand a
model the key's D ends. Every tag therefore goes through the corruption below, and the defaults
lean pessimistic. Until tag_signal.py measures the real tags against v3.1's stoppages, the
defaults come from public counts only (b2, from DATA.md, 2026-11-15): the real match has 59
SET PIECES tags (throw-in 26, corner 12, goal kick 10, free kick 5, kick-off after goal 4,
kick-off start 2). Every throw-in has team N/A, and 6 of the other 33 do (Jordan's
aggregate count with smart_tagging.py, 2026-11-15: teams None 32, H 16, A 11). A typical match
has about 40-50 throw-ins, 20-30 free kicks, 15-20 goal kicks and about 10 corners (these logs
agree: Metrica 28-49 throw-ins, DFL 36-54), so the real tagger misses roughly half the throw-ins,
most free kicks and a third of the goal kicks, and finds nearly every corner and kick-off.

What the start and end of a real tag mean (the stoppage, the ball being placed, or the kick) is
not documented. Both are parameters here: start = restart - START_LEAD_S, end = start +
DURATION_S. tag_signal.py prints the numbers that should settle them.
"""

import argparse
import collections
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Params
FPS = 25
TYPES = ("throw_in", "corner", "goal_kick", "free_kick", "kickoff_goal", "kickoff_start")

# Corruption. Every number is a default from public counts, to be matched with tag_signal.py.
MISS_RATE = {                  # share of logged restarts that get no tag
    "throw_in": 0.45,          # ~26 tagged of a typical 45-50
    "corner": 0.10,            # 12 tagged of a typical ~10: nearly all found
    "goal_kick": 0.35,         # 10 tagged of a typical 15-17
    "free_kick": 0.80,         # 5 tagged of a typical 20-30
    "kickoff_goal": 0.0,       # 4 + 2 kick-offs: assume all found
    "kickoff_start": 0.0,
}
START_LEAD_S = 0.0             # tag start this long before the restart kick (unknown: see above)
START_JITTER_S = 2.0           # + uniform(-this, +this) on the start
DURATION_S = 3.0               # tag end = start + this ...
DURATION_JITTER_S = 1.0        # ... + uniform(-this, +this), independent of the start jitter,
MIN_DURATION_S = 0.5           #     never shorter than this
NA_TEAM_TYPES = ("throw_in",)  # these types never have a team (all real throw-ins are N/A)
NA_RATE = 0.18                 # other types: team N/A (None) this often (6 of 33 real tags)
WRONG_TEAM_RATE = 0.05         # ... or the other team this often
FALSE_POSITIVES_PER_90 = 4.0   # invented tags per 90 min, Poisson, uniform over the halves,
FALSE_POSITIVE_TYPES = {       # with these type weights (the real mix of tagged types)
    "throw_in": 26, "corner": 12, "goal_kick": 10, "free_kick": 5,  # no invented kick-offs
}
RETAKE_S = 60.0                # a Metrica kick-off this soon after the last one is a retake
SOURCES = ("metrica", "dfl")


# -- clean restarts from the event logs --------------------------------------------------------

def metrica_restarts(game):
    """(kick row, team, type) for every logged restart in Metrica game 1-3. Rows are Metrica
    frames, which index metrica_tracking's arrays directly. Penalties are not a tag type."""
    import metrica_labels as ml
    out, last_kickoff = [], {}
    for e in ml.load_events(game):
        sub = e["sub"]
        if e["type"] == "SET PIECE":
            if "THROW IN" in sub:
                kind = "throw_in"
            elif "CORNER KICK" in sub:
                kind = "corner"
            elif "FREE KICK" in sub:
                kind = "free_kick"
            elif "KICK OFF" in sub:
                # A kick-off soon after the previous one in the half is that one taken again.
                prev = last_kickoff.get(e["period"])
                if prev is None:
                    kind = "kickoff_start"
                elif e["start"] - prev[0] <= RETAKE_S * FPS:
                    kind = prev[1]
                else:
                    kind = "kickoff_goal"
                last_kickoff[e["period"]] = (e["start"], kind)
            else:
                continue
        elif e["type"] in ("PASS", "BALL LOST") and "GOAL KICK" in sub:
            kind = "goal_kick"
        else:
            continue
        out.append((int(e["start"]), e["team"], kind))
    return out


DFL_TYPES = {"ThrowIn": "throw_in", "CornerKick": "corner", "GoalKick": "goal_kick",
             "FreeKick": "free_kick"}


def dfl_restarts(match):
    """(kick row, team, type) for every logged restart in a DFL match, placed on the tracking
    rows by the same sync the answer keys use (dfl_sync.snap). Penalties and drop balls are not
    tag types. The first kick-off of each half is kickoff_start, later ones kickoff_goal."""
    import dfl_labels as dl
    import dfl_sync as ds
    import dfl_tracking as dt
    game, extras = dt.load_game(match), dt.load_extras(match)
    info = dt.match_info(match)
    events = dt.read_events(match)
    ds.snap(game, extras, events)
    period = game["period"]
    out, halves_seen = [], set()
    for e in sorted(events, key=lambda e: e["row"]):
        if e["tag"] == "KickOff":
            half = int(period[e["row"]])
            kind = "kickoff_goal" if half in halves_seen else "kickoff_start"
            halves_seen.add(half)
        elif e["tag"] in DFL_TYPES:
            kind = DFL_TYPES[e["tag"]]
        else:
            continue
        out.append((int(e["row"]), dl.team_side(info, e["attrs"].get("Team")), kind))
    return out


def clean_restarts(source, game):
    if source == "metrica":
        return metrica_restarts(int(game))
    if source == "dfl":
        return dfl_restarts(game)
    raise ValueError("source must be one of {}".format(SOURCES))


# -- corruption --------------------------------------------------------------------------------

def rng_for(source, game, seed):
    """A generator fixed by (source, game, seed), stable across Python runs."""
    return np.random.default_rng([zlib.crc32("{}:{}".format(source, game).encode()), int(seed)])


def corrupt_team(team, kind, rng):
    if kind in NA_TEAM_TYPES:
        return None
    u = rng.random()
    if u < NA_RATE or team is None:
        return None
    if u < NA_RATE + WRONG_TEAM_RATE:
        return "A" if team == "H" else "H"
    return team


def make_tag(kick_row, team, kind, rng, raw, jitter_scale=1.0):
    start = kick_row + int(round(
        (-START_LEAD_S + rng.uniform(-1, 1) * START_JITTER_S * jitter_scale) * FPS))
    duration = max(MIN_DURATION_S,
                   DURATION_S + rng.uniform(-1, 1) * DURATION_JITTER_S * jitter_scale)
    return {"start_frame": start, "end_frame": start + int(round(duration * FPS)),
            "team": corrupt_team(team, kind, rng), "type": kind, "type_raw": raw}


def corrupt(restarts, period, rng, miss_scale=1.0, jitter_scale=1.0):
    """Corrupted tags from clean (kick row, team, type) restarts: misses, timing jitter, wrong
    or N/A team, and invented tags. period is the tracking's period array (0 outside the halves);
    tags are clipped to the rows that exist and sorted by start. miss_scale multiplies every miss
    rate (capped at 1) and jitter_scale both jitters (start and duration), e.g. 2 and 2 for a
    harsh robustness variant. The random draws don't depend on the scales, so a harsh variant
    misses a superset of the default's tags and moves each tag the same way, only further."""
    n = len(period)
    tags = []
    for kick_row, team, kind in restarts:
        draws = rng.random()  # drawn for every restart, so one rate change doesn't reshuffle all
        tag = make_tag(kick_row, team, kind, rng, "pretend", jitter_scale)
        if draws >= min(1.0, MISS_RATE[kind] * miss_scale):
            tags.append(tag)
    in_half = np.flatnonzero(period > 0)
    minutes = len(in_half) / FPS / 60.0
    kinds = list(FALSE_POSITIVE_TYPES)
    weights = np.array([FALSE_POSITIVE_TYPES[k] for k in kinds], dtype=float)
    for _ in range(rng.poisson(FALSE_POSITIVES_PER_90 * minutes / 90.0)):
        kind = kinds[rng.choice(len(kinds), p=weights / weights.sum())]
        team = ("H", "A")[rng.integers(2)]
        tags.append(make_tag(int(rng.choice(in_half)), team, kind, rng, "pretend_fp",
                             jitter_scale))
    for tag in tags:
        tag["start_frame"] = int(np.clip(tag["start_frame"], 0, n - 1))
        tag["end_frame"] = int(np.clip(tag["end_frame"], tag["start_frame"], n - 1))
    return sorted(tags, key=lambda t: t["start_frame"])


def period_of(source, game):
    if source == "metrica":
        import metrica_tracking as mt
        return mt.load_game(int(game))["period"]
    import dfl_tracking as dt
    return dt.load_game(game)["period"]


def pretend_tags(source, game, seed=0, period=None, miss_scale=1.0, jitter_scale=1.0):
    """Corrupted set-piece tags for one external game. seed picks the corruption draw (use the
    tracking copy's seed so a level/seed pair always sees the same tags). miss_scale and
    jitter_scale scale the miss rates (capped at 1) and the jitters; see corrupt()."""
    if period is None:
        period = period_of(source, game)
    return corrupt(clean_restarts(source, game), period, rng_for(source, game, seed),
                   miss_scale, jitter_scale)


def all_games():
    import dfl_tracking as dt
    return [("metrica", g) for g in (1, 2, 3)] + [("dfl", m) for m in dt.MATCHES]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("pretend set-piece tags, seed {}: logged restarts -> tags (by type, team H/A/None)"
          .format(args.seed))
    for source, game in all_games():
        clean = clean_restarts(source, game)
        tags = pretend_tags(source, game, args.seed)
        logged = collections.Counter(k for _, _, k in clean)
        kept = collections.Counter(t["type"] for t in tags)
        teams = collections.Counter(str(t["team"]) for t in tags)
        fps = sum(t["type_raw"] == "pretend_fp" for t in tags)
        print("{:7s} {:6s} {:3d} -> {:3d} tags ({} invented)  {}  teams {}".format(
            source, str(game), len(clean), len(tags), fps,
            " ".join("{} {}/{}".format(k, kept[k], logged[k]) for k in TYPES),
            dict(sorted(teams.items()))))


if __name__ == "__main__":
    main()
