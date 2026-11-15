"""Set-piece tags against predicted stoppages, in summary numbers: real Smart Tagging vs pretend.

v5 uses the provided match's Smart Tagging set pieces (external/smart_tagging.py). The model is
fitted on external games with pretend tags (external/pretend_tags.py) made from the event logs and
then corrupted. This prints the same aggregate numbers for both, so the corruption (miss rates,
timing, what a tag's start and end mean) can be matched to the real tagger:

    python external/tag_signal.py             # pretend tags on the external games only
    python external/tag_signal.py --match     # also the provided match: Jordan only

Stoppages come from v3.1 (models.touch_dead:TouchDeadShapeUnseen), fitted on the Metrica copies
(live20/35, seed 0, reception key) as everywhere else, and cached in data/external/ keyed by the
models/ tree. Its stoppages are what a model sees on the real match, where there's no key. On the
external games:
  * DFL TRAIN (4 matches) is out of sample for v3.1; Metrica 1-3 are in sample (v3.1 was fitted on
    them), so their stoppages are a little cleaner than the real match's would be.
  * "logged" rows use the uncorrupted restarts (start = the kick, duration DURATION_S), to show
    what the corruption does; "pretend vs key" uses the answer key's stoppages instead of v3.1's.
DFL TEST matches are never read.

Numbers, pooled over the tags of a row (seconds; signed, + = the tag is later):
  n, by type        tag counts; team H / A / None
  duration          tag end - start, p10 / p50 / p90
  start-E, start-S  tag start minus the nearest predicted stoppage END (dead -> live, i.e. the
                    restart) and the nearest stoppage START (live -> dead), p10 / p50 / p90
  end-E, end-S      the same for the tag end
  start-E by type   median start-E per type
  no stop 10 s      share of tags with no predicted stoppage end within 10 s of the tag start
  dead in tag       share of the tag's frames that are predicted dead
  team agree        tags with a team: share where it equals v3.1's majority possession team over
                    the TEAM_CHECK_S after the tag start (tags with no H/A frame there skipped)
With --match the output ends with VERDICTS: each real number against the min..max the pretend
tags span over the external game-levels (WITHIN / OUTSIDE), then two summary lines:
  team mapping check: ... -> mapping OK / FLIPPED / UNCLEAR   (FLIPPED: Barcelona = H is the
                                                               ATD's away team; tell b2)
  noise range check: all WITHIN / OUTSIDE on: ...              (OUTSIDE: v5 trained on tags
                                                               unlike the real ones)
    python external/tag_signal.py --match --quiet   # Monday: the match and the verdicts only Nothing per tag, frame or position is printed or saved, so
the output is safe to paste. This only measures: it changes no calibration.
"""

import argparse
import collections
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import pretend_tags as pt  # noqa: E402

# Params
FPS = 25
MODEL_SPEC = "models.touch_dead:TouchDeadShapeUnseen"   # v3.1: cheaper than v4, same signals
LEVELS = (20, 35)
SEED = 0                        # copies and pretend-tag draw
QUANTILES = (10, 50, 90)
NO_STOP_S = 10.0
TEAM_CHECK_S = 3.0
TEAM_OK = 0.55                  # team mapping check: at least this share agree -> OK,
TEAM_FLIPPED = 0.45             # at most this -> FLIPPED, in between -> UNCLEAR. Uncorrupted
                                # restarts agree with v3.1 ~71% on the external games (so a
                                # flip would show ~29%); pretend tags 59-74%; the real match has
                                # ~27 tags with a team, so +-9% sampling noise
CACHE = os.path.join(ROOT, "data", "external", "tag_signal_{}_{}.pkl")


# -- the stoppage model ------------------------------------------------------------------------

def models_tree():
    return subprocess.run(["git", "rev-parse", "HEAD:models"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()[:12] or "nogit"


def stoppage_model():
    """v3.1 fitted on the Metrica copies (the standard fit set), cached by the models/ tree."""
    import score_external as se
    path = CACHE.format(MODEL_SPEC.split(":")[1], models_tree())
    clean = not subprocess.run(["git", "status", "--porcelain", "models"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
    if clean and os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    print("fitting v3.1 on the Metrica copies (a few minutes) ...", flush=True)
    t0 = time.time()
    model = se.load_model(MODEL_SPEC)
    model.fit(se.metrica_train_set())
    print("  fitted in {:.0f} s".format(time.time() - t0), flush=True)
    if clean:  # models/ with uncommitted edits: don't cache under the committed tree's name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(model, handle)
    return model


def predict_labels(model, tracking):
    labels = np.asarray(model.predict(tracking)).astype("<U1")
    return np.where(tracking["period"] > 0, labels, "X")


# -- measuring ---------------------------------------------------------------------------------

def stoppages(labels, period):
    """(stoppage start rows, stoppage end rows) inside the halves. A start is the first dead row
    after live play; an end is the first live row after a dead stretch. The dead stretch before a
    half's first kick-off has an end but no start; the one after its last whistle, neither."""
    starts, ends = [], []
    for half in (1, 2):
        rows = np.flatnonzero(period == half)
        if not len(rows):
            continue
        dead = np.concatenate([[0], (labels[rows] == "D").astype(np.int8), [0]])
        a = np.flatnonzero(np.diff(dead) == 1)
        b = np.flatnonzero(np.diff(dead) == -1)
        starts += [rows[0] + i for i in a if i > 0]
        ends += [rows[0] + i for i in b if i < len(rows)]
    return np.array(sorted(starts), dtype=float), np.array(sorted(ends), dtype=float)


def nearest(x, points):
    """Signed x - nearest point, in seconds (NaN when there are no points)."""
    if not len(points):
        return np.full(len(x), np.nan)
    i = np.clip(np.searchsorted(points, x), 1, max(len(points) - 1, 1))
    before, after = points[i - 1], points[np.minimum(i, len(points) - 1)]
    best = np.where(np.abs(x - before) <= np.abs(x - after), before, after)
    return (x - best) / FPS


def measure(tags, labels, period):
    """Per-tag numbers for one game, to be pooled (see the docstring)."""
    out = collections.defaultdict(list)
    S, E = stoppages(labels, period)
    start = np.array([t["start_frame"] for t in tags], dtype=float)
    end = np.array([t["end_frame"] for t in tags], dtype=float)
    out["type"] = [str(t["type"]) for t in tags]
    out["team"] = [str(t["team"]) for t in tags]
    out["duration"] = list((end - start) / FPS)
    for name, x in (("start-E", start), ("start-S", start), ("end-E", end), ("end-S", end)):
        out[name] = list(nearest(x, E if name.endswith("E") else S))
    near = nearest(start, E)
    out["no stop"] = list(~(np.abs(near) <= NO_STOP_S))
    team_rows = int(round(TEAM_CHECK_S * FPS))
    for t in tags:
        a, b = int(t["start_frame"]), int(t["end_frame"])
        out["dead in tag"].append(float(np.mean(labels[a:b + 1] == "D")) if b >= a else np.nan)
        if t["team"] in ("H", "A"):
            window = labels[a:a + team_rows]
            h, aw = int(np.sum(window == "H")), int(np.sum(window == "A"))
            if h != aw:
                out["team agree"].append((t["team"] == "H") == (h > aw))
    return out


def pool(parts):
    out = collections.defaultdict(list)
    for part in parts:
        for key, values in part.items():
            out[key] += list(values)
    return out


def q(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "   -     -     -  "
    return " ".join("{:5.1f}".format(x) for x in np.percentile(v, QUANTILES))


def show(label, m, games):
    n = len(m["type"])
    types = collections.Counter(m["type"])
    teams = collections.Counter(m["team"])
    per = max(games, 1)
    print("\n{}  ({} game{}; per game: {:.1f} tags)".format(label, games, "s" * (games != 1),
                                                             n / per))
    print("  by type /game  " + "  ".join("{} {:.1f}".format(k, types[k] / per)
                                          for k in pt.TYPES + ("None",) if types[k]))
    print("  team /game     " + "  ".join("{} {:.1f}".format(k, teams[k] / per)
                                          for k in ("H", "A", "None")))
    print("  duration       {}".format(q(m["duration"])))
    print("  start-E {}   start-S {}".format(q(m["start-E"]), q(m["start-S"])))
    print("  end-E   {}   end-S   {}".format(q(m["end-E"]), q(m["end-S"])))
    by_type = collections.defaultdict(list)
    for kind, x in zip(m["type"], m["start-E"]):
        by_type[kind].append(x)
    print("  start-E median by type  " + "  ".join(
        "{} {:+.1f}".format(k, np.nanmedian(by_type[k])) for k in pt.TYPES + ("None",)
        if by_type[k]))
    agree = m["team agree"]
    print("  no stop {:.0f} s {:.0%}   dead in tag {:.0%}   team agree {} ({} tags)".format(
        NO_STOP_S, np.mean(m["no stop"]) if n else np.nan, np.nanmean(m["dead in tag"])
        if n else np.nan, "{:.0%}".format(np.mean(agree)) if agree else "-", len(agree)))


def logged_tags(restarts, n):
    """Uncorrupted tags: start = the kick, duration DURATION_S."""
    d = int(round(pt.DURATION_S * FPS))
    return [{"start_frame": r, "end_frame": min(r + d, n - 1), "team": team, "type": kind}
            for r, team, kind in restarts]


# -- the rows ----------------------------------------------------------------------------------

def external_rows(model, quiet=False):
    """Prints the external rows; returns the per-game PRETEND-vs-v3.1 measures (the trained
    noise, every group and level) that the --match verdicts compare against."""
    import score_external as se
    reference = []
    groups = [("Metrica 1-3, in sample", [("metrica", g) for g in se.GAMES]),
              ("DFL TRAIN, out of sample", [("dfl", m) for m in se.DFL_TRAIN])]
    for level in LEVELS:
        for name, games in groups:
            pretend, logged, keyed = [], [], []
            for source, game in games:
                assert not (source == "dfl" and game in se.DFL_TEST)
                tracking = se.load_tracking(game, level, SEED, sealed=source == "dfl")
                period = tracking["period"]
                labels = predict_labels(model, tracking)
                key = np.array(se.pm.LABELS)[se.load_answer_key(
                    game, se.HEADLINE_KEY, period, sealed=source == "dfl")]
                tags = pt.pretend_tags(source, game, SEED, period=period)
                pretend.append(measure(tags, labels, period))
                reference.append(pretend[-1])
                keyed.append(measure(tags, key, period))
                logged.append(measure(logged_tags(pt.clean_restarts(source, game), len(period)),
                                      labels, period))
            if quiet:
                continue
            show("PRETEND  {}  live{}  vs v3.1".format(name, level), pool(pretend), len(games))
            show("logged   {}  live{}  vs v3.1".format(name, level), pool(logged), len(games))
            show("pretend  {}  live{}  vs answer key".format(name, level), pool(keyed),
                 len(games))
    return reference


# -- verdicts ----------------------------------------------------------------------------------

def stat(m, name):
    """One number per check from a game's (or the match's) measures."""
    if name == "tags":
        return float(len(m["type"]))
    if name in ("throw_in", "corner", "goal_kick", "free_kick"):
        return float(sum(t == name for t in m["type"]))
    if name in ("start-E p50", "end-E p50", "duration p50"):
        v = np.asarray(m[name.split()[0]], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan
    if name == "start-E spread":
        v = np.asarray(m["start-E"], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.subtract(*np.percentile(v, (90, 10)))) if len(v) else np.nan
    if name == "no stop 10 s":
        return float(np.mean(m["no stop"])) if len(m["no stop"]) else np.nan
    raise KeyError(name)


VERDICT_STATS = ("tags", "throw_in", "corner", "goal_kick", "free_kick", "duration p50",
                 "start-E p50", "end-E p50", "start-E spread", "no stop 10 s")


def verdicts(real, reference):
    """One line per check: the real match's number against the range the pretend tags span over
    the external games and levels (each game counted separately, so the range is wide)."""
    print("\nVERDICTS (real match vs the trained pretend-tag noise: min..max over {} external "
          "game-levels)".format(len(reference)))
    outside = []
    for name in VERDICT_STATS:
        values = np.array([stat(m, name) for m in reference])
        values = values[np.isfinite(values)]
        x = stat(real, name)
        lo, hi = (values.min(), values.max()) if len(values) else (np.nan, np.nan)
        ok = np.isfinite(x) and lo <= x <= hi
        if not ok:
            outside.append(name)
        print("  {:16s} real {:7.2f}   trained {:7.2f} .. {:7.2f}   {}".format(
            name, x, lo, hi, "WITHIN" if ok else "OUTSIDE"))
    agree = real["team agree"]
    share = float(np.mean(agree)) if agree else float("nan")
    mapping = ("OK" if share >= TEAM_OK else "FLIPPED" if share <= TEAM_FLIPPED else "UNCLEAR")
    print("\nteam mapping check: {:.0%} agree ({} tags with a team and a predicted team) -> "
          "mapping {}".format(share, len(agree), mapping))
    print("noise range check: {}".format(
        "all WITHIN the trained range" if not outside
        else "OUTSIDE the trained range on: " + ", ".join(outside)))


def match_row(model):
    """The provided match, read exactly as predict_match reads it. Jordan only."""
    import games
    import predict_match
    import smart_tagging
    tracking = predict_match.read_match(game=games.GAME, data_dir=None)
    period = tracking["period"]
    n = len(period)
    assert n == games.source_frames("tactical", games.GAME), "rows are not the video frames"
    outside = period == 0
    tracking["ball"][outside] = np.nan
    tracking["players"][outside] = np.nan
    tags = smart_tagging.read_set_pieces(smart_tagging.default_path(games.GAME, None))
    assert all(0 <= t["start_frame"] < n for t in tags), "a tag starts outside the tracking"
    tags = [dict(t, end_frame=min(int(t["end_frame"]), n - 1)) for t in tags]
    labels = predict_labels(model, tracking)
    m = measure(tags, labels, period)
    show("REAL MATCH  Smart Tagging set pieces  vs v3.1", m, 1)
    outside_halves = sum(period[t["start_frame"]] == 0 for t in tags)
    print("  tags starting outside the graded halves: {}".format(outside_halves))
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--match", action="store_true",
                        help="also measure the provided match (run it yourself)")
    parser.add_argument("--quiet", action="store_true",
                        help="with --match: skip printing the external rows (still computed, as "
                             "the verdicts' reference)")
    args = parser.parse_args()
    model = stoppage_model()
    reference = external_rows(model, quiet=args.quiet and args.match)
    if args.match:
        verdicts(match_row(model), reference)


if __name__ == "__main__":
    main()
