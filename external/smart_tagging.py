"""Read the match's Smart Tagging (SportsCode XML): set pieces and phases, as video-frame intervals.

Written from the public spec in DATA.md ("Smart Tagging") only. Each <instance> has <start> and
<end> in seconds from the first frame of the tactical video (frame = round(s * 25)), a <code>, and
<label> elements with a <group> (tag group, e.g. Team, Type) and a <text> (its value). Codes and
values are matched case-insensitively, and whitespace is stripped.

    read_set_pieces(path)   code "SET PIECES" -> [{start_frame, end_frame, team, type, type_raw}]
    read_phases(path)       every other code with a Team tag -> [{start_frame, end_frame, team, code}]
    tag_arrays(items, n)    per-frame arrays over video frames 0..n-1, for the model and predict_match
    default_path()          the provided file, found the way predict_match finds atd_meta

team is "H" (Barcelona), "A" (Atletic Club) or None (N/A, missing or unknown). The tags are
Metrica's automatic output, not ground truth: evidence, not answers. Frames are VIDEO frames;
mapping them to tracking rows is the caller's job (the same mapping predict_match uses).

Standard library only (plus numpy for tag_arrays).

    python external/smart_tagging.py [path]    # prints counts only
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

# Params
FPS = 25
SET_PIECES = "set pieces"
TEAMS = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}
TYPES = {
    "throw-in": "throw_in",
    "corner kick": "corner",
    "goal kick": "goal_kick",
    "free kick": "free_kick",
    "kick off (after goal)": "kickoff_goal",
    "kick off (start)": "kickoff_start",
}
TEAM_CODE = {"H": 1, "A": -1, None: 0}


def _norm(text):
    """Lower case with runs of whitespace collapsed; None stays None."""
    if text is None:
        return None
    text = " ".join(text.split()).lower()
    return text or None


def _child_text(element, name):
    """Text of the first child whose tag matches name case-insensitively, else None."""
    for child in element:
        if child.tag.strip().lower() == name:
            return child.text
    return None


def _frame(seconds):
    return int(round(float(seconds) * FPS))


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_") or None


def read_instances(path):
    """Every <instance> with a readable start, end and code, as dicts:
    {start_frame, end_frame, code (lower case), tags {group (lower case): value (lower case)}}.
    Instances missing a start, end or code are skipped. A group given twice keeps the first."""
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter():
        if inst.tag.strip().lower() != "instance":
            continue
        start, end, code = (_child_text(inst, k) for k in ("start", "end", "code"))
        code = _norm(code)
        try:
            start_frame, end_frame = _frame(start), _frame(end)
        except (TypeError, ValueError):
            continue
        if code is None:
            continue
        tags = {}
        for label in inst:
            if label.tag.strip().lower() != "label":
                continue
            group, value = _norm(_child_text(label, "group")), _norm(_child_text(label, "text"))
            if group is not None and value is not None:
                tags.setdefault(group, value)
        out.append({"start_frame": min(start_frame, end_frame),
                    "end_frame": max(start_frame, end_frame), "code": code, "tags": tags})
    out.sort(key=lambda d: (d["start_frame"], d["end_frame"]))
    return out


def team_of(tags):
    """'H', 'A' or None from the Team tag (N/A, missing or an unknown name -> None)."""
    return TEAMS.get(tags.get("team"))


def read_set_pieces(path):
    """Set pieces: [{start_frame, end_frame, team, type, type_raw}], sorted by start."""
    out = []
    for d in read_instances(path):
        if d["code"] != SET_PIECES:
            continue
        raw = d["tags"].get("type")
        out.append({"start_frame": d["start_frame"], "end_frame": d["end_frame"],
                    "team": team_of(d["tags"]),
                    "type": None if raw is None else TYPES.get(raw, _slug(raw)),
                    "type_raw": raw})
    return out


def read_phases(path):
    """Phases of play: every code other than SET PIECES that has a Team tag (N/A included, as
    team None): [{start_frame, end_frame, team, code}], sorted by start."""
    return [{"start_frame": d["start_frame"], "end_frame": d["end_frame"],
             "team": team_of(d["tags"]), "code": d["code"]}
            for d in read_instances(path)
            if d["code"] != SET_PIECES and "team" in d["tags"]]


def tag_arrays(items, n_frames):
    """Per-frame arrays over video frames 0..n_frames-1 from read_set_pieces or read_phases:
    active (bool), team (int8: +1 H, -1 A, 0 none), start and end (bool, at the interval's first
    and last frame) and type (object: the item's type, or its code for phases, or None).
    Intervals are clipped to the range; where they overlap, the later one (by start) wins."""
    active = np.zeros(n_frames, bool)
    team = np.zeros(n_frames, np.int8)
    start = np.zeros(n_frames, bool)
    end = np.zeros(n_frames, bool)
    kind = np.full(n_frames, None, object)
    for d in sorted(items, key=lambda d: d["start_frame"]):
        a, b = max(d["start_frame"], 0), min(d["end_frame"], n_frames - 1)
        if 0 <= d["start_frame"] < n_frames:
            start[d["start_frame"]] = True
        if 0 <= d["end_frame"] < n_frames:
            end[d["end_frame"]] = True
        if a > b:
            continue
        active[a:b + 1] = True
        team[a:b + 1] = TEAM_CODE.get(d.get("team"), 0)
        kind[a:b + 1] = d.get("type", d.get("code"))
    return {"active": active, "team": team, "start": start, "end": end, "type": kind}


def default_path(game=None, data_dir=None):
    """The provided Smart Tagging file, via games.require (as predict_match finds atd_meta)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import games
    (path,) = games.require(["smart_tagging"], game or games.GAME, data_dir)
    return path


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else default_path()
    pieces, phases = read_set_pieces(path), read_phases(path)
    print("set pieces: {}".format(len(pieces)))
    print("  by type: {}".format(dict(Counter(p["type"] for p in pieces))))
    print("  by team: {}".format(dict(Counter(p["team"] for p in pieces))))
    print("phases with a Team tag: {}".format(len(phases)))
    print("  by team: {}".format(dict(Counter(p["team"] for p in phases))))


if __name__ == "__main__":
    main()
