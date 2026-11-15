"""Tests for external/smart_tagging.py on a small synthetic SportsCode XML (no match data).

    python external/test_smart_tagging.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smart_tagging as st  # noqa: E402

XML = """<?xml version="1.0" encoding="utf-8"?>
<file>
  <ALL_INSTANCES>
    <instance>
      <ID>1</ID>
      <start>10.0</start>
      <end>12.02</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>Barcelona</text></label>
      <label><group>Type</group><text>Kick Off (Start)</text></label>
      <label><group>Half</group><text>1st Half</text></label>
    </instance>
    <instance>
      <ID>2</ID>
      <start> 30.5 </start>
      <end>33</end>
      <code>  set pieces </code>
      <label><group>team</group><text> N/A </text></label>
      <label><group>TYPE</group><text>throw-in</text></label>
    </instance>
    <instance>
      <ID>3</ID>
      <start>40</start>
      <end>41</end>
      <code>SET PIECES</code>
      <label><group>Team</group><text>Atletic   Club</text></label>
      <label><group>Type</group><text>Penalty Kick</text></label>
    </instance>
    <instance>
      <ID>4</ID>
      <start>5</start>
      <end>6</end>
      <code>SET PIECES</code>
    </instance>
    <instance>
      <ID>5</ID>
      <start>11</start>
      <end>20</end>
      <code>BUILD UP</code>
      <label><group>Team</group><text>Atletic Club</text></label>
      <label><group>Side</group><text>Left</text></label>
    </instance>
    <instance>
      <ID>6</ID>
      <start>21</start>
      <end>22</end>
      <code>PLAYERS IN THE BOX</code>
      <label><group>Max Players in the box</group><text>7+</text></label>
    </instance>
    <instance>
      <ID>7</ID>
      <end>22</end>
      <code>BUILD UP</code>
      <label><group>Team</group><text>Barcelona</text></label>
    </instance>
    <instance>
      <ID>8</ID>
      <start>50</start>
      <end>51</end>
      <code>GOALS</code>
      <label><group>Team</group><text>N/A</text></label>
      <label><text>orphan text</text></label>
    </instance>
  </ALL_INSTANCES>
</file>
"""


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tags.xml")
        with open(path, "w") as f:
            f.write(XML)
        pieces = st.read_set_pieces(path)
        phases = st.read_phases(path)

    check(len(pieces) == 4, pieces)
    check([p["start_frame"] for p in pieces] == [125, 250, 762, 1000], pieces)
    p = pieces[1]  # kick off at 10.0-12.02 s
    check((p["start_frame"], p["end_frame"], p["team"], p["type"]) == (250, 300, "H",
                                                                       "kickoff_start"), p)
    check(pieces[0]["team"] is None and pieces[0]["type"] is None, pieces[0])   # no labels
    check(pieces[2]["team"] is None and pieces[2]["type"] == "throw_in", pieces[2])  # N/A, case
    check(pieces[2]["start_frame"] == 762, pieces[2])                           # round(30.5*25)
    check(pieces[3]["team"] == "A" and pieces[3]["type"] == "penalty_kick", pieces[3])
    check(pieces[3]["type_raw"] == "penalty kick", pieces[3])

    check([(d["code"], d["team"]) for d in phases] == [("build up", "A"), ("goals", None)], phases)

    arr = st.tag_arrays(pieces, 1010)
    check(arr["active"].sum() == (150 - 125 + 1) + (300 - 250 + 1) + (825 - 762 + 1) + 10,
          arr["active"].sum())  # last piece 1000-1025 clipped to 1000-1009
    check(arr["team"][260] == 1 and arr["team"][800] == 0 and arr["team"][1005] == -1, "team")
    check(arr["start"][250] and arr["end"][300] and not arr["end"][1009], "start/end")
    check(arr["type"][260] == "kickoff_start" and arr["type"][0] is None, "type")
    ph = st.tag_arrays(phases, 2000)
    check(ph["team"][300] == -1 and ph["type"][300] == "build up", "phases")
    check(st.tag_arrays([], 5)["active"].sum() == 0, "empty")
    print("smart_tagging tests passed")


if __name__ == "__main__":
    main()
