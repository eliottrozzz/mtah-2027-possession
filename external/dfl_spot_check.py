"""Write a page for eyeballing the DFL event sync: 20 random touches as short tracking clips.

    python external/dfl_spot_check.py          # writes data/external/dfl/spot_check.html

The touches are a fixed-seed random sample from the 5 held-out matches (not dfl_sync.DEV), drawn
from every logged touch, snapped or not. The page is blind: each clip starts at a random point
before the touch, the judge steps to the frame where the highlighted player touches the ball (or
says there is no touch), and only then sees where the snap put it. Open the file in a browser; it
needs no server and no internet.

What it can and can't show: it reads the same tracking the snap reads, so it catches gross errors
(the snap on the wrong touch, on the ball passing by, or a touch left unplaced) rather than fine
timing; dfl_video_check.md is the check for timing. It only measures: the sync was frozen
(commit b79114c) before this page existed, and nothing gets tuned on the result.

The page holds tracking data, so it stays under data/ (gitignored) and isn't published.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl_sync as ds  # noqa: E402
import dfl_tracking as dt  # noqa: E402

# Params
N_CLIPS = 20
SEED = 2027
CLIP_FRAMES = 125           # 5 s per clip
LEAD_FRAMES = (25, 100)     # the touch falls 1-4 s into the clip, at random
OUT = os.path.join(dt.OUT_DIR, "spot_check.html")
ACTION = {"Play": "pass", "ShotAtGoal": "shot", "OtherBallAction": "touch (e.g. clearance)",
          "BallClaiming": "claim / interception"}


def people(match):
    root = ET.parse(dt.fetch(match, "matchinformation")).getroot()
    return {p.get("PersonId"): (p.get("Shortname") or p.get("LastName"), p.get("ShirtNumber"))
            for p in root.iter("Player")}


def action(event):
    if event["tag"] in ("ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "Penalty"):
        return event["tag"]
    for tag in event["tags"]:
        if tag in ACTION:
            return "cross" if tag == "Play" and "Cross" in event["tags"] else ACTION[tag]
    return event["tag"]


def r(value):
    return None if np.isnan(value) else round(float(value), 2)


def clips():
    rng = np.random.default_rng(SEED)
    pool = []
    for match in dt.MATCHES:
        if match in ds.DEV:
            continue
        game, extras = dt.load_game(match), dt.load_extras(match)
        events = dt.read_events(match)
        ds.snap(game, extras, events)
        column = {pid: i for i, pid in enumerate(game["player"])}
        pool += [(match, e) for e in events if ds.touch_player(e) in column]
    picks = sorted(rng.choice(len(pool), N_CLIPS, replace=False))
    print("sampled {} of {} touches in the held-out matches".format(N_CLIPS, len(pool)))

    out, cache = [], {}
    for k, index in enumerate(rng.permutation(picks)):
        match, event = pool[index]
        if match not in cache:
            cache[match] = (dt.load_game(match), people(match))
        game, names = cache[match]
        n = len(game["period"])
        start = int(np.clip(event["row"] - rng.integers(*LEAD_FRAMES), 0, n - CLIP_FRAMES))
        rows = np.arange(start, start + CLIP_FRAMES)
        rows = rows[game["period"][rows] == game["period"][event["row"]]]
        ball = game["ball"][rows] * np.array(ds.PITCH)
        players = game["players"][rows] * np.array(ds.PITCH)
        pid = ds.touch_player(event)
        name, shirt = names.get(pid, ("?", ""))
        out.append({
            "id": k + 1, "match": match, "action": action(event),
            "player": "{} ({})".format(name, shirt) if shirt else name,
            "col": int(list(game["player"]).index(pid)),
            "team": [str(t) for t in game["team"]],
            "ball": [[r(x), r(y)] for x, y in ball],
            "players": [[[r(x), r(y)] for x, y in frame] for frame in players],
            "snap": int(event["row"] - rows[0]), "logged": int(event["lag_row"] - rows[0]),
            "snapped": bool(event["snapped"]),
        })
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL Sync Spot Check</title>
<style>
:root{--bg:#f6f6f3;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#d9d9d3;--pitch:#3f7d4e;--pline:#e9f2ea;--home:#2f6fd6;--away:#e07a1f;--ball:#111;--hi:#ffd400;--ok:#1f8a4c;--bad:#c23b2b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecece8;--muted:#9a9a93;--card:#21211f;--line:#3a3a36;--pitch:#2c5a38;--pline:#cfe0d2;--ball:#fff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}p{margin:6px 0}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
svg{width:100%;height:auto;display:block;border-radius:6px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg)}
input[type=range]{flex:1;min-width:160px}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;border:1px solid var(--line);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
textarea{width:100%;height:140px;font:12px ui-monospace,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.ok{color:var(--ok)}.bad{color:var(--bad)}
</style></head><body><main>
<h1>DFL sync spot check</h1>
<p class="muted">20 random touches from the 5 held-out matches. For each clip, step to the frame where the
<b>highlighted player</b> touches the ball and press <b>Mark this frame</b>, or <b>No touch</b> if you can't see one.
Where the snap put the touch stays hidden until every clip is judged. Keys: ← → step one frame, space play/pause, M mark.</p>
<p class="muted">The ball is black (white in dark mode) with a short trail. Blue is home, orange is away, the yellow ring is the logged player.
This reads the same tracking the snap reads, so it catches the snap landing on the wrong moment, not fine timing.</p>
<div class="card" id="clip"></div>
<div class="card" id="results"></div>
<script>
const CLIPS = __DATA__;
const KEY = "dfl-spot-check-v1";
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} };
let cur = CLIPS.findIndex(c => !(c.id in marks)); if (cur < 0) cur = 0;
let frame = 0, timer = null;
const NS = "http://www.w3.org/2000/svg";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v);

function pitch() {
  const l = css("--pline");
  return `<rect x="-3" y="-3" width="111" height="74" fill="${css("--pitch")}"/>
  <g fill="none" stroke="${l}" stroke-width="0.25"><rect x="0" y="0" width="105" height="68"/>
  <line x1="52.5" y1="0" x2="52.5" y2="68"/><circle cx="52.5" cy="34" r="9.15"/>
  <rect x="0" y="13.84" width="16.5" height="40.32"/><rect x="88.5" y="13.84" width="16.5" height="40.32"/>
  <rect x="0" y="24.84" width="5.5" height="18.32"/><rect x="99.5" y="24.84" width="5.5" height="18.32"/></g>`;
}

function draw() {
  const c = CLIPS[cur], n = c.ball.length;
  frame = Math.max(0, Math.min(n - 1, frame));
  let g = "";
  c.players[frame].forEach((p, i) => {
    if (p[0] === null) return;
    const col = c.team[i] === "H" ? css("--home") : css("--away");
    if (i === c.col) g += `<circle cx="${p[0]}" cy="${p[1]}" r="2.2" fill="none" stroke="${css("--hi")}" stroke-width="0.6"/>`;
    g += `<circle cx="${p[0]}" cy="${p[1]}" r="1.1" fill="${col}"/>`;
  });
  for (let k = Math.max(0, frame - 10); k < frame; k++) {
    const b = c.ball[k]; if (b[0] === null) continue;
    g += `<circle cx="${b[0]}" cy="${b[1]}" r="0.35" fill="${css("--ball")}" opacity="${0.15 + 0.05 * (k - frame + 10)}"/>`;
  }
  const b = c.ball[frame];
  if (b[0] !== null) g += `<circle cx="${b[0]}" cy="${b[1]}" r="0.7" fill="${css("--ball")}" stroke="${css("--pitch")}" stroke-width="0.2"/>`;
  document.getElementById("svg").innerHTML = pitch() + g;
  document.getElementById("slider").value = frame;
  document.getElementById("fr").textContent = `frame ${frame} / ${n - 1}  (${(frame / 25).toFixed(2)} s)`;
}

function render() {
  const c = CLIPS[cur], m = marks[c.id];
  const done = Object.keys(marks).length;
  document.getElementById("clip").innerHTML = `
    <div class="row" style="margin-top:0"><b>Clip ${cur + 1} of ${CLIPS.length}</b>
      <span class="tag">${c.action}</span><span class="tag">${c.player}</span>
      <span class="muted">${done} judged</span></div>
    <svg id="svg" viewBox="-3 -3 111 74"></svg>
    <div class="row"><button id="play">Play</button><button id="back">◀ frame</button><button id="fwd">frame ▶</button>
      <input type="range" id="slider" min="0" max="${c.ball.length - 1}" value="0"><span id="fr" class="muted"></span></div>
    <div class="row"><button class="primary" id="mark">Mark this frame</button><button id="none">No touch</button>
      <span class="muted">${m === undefined ? "not judged yet" : (m === null ? "your answer: no touch" : "your mark: frame " + m)}</span>
      <span style="flex:1"></span><button id="prev">← Previous clip</button><button id="next">Next clip →</button></div>`;
  const $ = id => document.getElementById(id);
  $("play").onclick = toggle; $("back").onclick = () => step(-1); $("fwd").onclick = () => step(1);
  $("slider").oninput = e => { frame = +e.target.value; draw(); };
  $("mark").onclick = () => judge(frame); $("none").onclick = () => judge(null);
  $("prev").onclick = () => go(cur - 1); $("next").onclick = () => go(cur + 1);
  draw(); results();
}

function toggle() {
  if (timer) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "Play"; return; }
  document.getElementById("play").textContent = "Pause";
  timer = setInterval(() => { if (frame >= CLIPS[cur].ball.length - 1) return toggle(); frame++; draw(); }, 40);
}
function step(d) { if (timer) toggle(); frame += d; draw(); }
function go(i) { if (timer) toggle(); cur = (i + CLIPS.length) % CLIPS.length; frame = 0; render(); }
function judge(v) { marks[CLIPS[cur].id] = v; save(); go(CLIPS.findIndex(c => !(c.id in marks)) >= 0 ? CLIPS.findIndex(c => !(c.id in marks)) : cur); }

function results() {
  const el = document.getElementById("results");
  const done = CLIPS.filter(c => c.id in marks).length;
  if (done < CLIPS.length) { el.innerHTML = `<b>Results</b><p class="muted">Shown once all ${CLIPS.length} clips are judged (${done} so far).</p>`; return; }
  let rows = "", csv = "clip,match,action,snapped,your_frame,snap_frame,logged_frame,snap_minus_you\n";
  let near = 0, close = 0, judged = 0;
  CLIPS.forEach(c => {
    const m = marks[c.id];
    const d = m === null ? null : c.snap - m;
    if (d !== null) { judged++; if (Math.abs(d) <= 3) near++; if (Math.abs(d) <= 12) close++; }
    const cls = d === null ? "muted" : (Math.abs(d) <= 3 ? "ok" : (Math.abs(d) > 12 ? "bad" : ""));
    rows += `<tr><td>${c.id}</td><td>${c.action}</td><td>${c.snapped ? "yes" : "no (lag only)"}</td>
      <td>${m === null ? "no touch" : m}</td><td>${c.snap}</td><td class="${cls}">${d === null ? "–" : (d > 0 ? "+" : "") + d}</td></tr>`;
    csv += [c.id, c.match, c.action, c.snapped, m === null ? "none" : m, c.snap, c.logged, d === null ? "" : d].join(",") + "\n";
  });
  el.innerHTML = `<b>Results</b>
    <p>Snap within 3 frames (0.12 s) of your mark: <b>${near} of ${judged}</b>; within 12 frames (0.5 s): <b>${close} of ${judged}</b>.
    ${CLIPS.length - judged} marked "no touch".</p>
    <table><tr><th>Clip</th><th>Action</th><th>Snapped</th><th>Your frame</th><th>Snap frame</th><th>Snap − you</th></tr>${rows}</table>
    <p class="muted">To share, copy this into the chat. It holds frame numbers only, no tracking data.</p>
    <textarea readonly>${csv}</textarea>
    <div class="row"><button id="reset">Clear my marks and start again</button></div>`;
  document.getElementById("reset").onclick = () => { if (confirm("Clear all marks?")) { marks = {}; save(); go(0); } };
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  else if (e.key === " ") { e.preventDefault(); toggle(); }
  else if (e.key === "m" || e.key === "M") judge(frame);
});
render();
</script></main></body></html>
"""


def main():
    data = clips()
    with open(OUT, "w") as handle:
        handle.write(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("written", os.path.relpath(OUT, dt.ROOT), "({:.0f} kB)".format(os.path.getsize(OUT) / 1e3))


if __name__ == "__main__":
    main()
