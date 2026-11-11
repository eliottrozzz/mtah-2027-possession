# Agent briefs

Paste-ready briefs for Claude Code agents working on side tracks. Each agent works in its own git
worktree and branch, on external data only. Its results come back through the scoring log and a
pull request into `jordan/scoring`.

## Setting up (about 5 minutes, once)

From a terminal:

```bash
cd "/Users/Bucha/Side Projects/mtah-2027-possession"
git checkout jordan/scoring && git pull

# One worktree (its own folder and branch) per track
git worktree add ../mtah-trackB -b jordan/dead-ball jordan/scoring
git worktree add ../mtah-trackC -b jordan/dfl jordan/scoring

# Share the external data only. The provided match never goes into an agent's folder.
for t in B C; do
  ln -s "/Users/Bucha/Side Projects/mtah-2027-possession/data/external" ../mtah-track$t/data/external
  ln -s "/Users/Bucha/Side Projects/mtah-2027-possession/.venv" ../mtah-track$t/.venv
  mkdir -p ../mtah-track$t/.claude
  cat > ../mtah-track$t/.claude/settings.local.json <<'EOF'
{ "permissions": { "deny": [
  "Read(//Users/Bucha/Library/CloudStorage/**)",
  "Bash(*CloudStorage*)",
  "Bash(*MTAH_DATA_DIR*)"
] } }
EOF
done
# The ball-gap summary (aggregate numbers only, safe to share) so Track C can calibrate its noise
cp data/ball_gap_stats.json ../mtah-trackC/data/
```

Then open **one terminal per track** and start Claude **without** the match data path:

```bash
cd "/Users/Bucha/Side Projects/mtah-trackB" && env -u MTAH_DATA_DIR claude
cd "/Users/Bucha/Side Projects/mtah-trackC" && env -u MTAH_DATA_DIR claude
```

Paste the brief below into each one. Look in on them every hour or two; they'll ask before pushing.

---

## Track B: dead-ball detection and ball-gap filling

```text
You're working on our entry for "More than a Hack 2027, Phase 1: ball possession". Read
TEAM_SETUP.md, POSSESSION_RULES.md, external/README.md, evaluation/score_external.py and
models/baselines.py first.

HARD RULES (from the organisers; breaking them disqualifies us):
- Use ONLY the external Metrica data in data/external/. Never read, search for or ask about the
  provided match (anything under ~/Library/CloudStorage, MTAH_DATA_DIR, the starter/ loaders or
  evaluation/gt/). Never use the ground truth or the video.
- Never use ball_reason as an input; it exists for analysis only.

GOAL: tell dead ball from live play mainly from the PLAYERS, so the model doesn't learn
"no ball = dead ball". In our real match the ball is missing on 41% of frames, much of it during
live play, so the ball can only be a supporting clue. Ideas: everyone slowing down or walking, the
team shape freezing, players gathering for a wall or a corner, someone standing at the touchline,
the ball still for a long time when it is visible.
Second, smaller part: fill SHORT ball gaps (a few seconds) when one player is clearly carrying
the ball. Leave long gaps missing.

DELIVERABLE, by the end of Nov 12:
1. models/dead_ball.py, with a model class that is baseline B (models.baselines:NeverDead) plus
   your dead-ball detector, using the same interface (name, settings, fit, predict). Any thresholds
   are learned in fit() from the training games it is given; never pick them by looking at test
   scores.
2. Score it with: .venv/bin/python evaluation/score_external.py --model models.dead_ball:<Class> --who agentB
   The floor to beat is 0.188 (baseline_never_dead). A version is only an improvement if the
   script prints KEEP: the headline goes up and no single result drops more than 0.02, including
   live50 and the kick key. macroF1 should also rise clearly (baseline B is 0.39, because it never
   predicts D).
3. Use seed 0 only. Never run --seed 1.
4. Commit your code and each logged run on branch jordan/dead-ball. Before pushing, show me a
   short summary: the best headline, by game, and what the detector looks at.

Keep it simple and readable. If something isn't working by mid-afternoon Nov 12, stop and
report what you have.
```

---

## Track C: DFL games as a sealed final test

```text
You're working on our entry for "More than a Hack 2027, Phase 1: ball possession". Read
TEAM_SETUP.md, POSSESSION_RULES.md, external/README.md, external/metrica_labels.py,
external/metrica_tracking.py, external/metrica_noise.py and evaluation/score_external.py first.

HARD RULES (from the organisers; breaking them disqualifies us):
- Use ONLY external data. Never read, search for or ask about the provided match (anything under
  ~/Library/CloudStorage, MTAH_DATA_DIR, the starter/ loaders or evaluation/gt/). Never use its
  ground truth or video.
- Check the licence of any dataset before using it, and record it.

GOAL: add the DFL / Bundesliga open dataset (Bassek et al., "An integrated dataset of
spatiotemporal and event data in elite soccer", on figshare, about 7 matches of official tracking
at 25 fps plus official events) as a SEALED TEST SET. Nobody tunes on it. It's only run on the 2–3
final candidates around Nov 14–15.

STEPS:
1. Find the dataset, confirm the licence allows our use (we believe it is CC BY 4.0), and download
   it to data/external/dfl/ (gitignored). Write the licence and citation into external/README.md.
2. external/dfl_tracking.py: convert each match into the SAME dict format as
   metrica_tracking.load_game (ball, players, team 'H'/'A', player, period; 0–1 pitch units; one
   row per frame), cached as npz. The kloppy library (sportec provider) may help. Check it the way
   metrica_tracking.py does (ball coverage; the passer is near the ball at pass start).
3. external/dfl_labels.py: an answer key per match, following POSSESSION_RULES.md and copying
   metrica_labels.py's logic: possession = the last deliberate touch; deflections, blocked shots
   and saves don't switch it; D from the stoppage until the restart pass is RECEIVED (keeper
   restarts alive at the roll or kick); write both --restart reception and kick versions. DFL logs
   restarts explicitly, which helps. If the DFL tracking has per-frame ball status or possession
   columns, also save them as a separate "dfl_provider" key, for comparison only.
   Print the same summary stats as metrica_labels.py (possessions, dead share, under-1 s count,
   home share) and flag anything far from Metrica's (dead about 33–43%).
4. Make noisy copies (live20/35/50, seed 0) with metrica_noise.py's logic and
   --calibration data/ball_gap_stats.json (summary numbers only; already in your folder). Adapt
   it minimally for DFL.
5. Add a --sealed option to evaluation/score_external.py that scores a model on the DFL games
   (test only, no fitting on them), logs it as a milestone run, and is clearly separate from the
   Metrica headline. Test it ONLY with the oracle check (the answer key itself should score 1.0)
   and NOT with any real model. Nobody looks at real models' DFL scores before the milestone.
6. Commit on branch jordan/dfl. Before pushing, show me a summary.

Deadline: the end of Nov 13. If the data or licence turns out to be a problem, stop early and
tell me; this track is optional.
```

---

## What to do with the results

When a track is done, say "review jordan/dead-ball" (or jordan/dfl) in the main session. It
re-runs the scores, checks the keep rule and merges into `jordan/scoring`.
