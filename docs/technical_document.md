# [Team name]: ball possession from tracking alone

*More than a Hack 2027, Phase 1. DRAFT: numbers in [brackets] or marked "to update" change
before submission.*

## 1. Approach

We label every graded frame of the match `H` (home in possession), `A` (away) or `D` (ball dead)
from the provided tracking data only. Two parts, each fitted on other games:

**Who has the ball: the last deliberate touch.** A touch is a sudden change in the ball's velocity
with a player close by. Being near the ball isn't enough, because the tracking is 2D and a high
ball passes "through" the players under it. Possession goes to the team that touched the ball
last, with three refinements:

- **Deflections:** a touch by the other team, followed by the original team touching again
  within 0.5 s, doesn't switch possession.
- **Reappearing ball:** when the ball reappears after a tracking gap, the nearest player counts
  as touching it, because possession often changed while it was unseen.
- **Restarts:** after a stoppage, the team comes from the first touch, not from the team that had
  the ball before it.

**Whether the ball is in play: mostly the players.** In the provided match the ball is missing on
41% of graded frames, often during live play, so "no ball" can't mean "dead ball". A logistic
regression estimates P(dead) from what the players do. It sees:

- how fast they move, and how many walk or sprint;
- whether each team's shape is frozen;
- players gathered for a wall or a corner;
- a player standing at the touchline;
- whether the ball is off the pitch or lying still, but only while the ball is visible. Whether
  it is missing is never an input.

Details are in the dead-ball section.

**Why this and not something else.** A learned end-to-end model needs a lot of labelled tracking,
and we may not train on the provided match. A small number of physically meaningful rules, whose
few thresholds are fitted on other games, transfers better across matches, and every decision can
be explained.

Every number the models use is chosen in `fit()` on the external games, by trying a small grid
and keeping the best score: the touch strength and radius, the measuring window, the dead-ball
threshold and minimum run lengths. None is picked by hand, and none comes from the provided
match.

## 2. Assets used, and dropped

| Asset | Use |
|---|---|
| Tactical tracking (ATD) | The model's only input when predicting the match |
| Tactical tracking, summary statistics | To make our training data look like it (section 3). Only aggregate numbers, never frames or labels |
| Smart Tagging | Not a model input. Once, in aggregate, to estimate how much of the missing ball falls in live play (section 3) |
| Tactical video | Not an input. Watched for the public 5 minutes only, to understand what the ground truth calls possession (section 4) |
| Homography, broadcast video and tracking | Not used |
| Public 5 minutes of ground truth | To understand the definition only, written up as rules. Never to set a number or choose between versions |

**Tried and dropped:**

| Idea | Why it was dropped |
|---|---|
| "No ball = dead ball" baseline | It splits real possessions wherever the ball is lost during play |
| Nearest-player possession | Switches whenever an opponent is close without touching the ball. Our first floor: [0.165] |
| A minimum hold time (2 s) before possession counts | Erases the real sub-second possessions the ground truth contains |
| Hysteresis thresholds for dead runs | No gain |
| Learned shifts of dead-run edges | No gain: timing errors fall both early and late |
| Restart timing from detected kicks or touches | Not reliable on smoothed tracking |

## 3. Training data

**Games.** Metrica Sports' three open sample games (github.com/metrica-sports/sample-data):
tracking at 25 fps plus event data. *[To confirm: licence wording.]* No data from the provided
match is used for training, fitting or tuning.

**Labels.** The games have no possession labels, so we built them from Metrica's events, following
our possession rules (section 4). It is done by a script, `external/metrica_labels.py`, with no
hand labelling.

- Possession goes to the team of each pass, shot, recovery, carry or set piece.
- Challenges and ball losses don't switch possession on their own.
- Blocks and saves switch only if the defending team plays the ball next.
- The ball is dead from a ball out, foul, goal, offside, handball or end of half, until the
  restart pass is received. Goal kicks included: a keeper picking up the ball doesn't end the
  stoppage.

We keep two versions of the labels: one ending each stoppage at the reception of the restart
pass (our reading of the ground truth), and one ending it at the kick, as a check.

**Making them look like our match.** Metrica's tracking is much cleaner than the provided match,
so a model fitted on it learns shortcuts that fail there. We fit and score on noisy copies
(`external/metrica_noise.py`):

- **Missing ball:** the ball is hidden on 20, 35 or 50% of live frames. It is hidden mostly in
  the air (long passes, crosses, keeper kicks) and in random gaps that start more often on slow
  balls, in crowds and near a goal.
- **Players:** tracks are cut into anonymous slots per team.
- **Ball position:** it is smoothed over about 0.8 s.

These settings copy summary statistics we measured on the provided tracking: gap counts and
lengths, where gaps start, and the smoothing. The script that measures them,
`external/measure_ball_gaps.py`, prints aggregate numbers only. The 20–35% range for live play
comes from the same tracking. Its gaps end as a Smart Tagging set piece starts more often than
chance would give (47% against 35%), so a good part of the 41% is dead ball, not live play.

**Validation.** Three rounds, each fitting on two games (20% and 35% copies) and scoring the
third. We report the headline, the spellF1 averaged over 20% and 35% and over the three games,
always with each game's own number. 50% and the kick-time labels are checks. A change is kept
only if the headline rises and no single result drops by more than 0.02. A second noise seed is
held back for milestone checks. [To update: a sealed test on open DFL/Bundesliga data (Bassek et
al.), if run.]

**Results on the external games** (seed 0; to update with the final model):

| Model | Headline (game 1 / 2 / 3) | macroF1 |
|---|---|---|
| Nearest team, "no ball = dead" | 0.131 | |
| Nearest team, never dead | 0.165 | 0.38 |
| Same + dead-ball detector | 0.196 (0.22 / 0.20 / 0.17) | 0.79 |
| Last touch, never dead (v1) | 0.354 (0.35 / 0.34 / 0.37) | |
| Last touch + dead-ball detector (v2) | **0.479 (0.51 / 0.47 / 0.46)** | about 0.83 |

## 4. Possession rules

We worked these out from CHALLENGE.md and from comparing the public 5 minutes of ground truth with
the video. Full reasoning and evidence are in `POSSESSION_RULES.md`.

- **Who has it:** the team whose player last touched the ball **on purpose**: a pass, header,
  clearance, tackle that plays the ball, control, shot or keeper catch. No control is needed: a
  single touch counts, so possessions under a second are real.
- **Loose balls and challenges:** stay with the last team with a deliberate touch until the
  other team's first deliberate touch. Accidental deflections, blocked shots and shielding don't
  switch.
- **Saves:** switch only if the defending team plays the ball next.
- **Dead ball:** from the ball crossing a line or the whistle, until the restart pass is
  received. Goal kicks included: a keeper picking up the ball during a stoppage doesn't end it.
  Advantage is play on.
- **Restarts:** the team that receives the restart has the ball.

## 5. Limitations

- **Restart timing.** About half of our predicted restarts land within 1 s of the true moment
  play resumes, on the external games. The smoothed tracking blurs the restart kick, and the
  players start moving before the pass is received.
- **Short stoppages.** Stoppages of a few seconds, and quick restarts followed straight away by
  another whistle, are often missed or merged.
- **Touches the tracking can't see.** These include headers and touches on a high ball, and
  touches while the ball is missing. Between them, possession is carried forward, so a switch
  while the ball is unseen shows up late.
- **Synthetic noise.** Our training noise matches the provided tracking's summary statistics, not
  every detail. Scores on our own external test are a guide to which version is better, not a
  forecast of the score on the match.
- **Offline only.** Some signals use the whole half (speed scaling) or look a few seconds ahead,
  so this labels a recorded match, not a live one.
- **With more time:** a ball detector on the tactical video for the frames the tracking misses,
  and more labelled matches to fit on.

## 6. Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-starter.txt
python external/metrica_labels.py                # labels for the external games
python external/metrica_labels.py --restart kick
python external/metrica_tracking.py              # their tracking
python external/metrica_noise.py --calibration data/ball_gap_stats.json   # noisy copies
python evaluation/score_external.py --model models.touch_dead:TouchDead   # external scores
python predict_match.py --model models.touch_dead:TouchDead --out evaluation/results/[team].csv
```

`predict_match.py` fits on the three external games, reads the provided tracking (set
`MTAH_DATA_DIR` to the data bundle), writes one label per video frame and runs the official
format check. Fitting is deterministic: two fits give identical settings and identical
predictions.
