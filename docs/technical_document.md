# [Team name]: ball possession from tracking alone

*More than a Hack 2027, Phase 1. DRAFT: numbers in [brackets] or marked "to update" change
before submission.*

## 1. Approach

We label every graded frame of the match `H` (home in possession), `A` (away) or `D` (ball dead)
from the provided tracking data only. Two parts, each fitted on other games:

**Who has the ball: the last deliberate touch.** A touch is a sudden change in the ball's velocity
with a player close by. Being near the ball isn't enough, because the tracking is 2D and a high
ball passes "through" the players under it. Possession goes to the team that touched the ball
last, with these refinements:

- **Deflections:** a touch by the other team, followed by the original team touching again
  within 0.5 s, doesn't switch possession.
- **Reappearing ball:** when the ball reappears after a tracking gap, the nearest player counts
  as touching it, because possession often changed while it was unseen.
- **Long ball gaps:** when the ball reappears near the other team after a long gap, the switch is
  moved back to when that team's centre started pushing toward the other goal faster than the
  old team's.
- **Restarts:** after a stoppage, the team comes from the first touch, not from the team that had
  the ball before it. The end of each stoppage is moved to the restart kick when one is detected
  close by.

**Where the ball is unseen: the team shapes.** A second model reads who has the ball from the
players alone: the team in possession spreads out and pushes forward, the other narrows and drops
back. It treats both teams the same, so it can't learn "the home team usually has it". On frames
where the ball isn't tracked, it overrules the touches when it is sure enough for long enough.
Where the ball is tracked, the touches always decide.

**Whether the ball is in play: mostly the players.** In the provided match the ball is missing on
41% of graded frames, often during live play, so "no ball" can't mean "dead ball". A logistic
regression estimates P(dead) from what the players do. It sees:

- how fast they move, and how many walk or sprint;
- whether each team's shape is frozen;
- players gathered for a wall or a corner;
- a player standing at the touchline;
- whether the ball is off the pitch or lying still (for 2 s or more, as when it is placed for a
  set piece), but only while the ball is visible. Whether it is missing is never an input.

Details are in the dead-ball section.

**Why this and not something else.** A learned end-to-end model needs a lot of labelled tracking,
and we may not train on the provided match. A small number of physically meaningful rules, whose
few thresholds are fitted on other games, transfers better across matches, and every decision can
be explained.

Every number the models use is chosen in `fit()` on the external games, by trying a small grid
and keeping the best score: the touch strength and radius, the measuring window, the long-gap and
restart settings, the dead-ball threshold and minimum run lengths, and how sure the team-shape
model must be. None is picked by hand, and none comes from the provided match.

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
| A 0.8 s deflection window | Erases the sub-second possessions the ground truth counts; 0.5 s kept |
| Touch goes to the player the ball moves away from | +0.006 on noisy copies, but clean tracking fell by up to 0.16: it only fitted our simulated blur |
| Touch threshold relative to the match's own touch strengths | No change, including on a blurrier ball |
| Team shapes overruling the touches everywhere | 0.527, but clean results fell by 0.056–0.077: it overruled correct touches |
| Team shapes overruling only "doubtful" touches (crowd, just reappeared, weak) | 0.508, but clean fell by up to 0.086: doubtful touches are as common on clean tracking |
| Team shapes overruling seen touches under a harm guard | 0.488, below the model kept |
| Shorter minimum live run (1 s) | 0.486, below the model kept |
| Longer deflection window (0.75 s, or up to 1.5 s) | A tie at 0.75 s; up to 1.5 s, clean fell by up to 0.069 |
| Switches confirmed by the ball's movement | 0.488, below the model kept |
| A learned classifier adding weak touches that switch possession | Not kept: +0.004 on seed 0, −0.006 on seed 1 |

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
- **Ball position:** it is smoothed over 0.84 s with a slow 0.1 m wobble. A ball shown in a
  stoppage is carried to the restart spot and often lies still there before the restart.

These settings copy summary statistics we measured on the provided tracking: gap counts and
lengths, where gaps start, and the smoothing. The script that measures them,
`external/measure_ball_gaps.py`, prints aggregate numbers only. The 20–35% range for live play
comes from the same tracking. Its gaps end as a Smart Tagging set piece starts more often than
chance would give (47% against 35%), so a good part of the 41% is dead ball, not live play.

**Validation.** Three rounds, each fitting on two games (20% and 35% copies) and scoring the
third. We report the headline, the spellF1 averaged over 20% and 35% and over the three games,
always with each game's own number. 50% and the kick-time labels are checks. A change is kept
only if the headline rises, no single result drops by more than 0.02, and no result on the clean
(noise-free) tracking drops by more than 0.05. That last guard stops changes that only fit our
simulated noise. A second noise seed is held back for milestone checks.

**Results on the external games** (seed 0, current noisy copies):

| Model | Headline (game 1 / 2 / 3) | macroF1 |
|---|---|---|
| Nearest team, "no ball = dead" | 0.122 (0.13 / 0.12 / 0.11) | 0.57 |
| Nearest team, never dead | 0.150 (0.15 / 0.16 / 0.14) | 0.38 |
| Last touch, never dead (v1) | 0.350 (0.34 / 0.35 / 0.36) | 0.42 |
| Last touch + dead-ball detector and refinements (v3) | 0.479 (0.50 / 0.48 / 0.46) | 0.84 |
| v3 + team shapes where the ball is unseen (v3.1) | **0.491 (0.51 / 0.48 / 0.48)**; seed 1: 0.488 | 0.85 |
| v4 | [In progress] | |

Each version was kept only after passing the rule above. Differences of a few thousandths are
within noise: the scores move by about ±0.04 between bootstrap samples of a game.

**Sealed check on other data.** As a test the models never saw, we scored them once on open
DFL/Bundesliga games (Bassek et al.): v3 0.372, v3.1 0.390, and a two-tier variant 0.393. The
order matches the Metrica results, so the gains carry over to another league and provider. The
team-shape model picks the right team on 84% of in-play frames there (89% on Metrica).

**Stress tests: extreme checks only.** We also scored each test game on copies with one noise
setting deliberately wrong (`evaluation/stress_test.py`, `docs/stress_test_v2.md`). They are
never used to choose anything. The player-noise ones are far beyond what the provided tracking
shows: its player jitter is about 0.001 m, like Metrica's.

| Variant (35% missing, model v3 before restart snapping) | Change |
|---|---|
| Player position wobble of 0.1 m per frame | −0.007 |
| Player tracks broken twice as often / half as often | −0.010 / +0.001 |

An earlier version measured player speed frame to frame and lost 0.071 under the wobble. Measuring
it over 0.4 s cut that to 0.006, and we kept it as insurance. On the ball side, a blurrier ball
(×1.5) costs 0.06 and missing 65% of the ball costs 0.11. Both degrade smoothly, with no collapse.

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

- **Restart timing.** This is still our largest source of error on the external games: many
  predicted restarts land more than 1 s from the true moment play resumes. The smoothed tracking
  blurs the restart kick, and the players start moving before the pass is received. Moving each
  restart to a detected kick helps, but only where the kick is visible.
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
python evaluation/score_external.py --model models.touch_dead:TouchDeadShapeUnseen   # external scores (v3.1)
python predict_match.py --model models.touch_dead:TouchDeadShapeUnseen --out evaluation/results/[team].csv
```

`predict_match.py` fits on the three external games, reads the provided tracking (set
`MTAH_DATA_DIR` to the data bundle), writes one label per video frame and runs the official
format check. Fitting is deterministic: two fits give identical settings and identical
predictions.
