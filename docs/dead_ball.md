# Dead-ball detection

*Background for the technical document (an earlier dead-ball model; the submitted v4 builds on it). Code: `models/dead_ball.py`.*

## Approach

The model has to label every frame as `H`, `A` or `D` (ball dead). The obvious shortcut is "no ball
tracked = dead ball", and it fails on the provided match: the ball is missing on 41% of graded
frames, much of that during live play. A model that leans on it would cut real possessions into
pieces. So we decide dead or alive **mainly from what the players do**, which looks the same
whether or not the ball is tracked. The ball is a supporting clue only while it is visible.
Whether it is missing is never an input.

Signals, computed per frame:

| Signal | What it captures |
|---|---|
| Median and 90th-percentile player speed, scaled to the half's normal pace | Everyone slows down after a whistle |
| Share of players walking, share sprinting | Same, more robustly when tracks are noisy |
| Speed of each team's centre plus the change in its spread | The team shapes freeze during a stoppage |
| Most players within 4 m of a single player | A wall, a corner, a crowd at a free kick |
| A slow player on or beyond a touchline | A throw-in being taken or waited for |
| Ball off the pitch (only when the ball is visible) | It went out |
| Ball lying still (only when the ball is visible) | It has been placed for a restart |

Each signal is averaged over windows centred on the frame (2, 8 and 20 s) and over the 6 s just
before and just after it, because play slows after a whistle and speeds up after a restart. A
logistic regression turns them into P(dead). Frames above a threshold are dead, and dead or live
runs that are too short to be real are removed.

**Nothing is set by hand from scores.** The regression weights, the threshold (0.35–0.65) and
the minimum dead and live run lengths (2–12 s) are all learned in `fit()` from the training games.
The threshold and run lengths are chosen to give the best possession score (spellF1) on those
games. `DeadBallDetector` can be tuned for any possession model. Our final model passes its own
predictions in, so the dead-ball settings are chosen for the model that actually uses them.

**Short ball gaps.** When one player is within 2 m of the ball both just before and just after a
gap and stays tracked through it, we put the ball at that player's feet for the gap. The longest
gap filled is learned in `fit()`. Stretches of visible ball in the training games are hidden and
re-filled, and the longest length (1–6 s) where the filled ball is typically within 2 m of the
real one is kept. Longer gaps, and gaps nobody clearly carries the ball through, stay missing.
The filled ball feeds the dead-ball signals only, never the possession model's touch detection.

## Training data

- **Games:** Metrica Sports' three open sample games (github.com/metrica-sports/sample-data), with
  tracking and event data, credited to Metrica Sports as their repository asks.
- **Answer key:** built by our team from Metrica's events, following our possession rules (see
  that section). A dead run starts at the ball going out, a foul, a goal, an offside, a handball
  or the end of a half. It ends when the restart pass is received, goal kicks included. A
  keeper picking up the ball during a stoppage doesn't end it. Nothing comes from the provided
  match.
- **Noisy copies:** the training data is a noisy copy of the Metrica tracking made to look like
  the provided match.
  - The ball is hidden on 20, 35 or 50% of live frames, in long bursts on slow balls and in crowds.
  - Player tracks are broken into anonymous team slots.
  - The ball is smoothed the way the provided tracking is.
  - The noise was calibrated from summary statistics of the provided tracking only: how often the
    ball is missing, gap lengths, and smoothing.
- **Train/test:** three rounds, each training on two games and testing on the third. Fitting uses
  the 20% and 35% copies. Results are reported per game, and the 50% copy is a stress test.

## What we tried and dropped

| Version | Change | Result |
|---|---|---|
| v1 | Player speeds and walking, plus visible ball out or still | Kept |
| v2 | + team-shape freeze, gathering, player at the touchline | Kept; the shape signal is the strongest single input |
| v3 | Hysteresis: a high threshold to start a dead run, a lower one to keep it | Dropped: slightly worse |
| v4 | + short ball-gap filling | Kept, but no measurable effect: few gaps in the noisy copies have a clear carrier |
| — | "Ball missing" as an input | Never used, by design (see Approach) |

On the final noisy copies, adding the detector to the simplest possession rule (nearest team,
never dead) raises the possession score from **0.165 to 0.196** and per-frame macroF1 from
**0.38 to about 0.79**, averaged over the three test games. In the final model, adding the
detector to the last-touch model (together with taking the restart team from the first touch
after a stoppage) raises the possession score from 0.354 to 0.468.

## Limitations

- **Short stoppages are missed.** On the training games about a quarter of stoppages were missed,
  mostly short ones (under about 8 s). Fixing that also created false stoppages in live play,
  which cost more.
- **Boundaries are within about a second.** Predicted dead runs start and end within about 1 s of
  the answer key (median). The restart rule ("dead until the restart pass is received") is
  learned only indirectly, through the 6 s before and after windows.
- **Whole halves only.** The speed scaling uses the whole half, so the method runs on complete
  halves, not live.
- **Tracking assumptions.** It assumes most players are tracked at any moment. Long stretches with
  few tracked players weaken every signal.
- **Synthetic noise.** It was tuned on synthetic noise. The noisy copies match the provided
  tracking's summary statistics but not every detail, so real-match results can differ.

## Reproduction

```bash
.venv/bin/python external/metrica_labels.py
.venv/bin/python external/metrica_tracking.py
.venv/bin/python external/metrica_noise.py --calibration data/ball_gap_stats.json
.venv/bin/python evaluation/score_external.py --model models.dead_ball:DeadBallB
```
