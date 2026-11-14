# External data: Metrica's three open games

We may only train and tune on other games, so this folder turns Metrica's open sample games (github.com/metrica-sports/sample-data) into what we need: an answer key and tracking, both indexed by the same frame numbers. Everything is downloaded into `data/external/metrica/`, which is gitignored.

```bash
.venv/bin/python external/metrica_labels.py      # answer key: game{1,2,3}_labels_reception.csv
.venv/bin/python external/metrica_tracking.py    # tracking: game{1,2,3}_tracking.npz, plus checks
```

## For Person 2: loading the tracking

```python
import sys; sys.path.insert(0, "external")
from metrica_tracking import load_game

game = load_game(1)    # about 15 s the first time (download and convert), then instant
```

`game` is a dict of numpy arrays. Row *i* is Metrica frame *i*, the same row as the answer key CSV:

| key | shape | contents |
|---|---|---|
| `ball` | (frames, 2) | ball x, y in 0–1 pitch units; NaN where it isn't tracked |
| `players` | (frames, players, 2) | each player's x, y in 0–1; NaN when they're not on the pitch |
| `team` | (players,) | `'H'` or `'A'` |
| `player` | (players,) | Metrica's player id within that game |
| `period` | (frames,) | 1 or 2; 0 where there's no tracking |

* Multiply x by 105 and y by 68 to get metres (`metrica_tracking.PITCH`).
* 25 frames per second. Row 0 is always empty, because Metrica frames start at 1.
* Unlike our match, players keep the same column all game; in our match, tracks drop out and swap players.

## Read before building features: the ball is missing for different reasons

In the Metrica games the ball is tracked on 98–99% of live frames and is mostly missing when the ball is dead. In our match the ball is missing on 41% of graded frames, including a lot of live play.

So **"ball missing" means "dead ball" in Metrica, but not in our match.** A model trained on clean Metrica tracking would learn that shortcut and then call live play dead on our match, splitting real possessions.

We avoid it in three ways, together:

1. **Noise (Person 1).** In the Metrica copy, delete the ball during live play too, in bursts, so it's missing about as often as in our match. All training, tuning and scoring uses the noisy copy, which also shows whether a model still falls for the shortcut.
2. **Dead-ball detection from the players (Person 2).** Decide "dead or alive" mainly from the players, not the ball: everyone slowing down, players walking, a wall forming, someone waiting at the touchline. That behaves the same whether or not the ball is tracked, so it carries over to our match. The ball is a supporting clue only.
3. **Ball-gap filling (Person 2).** Estimate where the ball is during short gaps, from its last position and who's nearby, so fewer frames have no ball at all. Long gaps stay missing rather than guessed.

## Noisy copies: what everything trains and scores on

```bash
.venv/bin/python external/measure_ball_gaps.py                                   # on the provided match
.venv/bin/python external/metrica_noise.py --calibration data/ball_gap_stats.json
```

`measure_ball_gaps.py` reads the provided tracking (not the ground truth, not the video) and prints only summary numbers: how often and for how long the ball goes missing, and how much more often on fast balls and in crowds. Run it locally; its printout is safe to share. Without it, `metrica_noise.py` uses defaults.

`metrica_noise.py` writes `data/external/metrica/noisy/game{g}_live{level}_seed{s}.npz`, the same format as above. Every setting below is calibrated from aggregate statistics of the provided tracking only (`data/ball_gap_stats.json`), never from its frames, labels or ground truth:

* **Ball blur:** the ball is smoothed over 21 frames (0.84 s) and given a slow 0.1 m wobble, so a kick looks as blurred as in the provided tracking (its touch-signal percentiles).
* **Missing ball:** most of long passes, crosses, keeper kicks, throws and shots, plus random gaps with the measured length distribution. Gaps start more often on slow balls, in crowds and near a goal or touchline.
* **Stoppage ball:** in half the stoppages the ball is shown for part of it. It is carried to the restart spot at 4–6 m/s; then, in 70% of them, it lies still at the spot before the restart (median 4.5 s). This matches the real tracking's touch signal and its still-ball summary: still for 2 s or more on 5.2% of visible frames.
* **Players:** tracks are broken into anonymous slots per team, and positions get a tiny jitter like the real tracking's (about 0.001 m).

* `live20/35/50`: the share of live frames with no ball. We can't measure the real split between live and dead play without the ground truth, but the provided tracking points to between 20% and 35%: its 41% missing comes in long gaps, and more of them end as a Smart Tagging set piece starts than chance would give (47% against 35%). So **tune on `live20` and `live35`, and use `live50` as a stress test.** Don't trust a model that only works at one level.
* What the provided tracking showed (`measure_ball_gaps.py`): 150 gaps in 106 minutes, median 10.7 s; gaps start most often on a slow or still ball (×1.7), in crowds (×2.6 with 3+ players near), near a goal (×1.8) or a touchline (×1.5), and almost never on a fast ball. The noise copies that pattern.
* `seed0` is for tuning, `seed1` for testing.
* `ball_reason` says why the ball is missing on each frame (0 seen, 1 missing in the clean data, 2 in the air, 3 random gap, 4 ball shown in a stoppage). Use it for analysis only, never as a model input: our match has no such column.

## Also in the tracking

* **2D ball:** a high ball passes straight through the positions of the players it flies over. Being near the ball is not a touch. A touch shows up as a sudden change in the ball's speed or direction.
* **Clean tracking:** these games are much cleaner than ours, so scores on them are too optimistic until the noise is added.

# Sealed test set: the DFL / Bundesliga open dataset

Seven official matches (2 Bundesliga, 5 2. Bundesliga, 2022/23), with the league's own in-stadium tracking at 25 fps and official event data. It was a sealed test for the final candidates, with two looks: look 1 (v3 0.372, v3.1 0.390, two-tier 0.393; 2026-11-13) and look 2 (v4 0.406; 2026-11-14). **Both looks are spent**, so `--sealed` now refuses real models; only `--oracle` still runs.

## DFL split: 4 TRAIN, 3 TEST (2026-11-14)

After both looks, Jordan split the 7 matches so later v4 work can train on DFL. The split is **fixed by a rule, not by scores**, and written into `evaluation/score_external.py` (`DFL_TRAIN`, `DFL_TEST`) before anyone trained on DFL:

* **TEST = the 3 highest match IDs among the 5 sync-holdout matches:** J03WPY, J03WQQ, J03WR9. The 2 sync development matches (`dfl_sync.DEV`: J03WMX, J03WOH) are never test.
* **TRAIN = the other 4:** J03WMX, J03WN1, J03WOH, J03WOY. Free to fit and tune on.

| | test-3 headline | train-4 | all 7 |
|---|---|---|---|
| v3 | 0.387 | 0.361 | 0.372 |
| v3.1 | 0.407 | 0.378 | 0.390 |
| two-tier | 0.409 | 0.382 | 0.393 |
| v4 | 0.415 | 0.399 | 0.406 |

These baselines come from the look-1 and look-2 logs (fit on Metrica only), not from new runs. With 3 matches, differences under about 0.01 are noise.

```python
import score_external as se                 # evaluation/ on sys.path
train = se.metrica_train_set() + se.dfl_train_set()   # [(tracking, reception key)], live20/35, seed 0
```

```bash
.venv/bin/python evaluation/score_external.py --dfl-test --oracle                  # must print PASS
.venv/bin/python evaluation/score_external.py --dfl-test --model <candidate>       # Metrica-only fit
.venv/bin/python evaluation/score_external.py --dfl-test --model <candidate> --fit-dfl-train
```

**One test-3 run per candidate, only after it passes both Metrica seeds.** The script refuses a model (matched by name or `module:Class`) with no logged seed-1 run, or one that already has a test-3 run (`--again` overrides; give the reason in `--note`). Runs are logged with seed `dfl-test` and listed by `--summary`. Nothing is tuned in response to a test score.

**Licence: CC BY 4.0**, confirmed on the Figshare record (checked 2026-11-11). We use it unchanged for research and keep it out of git (`data/` is gitignored); only code and scores are published.

**Citation:** Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data*, 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Data: Bassek, Rein, Weber, Memmert (2025), figshare, https://doi.org/10.6084/m9.figshare.28196177.v1. Data owner: Deutsche Fußball Liga (DFL).

```bash
.venv/bin/python external/dfl_tracking.py      # download (MD5-checked), convert, check: {match}_tracking.npz
.venv/bin/python external/dfl_sync.py          # event sync report (development matches)
.venv/bin/python external/dfl_labels.py        # answer key; add --restart kick for the kick key
.venv/bin/python external/dfl_noise.py --calibration data/ball_gap_stats.json   # live20/35/50, seed 0
.venv/bin/python evaluation/score_external.py --sealed --oracle                 # must print PASS
```

Everything lives in `data/external/dfl/`. Tracking rows are not DFL frame numbers: row 0 is empty, then the first half, one empty row, then the second half (DFL numbers them from 10000 and 100000).

## What differs from Metrica, and how we handled it

* **Event clock.** DFL's events come from a live manual log: they lag the tracking by −1.5 to +1.5 s per half and scatter by about ±0.7 s each. `dfl_sync.py` snaps every touch to the largest ball-velocity change next to the named player, in logged order. The method was fixed on 2 development matches and frozen (commit b79114c) before the 5 holdout matches were run once. Main test, which the snap never reads: is the first player to reach a completed pass its logged recipient? Development 82%, holdout 83.5% (80–87% per match), Metrica 83–84%; 3–7% of touches can't be placed and keep a lag-corrected time (about ±1 s).
* **No ball-out event.** D starts where the tracked ball reaches a line. J03WQQ's tracking often stops the ball short of the line; there, and wherever no crossing is found, D starts at DFL's decision time minus the typical gap for that restart type (throw-in 0.5 s, goal kick 2.0 s, corner 2.2 s).
* **Fouls and offsides** start D at the lag-corrected event (±0.7 s), or after the last logged touch when advantage was played. **Drop balls**: 1 s after the last touch (a guess; nothing marks the stoppage).
* **Stoppage starts then follow DFL's own ball status** (decided 2026-11-12), unless logged touches contradict it: a touch after DFL's stop keeps the ball live until 0.7 s after it, and DFL stopping later than us counts only with a touch in between. This fixes fouls logged late, video-referee reviews and halves ending before a restart. Restarts keep our reception rule: DFL marks them alive at the kick.
* **Ball in stoppages.** DFL tracks it; Metrica's clean data mostly doesn't. The noisy copies delete it on dead frames first, then apply exactly the Metrica noise (same calibration).
* **Unlogged touches:** DFL doesn't log an opponent collecting a misplaced pass or a clearance. Between two logged touches by different teams, the key places the switch at the new team's first touch in the tracking (the sync's touch thresholds); the log still decides which team. Won duels with result "ballClaimed" count as a touch. Other unlogged touches still can't switch possession.
* **More dead time** in four matches (0.42–0.53, Metrica 0.33–0.43) and more, shorter possessions. DFL's own per-frame ball status shows the same, so it is the matches, not our rules. Our key agrees with DFL's own per-frame columns (`dfl_provider`) on 91–93% of frames, but that is partly by construction: since `answer-key-dfl-v1` our stoppage starts come from DFL's ball status. The independent check is which team has the ball on frames both call live (printed as "same team when both live"); `dfl_rule_map.md` says where the rest comes from.
* **Video check (optional).** `dfl_video_check.md` lists the build-up to all 19 goals with game-clock times, to time touch gaps against public highlights. It measures the frozen sync; nothing is tuned on it.

# PFF FC World Cup 2022 (paused; optional second sealed set)

Broadcast tracking and events for the 64 matches, free on request via PFF FC's form, delivered as a Google Drive folder (`data/external/pff/`, gitignored; 21 games downloaded, 2 complete so far).

**Licence:** no dataset-specific terms were found in the request form, the Drive folder or PFF's release posts. PFF's general Terms of Use (pff.com/terms) limit PFF data to personal, non-commercial use. Jordan decided on 2026-11-11 to proceed; no PFF data is committed or redistributed, only code and scores.

**Warning: never use PFF's smoothed fields.** `ballsSmoothed` (what kloppy reads) puts the ball exactly on the player PFF's analysts logged as on the ball, on 98–99% of those frames (DFL: 0.1%). That is the answer key leaking into the input. `external/pff_tracking.py` reads the raw `balls`, `homePlayers` and `awayPlayers`; with them, PFF's events lag the tracking by a steady 0.2–0.3 s.
